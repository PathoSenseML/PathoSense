"""Baseline ML-architecture sweep on the current competition data.

Goal: a *controlled* comparison of model families on our current feature set,
scored the way we actually get scored — F1 at the per-panel TEST prior (the
şartname v2.0 "Klinik Stres Testi" makes the test set Benign-dominant), NOT the
inflated training-prior F1.

To isolate the *architecture* effect, every model runs through the SAME deployed
chain as run_pipeline.run_panel:
    load -> normalize_sentinels -> drop_all_nan_rows -> to_categorical
         -> engineer_aa_features (+ ACMG for pah) -> leakage drop
         -> 85/15 split -> preprocess_data (fit on train) -> mRMR select
Only the estimator changes. Each model is then evaluated identically:
    StratifiedKFold OOF probs -> Platt calibration -> threshold tuned to the
    per-panel TEST prior (optimize_threshold_f1_balanced) -> F1 + MCC at prior.

This mirrors run_panel's honest metric (`f1_calibrated_value`), so the LightGBM
rows here are directly comparable to the deployed system, and the Dummy row
gives the always-Pathogenic F1 floor *at the test prior* (much lower than at the
training prior — that is the whole point).

Scope note: uses the BASE feature set (AA engineering + ACMG-for-pah) + the
deployed preprocessing/mRMR. It does NOT add the per-panel optional tracks
(score-agreement, interactions) that run_pipeline's BEST_CONFIG enables for
cancer/cftr — those are model-agnostic add-ons; left out here for a clean
architecture comparison. The `lightgbm_ensemble` row IS the deployed model
family (multi-seed LGBM) on this base set.

Run:  .venv/bin/python analysis/baseline_architectures.py --panel all
      .venv/bin/python analysis/baseline_architectures.py --panel cftr --no-slow
"""
import os
import sys
import time
import json
import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipelines"))

from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import roc_auc_score

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier,
)
from lightgbm import LGBMClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

from config import SEED
from schema import (
    normalize_sentinels, drop_all_nan_rows, to_categorical, partition_columns,
)
from feature_engineering import engineer_aa_features, engineer_acmg_features
from calibration import fit_platt, apply_platt
from utils import (
    detect_leaky_columns, optimize_threshold_f1, optimize_threshold_f1_balanced,
    mcc_at_prior, KNOWN_LEAKY, PANELS, TEST_PRIORS, ACMG_PANELS,
)
from run_pipeline import (
    preprocess_data, select_features_mrmr,
    train_multi_seed_ensemble, get_model_params,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)

PANELS_ALL = ["general", "cancer", "pah", "cftr"]
SLOW_MODELS = {"linear_svm", "rbf_svm", "mlp"}


# ---------------------------------------------------------------------------
# Shared data prep (identical to run_pipeline.run_panel up through mRMR)
# ---------------------------------------------------------------------------

def prepare_panel(panel_name):
    """Return (X, y, prior, meta) — X is the post-mRMR feature frame with
    pd.Categorical columns intact; y is the aligned label Series (0..n-1 index).
    """
    df = pd.read_csv(PANELS[panel_name])
    df = normalize_sentinels(df)
    df = drop_all_nan_rows(df, verbose=False)
    df = to_categorical(df)
    df = engineer_aa_features(df)
    if panel_name in ACMG_PANELS:
        df = engineer_acmg_features(df)

    _, num_cols, cat_cols, tgt = partition_columns(df)
    feature_cols = num_cols + cat_cols
    y_all = df[tgt].reset_index(drop=True)
    X_all = df[feature_cols].reset_index(drop=True)

    # Leakage drop (same as deployed)
    leaky = detect_leaky_columns(X_all, y_all, auc_threshold=0.99, verbose=False)
    all_leaky = sorted(set(leaky) | (KNOWN_LEAKY & set(X_all.columns)))
    if all_leaky:
        X_all = X_all.drop(columns=all_leaky)

    # 85/15 split, fit preprocessing on train (deployed protocol)
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.15, random_state=SEED, stratify=y_all)
    for d in (X_train, X_val):
        d.reset_index(drop=True, inplace=True)
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)

    X_train_pp, X_val_pp, _, _ = preprocess_data(
        X_train.copy(), X_val.copy(), X_val.copy(), corr_threshold=0.95)
    X_train_val = pd.concat([X_train_pp, X_val_pp], ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], ignore_index=True)

    # mRMR selection (fit on train slice), then restrict X_train_val
    X_train_sel, _, _, fs_info = select_features_mrmr(
        X_train_pp, y_train, X_val_pp, X_val_pp)
    selected = list(X_train_sel.columns)
    X = X_train_val[selected].reset_index(drop=True)

    meta = {
        "n": int(len(X)),
        "n_features": len(selected),
        "n_pos": int(y_train_val.sum()),
        "n_neg": int((y_train_val == 0).sum()),
        "train_prior": round(float(y_train_val.mean()), 4),
        "test_prior": TEST_PRIORS.get(panel_name, 0.5),
    }
    return X, y_train_val.reset_index(drop=True), TEST_PRIORS.get(panel_name, 0.5), meta


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def split_col_types(X):
    cat_cols = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return cat_cols, num_cols


def to_string_cats(X, cat_cols):
    """Cast categorical columns to plain string, mapping missing -> 'NA'.

    Must be explicit: pandas 3.0's Categorical.astype(str) does not reliably
    stringify pd.NA, which would leak NaN into the encoders (only the legacy
    GradientBoostingClassifier trips on it — every other tree here tolerates NaN).
    """
    Xs = X.copy()
    for c in cat_cols:
        s = X[c].astype(object)
        Xs[c] = s.where(s.notna(), "NA").astype(str)
    return Xs


def make_tree_prep(cat_cols, num_cols):
    transformers = []
    if cat_cols:
        transformers.append((
            "cat",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                           encoded_missing_value=-2),
            cat_cols,
        ))
    if num_cols:
        transformers.append(("num", "passthrough", num_cols))
    return ColumnTransformer(transformers)


def make_linear_prep(cat_cols, num_cols):
    transformers = []
    if cat_cols:
        transformers.append((
            "cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols,
        ))
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    return ColumnTransformer(transformers)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def build_models(n, spw, include_slow=True):
    """Return list of (name, family, prep_kind, estimator_thunk).

    prep_kind: 'tree' (ordinal-encoded cats, no scaling) or
               'linear' (one-hot cats + standardized numerics).
    estimator_thunk returns a *fresh* estimator each call (no state reuse).
    """
    lgb_params = get_model_params(n, spw)

    models = [
        ("dummy_majority", "reference", "tree",
         lambda: DummyClassifier(strategy="constant", constant=1)),
        ("logreg", "linear", "linear",
         lambda: LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")),
        ("knn", "instance", "linear",
         lambda: KNeighborsClassifier(n_neighbors=min(15, max(3, n // 25)))),
        ("gaussian_nb", "probabilistic", "linear",
         lambda: GaussianNB()),
        ("decision_tree", "tree", "tree",
         lambda: DecisionTreeClassifier(class_weight="balanced", random_state=SEED)),
        ("random_forest", "bagging", "tree",
         lambda: RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                        n_jobs=4, random_state=SEED)),
        ("extra_trees", "bagging", "tree",
         lambda: ExtraTreesClassifier(n_estimators=400, class_weight="balanced",
                                      n_jobs=4, random_state=SEED)),
        ("gradient_boosting", "boosting", "tree",
         lambda: GradientBoostingClassifier(random_state=SEED)),
        ("hist_gb", "boosting", "tree",
         lambda: HistGradientBoostingClassifier(random_state=SEED)),
        ("lightgbm_single", "boosting", "tree",
         lambda: LGBMClassifier(**lgb_params, random_state=SEED)),
    ]

    # XGBoost / CatBoost get the same size-adaptive GBDT params as LightGBM and
    # the ordinal-encoded tree_prep (NOT their native categorical handling — kept
    # uniform with the other tree models; cats are few and low-cardinality here).
    if HAS_XGB:
        models.append((
            "xgboost", "boosting", "tree",
            lambda: XGBClassifier(
                n_estimators=lgb_params["n_estimators"],
                learning_rate=lgb_params["learning_rate"],
                max_depth=lgb_params["max_depth"],
                subsample=lgb_params["subsample"],
                colsample_bytree=lgb_params["colsample_bytree"],
                reg_alpha=lgb_params["reg_alpha"], reg_lambda=lgb_params["reg_lambda"],
                scale_pos_weight=spw, tree_method="hist", eval_metric="logloss",
                random_state=SEED, verbosity=0, n_jobs=4)))
    if HAS_CAT:
        models.append((
            "catboost", "boosting", "tree",
            lambda: CatBoostClassifier(
                iterations=500, learning_rate=0.03, depth=lgb_params["max_depth"],
                l2_leaf_reg=3.0, scale_pos_weight=spw, random_seed=SEED,
                verbose=0, allow_writing_files=False, thread_count=4)))

    if include_slow:
        models += [
            ("linear_svm", "svm", "linear",
             lambda: SVC(kernel="linear", probability=True,
                         class_weight="balanced", random_state=SEED)),
            ("rbf_svm", "svm", "linear",
             lambda: SVC(kernel="rbf", probability=True,
                         class_weight="balanced", random_state=SEED)),
            ("mlp", "neural", "linear",
             lambda: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000,
                                   early_stopping=True, random_state=SEED)),
        ]
    return models


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _safe_auc(y, probs):
    try:
        if np.unique(probs).size < 2:
            return float("nan")
        return round(float(roc_auc_score(y, probs)), 4)
    except Exception:
        return float("nan")


def get_oof(make_pipe, X, y, n_splits, seed=SEED):
    """StratifiedKFold out-of-fold class-1 probabilities for an sklearn pipeline."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        pipe = make_pipe()
        pipe.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = pipe.predict_proba(X.iloc[va])[:, 1]
    return oof


def score_oof(oof, y, prior):
    """Apply the deployed honest scoring chain to OOF probabilities.

    Platt-calibrate (when probs are non-constant), then tune the threshold to the
    per-panel test prior and report F1 + MCC at that prior. Returns a dict.
    """
    y_arr = y.values if hasattr(y, "values") else np.asarray(y)
    auc = _safe_auc(y_arr, oof)
    raw_t, raw_f1 = optimize_threshold_f1(y_arr, oof)  # training-prior F1 (inflated)

    if np.unique(oof).size > 1:
        platt = fit_platt(oof, y_arr)
        oof_cal = apply_platt(oof, platt)
    else:
        oof_cal = oof  # constant probs (dummy) — calibration is a no-op

    t_bal, f1_bal, f1_std = optimize_threshold_f1_balanced(
        y_arr, oof_cal, target_prior=prior, n_bootstrap=200)
    mcc, mcc_std = mcc_at_prior(y_arr, oof_cal, t_bal, target_prior=prior, n_bootstrap=200)

    return {
        "auc": auc,
        "raw_f1": raw_f1,                 # F1 at training prior (inflated reference)
        "test_prior_f1": f1_bal,          # PRIMARY: honest F1 at test prior
        "test_prior_f1_std": f1_std,
        "mcc": mcc,
        "mcc_std": mcc_std,
        "threshold": t_bal,
    }


def evaluate_panel(panel_name, include_slow=True):
    print(f"\n{'='*72}\n  {panel_name.upper()}\n{'='*72}")
    X, y, prior, meta = prepare_panel(panel_name)
    n = meta["n"]
    spw = meta["n_neg"] / max(meta["n_pos"], 1)
    n_splits = 3 if n < 200 else 5
    print(f"  n={n}  features={meta['n_features']}  train_prior={meta['train_prior']}  "
          f"test_prior={prior:.3f}  spw={spw:.3f}  cv={n_splits}-fold")

    cat_cols, num_cols = split_col_types(X)
    X_str = to_string_cats(X, cat_cols)  # for sklearn encoders
    print(f"  categorical={len(cat_cols)}  numeric={len(num_cols)}")

    rows = []

    # --- standard sklearn / lightgbm architectures ---
    for name, family, prep_kind, thunk in build_models(n, spw, include_slow):
        t0 = time.time()
        try:
            prep = (make_tree_prep(cat_cols, num_cols) if prep_kind == "tree"
                    else make_linear_prep(cat_cols, num_cols))
            make_pipe = (lambda _prep=prep, _thunk=thunk:
                         Pipeline([("prep", _prep), ("clf", _thunk())]))
            oof = get_oof(make_pipe, X_str, y, n_splits)
            sc = score_oof(oof, y, prior)
            sc.update({"panel": panel_name, "model": name, "family": family,
                       "seconds": round(time.time() - t0, 1),
                       "n": n, "n_features": meta["n_features"]})
            rows.append(sc)
            print(f"  {name:18s} F1*={sc['test_prior_f1']:.4f}±{sc['test_prior_f1_std']:.3f}  "
                  f"MCC*={sc['mcc']:.4f}  AUC={sc['auc'] if not np.isnan(sc['auc']) else float('nan'):.4f}  "
                  f"rawF1={sc['raw_f1']:.4f}  t={sc['threshold']:.3f}  ({sc['seconds']}s)")
        except Exception as e:
            print(f"  {name:18s} FAILED: {type(e).__name__}: {e}")

    # --- deployed multi-seed LightGBM ensemble (native categorical) ---
    t0 = time.time()
    try:
        params = get_model_params(n, spw)
        _, ens_oof, _, _ = train_multi_seed_ensemble(
            X, y, X, params, n_seeds=5, n_folds=n_splits)
        sc = score_oof(ens_oof, y, prior)
        sc.update({"panel": panel_name, "model": "lightgbm_ensemble", "family": "boosting(deployed)",
                   "seconds": round(time.time() - t0, 1),
                   "n": n, "n_features": meta["n_features"]})
        rows.append(sc)
        print(f"  {'lightgbm_ensemble':18s} F1*={sc['test_prior_f1']:.4f}±{sc['test_prior_f1_std']:.3f}  "
              f"MCC*={sc['mcc']:.4f}  AUC={sc['auc']:.4f}  rawF1={sc['raw_f1']:.4f}  "
              f"t={sc['threshold']:.3f}  ({sc['seconds']}s)")
    except Exception as e:
        print(f"  {'lightgbm_ensemble':18s} FAILED: {type(e).__name__}: {e}")

    return rows, meta


def main():
    ap = argparse.ArgumentParser(description="Baseline ML-architecture sweep")
    ap.add_argument("--panel", default="all",
                    choices=PANELS_ALL + ["all"])
    ap.add_argument("--no-slow", action="store_true",
                    help="skip SVMs and MLP (faster iteration)")
    args = ap.parse_args()

    panels = PANELS_ALL if args.panel == "all" else [args.panel]
    all_rows, metas = [], {}
    t_start = time.time()
    for pn in panels:
        rows, meta = evaluate_panel(pn, include_slow=not args.no_slow)
        all_rows.extend(rows)
        metas[pn] = meta

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(OUT_DIR, "baseline_architectures.csv")
    df.to_csv(csv_path, index=False)

    # Per-panel ranked tables (by honest test-prior F1)
    print(f"\n{'#'*72}\n  SUMMARY — F1* = honest F1 at the per-panel TEST prior "
          f"(Benign-dominant)\n{'#'*72}")
    for pn in panels:
        sub = df[df.panel == pn].sort_values("test_prior_f1", ascending=False)
        m = metas[pn]
        print(f"\n  {pn.upper()}  (n={m['n']}, feats={m['n_features']}, "
              f"test_prior={m['test_prior']:.3f}, always-P floor≈"
              f"{2*m['test_prior']/(1+m['test_prior']):.3f})")
        print(f"    {'model':20s} {'F1*':>8s} {'±':>6s} {'MCC*':>8s} {'AUC':>7s} "
              f"{'rawF1':>7s} {'t':>6s}")
        for _, r in sub.iterrows():
            auc = r["auc"]
            print(f"    {r['model']:20s} {r['test_prior_f1']:>8.4f} "
                  f"{r['test_prior_f1_std']:>6.3f} {r['mcc']:>8.4f} "
                  f"{auc:>7.4f} {r['raw_f1']:>7.4f} {r['threshold']:>6.3f}")

    # Combined pivot: model x panel -> F1*
    pivot = df.pivot_table(index="model", columns="panel",
                           values="test_prior_f1", aggfunc="first")
    pivot = pivot.reindex(columns=[p for p in PANELS_ALL if p in pivot.columns])
    pivot["mean"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("mean", ascending=False)
    pivot_path = os.path.join(OUT_DIR, "baseline_architectures_pivot.csv")
    pivot.to_csv(pivot_path)
    print(f"\n{'#'*72}\n  PIVOT — test-prior F1* (rows=model, cols=panel), "
          f"sorted by mean\n{'#'*72}")
    print(pivot.round(4).to_string())

    print(f"\n  Saved: {csv_path}")
    print(f"         {pivot_path}")
    print(f"  Total wall time: {round(time.time()-t_start,1)}s")


if __name__ == "__main__":
    main()
