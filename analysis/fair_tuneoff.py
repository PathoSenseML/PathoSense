"""Fair best-vs-best tune-off for the top architectures, per panel.

Built to answer one question honestly: on a given panel, which model family wins
when each is given a *fair* shot? It fixes every asymmetry in
`baseline_architectures.py` that made that sweep a poor basis for a submission swap:

  1. Per-model hyperparameter search (not LightGBM's params borrowed by everyone).
  2. Native categorical handling per library:
       LightGBM  -> categorical_feature
       XGBoost   -> enable_categorical
       CatBoost  -> cat_features
       RandomForest -> ordinal codes (its normal mode; no native cats)
  3. Early stopping for every booster (held-out ES set carved from train only).
  4. Repeated outer CV -> honest cross-split variance (not bootstrap-over-prior).
  5. Calibration (Platt) AND the decision threshold fit on TRAIN folds only, then
     frozen and applied to the held-out outer-test fold. No threshold peeking.

Protocol = tune-then-repeated-CV (two stage):
  STAGE 1 (tune, once per model): inner StratifiedKFold(3) over the full panel;
    each HP config scored by inner-OOF AUC (threshold-independent + cheap → no
    bootstrap in the hot loop). Pick the best config per model. Each inner fit
    early-stops on a slice carved from inner-train, so the inner-val stays clean.
  STAGE 2 (evaluate): outer = RepeatedStratifiedKFold(5 x 2). For each outer
    (train, test) fold and each candidate, using its STAGE-1 best config:
    - 3-fold inner-OOF on outer-train -> Platt + test-prior threshold (frozen).
    - refit on all outer-train (ES carved from it).
    - predict outer-test -> apply frozen Platt + threshold -> F1*/MCC at the test
      prior (bootstrap-resample outer-test to the prior); AUC on raw probs.
  Report per-candidate mean +/- std over the 10 outer folds, and PAIRED deltas
  (same folds for all candidates) vs the strongest, so small differences are
  testable rather than guessed.

  Tradeoff vs full nested CV: HP is tuned once on all data (mild, equal-for-all
  optimism) rather than re-tuned inside every outer fold, so HP-selection variance
  is NOT in the error bars. Everything that made the first sweep unfair — per-model
  tuning, native cats, early stopping, honest cross-split variance, and
  threshold/calibration fit on train only — IS fixed. Done for tractability; the
  --nested flag is left as a TODO if full nesting is ever wanted.

Shared preprocessing (preprocess_data + mRMR) is fit ONCE on the full panel and
reused for every candidate -- identical for all, so it cannot bias the *model*
choice (it is not one of the five fairness gaps). HP/calibration/threshold, which
were the unfair/optimistic parts, are all per-fold.

Run:  .venv/bin/python analysis/fair_tuneoff.py --panel pah
      .venv/bin/python analysis/fair_tuneoff.py --panel pah --quick
"""
import os
import sys
import time
import json
import argparse
import warnings
import itertools

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipelines"))

from sklearn.model_selection import (
    RepeatedStratifiedKFold, StratifiedKFold, train_test_split,
)
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier

import lightgbm as lgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from config import SEED
from schema import (
    normalize_sentinels, drop_all_nan_rows, to_categorical, partition_columns,
)
from feature_engineering import engineer_aa_features, engineer_acmg_features
from calibration import fit_platt, apply_platt
from utils import (
    detect_leaky_columns, optimize_threshold_f1_balanced,
    KNOWN_LEAKY, PANELS, TEST_PRIORS, ACMG_PANELS,
)
from run_pipeline import (
    preprocess_data, select_features_mrmr,
    train_multi_seed_ensemble, get_model_params,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)

# Current deployed (LightGBM multi-seed ensemble) honest F1* per panel, from
# `pipelines/run_pipeline.py --panel all` on 2026-06-07 — the incumbent to beat.
DEPLOYED_F1 = {"general": 0.5176, "cancer": 0.7127, "pah": 0.6446, "cftr": 0.6760}


# ---------------------------------------------------------------------------
# Shared data prep (fit once on the full panel; identical for all candidates)
# ---------------------------------------------------------------------------

def prepare_panel(panel_name):
    df = pd.read_csv(PANELS[panel_name])
    df = normalize_sentinels(df)
    df = drop_all_nan_rows(df, verbose=False)
    df = to_categorical(df)
    df = engineer_aa_features(df)
    if panel_name in ACMG_PANELS:
        df = engineer_acmg_features(df)

    _, num_cols, cat_cols, tgt = partition_columns(df)
    y = df[tgt].reset_index(drop=True)
    X = df[num_cols + cat_cols].reset_index(drop=True)

    leaky = detect_leaky_columns(X, y, auc_threshold=0.99, verbose=False)
    drop = sorted(set(leaky) | (KNOWN_LEAKY & set(X.columns)))
    if drop:
        X = X.drop(columns=drop)

    # Fit preprocessing + mRMR once on the whole panel (shared by all candidates).
    Xpp, _, _, _ = preprocess_data(X.copy(), X.copy(), X.copy(), corr_threshold=0.95)
    Xsel, _, _, _ = select_features_mrmr(Xpp, y, Xpp, Xpp)
    feats = list(Xsel.columns)
    X = Xpp[feats].reset_index(drop=True)

    cat_feats = [c for c in feats if isinstance(X[c].dtype, pd.CategoricalDtype)]
    return X, y, cat_feats, TEST_PRIORS.get(panel_name, 0.5)


def build_representations(X, cat_feats):
    """Three model-specific views of the same prepared matrix (same row order)."""
    # native: category dtype kept (LightGBM / XGBoost enable_categorical)
    X_native = X.copy()

    # catboost: categoricals as plain strings, missing -> 'NA' (CatBoost forbids NaN cats)
    X_cat = X.copy()
    for c in cat_feats:
        s = X[c].astype(object)
        X_cat[c] = s.where(s.notna(), "NA").astype(str)

    # randomforest: ordinal-encoded cats (RF has no native categorical support)
    X_ord = X.copy()
    if cat_feats:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                             encoded_missing_value=-2)
        X_ord[cat_feats] = enc.fit_transform(X_cat[cat_feats])
    return X_native, X_cat, X_ord


# ---------------------------------------------------------------------------
# Candidate models — each owns its representation, grid, fit (with ES), predict
# ---------------------------------------------------------------------------

def _grid(d):
    keys = list(d.keys())
    return [dict(zip(keys, vals)) for vals in itertools.product(*[d[k] for k in keys])]


class Candidate:
    def __init__(self, name, X_rep, y, spw, cat_feats, grid, quick=False):
        self.name = name
        self.X = X_rep
        self.y = y
        self.spw = spw
        self.cat_feats = cat_feats
        self.cat_idx = [X_rep.columns.get_loc(c) for c in cat_feats]
        self.grid = grid[: max(4, len(grid) // 2)] if quick else grid

    # ---- per-model fit (with early stopping where applicable) ----
    def _fit(self, hp, tr_idx, es_idx):
        Xtr, ytr = self.X.iloc[tr_idx], self.y.iloc[tr_idx]
        if self.name == "random_forest":
            m = RandomForestClassifier(class_weight="balanced", max_features="sqrt",
                                       n_jobs=4, random_state=SEED, **hp)
            m.fit(Xtr, ytr)
            return m
        Xes, yes = self.X.iloc[es_idx], self.y.iloc[es_idx]
        if self.name == "lightgbm":
            m = LGBMClassifier(n_estimators=2000, subsample=0.8, colsample_bytree=0.8,
                               reg_alpha=0.3, reg_lambda=1.5, scale_pos_weight=self.spw,
                               random_state=SEED, verbose=-1, **hp)
            m.fit(Xtr, ytr, eval_set=[(Xes, yes)],
                  callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
                  categorical_feature=self.cat_feats or "auto")
            return m
        if self.name == "xgboost":
            m = XGBClassifier(n_estimators=2000, subsample=0.8, colsample_bytree=0.8,
                              reg_lambda=1.5, scale_pos_weight=self.spw,
                              enable_categorical=True, tree_method="hist",
                              eval_metric="logloss", early_stopping_rounds=40,
                              random_state=SEED, verbosity=0, n_jobs=4, **hp)
            m.fit(Xtr, ytr, eval_set=[(Xes, yes)], verbose=False)
            return m
        if self.name == "catboost":
            m = CatBoostClassifier(iterations=1200, scale_pos_weight=self.spw,
                                   cat_features=self.cat_idx, random_seed=SEED,
                                   verbose=0, allow_writing_files=False,
                                   early_stopping_rounds=40, thread_count=4, **hp)
            m.fit(Xtr, ytr, eval_set=(Xes, yes))
            return m
        raise ValueError(self.name)

    def _proba(self, model, idx):
        return model.predict_proba(self.X.iloc[idx])[:, 1]

    def _carve_es(self, idx, seed):
        """Split a held-out early-stopping slice (15%) out of a train index."""
        if self.name == "random_forest":
            return idx, idx  # RF ignores ES
        tr, es = train_test_split(idx, test_size=0.15, random_state=seed,
                                  stratify=self.y.iloc[idx])
        return tr, es

    def inner_oof(self, hp, train_idx, prior, n_inner, seed):
        """Inner-OOF probabilities over outer-train for one HP config."""
        sub_y = self.y.iloc[train_idx].values
        oof = np.zeros(len(train_idx))
        skf = StratifiedKFold(n_inner, shuffle=True, random_state=seed)
        for k, (itr_rel, iva_rel) in enumerate(skf.split(train_idx, sub_y)):
            itr = train_idx[itr_rel]
            iva = train_idx[iva_rel]
            i2, ies = self._carve_es(itr, seed + k)
            model = self._fit(hp, i2, ies)
            oof[iva_rel] = self._proba(model, iva)
        return oof

    def tune(self, prior, n_inner, seed):
        """STAGE 1: pick the best HP config on the full panel by inner-OOF AUC."""
        all_idx = np.arange(len(self.y))
        y_all = self.y.values
        best = None
        for hp in self.grid:
            oof = self.inner_oof(hp, all_idx, prior, n_inner, seed)
            try:
                auc = roc_auc_score(y_all, oof)
            except Exception:
                auc = 0.0
            if best is None or auc > best[1]:
                best = (hp, auc)
        self.best_hp, self.tune_auc = best
        return best

    def eval_fold(self, train_idx, test_idx, prior, n_inner, seed):
        """STAGE 2: with the frozen best HP, calibrate+threshold on outer-train,
        refit, and score the held-out outer-test fold at the test prior."""
        sub_y = self.y.iloc[train_idx].values
        oof = self.inner_oof(self.best_hp, train_idx, prior, n_inner, seed)
        platt = fit_platt(oof, sub_y)
        thr, _, _ = optimize_threshold_f1_balanced(
            sub_y, apply_platt(oof, platt), target_prior=prior, n_bootstrap=150)
        tr, es = self._carve_es(train_idx, seed + 99)
        model = self._fit(self.best_hp, tr, es)
        probs = apply_platt(self._proba(model, test_idx), platt)
        y_te = self.y.iloc[test_idx].values
        f1, mcc = metrics_at_prior(y_te, probs, thr, prior)
        try:
            auc = roc_auc_score(y_te, probs)
        except Exception:
            auc = float("nan")
        return f1, mcc, auc, thr


# ---------------------------------------------------------------------------
# Holdout scoring at the test prior
# ---------------------------------------------------------------------------

def metrics_at_prior(y_true, probs, threshold, prior, n_boot=300, seed=SEED):
    """F1 and MCC on a held-out fold, bootstrap-resampled to the test prior."""
    y = np.asarray(y_true)
    preds = (np.asarray(probs) >= threshold).astype(int)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0, 0.0
    n = len(y)
    n_pos = max(1, int(round(n * prior)))
    n_neg = max(1, n - n_pos)
    rng = np.random.RandomState(seed)
    f1s, mccs = [], []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, n_pos, replace=True),
                              rng.choice(neg, n_neg, replace=True)])
        f1s.append(f1_score(y[idx], preds[idx], zero_division=0))
        mccs.append(matthews_corrcoef(y[idx], preds[idx]))
    return float(np.mean(f1s)), float(np.mean(mccs))


def eval_ensemble_fold(X_native, y, spw, train_idx, test_idx, prior):
    """The DEPLOYED model (multi-seed LightGBM ensemble) through one outer fold,
    so the incumbent gets the same honest CV treatment as the single candidates.
    Reuses run_pipeline.train_multi_seed_ensemble verbatim."""
    Xtr = X_native.iloc[train_idx].reset_index(drop=True)
    ytr = y.iloc[train_idx].reset_index(drop=True)
    Xte = X_native.iloc[test_idx].reset_index(drop=True)
    n_folds = 3 if len(Xtr) < 200 else 5
    params = get_model_params(len(Xtr), spw)
    _, oof, test_probs, _ = train_multi_seed_ensemble(
        Xtr, ytr, Xte, params, n_seeds=5, n_folds=n_folds)
    platt = fit_platt(oof, ytr.values)
    thr, _, _ = optimize_threshold_f1_balanced(
        ytr.values, apply_platt(oof, platt), target_prior=prior, n_bootstrap=150)
    probs = apply_platt(test_probs, platt)
    y_te = y.iloc[test_idx].values
    f1, mcc = metrics_at_prior(y_te, probs, thr, prior)
    try:
        auc = roc_auc_score(y_te, probs)
    except Exception:
        auc = float("nan")
    return f1, mcc, auc, thr


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

GRIDS = {
    "lightgbm": _grid({"num_leaves": [15, 31, 63], "learning_rate": [0.02, 0.05],
                       "min_child_samples": [5, 20]}),                       # 12
    "xgboost":  _grid({"max_depth": [3, 5, 7], "learning_rate": [0.02, 0.05],
                       "min_child_weight": [1, 5]}),                          # 12
    "catboost": _grid({"depth": [4, 6, 8], "learning_rate": [0.03, 0.06],
                       "l2_leaf_reg": [3.0, 9.0]}),                           # 12
    "random_forest": _grid({"n_estimators": [400, 800], "max_depth": [None, 8, 16],
                            "min_samples_leaf": [1, 3]}),                     # 12
}
REP_KEY = {"lightgbm": "native", "xgboost": "native",
           "catboost": "cat", "random_forest": "ord"}


def run_panel(panel_name, quick=False, n_repeats=2):
    print(f"\n{'#'*72}\n  FAIR TUNE-OFF — {panel_name.upper()}\n{'#'*72}")
    X, y, cat_feats, prior = prepare_panel(panel_name)
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    X_native, X_cat, X_ord = build_representations(X, cat_feats)
    reps = {"native": X_native, "cat": X_cat, "ord": X_ord}
    n_inner = 3
    n_splits = 5
    if quick:
        n_repeats = 1
    print(f"  n={len(X)}  features={len(X.columns)}  categorical={len(cat_feats)}  "
          f"test_prior={prior:.3f}  spw={spw:.3f}")
    print(f"  outer={n_splits}x{n_repeats}  inner={n_inner}-fold  "
          f"deployed F1*={DEPLOYED_F1.get(panel_name, float('nan')):.4f}  "
          f"{'[QUICK]' if quick else ''}")

    candidates = {name: Candidate(name, reps[REP_KEY[name]], y, spw, cat_feats,
                                  GRIDS[name], quick=quick)
                  for name in GRIDS}
    t0 = time.time()

    # ---- STAGE 1: tune each model once (inner CV on full panel, AUC) ----
    print(f"\n  [stage 1] tuning (inner {n_inner}-fold, AUC):", flush=True)
    for name, c in candidates.items():
        tt = time.time()
        hp, auc = c.tune(prior, n_inner, seed=SEED)
        print(f"    {name:14s} rep={REP_KEY[name]:6s} {len(c.grid)} configs -> "
              f"AUC={auc:.4f}  best={hp}  ({round(time.time()-tt,1)}s)", flush=True)

    # ---- STAGE 2: evaluate frozen configs over repeated outer folds ----
    print(f"\n  [stage 2] repeated outer CV:", flush=True)
    outer = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                    random_state=SEED)
    per_fold = {name: [] for name in candidates}  # list of (f1, mcc, auc)
    per_fold["lgbm_ensemble"] = []                 # the deployed incumbent, in-CV
    rows = []
    for fold, (tr_idx, te_idx) in enumerate(outer.split(X, y)):
        tr_idx = np.asarray(tr_idx)
        te_idx = np.asarray(te_idx)
        print(f"\n  --- outer fold {fold+1}/{n_splits*n_repeats} "
              f"(train={len(tr_idx)} test={len(te_idx)}) ---", flush=True)
        for name, c in candidates.items():
            ft = time.time()
            f1, mcc, auc, thr = c.eval_fold(tr_idx, te_idx, prior, n_inner,
                                            seed=SEED + fold)
            per_fold[name].append((f1, mcc, auc))
            rows.append({"panel": panel_name, "fold": fold, "model": name,
                         "f1": f1, "mcc": mcc, "auc": auc, "threshold": thr,
                         "best_hp": json.dumps(c.best_hp),
                         "seconds": round(time.time() - ft, 1)})
            print(f"    {name:14s} F1*={f1:.4f}  MCC*={mcc:.4f}  AUC={auc:.4f}  "
                  f"t={thr:.3f}  ({round(time.time()-ft,1)}s)", flush=True)
        # deployed multi-seed LightGBM ensemble through the same fold
        ft = time.time()
        f1, mcc, auc, thr = eval_ensemble_fold(X_native, y, spw, tr_idx, te_idx, prior)
        per_fold["lgbm_ensemble"].append((f1, mcc, auc))
        rows.append({"panel": panel_name, "fold": fold, "model": "lgbm_ensemble",
                     "f1": f1, "mcc": mcc, "auc": auc, "threshold": thr,
                     "best_hp": "deployed", "seconds": round(time.time() - ft, 1)})
        print(f"    {'lgbm_ensemble':14s} F1*={f1:.4f}  MCC*={mcc:.4f}  AUC={auc:.4f}  "
              f"t={thr:.3f}  ({round(time.time()-ft,1)}s)  [deployed]", flush=True)

    # ---- aggregate ----
    summ = {}
    for name, vals in per_fold.items():
        arr = np.array(vals)  # (folds, 3)
        summ[name] = {
            "f1_mean": float(arr[:, 0].mean()), "f1_std": float(arr[:, 0].std()),
            "mcc_mean": float(arr[:, 1].mean()), "mcc_std": float(arr[:, 1].std()),
            "auc_mean": float(np.nanmean(arr[:, 2])),
            "f1_folds": arr[:, 0].tolist(),
        }
    order = sorted(summ, key=lambda k: summ[k]["f1_mean"], reverse=True)
    champ = order[0]
    champ_folds = np.array(summ[champ]["f1_folds"])

    print(f"\n{'#'*72}\n  RESULT — {panel_name.upper()} "
          f"(mean +/- std F1* over {len(per_fold[champ])} outer folds)\n{'#'*72}")
    print(f"  deployed LightGBM ensemble F1* = {DEPLOYED_F1.get(panel_name, float('nan')):.4f} "
          f"(incumbent)\n")
    print(f"  {'model':14s} {'F1*':>8s} {'±':>6s} {'MCC*':>8s} {'AUC':>7s} "
          f"{'Δvs'+champ[:6]:>12s} {'winrate':>8s}")
    for name in order:
        s = summ[name]
        d = np.array(s["f1_folds"]) - champ_folds  # paired
        win = float((np.array(s["f1_folds"]) >= champ_folds).mean()) if name != champ else 1.0
        dstr = "  champ" if name == champ else f"{d.mean():+.4f}"
        print(f"  {name:14s} {s['f1_mean']:>8.4f} {s['f1_std']:>6.3f} "
              f"{s['mcc_mean']:>8.4f} {s['auc_mean']:>7.4f} {dstr:>12s} "
              f"{win*100:>7.0f}%")

    # paired delta champ vs runner-up with simple significance read
    if len(order) > 1:
        ru = order[1]
        d = champ_folds - np.array(summ[ru]["f1_folds"])
        se = d.std() / max(np.sqrt(len(d)), 1)
        print(f"\n  paired Δ ({champ} − {ru}) = {d.mean():+.4f} ± {se:.4f} (SE over folds); "
              f"{champ} wins {int((d>0).sum())}/{len(d)} folds")
        print(f"  vs deployed: {champ} {summ[champ]['f1_mean']:.4f} "
              f"{'beats' if summ[champ]['f1_mean']>DEPLOYED_F1.get(panel_name,9) else 'does NOT beat'} "
              f"incumbent {DEPLOYED_F1.get(panel_name, float('nan')):.4f}")

    df = pd.DataFrame(rows)
    csv = os.path.join(OUT_DIR, f"fair_tuneoff_{panel_name}.csv")
    df.to_csv(csv, index=False)
    print(f"\n  Saved {csv}   (total {round(time.time()-t0,1)}s)")
    return summ


def main():
    ap = argparse.ArgumentParser(description="Fair per-panel model tune-off (nested CV)")
    ap.add_argument("--panel", default="pah", choices=list(PANELS))
    ap.add_argument("--quick", action="store_true", help="1 repeat + halved grids")
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()
    run_panel(args.panel, quick=args.quick, n_repeats=args.repeats)


if __name__ == "__main__":
    main()
