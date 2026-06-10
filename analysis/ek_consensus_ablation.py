"""Test-prior F1* ablation for the PROPOSED conservation feature.

Compares, through the DEPLOYED chain at the per-panel TEST prior with repeated CV:
  deployed              : the current feature set (AA + interaction + ACMG-for-pah)
  deployed+cons_consensus : + two fold-fitted, leakage-safe features:
        EKX_cons_consensus = mean of frozen-train rank ECDF of (EK_7,EK_9,EK_2)
        EKX_cons_x_disrupt = EKX_cons_consensus * BLOSUM-disruption

Rank ECDF is fit on the OUTER-TRAIN rows ONLY and applied to the held-out fold
(no leak). Paired deltas on identical folds.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipelines"))
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from lightgbm import LGBMClassifier

from config import SEED
from schema import normalize_sentinels, drop_all_nan_rows, to_categorical, partition_columns
from feature_engineering import engineer_aa_features, engineer_acmg_features, engineer_interaction_features
from calibration import fit_platt, apply_platt
from utils import (detect_leaky_columns, optimize_threshold_f1_balanced,
                   KNOWN_LEAKY, PANELS, TEST_PRIORS, ACMG_PANELS)
from run_pipeline import preprocess_data, select_features_mrmr, get_model_params
from fair_tuneoff import metrics_at_prior

CORE = ("EK_7", "EK_9", "EK_2")


def prepare(panel):
    df = pd.read_csv(PANELS[panel])
    df = normalize_sentinels(df); df = drop_all_nan_rows(df, verbose=False)
    df = to_categorical(df)
    df = engineer_aa_features(df)
    if panel in ACMG_PANELS:
        df = engineer_acmg_features(df)
    df = engineer_interaction_features(df)
    # raw conservation + disruption kept aside for the fold-fitted proposed feature
    raw = pd.DataFrame(index=range(len(df)))
    for c in CORE:
        raw[c] = df[c].astype(float).values if c in df.columns else np.nan
    blos = df["AA_blosum62"].astype(float).values
    raw["disrupt"] = np.clip((4.0 - blos) / 15.0, 0, 1)

    _, num_cols, cat_cols, tgt = partition_columns(df)
    y = df[tgt].reset_index(drop=True)
    X = df[num_cols + cat_cols].reset_index(drop=True)
    leaky = detect_leaky_columns(X, y, auc_threshold=0.99, verbose=False)
    drop = sorted(set(leaky) | (KNOWN_LEAKY & set(X.columns)))
    if drop:
        X = X.drop(columns=drop)
    Xpp, _, _, _ = preprocess_data(X.copy(), X.copy(), X.copy(), corr_threshold=0.95)
    Xsel, _, _, _ = select_features_mrmr(Xpp, y, Xpp, Xpp)
    feats = list(Xsel.columns)
    X = Xpp[feats].reset_index(drop=True)
    cat_feats = [c for c in feats if isinstance(X[c].dtype, pd.CategoricalDtype)]
    return X, y, raw, cat_feats


def add_consensus(X, raw, train_idx):
    """Fold-fitted, leakage-safe rank-consensus conservation features."""
    cons_cols = []
    for c in CORE:
        if c not in raw.columns:
            continue
        tr_vals = raw[c].values[train_idx]
        sv = np.sort(tr_vals[~np.isnan(tr_vals)])
        v = raw[c].values
        pct = np.searchsorted(sv, v, side="right") / max(len(sv), 1)
        pct = pct.astype(float); pct[np.isnan(v)] = np.nan
        cons_cols.append(pct)
    cons = np.nanmean(np.vstack(cons_cols), axis=0)
    Xn = X.copy()
    Xn["EKX_cons_consensus"] = cons
    Xn["EKX_cons_x_disrupt"] = cons * raw["disrupt"].values
    return Xn


def eval_fold(X, y, spw, cat_feats, tr, te, prior, seed):
    params = get_model_params(len(tr), spw)
    sub_y = y.iloc[tr].values

    def fit_es(a, b, sd):
        m = LGBMClassifier(**params, random_state=sd)
        m.fit(X.iloc[a], y.iloc[a], eval_set=[(X.iloc[b], y.iloc[b])],
              callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
              categorical_feature=cat_feats or "auto")
        return m

    oof = np.zeros(len(tr))
    skf = StratifiedKFold(3, shuffle=True, random_state=seed)
    for k, (itr, iva) in enumerate(skf.split(tr, sub_y)):
        a, b = tr[itr], tr[iva]
        a2, es = train_test_split(a, test_size=0.15, random_state=seed + k, stratify=y.iloc[a])
        oof[iva] = fit_es(a2, es, seed + k).predict_proba(X.iloc[b])[:, 1]
    platt = fit_platt(oof, sub_y)
    thr, _, _ = optimize_threshold_f1_balanced(sub_y, apply_platt(oof, platt),
                                               target_prior=prior, n_bootstrap=150)
    a2, es = train_test_split(tr, test_size=0.15, random_state=seed + 99, stratify=sub_y)
    model = fit_es(a2, es, seed)
    probs = apply_platt(model.predict_proba(X.iloc[te])[:, 1], platt)
    f1, mcc = metrics_at_prior(y.iloc[te].values, probs, thr, prior)
    return f1, mcc


def run(panel, n_repeats=3):
    prior = TEST_PRIORS.get(panel, 0.5)
    X, y, raw, cat_feats = prepare(panel)
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED)
    dep, prop = [], []
    t0 = time.time()
    print(f"[{panel}] prepared X shape={X.shape}, starting {5*n_repeats} folds...", flush=True)
    for fold, (tr, te) in enumerate(outer.split(X, y)):
        tr, te = np.asarray(tr), np.asarray(te)
        f1d, _ = eval_fold(X, y, spw, cat_feats, tr, te, prior, SEED + fold)
        Xn = add_consensus(X, raw, tr)
        f1p, _ = eval_fold(Xn, y, spw, cat_feats, tr, te, prior, SEED + fold)
        dep.append(f1d); prop.append(f1p)
        print(f"  fold {fold}: dep={f1d:.3f} prop={f1p:.3f} ({time.time()-t0:.0f}s)", flush=True)
    dep, prop = np.array(dep), np.array(prop)
    d = prop - dep
    se = d.std() / max(np.sqrt(len(d)), 1)
    verdict = ("HELPS" if d.mean() > 2 * se else "HURTS" if d.mean() < -2 * se else "noise")
    print(f"\n{panel.upper()} (prior={prior}, {len(dep)} folds, {time.time()-t0:.0f}s)")
    print(f"  deployed            F1*={dep.mean():.4f}±{dep.std():.3f}")
    print(f"  +cons_consensus     F1*={prop.mean():.4f}±{prop.std():.3f}")
    print(f"  paired Δ={d.mean():+.4f} ± {se:.4f} SE  (wins {int((d>0).sum())}/{len(d)})  -> {verdict}")
    return dep, prop


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", default="cancer,pah")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    for p in args.panels.split(","):
        run(p.strip(), n_repeats=args.repeats)
