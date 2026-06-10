"""Test-prior feature ablation — the CORRECT way to ask "does this feature help?"

A recurring mistake in early feature experiments here
is judging features by accuracy/F1 at threshold 0.5 on a
training-prior (Pathogenic-dominant) validation split. That inflates everything
(a dummy scores ~0.9) and shifts the operating point so genuinely-useful features
can look like they "hurt" F1 (AUC up, F1 "down").

This harness fixes it. For each feature config it runs the deployed chain
(engineer -> leakage drop -> preprocess -> mRMR) and scores with the deployed
honest protocol over REPEATED outer CV: inner-OOF -> Platt -> threshold tuned to
the per-panel TEST prior, frozen, applied to the held-out fold -> F1*/MCC at the
test prior. Configs are compared PAIRED on identical folds.

Use it as a template: drop a new leakage-safe feature function into CONFIGS and
read the per-panel test-prior delta with an honest error bar — never t=0.5.

Configs tested here:
  no_aa          : raw columns only (engineer_aa_features OFF)
  deployed_AA    : + engineer_aa_features (the deployed base)            <- reference
  AA+blosum      : + engineer_blosum_extras (Almira's non-redundant BLOSUM flags)
(ACMG-for-pah is held ON in all three, matching deployment, to isolate the AA var.)

Run:  .venv/bin/python -u analysis/feature_ablation.py --panel all
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

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from lightgbm import LGBMClassifier

from config import SEED
from schema import normalize_sentinels, drop_all_nan_rows, to_categorical, partition_columns
from feature_engineering import (
    engineer_aa_features, engineer_acmg_features, engineer_blosum_extras,
)
from calibration import fit_platt, apply_platt
from utils import (
    detect_leaky_columns, optimize_threshold_f1_balanced,
    KNOWN_LEAKY, PANELS, TEST_PRIORS, ACMG_PANELS,
)
from run_pipeline import preprocess_data, select_features_mrmr, get_model_params
from fair_tuneoff import metrics_at_prior   # reuse the held-out test-prior bootstrap

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)

PANELS_ALL = ["general", "cancer", "pah", "cftr"]
BLOSUM_COLS = ["AA_blosum_abs", "AA_blosum_neg_flag", "AA_blosum_zero_flag", "AA_blosum_pos_flag"]

CONFIGS = [
    ("no_aa",       dict(use_aa=False, extra=None)),
    ("deployed_AA", dict(use_aa=True,  extra=None)),                    # reference
    ("AA+blosum",   dict(use_aa=True,  extra=engineer_blosum_extras)),
]
REFERENCE = "deployed_AA"


def prepare_config(panel, use_aa, extra):
    """Deployed chain with a feature toggle; returns the post-mRMR matrix."""
    df = pd.read_csv(PANELS[panel])
    df = normalize_sentinels(df)
    df = drop_all_nan_rows(df, verbose=False)
    df = to_categorical(df)
    if use_aa:
        df = engineer_aa_features(df)
    if panel in ACMG_PANELS:
        df = engineer_acmg_features(df)
    if extra is not None:
        df = extra(df)

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
    blosum_kept = [c for c in BLOSUM_COLS if c in feats]
    return X, y, cat_feats, feats, blosum_kept


def eval_fold(X, y, spw, cat_feats, train_idx, test_idx, prior, seed):
    """Single deployed-params LightGBM through one outer fold; F1*/MCC at test prior.
    Threshold + Platt fit on outer-train inner-OOF only (never the test fold)."""
    params = get_model_params(len(train_idx), spw)
    sub_y = y.iloc[train_idx].values

    def fit_es(tr_pos, es_pos, sd):
        m = LGBMClassifier(**params, random_state=sd)
        m.fit(X.iloc[tr_pos], y.iloc[tr_pos],
              eval_set=[(X.iloc[es_pos], y.iloc[es_pos])],
              callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
              categorical_feature=cat_feats or "auto")
        return m

    # inner-OOF on outer-train for calibration + threshold
    oof = np.zeros(len(train_idx))
    skf = StratifiedKFold(3, shuffle=True, random_state=seed)
    for k, (itr_rel, iva_rel) in enumerate(skf.split(train_idx, sub_y)):
        itr, iva = train_idx[itr_rel], train_idx[iva_rel]
        i2, ies = train_test_split(itr, test_size=0.15, random_state=seed + k,
                                   stratify=y.iloc[itr])
        oof[iva_rel] = fit_es(i2, ies, seed + k).predict_proba(X.iloc[iva])[:, 1]
    platt = fit_platt(oof, sub_y)
    thr, _, _ = optimize_threshold_f1_balanced(
        sub_y, apply_platt(oof, platt), target_prior=prior, n_bootstrap=150)

    # final model on outer-train, score outer-test at the test prior
    i2, ies = train_test_split(train_idx, test_size=0.15, random_state=seed + 99,
                               stratify=sub_y)
    model = fit_es(i2, ies, seed)
    probs = apply_platt(model.predict_proba(X.iloc[test_idx])[:, 1], platt)
    y_te = y.iloc[test_idx].values
    f1, mcc = metrics_at_prior(y_te, probs, thr, prior)
    try:
        auc = roc_auc_score(y_te, probs)
    except Exception:
        auc = float("nan")
    return f1, mcc, auc


def run_panel(panel, n_splits=5, n_repeats=2):
    print(f"\n{'#'*72}\n  FEATURE ABLATION — {panel.upper()}\n{'#'*72}", flush=True)
    prior = TEST_PRIORS.get(panel, 0.5)
    summ, rows = {}, []
    for cfg_name, cfg in CONFIGS:
        X, y, cat_feats, feats, blosum_kept = prepare_config(panel, cfg["use_aa"], cfg["extra"])
        spw = (y == 0).sum() / max((y == 1).sum(), 1)
        outer = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                        random_state=SEED)
        vals, t0 = [], time.time()
        for fold, (tr, te) in enumerate(outer.split(X, y)):
            f1, mcc, auc = eval_fold(X, y, spw, cat_feats, np.asarray(tr), np.asarray(te),
                                     prior, seed=SEED + fold)
            vals.append((f1, mcc, auc))
            rows.append({"panel": panel, "config": cfg_name, "fold": fold,
                         "f1": f1, "mcc": mcc, "auc": auc})
        arr = np.array(vals)
        summ[cfg_name] = {"f1": arr[:, 0], "f1_mean": arr[:, 0].mean(),
                          "f1_std": arr[:, 0].std(), "mcc_mean": arr[:, 1].mean(),
                          "auc_mean": np.nanmean(arr[:, 2]), "n_features": len(feats),
                          "blosum_kept": blosum_kept}
        print(f"  {cfg_name:14s} F1*={arr[:,0].mean():.4f}±{arr[:,0].std():.3f}  "
              f"MCC*={arr[:,1].mean():.4f}  AUC={np.nanmean(arr[:,2]):.4f}  "
              f"feats={len(feats)}  blosum_selected={blosum_kept}  "
              f"({round(time.time()-t0,1)}s)", flush=True)

    # paired deltas vs the deployed reference
    ref = summ[REFERENCE]["f1"]
    print(f"\n  paired Δ vs {REFERENCE} (test-prior F1*, same folds):", flush=True)
    for cfg_name, _ in CONFIGS:
        if cfg_name == REFERENCE:
            continue
        d = summ[cfg_name]["f1"] - ref
        se = d.std() / max(np.sqrt(len(d)), 1)
        verdict = ("HELPS" if d.mean() > 2 * se else
                   "HURTS" if d.mean() < -2 * se else "no effect (within noise)")
        print(f"    {cfg_name:14s} Δ={d.mean():+.4f} ± {se:.4f} SE  "
              f"(wins {int((d>0).sum())}/{len(d)})  -> {verdict}", flush=True)
    return summ, rows


def main():
    ap = argparse.ArgumentParser(description="Test-prior feature ablation")
    ap.add_argument("--panel", default="all", choices=PANELS_ALL + ["all"])
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()
    panels = PANELS_ALL if args.panel == "all" else [args.panel]
    all_rows = []
    for p in panels:
        _, rows = run_panel(p, n_repeats=args.repeats)
        all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    out = os.path.join(OUT_DIR, "feature_ablation.csv")
    df.to_csv(out, index=False)
    print(f"\n  Saved {out}", flush=True)


if __name__ == "__main__":
    main()
