"""Augmentation feasibility gate (step b): does a model trained on ONLY the
reconstructable 'shared derived-feature space' retain the full-column model's
signal? If derived << full, augmentation in that space can't win -> abandon.

Derived space = features computable identically from contest columns AND from
gnomAD/dbNSFP for external variants:
  AA_* (Grantham/BLOSUM/physchem from the AA change), acmg_* (max gnomAD AF +
  flags, phastCons mean, conserved), EK_1/EK_2 (GERP NR/RS), EK_7/EK_9 (phyloP),
  AL_185 (allele number).
Compared against FULL = all numeric columns, in one identical CV harness.
"""
import sys, warnings
sys.path.insert(0, "pipelines"); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from schema import normalize_sentinels, drop_all_nan_rows
from feature_engineering import engineer_aa_features, engineer_acmg_features
from calibration import fit_platt, apply_platt
from utils import optimize_threshold_f1_balanced, TEST_PRIORS, PANELS

SEEDS = (0, 1, 2)

def oof(X, y, n_folds):
    spw = (len(y) - y.sum()) / max(y.sum(), 1)
    p = np.zeros(len(y))
    for sd in SEEDS:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=sd)
        for tr, va in skf.split(X, y):
            m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31,
                                   subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                                   reg_lambda=1.0, scale_pos_weight=spw, random_state=sd,
                                   n_jobs=-1, verbose=-1)
            m.fit(X.iloc[tr], y.iloc[tr])
            p[va] += m.predict_proba(X.iloc[va])[:, 1]
    return p / len(SEEDS)

def honest(y, p, prior):
    pc = apply_platt(p, fit_platt(p, y.values))
    t, f1, _ = optimize_threshold_f1_balanced(y.values, pc, target_prior=prior, n_bootstrap=200)
    return roc_auc_score(y, p), f1, t

print(f"{'panel':8} {'FULL AUC/F1*':>16} {'DERIVED AUC/F1*':>18} {'dF1':>7}  feat(full/deriv)")
for panel, path in PANELS.items():
    df = pd.read_csv(path, low_memory=False)
    df = normalize_sentinels(df); df = drop_all_nan_rows(df, verbose=False)
    df = engineer_aa_features(df); df = engineer_acmg_features(df)
    y = df["Label"].astype(int).reset_index(drop=True)
    num = df.select_dtypes("number").drop(columns=["Label"], errors="ignore").reset_index(drop=True)
    full = num.fillna(num.median())
    dcols = [c for c in num.columns if c.startswith("AA_") or c.startswith("acmg_")]
    dcols += [c for c in ["EK_1", "EK_2", "EK_7", "EK_9", "AL_185"] if c in num.columns]
    derived = num[dcols].fillna(num[dcols].median())
    nf = 5 if len(y) >= 200 else 3
    prior = TEST_PRIORS[panel]
    af, ff, _ = honest(y, oof(full, y, nf), prior)
    ad, fd, _ = honest(y, oof(derived, y, nf), prior)
    print(f"{panel:8}   {af:.3f}/{ff:.3f}      {ad:.3f}/{fd:.3f}       {fd-ff:+.3f}   {full.shape[1]}/{derived.shape[1]}")
