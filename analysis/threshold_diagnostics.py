"""Threshold diagnostics for the train/test prior gap.

For each panel:
  1. Run the standard pipeline to get OOF predictions
  2. Measure F1 @ threshold=0.5 on raw OOF (current training prior, ~73-83% P)
  3. Measure F1 @ threshold=0.5 on OOF bootstrap-resampled to 50/50
  4. Measure 95% CI on the current F1-optimal threshold via bootstrap

Decision rule (see findings printed at bottom):
  - All three F1s close                  -> ship threshold=0.5, no calibration
  - F1@0.5 raw worse, F1@0.5 balanced OK -> Bayes prior correction works
  - Both bad                             -> Platt scaling needed
  - Wide CI on tuned threshold           -> tuned value is partly noise either way

Run: python analysis/threshold_diagnostics.py
"""
import os
import sys
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipelines"))
from config import SEED
from schema import normalize_sentinels, to_categorical, partition_columns
from utils import (
    detect_leaky_columns, optimize_threshold_f1, KNOWN_LEAKY, PANELS,
)
from run_pipeline import (
    preprocess_data, select_features_mrmr, get_model_params,
    train_multi_seed_ensemble,
)

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "output")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# OOF generation (mirrors run_pipeline.run_panel up to OOF probs)
# ---------------------------------------------------------------------------

def get_oof(panel_name):
    df = pd.read_csv(PANELS[panel_name])
    df = normalize_sentinels(df)
    df = to_categorical(df)
    _, num_cols, cat_cols, tgt = partition_columns(df)
    y_all = df[tgt]
    X_all = df[num_cols + cat_cols]

    # Leakage
    leaky = detect_leaky_columns(X_all, y_all, auc_threshold=0.99, verbose=False)
    all_leaky = sorted(set(leaky) | (KNOWN_LEAKY & set(X_all.columns)))
    X_all = X_all.drop(columns=all_leaky)

    # Split
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_all, y_all, test_size=0.15, random_state=SEED, stratify=y_all)
    for d in (X_tr, X_va):
        d.reset_index(drop=True, inplace=True)
    for s in (y_tr, y_va):
        s.reset_index(drop=True, inplace=True)

    # Preprocess
    X_tr_pp, X_va_pp, X_te_pp, _ = preprocess_data(
        X_tr.copy(), X_va.copy(), X_va.copy(), corr_threshold=0.99)
    X_tv = pd.concat([X_tr_pp, X_va_pp], ignore_index=True)
    y_tv = pd.concat([y_tr, y_va], ignore_index=True)

    # mRMR
    X_tr_pp, X_va_pp, _, _ = select_features_mrmr(
        X_tr_pp, y_tr, X_va_pp, X_va_pp)
    sel = list(X_tr_pp.columns)
    X_tv = X_tv[sel]

    # Ensemble
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    params = get_model_params(len(X_tv), spw)
    n_folds = 3 if len(X_tv) < 200 else 5
    _, oof, _, _ = train_multi_seed_ensemble(
        X_tv, y_tv, X_va_pp[sel], params, n_seeds=5, n_folds=n_folds)
    return y_tv.values, oof, spw


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def f1_at_threshold(y, probs, t):
    return f1_score(y, (probs >= t).astype(int), zero_division=0)


def bootstrap_balanced_f1(y, probs, t, n_iter=200, target_prior=0.5, seed=42):
    """Mean F1 at threshold t over n_iter bootstraps resampled to target_prior."""
    rng = np.random.RandomState(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n = len(y)
    n_pos = int(round(n * target_prior))
    n_neg = n - n_pos
    f1s = []
    for _ in range(n_iter):
        s_pos = rng.choice(pos_idx, n_pos, replace=True)
        s_neg = rng.choice(neg_idx, n_neg, replace=True)
        idx = np.concatenate([s_pos, s_neg])
        f1s.append(f1_score(y[idx], (probs[idx] >= t).astype(int), zero_division=0))
    return float(np.mean(f1s)), float(np.std(f1s))


def bootstrap_threshold_ci(y, probs, n_iter=200, target_prior=None, seed=42):
    """Distribution of F1-optimal thresholds over n_iter bootstrap resamples.

    If target_prior is None, resample with replacement preserving prior.
    Otherwise resample to balanced prior before threshold finding.
    """
    rng = np.random.RandomState(seed)
    n = len(y)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    thresholds = []
    f1s = []
    for _ in range(n_iter):
        if target_prior is None:
            idx = rng.choice(n, n, replace=True)
        else:
            n_pos = int(round(n * target_prior))
            n_neg = n - n_pos
            idx = np.concatenate([
                rng.choice(pos_idx, n_pos, replace=True),
                rng.choice(neg_idx, n_neg, replace=True),
            ])
        t, f1 = optimize_threshold_f1(pd.Series(y[idx]), probs[idx])
        thresholds.append(t)
        f1s.append(f1)
    return {
        "threshold_mean":  float(np.mean(thresholds)),
        "threshold_p2.5":  float(np.percentile(thresholds, 2.5)),
        "threshold_p97.5": float(np.percentile(thresholds, 97.5)),
        "threshold_std":   float(np.std(thresholds)),
        "f1_mean":         float(np.mean(f1s)),
        "f1_std":          float(np.std(f1s)),
    }


def diagnostics_for_panel(panel_name):
    y, probs, spw = get_oof(panel_name)

    prior_train = float(y.mean())

    # Current pipeline's choice: F1-opt threshold on raw OOF
    t_naive, f1_naive = optimize_threshold_f1(pd.Series(y), probs)

    # Method 1: F1 at threshold 0.5 on raw OOF
    f1_05_raw = f1_at_threshold(y, probs, 0.5)

    # Method 2: F1 at threshold 0.5 on balanced-resampled OOF
    f1_05_bal, f1_05_bal_std = bootstrap_balanced_f1(y, probs, 0.5)

    # Method 3: bootstrap CI on F1-optimal threshold (preserving train prior)
    ci_train_prior = bootstrap_threshold_ci(y, probs, target_prior=None)

    # Method 4: bootstrap CI on F1-optimal threshold (resampling to balanced)
    ci_balanced = bootstrap_threshold_ci(y, probs, target_prior=0.5)

    # Mean predicted prob — is the model well-calibrated to balanced prior?
    mean_pred = float(probs.mean())

    return {
        "panel": panel_name,
        "n":      int(len(y)),
        "n_pos":  int(y.sum()),
        "prior_train":  round(prior_train, 4),
        "spw":          round(spw, 4),
        "mean_pred":    round(mean_pred, 4),
        "t_naive":              round(t_naive, 4),
        "f1_naive":             round(f1_naive, 4),
        "f1_at_0.5_raw":        round(f1_05_raw, 4),
        "f1_at_0.5_balanced":   round(f1_05_bal, 4),
        "f1_at_0.5_balanced_std": round(f1_05_bal_std, 4),
        "ci_threshold_train_prior":  {
            "mean": round(ci_train_prior["threshold_mean"], 4),
            "p2.5": round(ci_train_prior["threshold_p2.5"], 4),
            "p97.5": round(ci_train_prior["threshold_p97.5"], 4),
            "std":  round(ci_train_prior["threshold_std"], 4),
            "f1_mean": round(ci_train_prior["f1_mean"], 4),
        },
        "ci_threshold_balanced":  {
            "mean": round(ci_balanced["threshold_mean"], 4),
            "p2.5": round(ci_balanced["threshold_p2.5"], 4),
            "p97.5": round(ci_balanced["threshold_p97.5"], 4),
            "std":  round(ci_balanced["threshold_std"], 4),
            "f1_mean": round(ci_balanced["f1_mean"], 4),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    panels = ["general", "cancer", "pah", "cftr"]
    results = {}
    for p in panels:
        print(f"\n--- {p} ---")
        results[p] = diagnostics_for_panel(p)
        r = results[p]
        print(f"  n={r['n']} (P-rate={r['prior_train']}, spw={r['spw']})")
        print(f"  mean OOF prob = {r['mean_pred']}   (if ~0.5 the model is calibrated to balanced)")
        print(f"  current method (F1-opt on raw OOF):    t={r['t_naive']}  F1={r['f1_naive']}")
        print(f"  F1 @ t=0.5 on raw OOF:                 F1={r['f1_at_0.5_raw']}")
        print(f"  F1 @ t=0.5 on balanced-resampled OOF:  F1={r['f1_at_0.5_balanced']} ± {r['f1_at_0.5_balanced_std']}")
        ci = r["ci_threshold_train_prior"]
        print(f"  bootstrap F1-opt threshold (train prior): mean={ci['mean']} 95%CI=[{ci['p2.5']}, {ci['p97.5']}] std={ci['std']}")
        ci = r["ci_threshold_balanced"]
        print(f"  bootstrap F1-opt threshold (balanced):    mean={ci['mean']} 95%CI=[{ci['p2.5']}, {ci['p97.5']}] std={ci['std']}")

    # Save
    out_path = os.path.join(OUT_DIR, "threshold_diagnostics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")

    # Decision summary
    print("\n" + "=" * 72)
    print("DECISION SUMMARY")
    print("=" * 72)
    print(f"{'Panel':<10} {'tuned F1':<10} {'F1@.5 raw':<11} {'F1@.5 bal':<11} {'thresh CI':<22}")
    for p in panels:
        r = results[p]
        ci = r["ci_threshold_train_prior"]
        ci_str = f"[{ci['p2.5']:.2f}, {ci['p97.5']:.2f}]"
        print(f"{p:<10} {r['f1_naive']:<10.4f} {r['f1_at_0.5_raw']:<11.4f} "
              f"{r['f1_at_0.5_balanced']:<11.4f} {ci_str:<22}")
    print()
    print("How to read this:")
    print("  - If tuned F1 ≈ F1@.5 raw ≈ F1@.5 balanced → ship threshold=0.5, no calibration")
    print("  - If F1@.5 raw < F1@.5 balanced → balanced-test scoring will benefit from threshold tuning at balanced prior")
    print("  - If 95% CI on tuned threshold is wide → the tuned value is partly bootstrap noise")


if __name__ == "__main__":
    main()
