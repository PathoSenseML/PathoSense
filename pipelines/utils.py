"""Shared utility functions for the PathoSense pipeline."""
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score,
    precision_score, recall_score, brier_score_loss, matthews_corrcoef
)

from config import BASE_DIR

# ── Panel Definitions ──────────────────────────────────────────────
PANELS = {
    "general": os.path.join(BASE_DIR, "data", "panels", "YARISMA_TRAIN_MASTER.csv"),
    "cancer": os.path.join(BASE_DIR, "data", "panels", "YARISMA_TRAIN_KANSER.csv"),
    "pah": os.path.join(BASE_DIR, "data", "panels", "YARISMA_TRAIN_PAH.csv"),
    "cftr": os.path.join(BASE_DIR, "data", "panels", "YARISMA_TRAIN_CFTR.csv"),
}

# Competition has not released a test set; external-test branch in run_pipeline.py
# stays inert as long as TEST_PANELS is empty.
TEST_PANELS: dict = {}

# Per-panel PATHOGENIC fraction of the *test* set, from şartname v2.0 §3.2
# ("Klinik Stres Testi": training is Pathogenic-dominant, test is Benign-dominant).
# Test counts (≈): general 500P/3000B, cancer 100P/500B, pah 100P/250B, cftr 20P/100B.
# Used as target_prior for threshold selection so F1 is optimised for the
# distribution we are actually scored on — NOT the training prior, NOT 50/50.
TEST_PRIORS = {
    "general": 500 / 3500,   # 0.143
    "cancer":  100 / 600,    # 0.167
    "pah":     100 / 350,    # 0.286
    "cftr":     20 / 120,    # 0.167
}

# Panels where the ACMG allele-frequency feature (engineer_acmg_features) is enabled.
# Empirically it helps only PAH (small, frequency-driven: F1 +0.044, AUC +0.017) and
# is neutral/harmful elsewhere. Each panel is a separate model, so this is per-panel config.
ACMG_PANELS = {"pah"}

# Known leaky columns (will also be auto-detected)
KNOWN_LEAKY = {'AF', 'nSNP', 'Pos1', 'Positions'}


# ── Auto Leakage Detection ────────────────────────────────────────

def detect_leaky_columns(X: pd.DataFrame, y: pd.Series,
                         auc_threshold: float = 0.99,
                         min_value_count: int = 10,
                         verbose: bool = True) -> List[str]:
    """
    Flag columns whose values let you reconstruct the label.

    Numeric columns: dropped if single-feature AUC > auc_threshold, if their
    NaN mask perfectly predicts one class, or if they're effectively constant.

    Categorical columns: dropped if any single category value with
    >= min_value_count samples is >=99% one class. This catches the
    categorical equivalent of an AUC-style leak (e.g. one CAT value always
    means Pathogenic).
    """
    leaky = []
    for col in X.columns:
        vals = X[col]
        is_cat = isinstance(vals.dtype, pd.CategoricalDtype)

        # Missingness-pattern leak (applies to both numeric and categorical).
        # Flag only if the missing rows are PURE one class AND that purity is
        # implausible under the panel's base rate. A base-rate-naive nunique==1
        # test fires on small all-one-class runs that occur by chance (e.g. 11/11
        # of a 70%-majority class: p=0.7^11≈0.02), wrongly dropping informative
        # sparse columns. The true AF-missingness leak has n_missing in the
        # hundreds, all one class -> chance≈0, so it is still caught.
        n_missing = int(vals.isna().sum())
        if n_missing > min_value_count:
            missing_mask = vals.isna()
            ym = y[missing_mask]
            if ym.nunique() == 1:
                cls = int(ym.iloc[0])
                base = float((y == cls).mean())     # P(this class) in the panel
                chance = base ** n_missing           # P(all missing are this class) under base rate
                if chance < 1e-3:
                    if verbose:
                        print(f"    LEAKY (missingness): {col} — all {n_missing} missing "
                              f"samples are class {cls} (chance={chance:.1e})")
                    leaky.append(col)
                    continue

        if is_cat:
            # Per-value P-rate scan
            valid_mask = vals.notna()
            if valid_mask.sum() < 20:
                continue
            sub = pd.DataFrame({"v": vals[valid_mask].astype(object), "y": y[valid_mask].values})
            for value, group in sub.groupby("v"):
                if len(group) < min_value_count:
                    continue
                p_rate = group["y"].mean()
                if p_rate >= 0.99 or p_rate <= 0.01:
                    if verbose:
                        print(f"    LEAKY (cat value): {col}=={value!r} "
                              f"-> P-rate={p_rate:.3f} on {len(group)} samples")
                    leaky.append(col)
                    break
            continue

        # ----- numeric path -----
        valid_mask = vals.notna()
        if valid_mask.sum() < 20 or y[valid_mask].nunique() < 2:
            continue

        vals_clean = vals[valid_mask].values
        y_clean = y[valid_mask].values

        try:
            auc = roc_auc_score(y_clean, vals_clean)
            auc = max(auc, 1 - auc)  # handle inversely correlated
            if auc > auc_threshold:
                if verbose:
                    print(f"    LEAKY (AUC={auc:.4f}): {col}")
                leaky.append(col)
        except Exception:
            pass

        if vals.std() < 1e-8:
            if verbose:
                print(f"    CONSTANT: {col} (std={vals.std():.2e})")
            leaky.append(col)

    return list(set(leaky))


# ── Threshold Optimization ────────────────────────────────────────

def optimize_threshold_f1(y_true: pd.Series, probs: np.ndarray,
                          step: float = 0.005) -> Tuple[float, float]:
    """
    Maximize standard F1 score via full sweep on the data as-given.
    Competition metric is F1 = 2*TP / (2*TP + FP + FN).
    """
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.96, step):
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return round(best_t, 4), round(best_f1, 4)


def optimize_threshold_f1_balanced(
    y_true: np.ndarray,
    probs: np.ndarray,
    target_prior: float = 0.5,
    n_bootstrap: int = 200,
    step: float = 0.01,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Find the threshold that maximizes mean F1 on bootstraps resampled to target_prior.

    Use this instead of optimize_threshold_f1 when the training data has a
    different class prior from the test data. For TEKNOFEST 2026 v2.0 the test
    sets are deliberately Benign-dominant ("Klinik Stres Testi"), so callers pass
    the per-panel test prior from utils.TEST_PRIORS (NOT 0.5 — that was the stale
    v1.x assumption). The 0.5 default is kept only as a safe fallback.

    Returns:
        threshold     : F1-optimal threshold under the target prior
        mean_f1       : mean F1 at that threshold across n_bootstrap resamples
        f1_std        : std of F1 across resamples (uncertainty estimate)
    """
    y_arr = np.asarray(y_true)
    rng = np.random.RandomState(seed)
    pos_idx = np.where(y_arr == 1)[0]
    neg_idx = np.where(y_arr == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return 0.5, 0.0, 0.0

    n = len(y_arr)
    n_pos = max(1, int(round(n * target_prior)))
    n_neg = max(1, n - n_pos)

    # Generate the bootstrap resamples once
    resamples = []
    for _ in range(n_bootstrap):
        s_pos = rng.choice(pos_idx, n_pos, replace=True)
        s_neg = rng.choice(neg_idx, n_neg, replace=True)
        idx = np.concatenate([s_pos, s_neg])
        resamples.append(idx)

    thresholds = np.arange(0.05, 0.96, step)
    best_t, best_mean_f1, best_std = 0.5, 0.0, 0.0
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1s = [f1_score(y_arr[idx], preds[idx], zero_division=0) for idx in resamples]
        mean_f1 = float(np.mean(f1s))
        if mean_f1 > best_mean_f1:
            best_mean_f1 = mean_f1
            best_t = float(t)
            best_std = float(np.std(f1s))
    return round(best_t, 4), round(best_mean_f1, 4), round(best_std, 4)


# ── Metrics ───────────────────────────────────────────────────────

def compute_metrics(y_true: pd.Series, probs: np.ndarray, threshold: float = 0.5) -> Dict:
    """Compute comprehensive metrics at a given threshold."""
    preds = (probs >= threshold).astype(int)
    return {
        "f1": round(f1_score(y_true, preds, zero_division=0), 4),
        "f1_macro": round(f1_score(y_true, preds, average='macro', zero_division=0), 4),
        "mcc": round(matthews_corrcoef(y_true, preds), 4),
        "accuracy": round(accuracy_score(y_true, preds), 4),
        "roc_auc": round(roc_auc_score(y_true, probs), 4),
        "precision": round(precision_score(y_true, preds, zero_division=0), 4),
        "recall": round(recall_score(y_true, preds, zero_division=0), 4),
        "brier": round(brier_score_loss(y_true, probs), 6),
        "threshold": threshold,
    }


def mcc_at_prior(y_true, probs, threshold: float, target_prior: float = 0.5,
                 n_bootstrap: int = 200, seed: int = 42) -> Tuple[float, float]:
    """Mean Matthews correlation coefficient at a FIXED threshold, on bootstraps
    resampled to target_prior. Reported alongside F1 (v2.0 §7.3 permits MCC, which
    is more robust to the benign-dominant test imbalance). Returns (mean, std)."""
    y_arr = np.asarray(y_true)
    preds = (np.asarray(probs) >= threshold).astype(int)
    pos_idx = np.where(y_arr == 1)[0]
    neg_idx = np.where(y_arr == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return 0.0, 0.0
    n = len(y_arr)
    n_pos = max(1, int(round(n * target_prior)))
    n_neg = max(1, n - n_pos)
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_bootstrap):
        idx = np.concatenate([rng.choice(pos_idx, n_pos, replace=True),
                              rng.choice(neg_idx, n_neg, replace=True)])
        vals.append(matthews_corrcoef(y_arr[idx], preds[idx]))
    return round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)


# ── Subsampling ──────────────────────────────────────────────────

def subsample_random(df, n_p, n_b, seed):
    """Subsample a panel to competition sizes (80/20 P/B)."""
    y = df["Label"]
    patho_idx = y[y == 1].index.tolist()
    benign_idx = y[y == 0].index.tolist()
    rng = np.random.RandomState(seed)
    rng.shuffle(patho_idx)
    rng.shuffle(benign_idx)
    actual_p = min(n_p, len(patho_idx))
    actual_b = min(n_b, len(benign_idx))
    sel = patho_idx[:actual_p] + benign_idx[:actual_b]
    rng.shuffle(sel)
    return df.loc[sel].reset_index(drop=True), actual_p, actual_b
