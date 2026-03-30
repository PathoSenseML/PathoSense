"""Shared utility functions for the PathoSense pipeline."""
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score,
    precision_score, recall_score, brier_score_loss
)

from config import BASE_DIR

# ── Panel Definitions ──────────────────────────────────────────────
PANELS = {
    "general": os.path.join(BASE_DIR, "data", "panels", "general_dataset.csv"),
    "cancer": os.path.join(BASE_DIR, "data", "panels", "hereditary_cancer_panel.csv"),
    "pah": os.path.join(BASE_DIR, "data", "panels", "pah_panel.csv"),
    "cftr": os.path.join(BASE_DIR, "data", "panels", "cftr_panel.csv"),
}

TEST_PANELS = {
    "general": os.path.join(BASE_DIR, "data", "panels", "test_general.csv"),
    "cancer": os.path.join(BASE_DIR, "data", "panels", "test_cancer.csv"),
    "pah": os.path.join(BASE_DIR, "data", "panels", "test_pah.csv"),
    "cftr": os.path.join(BASE_DIR, "data", "panels", "test_cftr.csv"),
}

# Known leaky columns (will also be auto-detected)
KNOWN_LEAKY = {'AF', 'nSNP', 'Pos1', 'Positions'}


# ── Auto Leakage Detection ────────────────────────────────────────

def detect_leaky_columns(X: pd.DataFrame, y: pd.Series,
                         auc_threshold: float = 0.99,
                         verbose: bool = True) -> List[str]:
    """
    Scan each feature for single-feature AUC > threshold.
    Any such feature is considered leaky and should be dropped.
    Also checks for missingness-as-signal (all missing -> one class).
    """
    leaky = []
    for col in X.columns:
        vals = X[col].copy()

        # Check missingness pattern
        n_missing = vals.isna().sum()
        if n_missing > 0:
            missing_mask = vals.isna()
            if y[missing_mask].nunique() == 1 and n_missing > 10:
                leaky_class = y[missing_mask].iloc[0]
                if verbose:
                    print(f"    LEAKY (missingness): {col} — all {n_missing} missing "
                          f"samples are class {leaky_class}")
                leaky.append(col)
                continue

        # Check single-feature AUC
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

        # Check near-zero variance
        if vals.std() < 1e-8:
            if verbose:
                print(f"    CONSTANT: {col} (std={vals.std():.2e})")
            leaky.append(col)

    return list(set(leaky))


# ── Threshold Optimization ────────────────────────────────────────

def optimize_threshold_f1(y_true: pd.Series, probs: np.ndarray,
                          step: float = 0.005) -> Tuple[float, float]:
    """
    Maximize standard F1 score via full sweep.
    Competition metric is F1 = 2*TP / (2*TP + FP + FN).
    """
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.96, step):
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return round(best_t, 4), round(best_f1, 4)


# ── Metrics ───────────────────────────────────────────────────────

def compute_metrics(y_true: pd.Series, probs: np.ndarray, threshold: float = 0.5) -> Dict:
    """Compute comprehensive metrics at a given threshold."""
    preds = (probs >= threshold).astype(int)
    return {
        "f1": round(f1_score(y_true, preds, zero_division=0), 4),
        "f1_macro": round(f1_score(y_true, preds, average='macro', zero_division=0), 4),
        "accuracy": round(accuracy_score(y_true, preds), 4),
        "roc_auc": round(roc_auc_score(y_true, probs), 4),
        "precision": round(precision_score(y_true, preds, zero_division=0), 4),
        "recall": round(recall_score(y_true, preds, zero_division=0), 4),
        "brier": round(brier_score_loss(y_true, probs), 6),
        "threshold": threshold,
    }


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
