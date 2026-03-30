"""
PathoSense — Competition Pipeline

Missense variant pathogenicity classifier for TEKNOFEST 2026 AI in Health.
Binary classification (Pathogenic vs Benign) using LightGBM multi-seed ensemble.

Architecture:
    1. Auto leakage detection (per-feature AUC > 0.99)
    2. Preprocessing: outlier clip -> missing indicators -> impute -> low-var -> correlation
    3. mRMR feature selection (redundancy-penalized mutual information)
    4. Multi-seed LightGBM ensemble (5 seeds, K-fold OOF)
    5. Dual-threshold optimization (F1-optimal + balanced-accuracy-optimal)

Usage:
    python run_pipeline.py --panel general
    python run_pipeline.py --panel all
"""

import os
import sys
import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lightgbm as lgb
from datetime import datetime
from typing import Dict

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, balanced_accuracy_score,
)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SEED, RESULTS_DIR
from utils import (
    detect_leaky_columns, compute_metrics, optimize_threshold_f1,
    KNOWN_LEAKY, PANELS, TEST_PANELS,
)




# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_data(X_train, X_val, X_test, corr_threshold=0.99):
    """
    Clean and filter features. All statistics computed on X_train only.

    Steps:
        1. Clip outliers using IQR x 3 (fit on train, apply to all)
        2. Create boolean missing-data indicators for features with >5 % NaN
        3. Impute remaining NaN with training median
        4. Drop near-zero-variance columns (var < 0.001)
        5. Drop highly correlated columns (|r| > corr_threshold)

    Returns:
        X_train, X_val, X_test (cleaned), info dict
    """
    info = {}

    # 1. Outlier clipping
    Q1, Q3 = X_train.quantile(0.25), X_train.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 3 * IQR, Q3 + 3 * IQR
    clipped = 0
    for col in X_train.columns:
        if IQR[col] > 0:
            for df in [X_train, X_val, X_test]:
                mask = df[col].notna()
                df.loc[mask, col] = df.loc[mask, col].clip(lower[col], upper[col])
            clipped += 1
    info["outlier_clipped"] = clipped

    # 2. Missing-data indicators (before imputation destroys the signal)
    miss_rate = X_train.isna().mean()
    informative = miss_rate[(miss_rate > 0.05) & (miss_rate < 0.95)].index
    for col in informative:
        name = f"{col}_missing"
        for df in [X_train, X_val, X_test]:
            df[name] = df[col].isna().astype(int)
    info["missing_indicators_added"] = len(informative)

    # 3. Median imputation
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_val = X_val.fillna(medians)
    X_test = X_test.fillna(medians)

    # 4. Low-variance removal
    low_var = X_train.var()[lambda v: v < 0.001].index.tolist()
    if low_var:
        X_train = X_train.drop(columns=low_var)
        X_val = X_val.drop(columns=low_var)
        X_test = X_test.drop(columns=low_var)
    info["low_var_removed"] = len(low_var)

    # 5. High-correlation removal
    dropped_corr = []
    if len(X_train.columns) > 50:
        corr = X_train.corr().abs()
        upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        dropped_corr = [c for c in upper_tri.columns if any(upper_tri[c] > corr_threshold)]
        if dropped_corr:
            X_train = X_train.drop(columns=dropped_corr)
            X_val = X_val.drop(columns=dropped_corr)
            X_test = X_test.drop(columns=dropped_corr)
    info["high_corr_removed"] = len(dropped_corr)
    info["final_features"] = len(X_train.columns)

    # Fitted params (for inference replay)
    info["fitted"] = {
        "clip_lower": lower, "clip_upper": upper,
        "clip_cols": [col for col in X_train.columns if IQR[col] > 0],
        "miss_indicator_cols": list(informative),
        "medians": medians,
        "low_var_cols": low_var,
        "high_corr_cols": dropped_corr,
    }

    return X_train, X_val, X_test, info


# ---------------------------------------------------------------------------
# Feature selection — mRMR
# ---------------------------------------------------------------------------

def select_features_mrmr(X_train, y_train, X_val, X_test, max_features=0):
    """
    Minimum Redundancy Maximum Relevance feature selection.

    Greedy algorithm:
        1. Pick the feature with highest MI(feature, target)
        2. Iteratively add the feature maximising
           MI(feature, target) - mean(|corr(feature, already_selected)|)

    The number of features adapts to dataset size:
        n < 200 -> 20,  200-500 -> 40,  500+ -> 60
    """
    n = len(X_train)
    if max_features <= 0:
        if n < 200:
            max_features = min(20, len(X_train.columns))
        elif n < 500:
            max_features = min(40, len(X_train.columns))
        else:
            max_features = min(60, len(X_train.columns))

    n_before = len(X_train.columns)

    if n_before <= max_features:
        return X_train, X_val, X_test, {
            "method": "none_needed", "n_before": n_before, "n_after": n_before,
        }

    # Mutual information scores (computed on training data only)
    mi = mutual_info_classif(X_train, y_train, random_state=SEED,
                              n_neighbors=max(3, min(5, n // 30)))
    mi_map = dict(zip(X_train.columns, mi))

    selected = []
    candidates = list(X_train.columns)

    # First feature: pure highest MI
    best = max(candidates, key=lambda c: mi_map[c])
    selected.append(best)
    candidates.remove(best)

    # Greedy loop: relevance minus redundancy
    while len(selected) < max_features and candidates:
        scores = {}
        for c in candidates:
            relevance = mi_map[c]
            redundancy = X_train[selected].corrwith(X_train[c]).abs().mean()
            scores[c] = relevance - redundancy
        best = max(scores, key=scores.get)
        selected.append(best)
        candidates.remove(best)

    info = {
        "method": "mRMR",
        "n_before": n_before,
        "n_after": len(selected),
        "top_5": {c: round(mi_map[c], 4) for c in selected[:5]},
    }
    return X_train[selected], X_val[selected], X_test[selected], info


# ---------------------------------------------------------------------------
# Hyperparameters (size-adaptive)
# ---------------------------------------------------------------------------

def get_model_params(n_train, scale_pos_weight):
    """
    Return LightGBM hyperparameters adapted to training-set size.

    Three tiers:
        small  (n < 200) : conservative — fewer trees, more regularisation
        medium (200-400)  : balanced
        large  (400+)     : deeper trees, less regularisation
    """
    common = {
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "binary_logloss",
        "scale_pos_weight": scale_pos_weight,
        "verbose": -1,
    }

    if n_train < 200:
        return {**common,
            "n_estimators": 500, "learning_rate": 0.02,
            "num_leaves": 24, "max_depth": 5, "min_child_samples": 10,
            "subsample": 0.7, "colsample_bytree": 0.7,
            "reg_alpha": 0.3, "reg_lambda": 1.5,
        }
    elif n_train < 400:
        return {**common,
            "n_estimators": 800, "learning_rate": 0.03,
            "num_leaves": 31, "max_depth": 6, "min_child_samples": 10,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_alpha": 0.3, "reg_lambda": 1.5,
        }
    else:
        return {**common,
            "n_estimators": 2000, "learning_rate": 0.02,
            "num_leaves": 31, "max_depth": 6, "min_child_samples": 15,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_alpha": 0.3, "reg_lambda": 1.5,
        }


# ---------------------------------------------------------------------------
# Multi-seed ensemble
# ---------------------------------------------------------------------------

def train_multi_seed_ensemble(X_train_val, y_train_val, X_test,
                               params_base, n_seeds=5, n_folds=5):
    """
    Train *n_seeds* LightGBM models with different random seeds and average
    their predicted probabilities to reduce variance.

    For each seed:
        - Generate out-of-fold (OOF) predictions via K-fold CV
        - Train a final model on all training data (with 10 % holdout for ES)
        - Predict on X_test

    Returns:
        models       : list of final trained models
        avg_oof      : averaged OOF probabilities
        avg_test     : averaged test probabilities
        oof_std      : mean std of OOF predictions across seeds
    """
    models = []
    oof_all = np.zeros((n_seeds, len(y_train_val)))
    test_all = np.zeros((n_seeds, len(X_test)))
    patience = 50 if len(y_train_val) > 300 else 30

    for i in range(n_seeds):
        seed_i = SEED + i
        params = {**params_base, "random_state": seed_i}

        # OOF predictions
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed_i)
        oof = np.zeros(len(y_train_val))
        for tr_idx, va_idx in skf.split(X_train_val, y_train_val):
            m = lgb.LGBMClassifier(**params)
            m.fit(X_train_val.iloc[tr_idx], y_train_val.iloc[tr_idx],
                  eval_set=[(X_train_val.iloc[va_idx], y_train_val.iloc[va_idx])],
                  callbacks=[lgb.early_stopping(patience, verbose=False),
                             lgb.log_evaluation(0)])
            oof[va_idx] = m.predict_proba(X_train_val.iloc[va_idx])[:, 1]
        oof_all[i] = oof

        # Final model (small holdout for early stopping only)
        X_tr, X_es, y_tr, y_es = train_test_split(
            X_train_val, y_train_val, test_size=0.1,
            random_state=seed_i, stratify=y_train_val)
        final = lgb.LGBMClassifier(**params)
        final.fit(X_tr, y_tr,
                  eval_set=[(X_es, y_es)],
                  callbacks=[lgb.early_stopping(patience, verbose=False),
                             lgb.log_evaluation(0)])
        models.append(final)
        test_all[i] = final.predict_proba(X_test)[:, 1]

    return models, oof_all.mean(0), test_all.mean(0), oof_all.std(0).mean()


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _save_shap_plots(models, X_data, feature_names, panel_name, plots_dir):
    """Generate SHAP bar and beeswarm plots from the first model."""
    explainer = shap.TreeExplainer(models[0])
    sv = explainer.shap_values(X_data)
    if isinstance(sv, list):
        sv = sv[1]

    for ptype, fname in [("bar", "shap_global.png"), (None, "shap_beeswarm.png")]:
        fig, _ = plt.subplots(figsize=(10, 7))
        kwargs = {"plot_type": ptype} if ptype else {}
        shap.summary_plot(sv, X_data, show=False, max_display=20, **kwargs)
        plt.title(f"{panel_name.upper()} — SHAP")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, fname), dpi=150)
        plt.close()

    mean_shap = np.abs(sv).mean(axis=0)
    return dict(sorted(zip(feature_names, mean_shap), key=lambda x: -x[1])[:10])


def _save_confusion_plot(y_true, y_pred, panel_name, threshold, plots_dir):
    """Save a confusion-matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=16)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Benign", "Pathogenic"])
    ax.set_yticklabels(["Benign", "Pathogenic"])
    ax.set_title(f"{panel_name.upper()} — Confusion Matrix (t={threshold})")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)


def _save_calibration_plot(y_true, probs, panel_name, plots_dir):
    """Save a reliability diagram and return ECE."""
    prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=10)
    ece = float(np.mean(np.abs(prob_true - prob_pred)))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(prob_pred, prob_true, "o-", lw=2, label=f"ECE={ece:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"{panel_name.upper()} — Calibration")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "calibration.png"), dpi=150)
    plt.close(fig)
    return ece


def _save_roc_plot(y_true, probs, auc_val, panel_name, plots_dir):
    """Save a ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, probs)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, "b-", lw=2, label=f"AUC={auc_val:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"{panel_name.upper()} — ROC")
    ax.legend(loc="lower right"); fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "roc.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_panel(panel_name: str) -> Dict:
    """
    Run the full pipeline for a single gene panel.

    Steps:
        [1] Load data (training + optional external test)
        [2] Baseline metric (majority-class F1)
        [3] Auto leakage detection
        [4] Stratified 85/15 train/val split
        [5] Preprocessing
        [6] mRMR feature selection
        [7] Multi-seed ensemble training
        [8] Dual-threshold optimisation (F1-opt + BA-opt)
        [9] Evaluation, SHAP, error analysis, calibration
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.time()

    print(f"\n{'#' * 70}")
    print(f"  PathoSense — {panel_name.upper()}")
    print(f"  {ts}")
    print(f"{'#' * 70}")

    # -- [1] Load data -------------------------------------------------------
    print("\n[1/9] Loading data...")
    df_train = pd.read_csv(PANELS[panel_name])
    has_ext = panel_name in TEST_PANELS and os.path.exists(TEST_PANELS.get(panel_name, ""))
    df_ext = pd.read_csv(TEST_PANELS[panel_name]) if has_ext else None

    y_all = df_train["Label"]
    feature_cols = [c for c in df_train.columns if c != "Label"]
    X_all = df_train[feature_cols]
    y_ext = df_ext["Label"] if df_ext is not None else None
    X_ext = df_ext[feature_cols] if df_ext is not None else None

    n_pos, n_neg = int(y_all.sum()), int((y_all == 0).sum())
    print(f"  Training: {len(df_train)} ({n_pos}P / {n_neg}B)")
    if df_ext is not None:
        print(f"  External test: {len(df_ext)} ({int(y_ext.sum())}P / {int((y_ext==0).sum())}B)")

    # Duplicate check
    if df_ext is not None:
        overlap = len(set(pd.util.hash_pandas_object(X_all)) &
                      set(pd.util.hash_pandas_object(X_ext)))
        print(f"  Duplicate check: {'WARNING: ' + str(overlap) + ' duplicates!' if overlap else 'OK (0 overlapping rows)'}")

    # -- [2] Baseline --------------------------------------------------------
    print("\n[2/9] Baseline metrics...")
    majority = int(y_all.mode()[0])
    baseline_f1 = round(f1_score(y_all, np.full(len(y_all), majority), zero_division=0), 4)
    print(f"  Majority class F1: {baseline_f1:.4f}")

    # -- [3] Leakage detection -----------------------------------------------
    print("\n[3/9] Leakage detection...")
    leaky = detect_leaky_columns(X_all, y_all, auc_threshold=0.99, verbose=False)
    all_leaky = sorted(set(leaky) | (KNOWN_LEAKY & set(X_all.columns)))
    if all_leaky:
        print(f"  Dropping {len(all_leaky)} leaky columns")
        X_all = X_all.drop(columns=all_leaky)
        if X_ext is not None:
            X_ext = X_ext.drop(columns=[c for c in all_leaky if c in X_ext.columns])
    print(f"  Features after: {len(X_all.columns)}")

    # -- [4] Train / Val split -----------------------------------------------
    print("\n[4/9] Train/Val split (85/15)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.15, random_state=SEED, stratify=y_all)
    for df in [X_train, X_val, X_all]:
        df.reset_index(drop=True, inplace=True)
    for s in [y_train, y_val, y_all]:
        s.reset_index(drop=True, inplace=True)
    if X_ext is not None:
        X_ext.reset_index(drop=True, inplace=True)
        y_ext.reset_index(drop=True, inplace=True)
    print(f"  Train: {len(X_train)} | Val: {len(X_val)}")

    # -- [5] Preprocessing ---------------------------------------------------
    print("\n[5/9] Preprocessing (corr threshold=0.99)...")

    X_train_pp, X_val_pp, X_ext_pp, prep_info = preprocess_data(
        X_train.copy(), X_val.copy(),
        X_ext.copy() if X_ext is not None else X_val.copy(),
        corr_threshold=0.99)

    X_train_val = pd.concat([X_train_pp, X_val_pp], ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], ignore_index=True)

    if X_ext is None:
        X_ext_pp = X_val_pp

    print(f"  Low-var removed: {prep_info['low_var_removed']}")
    print(f"  High-corr removed: {prep_info['high_corr_removed']}")
    print(f"  Features after: {prep_info['final_features']}")

    # -- [6] mRMR feature selection ------------------------------------------
    print("\n[6/9] mRMR feature selection...")
    X_train_pp, X_val_pp, X_ext_pp, fs_info = select_features_mrmr(
        X_train_pp, y_train, X_val_pp, X_ext_pp)
    selected_features = list(X_train_pp.columns)
    X_train_val = X_train_val[[c for c in selected_features if c in X_train_val.columns]]

    print(f"  {fs_info['n_before']} -> {fs_info['n_after']} features")
    if "top_5" in fs_info:
        for feat, score in fs_info["top_5"].items():
            print(f"    {feat}: MI={score}")

    # -- [7] Multi-seed ensemble ---------------------------------------------
    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    params = get_model_params(len(X_train_val), spw)
    n_folds = 3 if len(X_train_val) < 200 else 5

    print(f"\n[7/9] Training ensemble (5 seeds, {n_folds}-fold)...")
    print(f"  depth={params['max_depth']}, leaves={params['num_leaves']}, "
          f"n_est={params['n_estimators']}, lr={params['learning_rate']}, "
          f"lambda={params['reg_lambda']}, spw={spw:.3f}")

    test_data = X_ext_pp if df_ext is not None else X_val_pp
    models, oof_probs, test_probs, oof_std = train_multi_seed_ensemble(
        X_train_val, y_train_val, test_data, params, n_seeds=5, n_folds=n_folds)

    oof_f1 = f1_score(y_train_val, (oof_probs >= 0.5).astype(int), zero_division=0)
    oof_auc = roc_auc_score(y_train_val, oof_probs)
    print(f"  OOF: F1={oof_f1:.4f}, AUC={oof_auc:.4f}, std={oof_std:.4f}")

    # -- [8] Dual-threshold optimisation -------------------------------------
    print("\n[8/9] Threshold optimisation...")

    # F1-optimal (competition scoring)
    f1_t, f1_val = optimize_threshold_f1(y_train_val, oof_probs)

    # Balanced-accuracy-optimal (clinical safety)
    ba_t, ba_val = 0.5, 0.0
    for t in np.arange(0.10, 0.95, 0.01):
        ba = balanced_accuracy_score(y_train_val, (oof_probs >= t).astype(int))
        if ba > ba_val:
            ba_val, ba_t = ba, round(t, 2)

    # Confusion stats at each threshold
    def _cm_stats(threshold):
        cm = confusion_matrix(y_train_val, (oof_probs >= threshold).astype(int))
        tn, fp, fn, tp = cm.ravel()
        spec = tn / max(tn + fp, 1)
        return tp, fp, tn, fn, spec

    tp_f, fp_f, tn_f, fn_f, spec_f = _cm_stats(f1_t)
    tp_b, fp_b, tn_b, fn_b, spec_b = _cm_stats(ba_t)
    n_ben = int((y_train_val == 0).sum())

    print(f"  F1-opt  t={f1_t}  F1={f1_val:.4f}  Spec={spec_f:.3f}  "
          f"BenErr={fp_f}/{n_ben}={fp_f/max(n_ben,1)*100:.1f}%")
    print(f"  BA-opt  t={ba_t}  BA={ba_val:.4f}  Spec={spec_b:.3f}  "
          f"BenErr={fp_b}/{n_ben}={fp_b/max(n_ben,1)*100:.1f}%")

    # -- [9] Evaluation & analysis -------------------------------------------
    print("\n[9/9] Evaluation")

    results = {
        "timestamp": ts,
        "baseline_f1": baseline_f1,
        "leaky_dropped": len(all_leaky),
        "preprocessing": {k: v for k, v in prep_info.items() if k != "fitted"},
        "feature_selection": fs_info,
        "n_train_val": len(X_train_val),
        "n_features": len(selected_features),
        "hyperparams": {k: v for k, v in params.items() if k != "verbose"},
        "ensemble": {"n_seeds": 5, "n_folds": n_folds, "oof_std": round(oof_std, 4)},
        "threshold": {
            "f1_optimal": f1_t, "f1_value": f1_val,
            "ba_optimal": ba_t, "ba_value": round(ba_val, 4),
            "specificity_f1": round(spec_f, 4),
            "specificity_ba": round(spec_b, 4),
            "benign_error_f1": round(fp_f / max(n_ben, 1), 4),
            "benign_error_ba": round(fp_b / max(n_ben, 1), 4),
        },
        "oof_metrics": {"f1": round(oof_f1, 4), "auc": round(oof_auc, 4)},
    }

    # Val evaluation
    val_probs = np.mean([m.predict_proba(X_val_pp[selected_features])[:, 1] for m in models], axis=0)
    results["val_default"] = compute_metrics(y_val, val_probs, 0.5)
    results["val_optimal"] = compute_metrics(y_val, val_probs, f1_t)
    print(f"\n  Val ({len(y_val)}): F1@0.5={results['val_default']['f1']:.4f}  "
          f"AUC={results['val_default']['roc_auc']:.4f}")

    # External test evaluation
    if df_ext is not None:
        ext_def = compute_metrics(y_ext, test_probs, 0.5)
        ext_opt = compute_metrics(y_ext, test_probs, f1_t)
        results["ext_default"] = ext_def
        results["ext_optimal"] = ext_opt
        results["ext_size"] = len(y_ext)
        print(f"  Ext ({len(y_ext)}): F1@0.5={ext_def['f1']:.4f}  AUC={ext_def['roc_auc']:.4f}  "
              f"P={ext_def['precision']:.4f}  R={ext_def['recall']:.4f}")

    # --- Plots & analysis ---
    results_dir = os.path.join(RESULTS_DIR, panel_name)
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # SHAP
    if SHAP_AVAILABLE and models:
        try:
            print("\n  SHAP analysis...")
            shap_data = X_val_pp[selected_features]
            ranking = _save_shap_plots(models, shap_data, selected_features, panel_name, plots_dir)
            results["shap_top10"] = {k: round(float(v), 4) for k, v in ranking.items()}
            print(f"    Top: {list(ranking.keys())[:5]}")
        except Exception as e:
            print(f"    SHAP failed: {e}")

    # Error analysis on external test
    if df_ext is not None:
        print("\n  Error analysis...")
        preds = (test_probs >= f1_t).astype(int)
        tp = int(((preds == 1) & (y_ext.values == 1)).sum())
        fp = int(((preds == 1) & (y_ext.values == 0)).sum())
        tn = int(((preds == 0) & (y_ext.values == 0)).sum())
        fn = int(((preds == 0) & (y_ext.values == 1)).sum())
        results["error_analysis"] = {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
        print(f"    TP={tp} FP={fp} TN={tn} FN={fn}")

        _save_confusion_plot(y_ext, preds, panel_name, f1_t, plots_dir)

    # Calibration
    if df_ext is not None and y_ext.nunique() > 1:
        try:
            ece = _save_calibration_plot(y_ext, test_probs, panel_name, plots_dir)
            results["calibration_ece"] = round(ece, 4)
            print(f"\n  Calibration ECE={ece:.4f}")
        except Exception:
            pass

    # ROC
    if df_ext is not None and y_ext.nunique() > 1:
        _save_roc_plot(y_ext, test_probs, ext_def["roc_auc"], panel_name, plots_dir)

    elapsed = round(time.time() - t0, 1)
    results["elapsed_seconds"] = elapsed
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved -> metrics.json, plots/ ({elapsed}s)")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PathoSense Pipeline")
    parser.add_argument("--panel", required=True,
                        choices=["general", "cancer", "pah", "cftr", "all"])
    args = parser.parse_args()

    panels = ["general", "cancer", "pah", "cftr"] if args.panel == "all" else [args.panel]
    all_results = {}
    for pn in panels:
        all_results[pn] = run_panel(pn)

    # Summary table
    print(f"\n{'#' * 70}")
    print(f"  SUMMARY")
    print(f"{'#' * 70}")
    print(f"  {'Panel':<10} {'OOF F1':<10} {'OOF AUC':<10} "
          f"{'Ext F1':<10} {'Ext AUC':<10} {'F1 t':<8} {'BA t':<8}")
    print(f"  {'-'*66}")
    for pn in panels:
        r = all_results.get(pn, {})
        if not r:
            continue
        oof = r.get("oof_metrics", {})
        ext = r.get("ext_default", {})
        thr = r.get("threshold", {})
        print(f"  {pn:<10} {oof.get('f1',0):<10.4f} {oof.get('auc',0):<10.4f} "
              f"{ext.get('f1',0):<10.4f} {ext.get('roc_auc',0):<10.4f} "
              f"{thr.get('f1_optimal','?'):<8} {thr.get('ba_optimal','?'):<8}")


if __name__ == "__main__":
    main()
