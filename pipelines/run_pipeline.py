"""
PathoSense — Competition Pipeline

⚠️ LEGACY EVAL (2026-06-10): superseded by the rework. Deployment is now
`pipelines/predict.py` (per-panel recipe in `recipe.py` + cross-panel transfer in
`transfer.py`), and honest evaluation is `analysis/harness.py`. This script's
median-impute + mRMR-60 recipe, single 85/15 split, and `metrics.json` numbers are
INFLATED (it also fits feature selection on data it evaluates on). Kept for reference.

The 5 shared model/preprocessing helpers (`preprocess_data`, `_encode_for_mi`,
`select_features_mrmr`, `get_model_params`, `train_multi_seed_ensemble`) were moved to
`pipelines/models.py` (2026-06-26) — the deployed path imports them from there. They are
re-exported below so the many analysis/ scripts importing them from `run_pipeline` keep
working. What REMAINS here is purely legacy eval (`run_panel`, `main`, plot helpers).
See docs/METHODS.md.

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
from schema import (
    normalize_sentinels, to_categorical, partition_columns, classify_columns,
    drop_all_nan_rows,
    CAT_POPULATION_FREQ, CAT_CONSERVATION, CAT_SCORE_CENTERED,
    CAT_IN_SILICO, CAT_IN_SILICO_NEAR0, CAT_IN_SILICO_NEAR1,
    CAT_CADD_PHRED, CAT_BOUNDED_POS, CAT_AA_DERIVED,
    CAT_CONSTANT, CAT_SPARSE,
)
from feature_engineering import (
    engineer_aa_features, engineer_acmg_features, engineer_interaction_features,
    fit_score_agreement, apply_score_agreement,
    fit_quadrants, apply_quadrants,
)
from calibration import fit_platt, apply_platt, calibration_summary
from utils import (
    detect_leaky_columns, compute_metrics,
    optimize_threshold_f1, optimize_threshold_f1_balanced, mcc_at_prior,
    KNOWN_LEAKY, PANELS, TEST_PANELS, TEST_PRIORS, ACMG_PANELS,
)




# Deployed model/preprocessing helpers now live in pipelines/models.py (clean, matplotlib-free).
# Re-exported here for backward-compat with the many analysis/ scripts importing them from run_pipeline.
from models import (preprocess_data, _encode_for_mi, select_features_mrmr,
                    get_model_params, train_multi_seed_ensemble)



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

def run_panel(panel_name: str,
              use_platt: bool = True,
              use_score_agree: bool = True,
              use_interactions: bool = True,
              use_quadrants: bool = False) -> Dict:
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
    df_train = normalize_sentinels(df_train)
    df_train = drop_all_nan_rows(df_train)
    df_train = to_categorical(df_train)
    df_train = engineer_aa_features(df_train)
    if panel_name in ACMG_PANELS:
        df_train = engineer_acmg_features(df_train)
    if use_interactions:
        df_train = engineer_interaction_features(df_train)
    has_ext = panel_name in TEST_PANELS and os.path.exists(TEST_PANELS.get(panel_name, ""))
    df_ext = pd.read_csv(TEST_PANELS[panel_name]) if has_ext else None
    if df_ext is not None:
        df_ext = normalize_sentinels(df_ext)
        df_ext = drop_all_nan_rows(df_ext)
        df_ext = to_categorical(df_ext)
        df_ext = engineer_aa_features(df_ext)
        if panel_name in ACMG_PANELS:
            df_ext = engineer_acmg_features(df_ext)
        if use_interactions:
            df_ext = engineer_interaction_features(df_ext)

    _, num_cols, cat_cols, tgt = partition_columns(df_train)
    feature_cols = num_cols + cat_cols  # ID is excluded
    y_all = df_train[tgt]
    X_all = df_train[feature_cols]
    y_ext = df_ext["Label"] if df_ext is not None else None
    X_ext = df_ext[feature_cols] if df_ext is not None else None
    print(f"  features after AA engineering: {len(feature_cols)} "
          f"(numeric={len(num_cols)}, categorical={len(cat_cols)})")

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
    print("\n[5/9] Preprocessing (per-category transforms, corr threshold=0.95)...")

    X_train_pp, X_val_pp, X_ext_pp, prep_info = preprocess_data(
        X_train.copy(), X_val.copy(),
        X_ext.copy() if X_ext is not None else X_val.copy(),
        corr_threshold=0.95)

    X_train_val = pd.concat([X_train_pp, X_val_pp], ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], ignore_index=True)

    if X_ext is None:
        X_ext_pp = X_val_pp

    print(f"  Dropped constants: {len(prep_info.get('dropped_constants', []))}")
    print(f"  Low-var after transform: {len(prep_info.get('low_var_after_transform', []))}")
    print(f"  High-corr removed (NaN-aware): {len(prep_info.get('high_corr_removed', []))}")
    print(f"  Numeric kept: {prep_info['n_numeric']}, categorical: {prep_info['n_categorical']}, "
          f"total: {prep_info['final_features']}")
    cat_counts = prep_info.get('column_categories_count', {})
    if cat_counts:
        print(f"  Category breakdown: " +
              ", ".join(f"{k}={v}" for k, v in sorted(cat_counts.items(), key=lambda x: -x[1])[:6]))

    # -- [5.5] Score-agreement meta-features ---------------------------------
    sa_fitted = None
    if use_score_agree:
        print("\n[5.5/9] Engineering score-agreement features...")
        sa_fitted = fit_score_agreement(X_train_pp, y_train, top_k=7)
        print(f"  anchor set: {list(sa_fitted['anchors'].keys())}")
        X_train_pp = apply_score_agreement(X_train_pp, sa_fitted)
        X_val_pp   = apply_score_agreement(X_val_pp,   sa_fitted)
        X_ext_pp   = apply_score_agreement(X_ext_pp,   sa_fitted)
        # Rebuild X_train_val to include the new FX_ columns
        X_train_val = pd.concat([X_train_pp, X_val_pp], ignore_index=True)

    # -- [5.6] Quadrant indicator features (PAH-targeted by default) ---------
    qd_fitted = None
    if use_quadrants:
        print("\n[5.6/9] Engineering (conservation × biochem) quadrant indicators...")
        qd_fitted = fit_quadrants(X_train_pp)
        print(f"  conservation threshold (EK_7): {qd_fitted.get('cons_threshold')}")
        print(f"  biochem composite threshold: {qd_fitted.get('biochem_threshold')}")
        X_train_pp = apply_quadrants(X_train_pp, qd_fitted)
        X_val_pp   = apply_quadrants(X_val_pp,   qd_fitted)
        X_ext_pp   = apply_quadrants(X_ext_pp,   qd_fitted)
        X_train_val = pd.concat([X_train_pp, X_val_pp], ignore_index=True)

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

    # -- [8] Calibration + Threshold optimisation ---------------------------
    print("\n[8/9] Threshold optimisation...")

    # Test set is Benign-dominant (şartname v2.0 "Klinik Stres Testi"); tune the
    # threshold for the per-panel TEST prior, not the training prior or 50/50.
    panel_prior = TEST_PRIORS.get(panel_name, 0.5)
    print(f"  target test prior (pathogenic fraction): {panel_prior:.3f}")

    # Raw threshold (training prior, inflated) — always reported for comparison.
    f1_t_raw, f1_val_raw = optimize_threshold_f1(y_train_val, oof_probs)
    # Test-prior threshold on raw OOF.
    f1_t_bal, f1_val_bal, f1_std_bal = optimize_threshold_f1_balanced(
        y_train_val.values, oof_probs, target_prior=panel_prior, n_bootstrap=200)

    if use_platt:
        platt_fitted = fit_platt(oof_probs, y_train_val.values)
        oof_calibrated = apply_platt(oof_probs, platt_fitted)
        cal_before = calibration_summary(oof_probs, y_train_val.values)
        cal_after  = calibration_summary(oof_calibrated, y_train_val.values)
        print(f"  Platt fit: a={platt_fitted['a']:.3f}, b={platt_fitted['b']:.3f}")
        print(f"  Calibration ECE: raw={cal_before['ece']:.4f} -> "
              f"calibrated={cal_after['ece']:.4f}  (lower is better)")
        # Test-prior threshold on calibrated OOF (deployed)
        f1_t_cal, f1_val_cal, f1_std_cal = optimize_threshold_f1_balanced(
            y_train_val.values, oof_calibrated, target_prior=panel_prior, n_bootstrap=200)
        deployed_probs = oof_calibrated
        f1_t = f1_t_cal
    else:
        platt_fitted = None
        cal_before = calibration_summary(oof_probs, y_train_val.values)
        cal_after  = cal_before  # no calibration applied
        f1_t_cal, f1_val_cal, f1_std_cal = f1_t_bal, f1_val_bal, f1_std_bal
        deployed_probs = oof_probs
        f1_t = f1_t_bal

    # Balanced-accuracy-optimal (clinical safety, on deployed-probs OOF)
    ba_t, ba_val = 0.5, 0.0
    for t in np.arange(0.10, 0.95, 0.01):
        ba = balanced_accuracy_score(y_train_val, (deployed_probs >= t).astype(int))
        if ba > ba_val:
            ba_val, ba_t = ba, round(t, 2)

    def _cm_stats(probs, threshold):
        cm = confusion_matrix(y_train_val, (probs >= threshold).astype(int))
        tn, fp, fn, tp = cm.ravel()
        spec = tn / max(tn + fp, 1)
        return tp, fp, tn, fn, spec

    tp_b, fp_b, tn_b, fn_b, spec_b = _cm_stats(deployed_probs, ba_t)
    n_ben = int((y_train_val == 0).sum())

    print(f"  Raw F1-opt          t={f1_t_raw}  F1={f1_val_raw:.4f}  (training prior; inflated)")
    print(f"  Test-prior F1 (raw) t={f1_t_bal}  F1={f1_val_bal:.4f} ± {f1_std_bal:.4f}")
    if use_platt:
        print(f"  Test-prior F1 (cal) t={f1_t_cal}  F1={f1_val_cal:.4f} ± {f1_std_cal:.4f}  <- deployed")
    print(f"  BA-opt            t={ba_t}  BA={ba_val:.4f}  Spec={spec_b:.3f}  "
          f"BenErr={fp_b}/{n_ben}={fp_b/max(n_ben,1)*100:.1f}%")
    mcc_val, mcc_std = mcc_at_prior(y_train_val.values, deployed_probs, f1_t,
                                    target_prior=panel_prior, n_bootstrap=200)
    print(f"  MCC (test prior)    {mcc_val:.4f} ± {mcc_std:.4f}  (at deployed t={f1_t})")

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
            "deployed":              f1_t,
            "f1_raw_optimal":        f1_t_raw,
            "f1_raw_value":          f1_val_raw,   # inflated (training-prior F1)
            "f1_balanced_optimal":   f1_t_bal,
            "f1_balanced_value":     f1_val_bal,   # honest estimate, no calibration
            "f1_balanced_std":       f1_std_bal,
            "f1_calibrated_optimal": f1_t_cal,
            "f1_calibrated_value":   f1_val_cal,   # deployed F1 (Platt + balanced)
            "f1_calibrated_std":     f1_std_cal,
            "mcc_test_prior":        mcc_val,      # honest MCC at deployed t, test prior
            "mcc_test_prior_std":    mcc_std,
            "ba_optimal":            ba_t,
            "ba_value":              round(ba_val, 4),
            "specificity_ba":        round(spec_b, 4),
            "benign_error_ba":       round(fp_b / max(n_ben, 1), 4),
        },
        "calibration": {
            "platt_a":               round(platt_fitted["a"], 4) if platt_fitted else None,
            "platt_b":               round(platt_fitted["b"], 4) if platt_fitted else None,
            "ece_raw":               cal_before["ece"],
            "ece_calibrated":        cal_after["ece"],
            "mean_pred_raw":         cal_before["mean_predicted"],
            "mean_pred_calibrated":  cal_after["mean_predicted"],
            "mean_actual":           cal_before["mean_actual"],
        },
        "ablation_flags": {
            "use_platt":        use_platt,
            "use_score_agree":  use_score_agree,
            "use_interactions": use_interactions,
            "use_quadrants":    use_quadrants,
        },
        "oof_metrics": {"f1": round(oof_f1, 4), "auc": round(oof_auc, 4)},
    }

    # Val evaluation (calibrated if Platt enabled)
    val_probs_raw = np.mean(
        [m.predict_proba(X_val_pp[selected_features])[:, 1] for m in models], axis=0)
    val_probs = apply_platt(val_probs_raw, platt_fitted) if platt_fitted else val_probs_raw
    results["val_default"] = compute_metrics(y_val, val_probs, 0.5)
    results["val_optimal"] = compute_metrics(y_val, val_probs, f1_t)
    cal_tag = " (Platt-calibrated)" if platt_fitted else ""
    print(f"\n  Val ({len(y_val)}): F1@0.5={results['val_default']['f1']:.4f}  "
          f"AUC={results['val_default']['roc_auc']:.4f}{cal_tag}")

    # External test evaluation
    if df_ext is not None:
        if platt_fitted:
            test_probs = apply_platt(test_probs, platt_fitted)
        ext_def = compute_metrics(y_ext, test_probs, 0.5)
        ext_opt = compute_metrics(y_ext, test_probs, f1_t)
        results["ext_default"] = ext_def
        results["ext_optimal"] = ext_opt
        results["ext_size"] = len(y_ext)
        print(f"  Ext ({len(y_ext)}): F1@0.5={ext_def['f1']:.4f}  AUC={ext_def['roc_auc']:.4f}  "
              f"P={ext_def['precision']:.4f}  R={ext_def['recall']:.4f}{cal_tag}")

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

# Per-panel best ablation configs, chosen by analysis/ablate.py on 2026-05-18.
# E1 (Platt) is F1-neutral but kept everywhere for well-calibrated probabilities.
# E2 (score-agreement) and E3 (interactions) help small panels (cftr, cancer)
# but hurt large/distinct panels (general, pah) - see analysis/feature_ablation.py.
BEST_CONFIG_PER_PANEL = {
    "general": dict(use_platt=True, use_score_agree=False, use_interactions=False, use_quadrants=False),
    "cancer":  dict(use_platt=True, use_score_agree=True,  use_interactions=True,  use_quadrants=False),
    # PAH: E1_only. Quadrant features (G-track, 2026-05-18) were tested and
    # REJECTED — they hurt PAH F1 by ~0.02. The mirror-image FN/FP regimes
    # identified by analysis/pah_error_analysis.py are real, but discrete
    # threshold flags don't help the model carve them. Bigger interventions
    # (different model family, data cleaning of suspect benigns) needed.
    "pah":     dict(use_platt=True, use_score_agree=False, use_interactions=False, use_quadrants=False),
    "cftr":    dict(use_platt=True, use_score_agree=True,  use_interactions=True,  use_quadrants=False),
}


def main():
    parser = argparse.ArgumentParser(description="PathoSense Pipeline")
    parser.add_argument("--panel", required=True,
                        choices=["general", "cancer", "pah", "cftr", "all"])
    parser.add_argument("--all-features", action="store_true",
                        help="Override per-panel defaults and enable all 3 feature tracks "
                             "(E1+E2+E3) for every panel. Useful for ablation only.")
    parser.add_argument("--no-features", action="store_true",
                        help="Disable all 3 feature tracks. Useful for baseline-comparison runs.")
    args = parser.parse_args()

    panels = ["general", "cancer", "pah", "cftr"] if args.panel == "all" else [args.panel]
    all_results = {}
    for pn in panels:
        if args.all_features:
            cfg = dict(use_platt=True, use_score_agree=True, use_interactions=True, use_quadrants=True)
        elif args.no_features:
            cfg = dict(use_platt=False, use_score_agree=False, use_interactions=False, use_quadrants=False)
        else:
            cfg = BEST_CONFIG_PER_PANEL.get(pn, dict(use_platt=True))
        all_results[pn] = run_panel(pn, **cfg)

    # Summary table
    print(f"\n{'#' * 70}")
    print(f"  SUMMARY  (* = honest est. F1 at the per-panel test prior; test set is Benign-dominant, v2.0)")
    print(f"{'#' * 70}")
    print(f"  {'Panel':<8} {'AUC':<7} {'raw F1':<8} "
          f"{'bal F1*':<9} {'cal F1*':<9} {'MCC*':<8} {'cal t':<7} {'ECE':<7}")
    print(f"  {'-'*60}")
    for pn in panels:
        r = all_results.get(pn, {})
        if not r:
            continue
        oof = r.get("oof_metrics", {})
        thr = r.get("threshold", {})
        cal = r.get("calibration", {})
        print(f"  {pn:<8} {oof.get('auc',0):<7.4f} "
              f"{thr.get('f1_raw_value',0):<8.4f} "
              f"{thr.get('f1_balanced_value',0):<9.4f} "
              f"{thr.get('f1_calibrated_value',0):<9.4f} "
              f"{thr.get('mcc_test_prior',0):<8.4f} "
              f"{thr.get('f1_calibrated_optimal','?'):<7} "
              f"{cal.get('ece_calibrated',0):<7.4f}")


if __name__ == "__main__":
    main()
