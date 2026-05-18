"""
PathoSense — Train/Save & Inference Pipeline

Usage:
    # Train on panel data and save artifacts
    python predict.py train --panel general

    # Predict on new unlabeled data
    python predict.py infer --panel general --input test.csv --output predictions.csv
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from config import SEED
from schema import normalize_sentinels, to_categorical, partition_columns, drop_all_nan_rows
from feature_engineering import (
    engineer_aa_features, engineer_interaction_features,
    fit_score_agreement, apply_score_agreement,
    fit_quadrants, apply_quadrants,
)
from calibration import fit_platt, apply_platt
from utils import (
    detect_leaky_columns, optimize_threshold_f1, optimize_threshold_f1_balanced,
    KNOWN_LEAKY, PANELS,
)
from run_pipeline import (
    preprocess_data, select_features_mrmr, get_model_params,
    train_multi_seed_ensemble, BEST_CONFIG_PER_PANEL,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


# ---------------------------------------------------------------------------
# Train & save
# ---------------------------------------------------------------------------

def train_and_save(panel_name):
    """Run pipeline on full training data and save all artifacts for inference.

    Uses the per-panel best feature configuration determined by
    analysis/ablate.py (see BEST_CONFIG_PER_PANEL in run_pipeline.py).
    """
    cfg = BEST_CONFIG_PER_PANEL.get(panel_name, dict(use_platt=True))
    print(f"  config for {panel_name}: {cfg}")

    panel_dir = os.path.join(MODELS_DIR, panel_name)
    os.makedirs(panel_dir, exist_ok=True)

    # Load + normalize + drop all-NaN rows + cast + engineer
    df = pd.read_csv(PANELS[panel_name])
    df = normalize_sentinels(df)
    df = drop_all_nan_rows(df)
    df = to_categorical(df)
    df = engineer_aa_features(df)
    if cfg.get("use_interactions", True):
        df = engineer_interaction_features(df)
    _, num_cols, cat_cols, tgt = partition_columns(df)
    y = df[tgt]
    X = df[num_cols + cat_cols]
    print(f"  {panel_name.upper()}: {len(df)} samples ({int(y.sum())}P / {int((y==0).sum())}B)")

    # Leakage detection (numeric features only; categorical leakage handled in detect_leaky_columns)
    leaky = detect_leaky_columns(X[num_cols], y, auc_threshold=0.99, verbose=False)
    all_leaky = sorted(set(leaky) | (KNOWN_LEAKY & set(X.columns)))
    X = X.drop(columns=all_leaky)
    print(f"  Dropped {len(all_leaky)} leaky columns -> {len(X.columns)} features")

    # 85/15 split (val is for preprocessing + early stopping only)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=SEED, stratify=y)
    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)

    # Preprocess
    X_train_pp, X_val_pp, _, prep_info = preprocess_data(
        X_train.copy(), X_val.copy(), X_val.copy(), corr_threshold=0.95)
    print(f"  Preprocessing: {prep_info['final_features']} features")

    # Score-agreement meta-features (conditional per panel)
    sa_fitted = None
    if cfg.get("use_score_agree", True):
        sa_fitted = fit_score_agreement(X_train_pp, y_train, top_k=7)
        X_train_pp = apply_score_agreement(X_train_pp, sa_fitted)
        X_val_pp   = apply_score_agreement(X_val_pp,   sa_fitted)
        print(f"  Score-agreement anchors: {list(sa_fitted['anchors'].keys())}")

    # Quadrant indicator features (conditional per panel; PAH default)
    qd_fitted = None
    if cfg.get("use_quadrants", False):
        qd_fitted = fit_quadrants(X_train_pp)
        X_train_pp = apply_quadrants(X_train_pp, qd_fitted)
        X_val_pp   = apply_quadrants(X_val_pp,   qd_fitted)
        ct = qd_fitted.get("cons_threshold")
        bt = qd_fitted.get("biochem_threshold")
        print(f"  Quadrant thresholds: cons={ct if ct is None else f'{ct:.3f}'}, "
              f"biochem={bt if bt is None else f'{bt:.3f}'}")

    X_train_val = pd.concat([X_train_pp, X_val_pp], ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], ignore_index=True)

    # mRMR feature selection
    X_train_pp, X_val_pp, _, fs_info = select_features_mrmr(
        X_train_pp, y_train, X_val_pp, X_val_pp)
    selected = list(X_train_pp.columns)
    X_train_val = X_train_val[selected]
    print(f"  mRMR: {fs_info['n_before']} -> {fs_info['n_after']} features")

    # Train ensemble
    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    params = get_model_params(len(X_train_val), spw)
    n_folds = 3 if len(X_train_val) < 200 else 5

    models, oof_probs, _, oof_std = train_multi_seed_ensemble(
        X_train_val, y_train_val, X_val_pp[selected], params, n_seeds=5, n_folds=n_folds)

    print(f"  OOF: F1={f1_score(y_train_val, (oof_probs >= 0.5).astype(int)):.4f}, "
          f"AUC={roc_auc_score(y_train_val, oof_probs):.4f}")

    # Calibration (Platt) is conditional per panel; balanced threshold always used.
    platt_fitted = None
    f1_t_raw, f1_val_raw = optimize_threshold_f1(y_train_val, oof_probs)
    f1_t_bal, f1_val_bal, f1_std_bal = optimize_threshold_f1_balanced(
        y_train_val.values, oof_probs, target_prior=0.5, n_bootstrap=200)

    if cfg.get("use_platt", True):
        platt_fitted = fit_platt(oof_probs, y_train_val.values)
        oof_calibrated = apply_platt(oof_probs, platt_fitted)
        f1_t_cal, f1_val_cal, f1_std_cal = optimize_threshold_f1_balanced(
            y_train_val.values, oof_calibrated, target_prior=0.5, n_bootstrap=200)
        deployed_t = f1_t_cal
        deployed_f1 = f1_val_cal
    else:
        f1_t_cal, f1_val_cal, f1_std_cal = f1_t_bal, f1_val_bal, f1_std_bal
        deployed_t = f1_t_bal
        deployed_f1 = f1_val_bal

    print(f"  Threshold (raw, training prior):     {f1_t_raw} (F1={f1_val_raw:.4f}, inflated)")
    print(f"  Threshold (raw OOF, balanced prior): {f1_t_bal} (F1={f1_val_bal:.4f}±{f1_std_bal:.4f})")
    if platt_fitted:
        print(f"  Threshold (Platt + balanced):        {f1_t_cal} (F1={f1_val_cal:.4f}±{f1_std_cal:.4f})  <- deployed")
    else:
        print(f"  Deployed = balanced threshold (no Platt for this panel)")

    # Save artifacts
    artifacts = {
        "leaky_cols": all_leaky,
        "feature_cols": list(X.columns),  # after leakage removal
        "prep_fitted": prep_info["fitted"],
        "score_agreement_fitted": sa_fitted,
        "quadrants_fitted": qd_fitted,
        "platt_fitted": platt_fitted,
        "selected_features": selected,
        "models": models,
        "threshold": deployed_t,
        "threshold_raw": f1_t_raw,
        "threshold_balanced": f1_t_bal,
        "threshold_calibrated": f1_t_cal,
        "f1_estimate_balanced":   f1_val_bal,
        "f1_estimate_calibrated": f1_val_cal,
        "f1_estimate_calibrated_std": f1_std_cal,
        "f1_deployed": deployed_f1,
        "training_prior": float(y_train_val.mean()),
        "config": cfg,
        "params": {k: v for k, v in params.items() if k != "verbose"},
    }
    path = os.path.join(panel_dir, "artifacts.joblib")
    joblib.dump(artifacts, path)
    print(f"  Saved to {path}")
    return artifacts


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def apply_preprocessing(X, fitted):
    """Replay the training-time per-category preprocessing on new data.

    Mirrors preprocess_data step-by-step:
      1. Normalize sentinels + cast categoricals + engineer AA features
      2. Drop the same constants
      3. Apply per-category transforms (population rank, conservation clip)
      4. Add missing-indicators
      5. Median-impute remaining NaN
      6. Drop the same low-var / high-corr columns
      7. Reassemble in the same final column order
    """
    X = normalize_sentinels(X)
    X = to_categorical(X)
    X = engineer_aa_features(X)
    X = engineer_interaction_features(X)  # safe even if not used at training — extra cols get dropped at reassembly

    # 2. Drop training-time constants
    drop_const = fitted.get("drop_constants", [])
    X.drop(columns=[c for c in drop_const if c in X.columns],
           inplace=True, errors="ignore")

    # 3a. Population: replay rank transform using saved quantile grids
    for col, (grid_vals, grid_ranks) in fitted.get("pop_quantile_grids", {}).items():
        if col not in X.columns or len(grid_vals) == 0:
            continue
        vals = X[col].values.astype(float)
        mapped = np.full_like(vals, np.nan, dtype=float)
        present = ~np.isnan(vals)
        if present.any():
            mapped[present] = np.interp(
                vals[present], grid_vals, grid_ranks,
                left=grid_ranks[0], right=grid_ranks[-1])
        X[col] = mapped

    # 3b. Conservation: replay clip
    for col, (lo, hi) in fitted.get("conservation_clip", {}).items():
        if col in X.columns:
            X[col] = X[col].clip(lo, hi)

    # 3c. Missing-indicators (numeric)
    for col in fitted.get("miss_indicator_cols", []):
        if col in X.columns:
            X[f"{col}_missing"] = X[col].isna().astype(int)

    # 3d. Missing-indicators (categorical)
    for col in fitted.get("cat_miss_indicator_cols", []):
        if col in X.columns:
            X[f"{col}_missing"] = X[col].isna().astype(int)

    # 3e. Median imputation
    X = X.fillna(fitted["medians"])

    # 4. Drop training-time low-variance + correlated columns
    for col in fitted.get("low_var_after_transform", []) + fitted.get("high_corr_cols", []):
        if col in X.columns:
            X.drop(columns=[col], inplace=True)

    # 5. Reassemble in training column order
    num_kept = fitted.get("numeric_cols_kept", [])
    cat_kept = fitted.get("categorical_cols", [])
    for c in num_kept:
        if c not in X.columns:
            X[c] = 0  # defensive: zero-fill missing-indicator cols if source absent
    final_cols = [c for c in num_kept if c in X.columns] + [c for c in cat_kept if c in X.columns]
    return X[final_cols]


def predict(input_csv, panel_name, output_csv):
    """Load saved artifacts and predict on new data."""
    panel_dir = os.path.join(MODELS_DIR, panel_name)
    art_path = os.path.join(panel_dir, "artifacts.joblib")

    if not os.path.exists(art_path):
        print(f"ERROR: No saved model at {art_path}")
        print(f"Run 'python predict.py train --panel {panel_name}' first.")
        sys.exit(1)

    art = joblib.load(art_path)
    print(f"  Loaded model for {panel_name.upper()}")
    print(f"  Features: {len(art['selected_features'])}, Threshold: {art['threshold']}")

    # Load input
    df = pd.read_csv(input_csv)
    print(f"  Input: {len(df)} samples, {len(df.columns)} columns")

    # Preserve Variant_ID for output if present (it is not a feature)
    variant_ids = df["Variant_ID"].copy() if "Variant_ID" in df.columns else None

    # Drop Label column if present
    if "Label" in df.columns:
        df = df.drop(columns=["Label"])

    # Keep only feature columns the model expects (after leakage removal)
    available = [c for c in art["feature_cols"] if c in df.columns]
    missing = set(art["feature_cols"]) - set(df.columns)
    if missing:
        print(f"  WARNING: {len(missing)} expected columns missing from input")
    X = df[available].copy()

    # Apply preprocessing (sentinel-normalise, cast categoricals, replay numeric transforms)
    X = apply_preprocessing(X, art["prep_fitted"])

    # Apply saved score-agreement feature engineering (FX_score_agree_*)
    sa_fitted = art.get("score_agreement_fitted")
    if sa_fitted is not None:
        X = apply_score_agreement(X, sa_fitted)

    # Apply saved quadrant indicator features (Q_*)
    qd_fitted = art.get("quadrants_fitted")
    if qd_fitted is not None:
        X = apply_quadrants(X, qd_fitted)

    # Select mRMR features
    selected = [c for c in art["selected_features"] if c in X.columns]
    missing_feats = set(art["selected_features"]) - set(X.columns)
    if missing_feats:
        print(f"  WARNING: {len(missing_feats)} selected features missing after preprocessing")
        for col in missing_feats:
            X[col] = 0  # zero-fill missing features
        selected = art["selected_features"]
    X = X[selected]

    # Predict (average across all seed models) then Platt-calibrate
    probs_raw = np.mean([m.predict_proba(X)[:, 1] for m in art["models"]], axis=0)
    platt_fitted = art.get("platt_fitted")
    probs = apply_platt(probs_raw, platt_fitted) if platt_fitted else probs_raw
    preds = (probs >= art["threshold"]).astype(int)

    # Output (include Variant_ID if it was present; ship both raw and calibrated probs)
    out_data = {}
    if variant_ids is not None:
        out_data["Variant_ID"] = variant_ids.values
    out_data["prediction"] = preds
    out_data["probability"] = probs.round(6)
    out_data["probability_raw"] = probs_raw.round(6)
    out = pd.DataFrame(out_data)
    out.to_csv(output_csv, index=False)

    n_path = int(preds.sum())
    n_ben = len(preds) - n_path
    print(f"  Predictions: {n_path} Pathogenic, {n_ben} Benign")
    print(f"  Saved to {output_csv}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PathoSense Train/Predict")
    sub = parser.add_subparsers(dest="command", required=True)

    # Train
    p_train = sub.add_parser("train", help="Train and save model artifacts")
    p_train.add_argument("--panel", required=True,
                         choices=["general", "cancer", "pah", "cftr", "all"])

    # Infer
    p_infer = sub.add_parser("infer", help="Predict on new data")
    p_infer.add_argument("--panel", required=True,
                         choices=["general", "cancer", "pah", "cftr"])
    p_infer.add_argument("--input", required=True, help="Input CSV path")
    p_infer.add_argument("--output", default="predictions.csv", help="Output CSV path")

    args = parser.parse_args()

    if args.command == "train":
        panels = ["general", "cancer", "pah", "cftr"] if args.panel == "all" else [args.panel]
        for pn in panels:
            print(f"\n{'='*60}")
            train_and_save(pn)

    elif args.command == "infer":
        predict(args.input, args.panel, args.output)


if __name__ == "__main__":
    main()
