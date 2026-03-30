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
from utils import detect_leaky_columns, optimize_threshold_f1, KNOWN_LEAKY, PANELS
from run_pipeline import (
    preprocess_data, select_features_mrmr, get_model_params,
    train_multi_seed_ensemble,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


# ---------------------------------------------------------------------------
# Train & save
# ---------------------------------------------------------------------------

def train_and_save(panel_name):
    """Run pipeline on full training data and save all artifacts for inference."""
    panel_dir = os.path.join(MODELS_DIR, panel_name)
    os.makedirs(panel_dir, exist_ok=True)

    # Load
    df = pd.read_csv(PANELS[panel_name])
    y = df["Label"]
    X = df[[c for c in df.columns if c != "Label"]]
    print(f"  {panel_name.upper()}: {len(df)} samples ({int(y.sum())}P / {int((y==0).sum())}B)")

    # Leakage detection
    leaky = detect_leaky_columns(X, y, auc_threshold=0.99, verbose=False)
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
        X_train.copy(), X_val.copy(), X_val.copy(), corr_threshold=0.99)
    X_train_val = pd.concat([X_train_pp, X_val_pp], ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], ignore_index=True)
    print(f"  Preprocessing: {prep_info['final_features']} features")

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

    # Threshold
    f1_t, f1_val = optimize_threshold_f1(y_train_val, oof_probs)
    print(f"  Threshold: {f1_t} (F1={f1_val:.4f})")

    # Save artifacts
    artifacts = {
        "leaky_cols": all_leaky,
        "feature_cols": list(X.columns),  # after leakage removal
        "prep_fitted": prep_info["fitted"],
        "selected_features": selected,
        "models": models,
        "threshold": f1_t,
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
    """Apply saved preprocessing params to new data."""
    # 1. Clip outliers (only columns with IQR > 0, matching preprocess_data)
    lower, upper = fitted["clip_lower"], fitted["clip_upper"]
    for col in fitted["clip_cols"]:
        if col in X.columns:
            mask = X[col].notna()
            X.loc[mask, col] = X.loc[mask, col].clip(lower[col], upper[col])

    # 2. Missing indicators
    for col in fitted["miss_indicator_cols"]:
        if col in X.columns:
            X[f"{col}_missing"] = X[col].isna().astype(int)

    # 3. Median imputation
    X = X.fillna(fitted["medians"])

    # 4. Drop low-var and high-corr columns
    drop = [c for c in fitted["low_var_cols"] + fitted["high_corr_cols"] if c in X.columns]
    X = X.drop(columns=drop)

    return X


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

    # Drop Label column if present
    if "Label" in df.columns:
        df = df.drop(columns=["Label"])

    # Keep only feature columns the model expects (after leakage removal)
    available = [c for c in art["feature_cols"] if c in df.columns]
    missing = set(art["feature_cols"]) - set(df.columns)
    if missing:
        print(f"  WARNING: {len(missing)} expected columns missing from input")
    X = df[available].copy()

    # Apply preprocessing
    X = apply_preprocessing(X, art["prep_fitted"])

    # Select mRMR features
    selected = [c for c in art["selected_features"] if c in X.columns]
    missing_feats = set(art["selected_features"]) - set(X.columns)
    if missing_feats:
        print(f"  WARNING: {len(missing_feats)} selected features missing after preprocessing")
        for col in missing_feats:
            X[col] = 0  # zero-fill missing features
        selected = art["selected_features"]
    X = X[selected]

    # Predict (average across all seed models)
    probs = np.mean([m.predict_proba(X)[:, 1] for m in art["models"]], axis=0)
    preds = (probs >= art["threshold"]).astype(int)

    # Output
    out = pd.DataFrame({
        "prediction": preds,
        "probability": probs.round(6),
    })
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
