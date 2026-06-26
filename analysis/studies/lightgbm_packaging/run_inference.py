"""
run_inference.py — Almira LightGBM Final Model Inference
=========================================================
Usage:
    python run_inference.py --input input.csv --output predictions.csv

Input CSV must contain the same columns as the training data
(YARISMA_TRAIN_MASTER.csv format). Label column is optional.

Output columns:
    Variant_ID, final_probability, final_prediction,
    locked_threshold, model_name
"""
import argparse
import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings("ignore")

MODEL_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_PATH = os.path.join(MODEL_PKG_DIR, "..", "models", "lightgbm_generalmodel.pkl")
MODEL_NAME = "almira_lightgbm_model"


def load_package(pkl_path: str) -> dict:
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Model file not found: {pkl_path}\n"
                                f"Run build script first.")
    pkg = joblib.load(pkl_path)
    if pkg.get("model_name") != MODEL_NAME:
        raise ValueError(f"Unexpected model_name in pkl: {pkg.get('model_name')}")
    return pkg


def apply_pipeline(X_raw: pd.DataFrame, pkg: dict) -> np.ndarray:
    """Apply feature engineering pipeline and return final mean probabilities."""
    leaky_cols     = pkg["leaky_cols"]
    al_cols        = pkg["al_cols"]
    cap_bounds     = pkg["cap_bounds"]
    cat_cols       = pkg["cat_cols"]
    encoder        = pkg["encoder"]
    selected_feats = pkg["feature_cols"]
    models         = pkg["models"]

    X = X_raw.copy()

    # Drop leaky columns
    X = X.drop(columns=[c for c in leaky_cols if c in X.columns], errors="ignore")

    # Capped signed_log AL features
    for c in al_cols:
        if c in X.columns and c in cap_bounds:
            vals = pd.to_numeric(X[c], errors="coerce")
            lo   = cap_bounds[c]["lower"]
            hi   = cap_bounds[c]["upper"]
            clipped = vals.clip(lo, hi)
            feat_name = f"ALlogcap_{c}_signed_log"
            X[feat_name] = (
                np.sign(clipped) * np.log1p(np.abs(clipped))
            ).replace([np.inf, -np.inf], np.nan)

    # Ordinal encode categoricals
    cats_present = [c for c in cat_cols if c in X.columns]
    if cats_present:
        s = X[cats_present].astype(str).fillna("__NA__")
        X[cats_present] = encoder.transform(s)

    # Align to selected features
    for c in selected_feats:
        if c not in X.columns:
            X[c] = np.nan
    X = X[selected_feats]

    X_np = X.to_numpy()

    # Mean probability over 40 seed models
    probs_per_seed = np.array([m.predict_proba(X_np)[:, 1] for m in models])
    return probs_per_seed.mean(axis=0)


def main():
    parser = argparse.ArgumentParser(description="Almira LightGBM Model Inference")
    parser.add_argument("--input",  required=True, help="Input CSV file path")
    parser.add_argument("--output", required=True, help="Output predictions CSV path")
    parser.add_argument("--pkl",    default=PKL_PATH, help="Model pkl path (optional)")
    args = parser.parse_args()

    print(f"[{MODEL_NAME}] Loading model package...")
    pkg = load_package(os.path.abspath(args.pkl))
    threshold = pkg["threshold"]
    print(f"  n_seeds       : {pkg['n_seeds']}")
    print(f"  locked thresh : {threshold}")
    print(f"  n_features    : {len(pkg['feature_cols'])}")
    print(f"  final F1 (CV) : {pkg['final_exact_f1']}")

    print(f"\n[{MODEL_NAME}] Loading input: {args.input}")
    df_in = pd.read_csv(args.input)
    print(f"  Input shape: {df_in.shape}")

    id_col = "Variant_ID"
    ids = df_in[id_col].tolist() if id_col in df_in.columns else list(range(len(df_in)))
    X_raw = df_in.drop(columns=[c for c in [id_col, "Label"] if c in df_in.columns])

    print(f"[{MODEL_NAME}] Running inference...")
    probs = apply_pipeline(X_raw, pkg)
    preds = (probs >= threshold).astype(int)

    out_df = pd.DataFrame({
        "Variant_ID":        ids,
        "final_probability": probs.round(6),
        "final_prediction":  preds,
        "locked_threshold":  threshold,
        "model_name":        MODEL_NAME,
    })

    out_df.to_csv(args.output, index=False)
    print(f"[{MODEL_NAME}] Saved predictions: {args.output}")
    print(f"  Total rows  : {len(out_df)}")
    print(f"  Predicted P : {int(preds.sum())}")
    print(f"  Predicted B : {int((preds==0).sum())}")


if __name__ == "__main__":
    main()
