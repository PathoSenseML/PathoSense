"""G1: Detailed error analysis on the PAH panel.

PAH is the weakest panel (balanced F1 = 0.791 with E1_only). Engineered
features added 2026-05-18 hurt it (-0.04 to -0.07). This script:

  1. Re-trains PAH with the current best config (E1_only) and captures
     per-Variant_ID OOF probabilities.
  2. Computes confusion at the deployed (Platt + balanced) threshold.
  3. For each error class (FN, FP, TP, TN):
     - per-feature mean / std
     - top-5 features by mean-difference vs the correct class
  4. Identifies the most-confused Variant_IDs (probability near threshold).
  5. Looks specifically at the 62 benign variants — are FPs scattered or clustered?
  6. Writes a human-readable markdown report + per-variant CSV.

Output:
  analysis/output/pah_error_analysis.md
  analysis/output/pah_per_variant.csv
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipelines"))
from config import SEED
from schema import normalize_sentinels, to_categorical, partition_columns
from feature_engineering import engineer_aa_features
from utils import (
    detect_leaky_columns, optimize_threshold_f1_balanced,
    KNOWN_LEAKY, PANELS,
)
from run_pipeline import (
    preprocess_data, select_features_mrmr, get_model_params,
    train_multi_seed_ensemble,
)
from calibration import fit_platt, apply_platt

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)


def get_pah_oof_with_ids():
    """Train PAH with E1_only config (current BEST_CONFIG_PER_PANEL['pah'])
    and return per-Variant_ID OOF probabilities + the feature frame used."""
    df = pd.read_csv(PANELS["pah"])
    df = normalize_sentinels(df)
    df = to_categorical(df)
    df = engineer_aa_features(df)   # E1_only: no interactions

    _, num_cols, cat_cols, tgt = partition_columns(df)
    y_all = df[tgt]
    X_all = df[num_cols + cat_cols]
    variant_ids = df["Variant_ID"].copy()

    # Leakage (same as run_panel)
    leaky = detect_leaky_columns(X_all, y_all, auc_threshold=0.99, verbose=False)
    all_leaky = sorted(set(leaky) | (KNOWN_LEAKY & set(X_all.columns)))
    X_all = X_all.drop(columns=all_leaky)

    # Split (same seed)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_all, y_all, test_size=0.15, random_state=SEED, stratify=y_all)
    ids_tr = variant_ids.loc[X_tr.index].reset_index(drop=True)
    ids_va = variant_ids.loc[X_va.index].reset_index(drop=True)
    for d in (X_tr, X_va):
        d.reset_index(drop=True, inplace=True)
    for s in (y_tr, y_va):
        s.reset_index(drop=True, inplace=True)

    # Preprocess
    X_tr_pp, X_va_pp, _, _ = preprocess_data(
        X_tr.copy(), X_va.copy(), X_va.copy(), corr_threshold=0.95)
    X_tv = pd.concat([X_tr_pp, X_va_pp], ignore_index=True)
    y_tv = pd.concat([y_tr, y_va], ignore_index=True)
    ids_tv = pd.concat([ids_tr, ids_va], ignore_index=True)

    # mRMR
    X_tr_pp, X_va_pp, _, fs_info = select_features_mrmr(
        X_tr_pp, y_tr, X_va_pp, X_va_pp)
    selected = list(X_tr_pp.columns)
    X_tv = X_tv[selected]

    # Ensemble
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    params = get_model_params(len(X_tv), spw)
    n_folds = 3 if len(X_tv) < 200 else 5
    _, oof, _, _ = train_multi_seed_ensemble(
        X_tv, y_tv, X_va_pp[selected], params, n_seeds=5, n_folds=n_folds)

    # Platt
    platt = fit_platt(oof, y_tv.values)
    oof_cal = apply_platt(oof, platt)

    # Deployed threshold (Platt + balanced)
    t, _, _ = optimize_threshold_f1_balanced(
        y_tv.values, oof_cal, target_prior=0.5, n_bootstrap=200)

    # Return frame with raw + Platt-calibrated probs, labels, IDs, and the
    # selected-feature view of each variant.
    return {
        "ids":         ids_tv,
        "y":           y_tv,
        "oof_raw":     oof,
        "oof_cal":     oof_cal,
        "threshold":   t,
        "X_full":      X_tv,                          # selected features only
        "X_pre_mrmr":  pd.concat([X_tr_pp[selected].iloc[0:0]]),  # column hint only
        "selected_features": selected,
        "params":      params,
        "platt":       platt,
    }


def per_class_signatures(X, y, preds, selected_features, top_n=10):
    """For each error class, compute mean / std per feature and rank top features
    that separate that error class from its 'correct' counterpart.

    Returns dict: { 'FN_vs_TP': df, 'FP_vs_TN': df, 'all_means': df }
    """
    classes = {
        "TP": (y == 1) & (preds == 1),
        "FN": (y == 1) & (preds == 0),
        "TN": (y == 0) & (preds == 0),
        "FP": (y == 0) & (preds == 1),
    }
    means = pd.DataFrame({c: X[mask].mean(numeric_only=True) for c, mask in classes.items()})
    means = means.reindex(selected_features)

    diffs = {}
    if classes["FN"].sum() >= 2 and classes["TP"].sum() >= 2:
        diff_fn = (means["FN"] - means["TP"]).abs().dropna().sort_values(ascending=False)
        # Also include normalized version: divide by pooled std
        pooled_std = pd.concat([X[classes["FN"]], X[classes["TP"]]]).std(numeric_only=True)
        diff_fn_norm = ((means["FN"] - means["TP"]) / pooled_std.replace(0, 1)).abs().dropna().sort_values(ascending=False)
        diffs["FN_vs_TP_top"] = pd.DataFrame({
            "abs_diff": diff_fn.head(top_n),
            "abs_diff_normalized": diff_fn_norm.reindex(diff_fn.head(top_n).index),
            "mean_FN": means["FN"].reindex(diff_fn.head(top_n).index),
            "mean_TP": means["TP"].reindex(diff_fn.head(top_n).index),
        })

    if classes["FP"].sum() >= 2 and classes["TN"].sum() >= 2:
        diff_fp = (means["FP"] - means["TN"]).abs().dropna().sort_values(ascending=False)
        pooled_std = pd.concat([X[classes["FP"]], X[classes["TN"]]]).std(numeric_only=True)
        diff_fp_norm = ((means["FP"] - means["TN"]) / pooled_std.replace(0, 1)).abs().dropna().sort_values(ascending=False)
        diffs["FP_vs_TN_top"] = pd.DataFrame({
            "abs_diff": diff_fp.head(top_n),
            "abs_diff_normalized": diff_fp_norm.reindex(diff_fp.head(top_n).index),
            "mean_FP": means["FP"].reindex(diff_fp.head(top_n).index),
            "mean_TN": means["TN"].reindex(diff_fp.head(top_n).index),
        })

    return {"means": means, "diffs": diffs, "class_masks": classes}


def main():
    print("Re-training PAH with current best config (E1_only)...")
    res = get_pah_oof_with_ids()

    y         = res["y"].values
    oof_cal   = res["oof_cal"]
    oof_raw   = res["oof_raw"]
    t         = res["threshold"]
    preds     = (oof_cal >= t).astype(int)
    X         = res["X_full"]
    selected  = res["selected_features"]

    # Confusion
    tp = int(((preds == 1) & (y == 1)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    print(f"\nConfusion (at deployed t={t:.3f}):")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
    print(f"  Specificity={tn/max(tn+fp,1):.4f}  P-rate (training)={y.mean():.4f}")

    # Per-variant CSV
    per_variant = pd.DataFrame({
        "Variant_ID":  res["ids"].values,
        "y":           y,
        "prob_raw":    oof_raw.round(6),
        "prob_cal":    oof_cal.round(6),
        "pred":        preds,
        "error_class": np.where(
            preds == y,
            np.where(y == 1, "TP", "TN"),
            np.where(y == 1, "FN", "FP"),
        ),
        "abs_distance_from_threshold": np.abs(oof_cal - t).round(6),
    }).sort_values("prob_cal", ascending=False)
    per_variant.to_csv(os.path.join(OUT_DIR, "pah_per_variant.csv"), index=False)
    print(f"\nwrote pah_per_variant.csv ({len(per_variant)} rows)")

    # Per-error-class signatures
    sigs = per_class_signatures(X, y, preds, selected)
    means = sigs["means"]
    diffs = sigs["diffs"]

    # Most-confused variants (smallest distance to threshold, both classes)
    near_threshold = per_variant.nsmallest(15, "abs_distance_from_threshold")

    # Write report
    lines = []
    lines.append("# PAH Error Analysis (G1)\n")
    lines.append("Generated by `analysis/pah_error_analysis.py`. Goal: characterize PAH's "
                 "errors with the current `E1_only` configuration (Platt calibration only, "
                 "no engineered meta-features), to inform what intervention might lift its "
                 "balanced F1 above 0.791.\n")
    lines.append("## Confusion matrix\n")
    lines.append(f"At deployed threshold **{t:.3f}** (Platt-calibrated probabilities, "
                 f"balanced-prior tuned):\n")
    lines.append("| | Pred = Benign | Pred = Pathogenic |")
    lines.append("|---|---|---|")
    lines.append(f"| **Actual = Benign** | TN={tn} | FP={fp} |")
    lines.append(f"| **Actual = Pathogenic** | FN={fn} | TP={tp} |")
    lines.append("")
    lines.append(f"Precision: **{prec:.4f}** | Recall: **{rec:.4f}** | "
                 f"F1: **{f1:.4f}** | Specificity: **{tn/max(tn+fp,1):.4f}**")
    lines.append(f"\nTraining prior P-rate: **{y.mean():.4f}** "
                 f"(test prior per rules: 0.5; threshold tuned for that.)\n")

    # FN analysis
    if "FN_vs_TP_top" in diffs:
        lines.append(f"## False Negatives (FN, n={fn}) — pathogenic variants missed\n")
        lines.append("These are variants the model called Benign but were actually Pathogenic. "
                     "Top features where FN distribution differs from TP:\n")
        df_top = diffs["FN_vs_TP_top"]
        lines.append("| Feature | abs(diff) | |diff|/std | mean_FN | mean_TP |")
        lines.append("|---|---|---|---|---|")
        for feat, r in df_top.iterrows():
            lines.append(f"| {feat} | {r['abs_diff']:.4f} | "
                         f"{r['abs_diff_normalized']:.3f} | "
                         f"{r['mean_FN']:.4f} | {r['mean_TP']:.4f} |")
        lines.append("")

    # FP analysis (more interesting given user concern about benign class size)
    if "FP_vs_TN_top" in diffs:
        lines.append(f"## False Positives (FP, n={fp}) — benign variants called Pathogenic\n")
        lines.append("These are variants the model called Pathogenic but were actually Benign. "
                     "With only 62 benign variants total in PAH, every FP represents ~1.6% of "
                     "the benign class. Top features where FP distribution differs from TN:\n")
        df_top = diffs["FP_vs_TN_top"]
        lines.append("| Feature | abs(diff) | |diff|/std | mean_FP | mean_TN |")
        lines.append("|---|---|---|---|---|")
        for feat, r in df_top.iterrows():
            lines.append(f"| {feat} | {r['abs_diff']:.4f} | "
                         f"{r['abs_diff_normalized']:.3f} | "
                         f"{r['mean_FP']:.4f} | {r['mean_TN']:.4f} |")
        lines.append("")

    # Per-class means
    lines.append("## Mean of every selected feature, by error class\n")
    lines.append("| Feature | TN | FP | FN | TP |")
    lines.append("|---|---|---|---|---|")
    for feat in selected:
        if feat not in means.index:
            continue
        vals = []
        for c in ("TN", "FP", "FN", "TP"):
            v = means.loc[feat, c] if c in means.columns else float("nan")
            vals.append(f"{v:.4f}" if v == v else "—")
        lines.append(f"| {feat} | " + " | ".join(vals) + " |")
    lines.append("")

    # Most-confused variants
    lines.append("## Most-confused variants (probability nearest threshold)\n")
    lines.append("Variants where the model output sits near the decision boundary. These are "
                 "the variants most sensitive to any threshold change.\n")
    lines.append("| Variant_ID | y | prob_cal | error_class | abs distance from t |")
    lines.append("|---|---|---|---|---|")
    for _, r in near_threshold.iterrows():
        lines.append(f"| {r['Variant_ID']} | {r['y']} | {r['prob_cal']:.4f} "
                     f"| {r['error_class']} | {r['abs_distance_from_threshold']:.4f} |")
    lines.append("")

    # Summary of P-distribution across classes
    lines.append("## Prediction-probability summary by actual label\n")
    for label in (0, 1):
        sub = per_variant[per_variant["y"] == label]
        lab = "Benign" if label == 0 else "Pathogenic"
        lines.append(f"**{lab} (n={len(sub)})**: prob_cal mean={sub['prob_cal'].mean():.3f}, "
                     f"std={sub['prob_cal'].std():.3f}, "
                     f"min={sub['prob_cal'].min():.3f}, "
                     f"median={sub['prob_cal'].median():.3f}, "
                     f"max={sub['prob_cal'].max():.3f}")
        lines.append("")

    out_path = os.path.join(OUT_DIR, "pah_error_analysis.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
