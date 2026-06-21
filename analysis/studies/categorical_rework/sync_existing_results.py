"""Populate the central experiment log from artifacts already on disk."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from experiment_logger import DEFAULT_CSV, log_experiment_result, write_decision_log


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sync_preprocessing():
    specs = [
        ("MASTER", "strict"),
        ("MASTER", "no_rowdrop"),
        ("CANCER", "strict"),
        ("CANCER", "no_rowdrop"),
    ]
    for panel, mode in specs:
        meta_path = OUT / f"{panel}_preprocessing_metadata_{mode}.json"
        csv_path = OUT / f"{panel}_processed_numeric_{mode}.csv"
        if not meta_path.exists() or not csv_path.exists():
            continue
        meta = _json(meta_path)
        df = pd.read_csv(csv_path)
        y = df["Label"].astype(int)
        original_shape = meta["original_shape"]
        input_rows, input_cols = int(original_shape[0]), int(original_shape[1])
        log_experiment_result({
            "panel": panel,
            "experiment_name": f"preprocessing_{panel.lower()}_{mode}_synced",
            "pipeline_role": "ablation" if panel == "CANCER" and mode == "strict" else "primary",
            "preprocess_mode": mode,
            "row_drop_enabled": meta["row_drop_enabled"],
            "pca_enabled": False,
            "pca_variant": "none",
            "scaling_enabled": False,
            "outlier_control_enabled": False,
            "imputation_method": "median",
            "encoding_method": "safe ordinal / label encoding",
            "feature_selection_method": "none",
            "model_name": "n/a",
            "calibration_method": "n/a",
            "threshold_strategy": "n/a",
            "selected_threshold": "n/a",
            "blend_enabled": False,
            "blend_weight": "n/a",
            "general_transfer_weight": "n/a",
            "cancer_own_weight": "n/a",
            "cv_strategy": "n/a",
            "n_folds": "n/a",
            "input_rows": input_rows,
            "output_rows": len(df),
            "dropped_rows": meta["dropped_rows_count"],
            "input_features": input_cols - 1,
            "output_features": len(meta["final_feature_list"]),
            "dropped_features": len(meta["dropped_columns"]),
            "positive_ratio_before": "see preprocessing_mode_comparison.csv",
            "positive_ratio_after": float(y.mean()),
            "auc": "n/a",
            "f1_star": "n/a",
            "finals_adjusted_f1": "n/a",
            "precision": "n/a",
            "recall": "n/a",
            "pred_positive_mean": "n/a",
            "delta_f1_vs_baseline": "n/a",
            "notes": "Synced from existing preprocessing artifact.",
            "output_file": csv_path,
        })


def sync_pca():
    summary_path = OUT / "pca_experiments" / "KANSER_no_rowdrop_pca_evaluation_summary.csv"
    if not summary_path.exists():
        return
    df = pd.read_csv(summary_path)
    baseline = df.loc[df["experiment"].eq("baseline"), "f1_star"]
    baseline_f1 = float(baseline.iloc[0]) if not baseline.empty else None
    variants = {
        "baseline": ("none", False, False, False, "primary"),
        "exp1": ("pca_before_imputation", True, False, False, "rejected"),
        "exp2": ("impute_pca", True, False, True, "rejected"),
        "exp3": ("scale_pca", True, True, True, "rejected"),
    }
    for _, row in df.iterrows():
        variant, pca_enabled, scaling, imputed, role = variants.get(
            row["experiment"],
            (row["experiment"], True, False, True, "ablation"),
        )
        log_experiment_result({
            "panel": "CANCER",
            "experiment_name": f"pca_cancer_{row['experiment']}_no_rowdrop_synced",
            "pipeline_role": role if row["experiment"] != "baseline" else "primary",
            "preprocess_mode": "no_rowdrop",
            "row_drop_enabled": False,
            "pca_enabled": pca_enabled,
            "pca_variant": variant,
            "scaling_enabled": scaling,
            "outlier_control_enabled": row["experiment"] in ("baseline", "exp2", "exp3"),
            "imputation_method": "median" if imputed or row["experiment"] == "baseline" else "none",
            "encoding_method": "safe ordinal / label encoding",
            "feature_selection_method": "none",
            "model_name": "fixed LightGBM",
            "calibration_method": "Platt/sigmoid",
            "threshold_strategy": "train-fold threshold optimized at test prior",
            "selected_threshold": "per_fold",
            "blend_enabled": False,
            "blend_weight": "n/a",
            "general_transfer_weight": 0.0,
            "cancer_own_weight": 1.0,
            "cv_strategy": "RepeatedStratifiedKFold 5x1",
            "n_folds": row["completed_folds"],
            "input_rows": 388,
            "output_rows": 388,
            "dropped_rows": 0,
            "input_features": 364,
            "output_features": row["pca_components_mean"] if pca_enabled else 344,
            "dropped_features": "per_fold",
            "positive_ratio_before": 0.690722,
            "positive_ratio_after": 0.690722,
            "auc": row["auc"],
            "f1_star": row["f1_star"],
            "finals_adjusted_f1": row["finals_adjusted_f1"],
            "precision": row["precision"],
            "recall": row["recall"],
            "pred_positive_mean": "n/a",
            "delta_f1_vs_baseline": float(row["f1_star"]) - baseline_f1 if baseline_f1 is not None else "n/a",
            "notes": "Synced from existing PCA evaluation summary.",
            "output_file": summary_path,
        })


def sync_cancer_decision():
    sanity_path = OUT / "cancer_primary" / "cancer_primary_sanity_check.csv"
    if sanity_path.exists():
        sanity = pd.read_csv(sanity_path).iloc[0]
        log_experiment_result({
            "panel": "CANCER",
            "experiment_name": "cancer_primary_sanity_no_rowdrop_no_pca_synced",
            "pipeline_role": "sanity_check",
            "preprocess_mode": "no_rowdrop",
            "row_drop_enabled": False,
            "pca_enabled": False,
            "pca_variant": "none",
            "scaling_enabled": False,
            "outlier_control_enabled": True,
            "imputation_method": "median",
            "encoding_method": "safe ordinal / label encoding",
            "feature_selection_method": "none",
            "model_name": "fixed LightGBM cancer-own",
            "calibration_method": "Platt/sigmoid",
            "threshold_strategy": "train-fold threshold optimized at test prior",
            "selected_threshold": "per_fold",
            "blend_enabled": False,
            "blend_weight": "n/a",
            "general_transfer_weight": 0.0,
            "cancer_own_weight": 1.0,
            "cv_strategy": "RepeatedStratifiedKFold 5x6",
            "n_folds": sanity["n_folds_completed"],
            "input_rows": 388,
            "output_rows": 388,
            "dropped_rows": 0,
            "input_features": 364,
            "output_features": 344,
            "dropped_features": 19,
            "positive_ratio_before": 0.690722,
            "positive_ratio_after": 0.690722,
            "auc": sanity["auc"],
            "f1_star": sanity["f1_star"],
            "finals_adjusted_f1": sanity["finals_adjusted_f1"],
            "precision": sanity["precision"],
            "recall": sanity["recall"],
            "pred_positive_mean": "n/a",
            "delta_f1_vs_baseline": sanity["delta_vs_previous_reproduced_baseline"],
            "notes": sanity["interpretation"],
            "output_file": sanity_path,
        })
    blend_path = OUT / "cancer_primary" / "cancer_primary_blend_weight_scan.csv"
    if blend_path.exists():
        blend = pd.read_csv(blend_path).iloc[0]
        log_experiment_result({
            "panel": "CANCER",
            "experiment_name": "cancer_blend_weight_scan_incomplete_synced",
            "pipeline_role": "sanity_check",
            "preprocess_mode": "no_rowdrop",
            "row_drop_enabled": False,
            "pca_enabled": False,
            "pca_variant": "none",
            "scaling_enabled": False,
            "outlier_control_enabled": True,
            "imputation_method": "median",
            "encoding_method": "safe ordinal / label encoding",
            "feature_selection_method": "none",
            "model_name": "fixed LightGBM + general transfer",
            "calibration_method": "Platt/sigmoid",
            "threshold_strategy": "grid_0.70_0.90 planned",
            "selected_threshold": "n/a",
            "blend_enabled": True,
            "blend_weight": blend["requested_weights"],
            "general_transfer_weight": blend["requested_weights"],
            "cancer_own_weight": "1-w",
            "cv_strategy": blend["status"],
            "n_folds": 0,
            "input_rows": 388,
            "output_rows": "n/a",
            "dropped_rows": 0,
            "input_features": 364,
            "output_features": 344,
            "dropped_features": 19,
            "positive_ratio_before": 0.690722,
            "positive_ratio_after": 0.690722,
            "auc": "n/a",
            "f1_star": "n/a",
            "finals_adjusted_f1": "n/a",
            "precision": "n/a",
            "recall": "n/a",
            "pred_positive_mean": "n/a",
            "delta_f1_vs_baseline": "n/a",
            "notes": blend["reason"],
            "output_file": blend_path,
        })


def main():
    sync_preprocessing()
    sync_pca()
    sync_cancer_decision()
    write_decision_log([
        "Strict row-drop was rejected as the primary CANCER preprocessing because it dropped 158/388 rows and changed the positive class ratio from 69.07% to 59.13%.",
        "CANCER primary preprocessing is now no_rowdrop.",
        "PCA was rejected for the primary CANCER pipeline because all PCA variants underperformed the no-PCA baseline.",
        "Primary CANCER pipeline is currently updated CAT features + no_rowdrop + safe encoding + median imputation + outlier control + no PCA.",
        "Calibration strategy remains Platt/sigmoid for now.",
        "Threshold strategy remains aligned with the previous conservative range around 0.80-0.82 for later blend experiments.",
    ])
    print(f"Synced existing results into {DEFAULT_CSV}")


if __name__ == "__main__":
    main()
