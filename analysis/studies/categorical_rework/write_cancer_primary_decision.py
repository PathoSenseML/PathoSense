"""Write CANCER primary decision log from completed sanity/PCA artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from experiment_logger import log_experiment_result, write_decision_log


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "cancer_primary"
PCA = ROOT / "outputs" / "pca_experiments" / "KANSER_baseline_no_rowdrop_pca_results.json"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = json.loads(PCA.read_text(encoding="utf-8"))
    s = data["summary"]
    sanity = pd.DataFrame([
        {
            "panel": "CANCER",
            "preprocess_mode": "no_rowdrop",
            "pipeline_role": "primary",
            "pca": "rejected_ablation_only",
            "calibration": "platt_sigmoid_unchanged",
            "model_scope": "cancer_own_only_no_transfer_blend",
            "n_folds_completed": s["n_folds_completed"],
            "f1_star": s["f1_mean"],
            "f1_star_se": s["f1_se"],
            "finals_adjusted_f1": s["f1_finals_mean"],
            "auc": s["auc_mean"],
            "precision": s["precision_mean"],
            "recall": s["recall_mean"],
            "previous_reproduced_baseline_f1_star": 0.7077,
            "previous_threshold_stability_f1_star": 0.7136,
            "previous_calibration_threshold_grid_f1_star": 0.7162,
            "delta_vs_previous_reproduced_baseline": s["f1_mean"] - 0.7077,
            "interpretation": (
                "Lower than previous ~0.71 references because this sanity run is "
                "cancer-own only with updated CAT/no_rowdrop/no-PCA/no mRMR; the "
                "previous references used the historical cancer/general-transfer "
                "blend and threshold/calibration scans."
            ),
        }
    ])
    sanity_path = OUT / "cancer_primary_sanity_check.csv"
    sanity.to_csv(sanity_path, index=False)
    log_experiment_result({
        "panel": "CANCER",
        "experiment_name": "cancer_primary_sanity_no_rowdrop_no_pca",
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
        "n_folds": s["n_folds_completed"],
        "input_rows": 388,
        "output_rows": 388,
        "dropped_rows": 0,
        "input_features": 364,
        "output_features": 344,
        "dropped_features": 19,
        "positive_ratio_before": 0.690722,
        "positive_ratio_after": 0.690722,
        "auc": s["auc_mean"],
        "f1_star": s["f1_mean"],
        "finals_adjusted_f1": s["f1_finals_mean"],
        "precision": s["precision_mean"],
        "recall": s["recall_mean"],
        "pred_positive_mean": "n/a",
        "delta_f1_vs_baseline": s["f1_mean"] - 0.7077,
        "notes": sanity.iloc[0]["interpretation"],
        "output_file": sanity_path,
    })

    blend_status = pd.DataFrame([
        {
            "status": "incomplete_timeout",
            "requested_weights": "0.0,0.1,...,1.0",
            "requested_threshold_range": "0.70-0.90; track 0.80,0.81,0.82",
            "attempted_methods": (
                "nested OOF fixed LightGBM + source transfer; then holdout-calibration "
                "fast scan with capped estimators"
            ),
            "completed_result_rows": 0,
            "reason": "LightGBM fold training in blend scanner exceeded tool runtime before reports could be written.",
            "next_action": "Run blend scan in project environment/terminal with longer runtime, or reduce model budget explicitly.",
        }
    ])
    blend_path = OUT / "cancer_primary_blend_weight_scan.csv"
    blend_status.to_csv(blend_path, index=False)
    log_experiment_result({
        "panel": "CANCER",
        "experiment_name": "cancer_blend_weight_scan_requested_incomplete",
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
        "blend_weight": "0.0..1.0 planned",
        "general_transfer_weight": "0.0..1.0 planned",
        "cancer_own_weight": "1.0..0.0 planned",
        "cv_strategy": "attempted; incomplete timeout",
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
        "notes": blend_status.iloc[0]["reason"],
        "output_file": blend_path,
    })

    decision = {
        "frozen_decisions": [
            "strict row-drop rejected for CANCER; ablation only",
            "no_rowdrop selected as primary CANCER preprocessing",
            "PCA rejected for CANCER; ablation only",
            "Platt/sigmoid calibration unchanged for now",
        ],
        "primary_pipeline": "updated CAT features + no_rowdrop + safe encoding + median imputation + outlier control + no PCA",
        "sanity_result": sanity.iloc[0].to_dict(),
        "blend_scan_status": blend_status.iloc[0].to_dict(),
        "next_selected_blend_threshold_result": None,
    }
    decision_path = OUT / "cancer_decision_log.json"
    decision_path.write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")
    write_decision_log([
        "Strict row-drop was rejected as the primary CANCER preprocessing because it dropped 158/388 rows and changed the positive class ratio from 69.07% to 59.13%.",
        "CANCER primary preprocessing is now no_rowdrop.",
        "PCA was rejected for the primary CANCER pipeline because all PCA variants underperformed the no-PCA baseline.",
        "Primary CANCER pipeline is updated CAT features + no_rowdrop + safe encoding + median imputation + outlier control + no PCA.",
        "Calibration strategy remains Platt/sigmoid for now.",
        "Threshold strategy remains aligned with the previous conservative range around 0.80-0.82 for later blend experiments.",
    ])

    report = (
        "# CANCER Primary Decision Log\n\n"
        "- Strict row-drop rejected; ablation only.\n"
        "- `no_rowdrop` selected as primary preprocessing.\n"
        "- PCA rejected; ablation only.\n"
        "- Platt/sigmoid calibration unchanged for now.\n\n"
        "## Sanity Check\n\n"
        f"F1*: {s['f1_mean']:.4f}; finals-adjusted F1: {s['f1_finals_mean']:.4f}; "
        f"AUC: {s['auc_mean']:.4f}; precision: {s['precision_mean']:.4f}; "
        f"recall: {s['recall_mean']:.4f}.\n\n"
        "This is lower than the previous ~0.71 references because it is cancer-own only, "
        "uses the updated CAT/no-row-drop/no-PCA preprocessing, does not use mRMR, and "
        "does not include the final general-transfer blend.\n\n"
        "## Blend Scan\n\n"
        "Blend scan was attempted but did not complete within the tool runtime. No blend "
        "result is selected yet.\n"
    )
    (OUT / "cancer_primary_decision_log.md").write_text(report, encoding="utf-8")
    print(sanity.to_string(index=False))
    print(f"Saved {sanity_path}")
    print(f"Saved {blend_path}")
    print(f"Saved {decision_path}")


if __name__ == "__main__":
    main()
