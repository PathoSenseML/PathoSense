"""MASTER baseline v0.1: OOF-only threshold and prior-shift diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parent
TARGET_PRIOR = {0: 0.85, 1: 0.15}
REQUIRED = [
    "master_train_set.csv", "master_test_set.csv",
    "master_baseline_predictions_oof.csv", "master_baseline_predictions_test.csv",
    "master_baseline_model_comparison.csv", "master_baseline_test_report.csv",
    "master_baseline_metadata.json", "master_baseline_best_model.pkl",
    "master_baseline_report.md",
]
OUTPUTS = {
    "oof_candidates": ROOT / "master_v01_threshold_candidates_oof.csv",
    "test_candidates": ROOT / "master_v01_threshold_candidates_test_reporting_only.csv",
    "selected": ROOT / "master_v01_selected_threshold.csv",
    "test_predictions": ROOT / "master_v01_predictions_test_selected_threshold.csv",
    "missingness": ROOT / "master_v01_missingness_error_analysis.csv",
    "all_nan_train": ROOT / "master_v01_all_nan_rows_train.csv",
    "all_nan_test": ROOT / "master_v01_all_nan_rows_test.csv",
    "fp_analysis": ROOT / "master_v01_false_positive_analysis.csv",
    "fp_rows_v0": ROOT / "master_v01_false_positive_rows_v0.csv",
    "fp_rows_v01": ROOT / "master_v01_false_positive_rows_v01.csv",
    "report": ROOT / "master_v01_report.md",
    "metadata": ROOT / "master_v01_metadata.json",
}


def validate_inputs() -> None:
    missing = [name for name in REQUIRED if not (ROOT / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing v0 inputs: {missing}")


def error_types(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.select(
        [(y == 1) & (pred == 1), (y == 0) & (pred == 0),
         (y == 0) & (pred == 1), (y == 1) & (pred == 0)],
        ["TP", "TN", "FP", "FN"], default="UNKNOWN",
    )


def base_metrics(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else np.nan
    return {
        "f1": f1_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "specificity": specificity,
        "fpr": 1.0 - specificity,
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "predicted_positive_rate": pred.mean(),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def weighted_metrics(y: np.ndarray, prob: np.ndarray, threshold: float,
                     weights: np.ndarray) -> dict:
    pred = (prob >= threshold).astype(int)
    return {
        "weighted_final_like_f1": f1_score(y, pred, sample_weight=weights, zero_division=0),
        "weighted_final_like_precision": precision_score(y, pred, sample_weight=weights, zero_division=0),
        "weighted_final_like_recall": recall_score(y, pred, sample_weight=weights, zero_division=0),
        "weighted_final_like_f0_5": fbeta_score(y, pred, beta=0.5, sample_weight=weights, zero_division=0),
        "weighted_final_like_f0_75": fbeta_score(y, pred, beta=0.75, sample_weight=weights, zero_division=0),
    }


def threshold_grid(prob: np.ndarray) -> np.ndarray:
    # Every distinct prediction boundary plus endpoints; no test probabilities enter this grid.
    return np.unique(np.r_[0.0, np.asarray(prob, dtype=float), 1.0])


def choose_thresholds(y: np.ndarray, prob: np.ndarray, weights: np.ndarray,
                      original: float) -> list[dict]:
    grid = threshold_grid(prob)
    rows = []
    for threshold in grid:
        raw = base_metrics(y, prob, float(threshold))
        weighted = weighted_metrics(y, prob, float(threshold), weights)
        rows.append({"threshold": float(threshold), **raw, **weighted})
    search = pd.DataFrame(rows)

    def argmax(col: str) -> float:
        best = search.sort_values([col, "threshold"], ascending=[False, True]).iloc[0]
        return float(best["threshold"])

    def fp_limited(limit: float) -> float:
        eligible = search[search["fpr"] <= limit + 1e-15]
        best = eligible.sort_values(
            ["recall", "fpr", "threshold"], ascending=[False, True, True]
        ).iloc[0]
        return float(best["threshold"])

    predicted_rate = search.assign(distance=(search["predicted_positive_rate"] - 0.15).abs())
    ppr_threshold = float(predicted_rate.sort_values(
        ["distance", "threshold"], ascending=[True, True]
    ).iloc[0]["threshold"])

    return [
        {"threshold_name": "original_oof_f1_threshold", "threshold": original,
         "threshold_source": "v0 metadata; OOF train CV"},
        {"threshold_name": "default_0_5_threshold", "threshold": 0.5,
         "threshold_source": "predefined default; not selected on test"},
        {"threshold_name": "weighted_final_like_f1_threshold",
         "threshold": argmax("weighted_final_like_f1"),
         "threshold_source": "OOF train predictions; target prior weighted F1"},
        {"threshold_name": "weighted_final_like_f05_threshold",
         "threshold": argmax("weighted_final_like_f0_5"),
         "threshold_source": "OOF train predictions; target prior weighted F0.5"},
        {"threshold_name": "weighted_final_like_f075_threshold",
         "threshold": argmax("weighted_final_like_f0_75"),
         "threshold_source": "OOF train predictions; target prior weighted F0.75"},
        {"threshold_name": "balanced_accuracy_threshold", "threshold": argmax("balanced_accuracy"),
         "threshold_source": "OOF train predictions; maximum balanced accuracy"},
        {"threshold_name": "youden_j_threshold",
         "threshold": float(search.assign(youden=search.recall-search.fpr).sort_values(
             ["youden", "threshold"], ascending=[False, True]).iloc[0]["threshold"]),
         "threshold_source": "OOF train predictions; maximum Youden J"},
        {"threshold_name": "fp_limited_threshold_fpr_0_20", "threshold": fp_limited(0.20),
         "threshold_source": "OOF train predictions; benign FPR <= 0.20, maximum recall"},
        {"threshold_name": "fp_limited_threshold_fpr_0_15", "threshold": fp_limited(0.15),
         "threshold_source": "OOF train predictions; benign FPR <= 0.15, maximum recall"},
        {"threshold_name": "fp_limited_threshold_fpr_0_10", "threshold": fp_limited(0.10),
         "threshold_source": "OOF train predictions; benign FPR <= 0.10, maximum recall"},
        {"threshold_name": "predicted_positive_rate_0_15_threshold", "threshold": ppr_threshold,
         "threshold_source": "OOF train predictions; predicted positive rate nearest 0.15"},
    ]


def oof_table(y: np.ndarray, prob: np.ndarray, candidates: list[dict],
              weights: np.ndarray) -> pd.DataFrame:
    auc, ap = roc_auc_score(y, prob), average_precision_score(y, prob)
    result = []
    for candidate in candidates:
        raw = base_metrics(y, prob, candidate["threshold"])
        weighted = weighted_metrics(y, prob, candidate["threshold"], weights)
        result.append({
            **candidate, "oof_auc": auc, "oof_average_precision": ap,
            **{f"oof_{k}": v for k, v in raw.items()}, **weighted,
        })
    columns = [
        "threshold_name", "threshold", "threshold_source", "oof_auc", "oof_average_precision",
        "oof_f1", "oof_precision", "oof_recall", "oof_specificity", "oof_fpr",
        "oof_balanced_accuracy", "oof_predicted_positive_rate", "oof_tn", "oof_fp",
        "oof_fn", "oof_tp", "weighted_final_like_f1", "weighted_final_like_precision",
        "weighted_final_like_recall", "weighted_final_like_f0_5", "weighted_final_like_f0_75",
    ]
    return pd.DataFrame(result)[columns]


def test_table(y: np.ndarray, prob: np.ndarray, candidates: list[dict]) -> pd.DataFrame:
    auc, ap = roc_auc_score(y, prob), average_precision_score(y, prob)
    result = []
    for candidate in candidates:
        raw = base_metrics(y, prob, candidate["threshold"])
        result.append({
            "threshold_name": candidate["threshold_name"], "threshold": candidate["threshold"],
            "threshold_source": candidate["threshold_source"] + "; test reporting only",
            "test_auc": auc, "test_average_precision": ap,
            **{f"test_{k}": v for k, v in raw.items()},
        })
    columns = [
        "threshold_name", "threshold", "threshold_source", "test_auc", "test_average_precision",
        "test_f1", "test_precision", "test_recall", "test_specificity", "test_fpr",
        "test_balanced_accuracy", "test_predicted_positive_rate", "test_tn", "test_fp",
        "test_fn", "test_tp",
    ]
    return pd.DataFrame(result)[columns]


def missing_bucket(rate: float) -> str:
    if rate == 0:
        return "0.00"
    if rate <= 0.25:
        return "0.00-0.25"
    if rate <= 0.50:
        return "0.25-0.50"
    if rate <= 0.75:
        return "0.50-0.75"
    if rate < 1.0:
        return "0.75-0.99"
    return "1.00"


def row_missingness(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    missing_count = frame[features].isna().sum(axis=1)
    return pd.DataFrame({
        "missing_count": missing_count.astype(int),
        "missing_rate": missing_count / len(features),
        "is_all_features_missing": missing_count.eq(len(features)),
    })


def bucket_analysis(dataset: str, y: np.ndarray, prob: np.ndarray,
                    miss: pd.DataFrame, pred_v0: np.ndarray, pred_v01: np.ndarray) -> list[dict]:
    buckets = miss["missing_rate"].map(missing_bucket)
    result = []
    for bucket in ["0.00", "0.00-0.25", "0.25-0.50", "0.50-0.75", "0.75-0.99", "1.00"]:
        mask = buckets.eq(bucket).to_numpy()
        result.append({
            "dataset": dataset, "missing_rate_bucket": bucket, "row_count": int(mask.sum()),
            "benign_count": int(((y == 0) & mask).sum()),
            "pathogenic_count": int(((y == 1) & mask).sum()),
            "mean_pred_proba_pathogenic": float(prob[mask].mean()) if mask.any() else np.nan,
            "v0_error_counts": json.dumps(pd.Series(error_types(y[mask], pred_v0[mask])).value_counts().to_dict()),
            "v01_error_counts": json.dumps(pd.Series(error_types(y[mask], pred_v01[mask])).value_counts().to_dict()),
        })
    return result


def markdown_table(frame: pd.DataFrame) -> str:
    shown = frame.copy()
    for col in shown.select_dtypes(include="float").columns:
        shown[col] = shown[col].map(lambda x: "" if pd.isna(x) else f"{x:.6f}")
    head = "| " + " | ".join(shown.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(shown.columns)) + " |"
    rows = ["| " + " | ".join(map(str, x)) + " |" for x in shown.itertuples(index=False, name=None)]
    return "\n".join([head, sep, *rows])


def main() -> None:
    validate_inputs()
    train = pd.read_csv(ROOT / "master_train_set.csv")
    test = pd.read_csv(ROOT / "master_test_set.csv")
    oof = pd.read_csv(ROOT / "master_baseline_predictions_oof.csv")
    test_pred_v0_file = pd.read_csv(ROOT / "master_baseline_predictions_test.csv")
    metadata_v0 = json.loads((ROOT / "master_baseline_metadata.json").read_text(encoding="utf-8"))
    test_report_v0 = pd.read_csv(ROOT / "master_baseline_test_report.csv").iloc[0]
    comparison_v0 = pd.read_csv(ROOT / "master_baseline_model_comparison.csv")

    if len(train) != len(oof) or len(test) != len(test_pred_v0_file):
        raise ValueError("Prediction row counts do not match train/test sets")
    if not train["Variant_ID"].astype(str).equals(oof["Variant_ID"].astype(str)):
        raise ValueError("OOF predictions are not aligned with train Variant_ID order")
    if not test["Variant_ID"].astype(str).equals(test_pred_v0_file["Variant_ID"].astype(str)):
        raise ValueError("Test predictions are not aligned with test Variant_ID order")

    features = metadata_v0["feature_columns"]
    if "Label" in features or "Variant_ID" in features:
        raise ValueError("Baseline metadata contains excluded columns as features")
    y_oof = oof["true_label"].to_numpy(dtype=int)
    p_oof = oof["oof_proba_pathogenic"].to_numpy(dtype=float)
    y_test = test_pred_v0_file["true_label"].to_numpy(dtype=int)
    p_test = test_pred_v0_file["pred_proba_pathogenic"].to_numpy(dtype=float)
    original_threshold = float(metadata_v0["threshold"])

    shares = pd.Series(y_oof).value_counts(normalize=True).to_dict()
    class_weights = {0: TARGET_PRIOR[0] / shares[0], 1: TARGET_PRIOR[1] / shares[1]}
    sample_weights = np.where(y_oof == 0, class_weights[0], class_weights[1])
    candidates = choose_thresholds(y_oof, p_oof, sample_weights, original_threshold)
    oof_results = oof_table(y_oof, p_oof, candidates, sample_weights)
    test_results = test_table(y_test, p_test, candidates)
    oof_results.to_csv(OUTPUTS["oof_candidates"], index=False)
    test_results.to_csv(OUTPUTS["test_candidates"], index=False)

    selected_name = "weighted_final_like_f1_threshold"
    selected_threshold = float(oof_results.loc[
        oof_results.threshold_name.eq(selected_name), "threshold"
    ].iloc[0])
    selection_rule = "maximize OOF F1 under target final-like class prior weighting: benign=0.85, pathogenic=0.15"
    selected_row = {
        "selected_threshold_name": selected_name,
        "selected_threshold": selected_threshold,
        "selected_by": "weighted_final_like_oof_f1",
        "selection_source": "OOF train predictions only",
        "selection_rule": selection_rule,
        "warning": "Test metrics are reporting only and were not used to select or modify this threshold.",
    }
    pd.DataFrame([selected_row]).to_csv(OUTPUTS["selected"], index=False)

    pred_v0_train = (p_oof >= original_threshold).astype(int)
    pred_v01_train = (p_oof >= selected_threshold).astype(int)
    pred_v0_test = (p_test >= original_threshold).astype(int)
    pred_v01_test = (p_test >= selected_threshold).astype(int)
    miss_train = row_missingness(train, features)
    miss_test = row_missingness(test, features)

    selected_test_predictions = pd.DataFrame({
        "Variant_ID": test["Variant_ID"], "true_label": y_test,
        "pred_proba_pathogenic": p_test, "pred_label_v0": pred_v0_test,
        "pred_label_v01": pred_v01_test,
        "error_type_v0": error_types(y_test, pred_v0_test),
        "error_type_v01": error_types(y_test, pred_v01_test),
        "missing_count": miss_test.missing_count, "missing_rate": miss_test.missing_rate,
        "is_all_features_missing": miss_test.is_all_features_missing,
    })
    selected_test_predictions.to_csv(OUTPUTS["test_predictions"], index=False)

    missingness_rows = bucket_analysis(
        "train", y_oof, p_oof, miss_train, pred_v0_train, pred_v01_train
    ) + bucket_analysis("test", y_test, p_test, miss_test, pred_v0_test, pred_v01_test)
    missingness_table = pd.DataFrame(missingness_rows)
    missingness_table.to_csv(OUTPUTS["missingness"], index=False)

    all_nan_train_mask = miss_train.is_all_features_missing.to_numpy()
    all_nan_test_mask = miss_test.is_all_features_missing.to_numpy()
    train_feature_hash = pd.util.hash_pandas_object(train[features], index=False)
    test_feature_hash = pd.util.hash_pandas_object(test[features], index=False)
    shared_feature_hashes = set(train_feature_hash.tolist()) & set(test_feature_hash.tolist())
    shared_hash_is_all_nan = all(
        train.loc[train_feature_hash.eq(value), features].isna().all(axis=1).all()
        and test.loc[test_feature_hash.eq(value), features].isna().all(axis=1).all()
        for value in shared_feature_hashes
    )
    all_nan_train = pd.DataFrame({
        "Variant_ID": train.loc[all_nan_train_mask, "Variant_ID"],
        "true_label": y_oof[all_nan_train_mask],
        "oof_proba_pathogenic": p_oof[all_nan_train_mask],
        "pred_label_v0": pred_v0_train[all_nan_train_mask],
        "pred_label_v01": pred_v01_train[all_nan_train_mask],
        "error_type_v0": error_types(y_oof[all_nan_train_mask], pred_v0_train[all_nan_train_mask]),
        "error_type_v01": error_types(y_oof[all_nan_train_mask], pred_v01_train[all_nan_train_mask]),
        "missing_count": miss_train.loc[all_nan_train_mask, "missing_count"].to_numpy(),
        "missing_rate": miss_train.loc[all_nan_train_mask, "missing_rate"].to_numpy(),
    })
    all_nan_test = selected_test_predictions.loc[all_nan_test_mask].copy()
    all_nan_train.to_csv(OUTPUTS["all_nan_train"], index=False)
    all_nan_test.to_csv(OUTPUTS["all_nan_test"], index=False)

    fp_rows = []
    for label, threshold, pred in [
        ("v0", original_threshold, pred_v0_test), ("v0.1", selected_threshold, pred_v01_test)
    ]:
        fp_mask = (y_test == 0) & (pred == 1)
        fp_buckets = miss_test.loc[fp_mask, "missing_rate"].map(missing_bucket)
        fp_rows.append({
            "version": label, "threshold": threshold, "fp_count": int(fp_mask.sum()),
            "mean_fp_pred_proba_pathogenic": float(p_test[fp_mask].mean()) if fp_mask.any() else np.nan,
            "mean_fp_missing_rate": float(miss_test.loc[fp_mask, "missing_rate"].mean()) if fp_mask.any() else np.nan,
            "most_common_fp_missingness_bucket": fp_buckets.mode().iloc[0] if len(fp_buckets) else "none",
            "analysis_purpose": "diagnostics only; no threshold selection from test FP",
        })
    fp_analysis = pd.DataFrame(fp_rows)
    fp_analysis.to_csv(OUTPUTS["fp_analysis"], index=False)
    selected_test_predictions.loc[selected_test_predictions.error_type_v0.eq("FP")].head(50).to_csv(
        OUTPUTS["fp_rows_v0"], index=False
    )
    selected_test_predictions.loc[selected_test_predictions.error_type_v01.eq("FP")].head(50).to_csv(
        OUTPUTS["fp_rows_v01"], index=False
    )

    v0_test = test_results.loc[test_results.threshold_name.eq("original_oof_f1_threshold")].iloc[0]
    v01_oof = oof_results.loc[oof_results.threshold_name.eq(selected_name)].iloc[0]
    v01_test = test_results.loc[test_results.threshold_name.eq(selected_name)].iloc[0]
    all_nan_train_dist = pd.Series(y_oof[all_nan_train_mask]).value_counts().reindex([0, 1], fill_value=0).to_dict()
    all_nan_test_dist = pd.Series(y_test[all_nan_test_mask]).value_counts().reindex([0, 1], fill_value=0).to_dict()
    all_nan_train_share = pd.Series(y_oof[all_nan_train_mask]).value_counts(normalize=True).reindex([0, 1], fill_value=0).to_dict()
    all_nan_test_share = pd.Series(y_test[all_nan_test_mask]).value_counts(normalize=True).reindex([0, 1], fill_value=0).to_dict()

    created_files = [Path(__file__).name, *[p.name for p in OUTPUTS.values()]]
    metadata_v01 = {
        "project_folder": "TeknoPatho", "source_baseline": "master_baseline_v0",
        "selected_threshold_name": selected_name, "selected_threshold": selected_threshold,
        "selected_by": "weighted_final_like_oof_f1",
        "selection_source": "OOF train predictions only", "target_prior": {"benign": 0.85, "pathogenic": 0.15},
        "created_files": created_files,
        "warning_about_test_reporting_only": "Test set is used only for reporting, not for threshold selection.",
    }
    OUTPUTS["metadata"].write_text(json.dumps(metadata_v01, indent=2, ensure_ascii=False), encoding="utf-8")

    fp_comment = (
        f"The OOF-selected v0.1 threshold yields {int(v01_test.test_fp)} FP on test. "
        "This is diagnostic information only and does not alter the selected threshold."
    )
    winner_v0 = comparison_v0.loc[comparison_v0.model_name.eq(metadata_v0["selected_model"])].iloc[0]
    report = f"""# MASTER baseline v0.1 threshold / prior-shift analysis

## Baseline v0 summary

- Model: {metadata_v0['selected_model']}
- v0 OOF F1: {winner_v0.best_oof_f1:.6f}
- v0 threshold: {original_threshold:.12g}
- v0 test: F1={test_report_v0.test_f1:.6f}, precision={test_report_v0.test_precision:.6f}, recall={test_report_v0.test_recall:.6f}, TN={int(test_report_v0.test_tn)}, FP={int(test_report_v0.test_fp)}, FN={int(test_report_v0.test_fn)}, TP={int(test_report_v0.test_tp)}
- Diagnostic problem: high recall, low precision, and high false-positive count under the prior-shifted test distribution.
- Mechanism: the original OOF-F1 threshold already has benign FPR={float(oof_results.loc[oof_results.threshold_name.eq('original_oof_f1_threshold'), 'oof_fpr'].iloc[0]):.6f}. Pathogenic-heavy unweighted OOF F1 tolerates many benign errors; when the test population is benign-heavy, that FPR becomes a large absolute FP count. This is a prior/objective mismatch diagnosis, not a test-driven threshold choice.

## Protocol

**Threshold candidates are selected from OOF predictions only.**

**Test set is used only for reporting, not for threshold selection.** No model reselection, feature engineering, ensemble, SHAP, AA conversion, or AL transform was performed.

OOF observed shares: benign={shares[0]:.6f}, pathogenic={shares[1]:.6f}. Target prior: benign=0.85, pathogenic=0.15. Applied OOF sample weights: benign={class_weights[0]:.6f}, pathogenic={class_weights[1]:.6f}.

## OOF threshold candidates

{markdown_table(oof_results)}

## Test metrics — reporting only

{markdown_table(test_results)}

## v0.1 selection

- Selected threshold: {selected_threshold:.12g} (`{selected_name}`)
- Selected by: weighted_final_like_oof_f1
- Rule: {selection_rule}
- v0 vs v0.1 test confusion matrices: v0 TN={int(v0_test.test_tn)}, FP={int(v0_test.test_fp)}, FN={int(v0_test.test_fn)}, TP={int(v0_test.test_tp)}; v0.1 TN={int(v01_test.test_tn)}, FP={int(v01_test.test_fp)}, FN={int(v01_test.test_fn)}, TP={int(v01_test.test_tp)}.
- {fp_comment}

## Missingness and all-NaN analysis

- Feature count: {len(features)}
- Train all-NaN rows: {int(all_nan_train_mask.sum())}; label distribution: {all_nan_train_dist}; label shares: {all_nan_train_share}
- Test all-NaN rows: {int(all_nan_test_mask.sum())}; label distribution: {all_nan_test_dist}; label shares: {all_nan_test_share}
- Test all-NaN v0 errors: {pd.Series(error_types(y_test[all_nan_test_mask], pred_v0_test[all_nan_test_mask])).value_counts().to_dict()}
- Test all-NaN v0.1 errors: {pd.Series(error_types(y_test[all_nan_test_mask], pred_v01_test[all_nan_test_mask])).value_counts().to_dict()}
- Shared unique feature hashes across train/test: {len(shared_feature_hashes)}. All shared hashes represent all-feature-missing rows: {shared_hash_is_all_nan}. Thus the v0 warning is explained by repeated all-NaN feature vectors, not by shared Variant_IDs.

{markdown_table(missingness_table)}

## False-positive diagnostics

{markdown_table(fp_analysis)}

The FP analysis is diagnostic only. It was not used to select or modify the threshold.

## Suggested next step

Validate the OOF prior-weighted threshold policy across repeated train-only CV seeds or bootstrap resamples before changing the model family. Keep the test set reporting-only throughout that validation.

**v0.1 threshold was selected using OOF predictions under final-like prior weighting; test metrics are reported only for diagnostics.**
"""
    OUTPUTS["report"].write_text(report, encoding="utf-8")

    print("\n=== MASTER v0.1 threshold / prior-shift summary ===")
    print(f"v0 threshold: {original_threshold:.12g}")
    print(f"v0 test F1 / precision / recall / FP / FN: {v0_test.test_f1:.6f} / {v0_test.test_precision:.6f} / {v0_test.test_recall:.6f} / {int(v0_test.test_fp)} / {int(v0_test.test_fn)}")
    print(f"v0.1 selected threshold: {selected_threshold:.12g}")
    print("v0.1 selected-by rule: weighted_final_like_oof_f1")
    print(f"v0.1 OOF weighted-final-like F1: {v01_oof.weighted_final_like_f1:.6f}")
    print(f"v0.1 test reporting-only F1 / precision / recall / FP / FN: {v01_test.test_f1:.6f} / {v01_test.test_precision:.6f} / {v01_test.test_recall:.6f} / {int(v01_test.test_fp)} / {int(v01_test.test_fn)}")
    print(f"all-NaN train label distribution: {all_nan_train_dist}")
    print(f"all-NaN test label distribution: {all_nan_test_dist}")
    print(f"FP change: {int(v0_test.test_fp)} -> {int(v01_test.test_fp)}")
    print("v0.1 threshold was selected using OOF predictions under final-like prior weighting; test metrics are reported only for diagnostics.")


if __name__ == "__main__":
    main()
