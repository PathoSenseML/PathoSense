"""MASTER v0.2: repeated-seed CatBoost and OOF threshold-policy stability audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
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
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parent
SEEDS = [7, 13, 21, 42, 77, 101, 202, 340, 512, 999]
N_SPLITS = 5
TARGET_PRIOR = {0: 0.85, 1: 0.15}
REQUIRED = [
    "master_train_set.csv", "master_test_set.csv",
    "master_baseline_predictions_oof.csv", "master_baseline_predictions_test.csv",
    "master_baseline_model_comparison.csv", "master_baseline_test_report.csv",
    "master_baseline_metadata.json", "master_v01_threshold_candidates_oof.csv",
    "master_v01_threshold_candidates_test_reporting_only.csv",
    "master_v01_selected_threshold.csv", "master_v01_metadata.json", "master_v01_report.md",
]
OUTPUTS = {
    "oof": ROOT / "master_v02_oof_stability_by_seed.csv",
    "test": ROOT / "master_v02_test_reporting_only_by_seed.csv",
    "summary": ROOT / "master_v02_stability_summary.csv",
    "correlations": ROOT / "master_v02_oof_test_correlation_diagnostics.csv",
    "policy": ROOT / "master_v02_selected_policy.json",
    "report": ROOT / "master_v02_report.md",
}
PARTIAL_OOF = ROOT / ".master_v02_oof_partial.csv"
PARTIAL_TEST = ROOT / ".master_v02_test_partial.csv"


def validate_inputs() -> None:
    missing = [name for name in REQUIRED if not (ROOT / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")


def catboost_frame(frame: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for col in categorical:
        result[col] = result[col].where(result[col].notna(), "__MISSING__").astype(str)
    return result


def make_model(seed: int) -> CatBoostClassifier:
    # Kept identical to the v0 CatBoost baseline except for the audited seed.
    return CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


def classification_metrics(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
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
        "weighted_final_like_precision": precision_score(
            y, pred, sample_weight=weights, zero_division=0
        ),
        "weighted_final_like_recall": recall_score(
            y, pred, sample_weight=weights, zero_division=0
        ),
        "weighted_final_like_f0_5": fbeta_score(
            y, pred, beta=0.5, sample_weight=weights, zero_division=0
        ),
        "weighted_final_like_f0_75": fbeta_score(
            y, pred, beta=0.75, sample_weight=weights, zero_division=0
        ),
    }


def select_weighted_f1_threshold(y: np.ndarray, prob: np.ndarray,
                                 weights: np.ndarray) -> tuple[float, dict]:
    # Candidate boundaries come exclusively from this seed's train OOF predictions.
    thresholds = np.unique(np.r_[0.0, prob, 1.0])
    best_key = None
    best_threshold = None
    best_metrics = None
    for threshold in thresholds:
        metrics = weighted_metrics(y, prob, float(threshold), weights)
        # Deterministic tie-break: lower threshold (higher recall) for equal weighted F1.
        key = (metrics["weighted_final_like_f1"], -float(threshold))
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def summarize(frame: pd.DataFrame, scope: str, metrics: list[str]) -> list[dict]:
    rows = []
    for metric in metrics:
        values = frame[metric].astype(float)
        rows.append({
            "scope": scope,
            "metric": metric,
            "mean": values.mean(),
            "std": values.std(ddof=1),
            "min": values.min(),
            "max": values.max(),
            "median": values.median(),
            "q25": values.quantile(0.25),
            "q75": values.quantile(0.75),
        })
    return rows


def spearman_diagnostic(x: pd.Series, y: pd.Series) -> float:
    # Spearman equals Pearson correlation of average ranks; avoids an extra dependency.
    return float(x.rank(method="average").corr(y.rank(method="average")))


def markdown_table(frame: pd.DataFrame) -> str:
    shown = frame.copy()
    for col in shown.select_dtypes(include="float").columns:
        shown[col] = shown[col].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
    header = "| " + " | ".join(shown.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(shown.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in shown.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


def main() -> None:
    validate_inputs()
    train = pd.read_csv(ROOT / "master_train_set.csv")
    test = pd.read_csv(ROOT / "master_test_set.csv")
    baseline_metadata = json.loads(
        (ROOT / "master_baseline_metadata.json").read_text(encoding="utf-8")
    )
    v01_selected = pd.read_csv(ROOT / "master_v01_selected_threshold.csv").iloc[0]
    v01_metadata = json.loads((ROOT / "master_v01_metadata.json").read_text(encoding="utf-8"))
    baseline_test = pd.read_csv(ROOT / "master_baseline_test_report.csv").iloc[0]

    features = baseline_metadata["feature_columns"]
    if "Variant_ID" in features or "Label" in features:
        raise ValueError("Variant_ID/Label leakage in baseline feature list")
    if baseline_metadata["selected_model"] != "CatBoostClassifier":
        raise ValueError("v0 selected model is not CatBoostClassifier")
    if v01_selected["selected_threshold_name"] != "weighted_final_like_f1_threshold":
        raise ValueError("v0.1 threshold policy does not match the requested policy")

    categorical = train[features].select_dtypes(include=["object", "category"]).columns.tolist()
    x_train = catboost_frame(train[features], categorical)
    x_test = catboost_frame(test[features], categorical)
    y_train = train["Label"].to_numpy(dtype=int)
    y_test = test["Label"].to_numpy(dtype=int)
    shares = pd.Series(y_train).value_counts(normalize=True).to_dict()
    class_weights = {0: TARGET_PRIOR[0] / shares[0], 1: TARGET_PRIOR[1] / shares[1]}
    sample_weights = np.where(y_train == 0, class_weights[0], class_weights[1])

    if PARTIAL_OOF.exists() and PARTIAL_TEST.exists():
        oof_rows = pd.read_csv(PARTIAL_OOF).to_dict("records")
        test_rows = pd.read_csv(PARTIAL_TEST).to_dict("records")
        completed_seeds = {int(row["seed"]) for row in oof_rows} & {
            int(row["seed"]) for row in test_rows
        }
        oof_rows = [row for row in oof_rows if int(row["seed"]) in completed_seeds]
        test_rows = [row for row in test_rows if int(row["seed"]) in completed_seeds]
        print(f"Resuming completed seeds: {sorted(completed_seeds)}", flush=True)
    else:
        oof_rows, test_rows, completed_seeds = [], [], set()
    for seed in SEEDS:
        if seed in completed_seeds:
            continue
        print(f"Seed {seed}: 5-fold OOF", flush=True)
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        oof_prob = np.full(len(train), np.nan, dtype=float)
        for fold, (fit_idx, valid_idx) in enumerate(cv.split(x_train, y_train), 1):
            model = make_model(seed)
            model.fit(x_train.iloc[fit_idx], y_train[fit_idx], cat_features=categorical)
            oof_prob[valid_idx] = model.predict_proba(x_train.iloc[valid_idx])[:, 1]
            print(f"  fold {fold}/{N_SPLITS}", flush=True)
        if np.isnan(oof_prob).any():
            raise RuntimeError(f"NaN OOF predictions for seed {seed}")

        threshold, weighted = select_weighted_f1_threshold(
            y_train, oof_prob, sample_weights
        )
        raw = classification_metrics(y_train, oof_prob, threshold)
        oof_rows.append({
            "seed": seed,
            "selected_threshold": threshold,
            "oof_auc": roc_auc_score(y_train, oof_prob),
            "oof_average_precision": average_precision_score(y_train, oof_prob),
            "oof_f1_at_selected_threshold": raw["f1"],
            "oof_precision": raw["precision"],
            "oof_recall": raw["recall"],
            "oof_specificity": raw["specificity"],
            "oof_fpr": raw["fpr"],
            "oof_balanced_accuracy": raw["balanced_accuracy"],
            "oof_predicted_positive_rate": raw["predicted_positive_rate"],
            **weighted,
            "oof_tn": raw["tn"], "oof_fp": raw["fp"],
            "oof_fn": raw["fn"], "oof_tp": raw["tp"],
        })

        print(f"Seed {seed}: full-train fit; test reporting only", flush=True)
        final_model = make_model(seed)
        final_model.fit(x_train, y_train, cat_features=categorical)
        test_prob = final_model.predict_proba(x_test)[:, 1]
        test_metric = classification_metrics(y_test, test_prob, threshold)
        test_rows.append({
            "seed": seed,
            "selected_threshold_from_oof": threshold,
            "test_auc": roc_auc_score(y_test, test_prob),
            "test_average_precision": average_precision_score(y_test, test_prob),
            "test_f1": test_metric["f1"],
            "test_precision": test_metric["precision"],
            "test_recall": test_metric["recall"],
            "test_specificity": test_metric["specificity"],
            "test_fpr": test_metric["fpr"],
            "test_balanced_accuracy": test_metric["balanced_accuracy"],
            "test_predicted_positive_rate": test_metric["predicted_positive_rate"],
            "test_tn": test_metric["tn"], "test_fp": test_metric["fp"],
            "test_fn": test_metric["fn"], "test_tp": test_metric["tp"],
        })
        pd.DataFrame(oof_rows).to_csv(PARTIAL_OOF, index=False)
        pd.DataFrame(test_rows).to_csv(PARTIAL_TEST, index=False)
        print(f"Seed {seed}: checkpoint saved", flush=True)

    oof = pd.DataFrame(oof_rows)
    test_reporting = pd.DataFrame(test_rows)
    oof.to_csv(OUTPUTS["oof"], index=False)
    test_reporting.to_csv(OUTPUTS["test"], index=False)

    oof_summary_metrics = [
        "selected_threshold", "weighted_final_like_f1", "oof_precision", "oof_recall",
        "oof_fpr", "oof_balanced_accuracy", "oof_predicted_positive_rate",
    ]
    test_summary_metrics = [
        "test_f1", "test_precision", "test_recall", "test_fpr",
        "test_balanced_accuracy", "test_fp", "test_fn",
    ]
    summary = pd.DataFrame(
        summarize(oof, "OOF train CV", oof_summary_metrics)
        + summarize(test_reporting, "test reporting only", test_summary_metrics)
    )
    summary.to_csv(OUTPUTS["summary"], index=False)

    joined = oof.merge(test_reporting, on="seed", validate="one_to_one")
    correlation_specs = [
        ("oof_weighted_final_like_f1_vs_test_f1", "weighted_final_like_f1", "test_f1"),
        ("oof_fpr_vs_test_fpr", "oof_fpr", "test_fpr"),
        ("oof_precision_vs_test_precision", "oof_precision", "test_precision"),
    ]
    correlation_rows = []
    for name, x_col, y_col in correlation_specs:
        correlation_rows.append({
            "diagnostic": name,
            "x_metric": x_col,
            "y_metric": y_col,
            "spearman_correlation": spearman_diagnostic(joined[x_col], joined[y_col]),
            "n_seeds": len(joined),
            "usage": "diagnostic only; not used for model, seed, or threshold selection",
        })
    correlations = pd.DataFrame(correlation_rows)
    correlations.to_csv(OUTPUTS["correlations"], index=False)

    threshold_std = float(oof.selected_threshold.std(ddof=1))
    test_f1_std = float(test_reporting.test_f1.std(ddof=1))
    fp_mean = float(test_reporting.test_fp.mean())
    fp_range = float(test_reporting.test_fp.max() - test_reporting.test_fp.min())
    fp_wide_limit = max(20.0, 0.25 * fp_mean)
    threshold_stable = threshold_std <= 0.05
    performance_stable = test_f1_std <= 0.05
    fp_stable = fp_range <= fp_wide_limit
    overall_stable = threshold_stable and performance_stable and fp_stable
    stability_text = (
        "CatBoost + weighted final-like OOF threshold policy is reasonably stable across seeds."
        if overall_stable else
        "CatBoost + weighted final-like OOF threshold policy is not sufficiently stable across seeds."
    )
    recommendation = (
        "Proceed to Step 4 ensemble experiments while preserving OOF-only policy selection."
        if overall_stable else
        "Revise the threshold policy or validation strategy before Step 4."
    )

    policy = {
        "model_family": "CatBoostClassifier",
        "policy": "weighted_final_like_oof_f1_threshold",
        "target_prior": {"benign": 0.85, "pathogenic": 0.15},
        "seeds": SEEDS,
        "selection_source": "OOF train CV only",
        "test_usage": "reporting only",
        "stability_rules": {
            "selected_threshold_std_max": 0.05,
            "test_reporting_only_f1_std_max": 0.05,
            "test_fp_range_max": fp_wide_limit,
        },
        "diagnostic_stability": {
            "threshold_stable": threshold_stable,
            "test_performance_stable": performance_stable,
            "test_fp_stable": fp_stable,
            "overall_stable": overall_stable,
        },
        "recommendation": recommendation,
    }
    OUTPUTS["policy"].write_text(json.dumps(policy, indent=2), encoding="utf-8")

    threshold_summary = summary.query(
        "scope == 'OOF train CV' and metric == 'selected_threshold'"
    ).iloc[0]
    fp_summary = summary.query(
        "scope == 'test reporting only' and metric == 'test_fp'"
    ).iloc[0]
    fn_summary = summary.query(
        "scope == 'test reporting only' and metric == 'test_fn'"
    ).iloc[0]
    report = f"""# MASTER v0.2 / Step 3 CatBoost threshold-policy stability

## Context and purpose

- v0 selected model: {baseline_metadata['selected_model']}; OOF-selected threshold={baseline_metadata['threshold']:.12g}; test reporting F1={baseline_test.test_f1:.6f}.
- v0.1 selected policy: {v01_metadata['selected_threshold_name']}; threshold={float(v01_selected.selected_threshold):.12g}; target prior benign=0.85/pathogenic=0.15.
- This step measures repeatability across fixed seeds without feature engineering, model-family selection, hyperparameter selection, ensemble, SHAP, AA conversion, or AL transform.
- Seeds: {SEEDS}; each uses 5-fold StratifiedKFold and the unchanged v0 CatBoost parameters.

**Test set is reporting-only and was not used to choose threshold, model, seed, or hyperparameters.** Every seed's threshold was selected exclusively from that seed's train OOF predictions.

## Per-seed OOF results

{markdown_table(oof)}

## Per-seed test results — reporting only

{markdown_table(test_reporting)}

## Stability summary

{markdown_table(summary)}

- Threshold std={threshold_std:.6f}; min={threshold_summary['min']:.6f}; max={threshold_summary['max']:.6f}. Rule: std <= 0.05.
- Test reporting-only F1 std={test_f1_std:.6f}. Rule: std <= 0.05.
- Test reporting-only FP mean={fp_summary['mean']:.3f}, std={fp_summary['std']:.3f}, min={fp_summary['min']:.0f}, max={fp_summary['max']:.0f}. Explicit diagnostic wide-range rule: range <= max(20, 25% of mean) = {fp_wide_limit:.3f}.
- Test reporting-only FN mean={fn_summary['mean']:.3f}, std={fn_summary['std']:.3f}, min={fn_summary['min']:.0f}, max={fn_summary['max']:.0f}.

## OOF–test correlation diagnostics

{markdown_table(correlations)}

These correlations are diagnostic only and did not select a seed, threshold, model, or hyperparameter.

## Stability interpretation

- Threshold stable: {threshold_stable}
- Test performance reasonably stable (reporting-only): {performance_stable}
- FP control stable under the explicit range rule (reporting-only): {fp_stable}
- Overall diagnostic conclusion: **{stability_text}**

## Recommended Step 4

{recommendation}

**Test set is reporting-only and was not used to choose threshold, model, seed, or hyperparameters.**
"""
    OUTPUTS["report"].write_text(report, encoding="utf-8")
    PARTIAL_OOF.unlink(missing_ok=True)
    PARTIAL_TEST.unlink(missing_ok=True)

    print("\n=== MASTER v0.2 stability summary ===")
    print(f"Seed count: {len(SEEDS)}")
    print(f"Mean selected threshold: {oof.selected_threshold.mean():.6f}")
    print(f"Std selected threshold: {threshold_std:.6f}")
    print(f"Mean OOF weighted-final-like F1: {oof.weighted_final_like_f1.mean():.6f}")
    print(f"Mean test reporting-only F1: {test_reporting.test_f1.mean():.6f}")
    print(f"Mean test precision: {test_reporting.test_precision.mean():.6f}")
    print(f"Mean test recall: {test_reporting.test_recall.mean():.6f}")
    print(f"Mean test FP: {test_reporting.test_fp.mean():.3f}")
    print(f"Mean test FN: {test_reporting.test_fn.mean():.3f}")
    print(f"Stability interpretation: {stability_text}")
    print(f"Recommended Step 4: {recommendation}")


if __name__ == "__main__":
    main()
