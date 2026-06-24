"""MASTER Step 3b: repeated split CatBoost stability and leakage audit."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from master_v02_catboost_threshold_stability import (
    catboost_frame,
    classification_metrics,
    make_model,
    markdown_table,
    select_weighted_f1_threshold,
)


ROOT = Path(__file__).resolve().parent
EXPECTED_INPUT = Path(r"C:\Users\AET\Documents\Teknofest sağlık\data\panels\YARISMA_TRAIN_MASTER.csv")
SEARCH_ROOTS = [
    Path(r"C:\Users\AET\Documents\Teknofest sağlık"),
    Path(r"C:\Users\AET\Desktop\PathoSense"),
    ROOT,
]
SEEDS = [
    7, 13, 21, 42, 77, 101, 202, 340, 512, 999,
    1234, 2024, 3141, 4096, 7777, 10001, 17017, 27182, 42424, 65535,
]
N_SPLITS = 5
TARGET_PRIOR = {0: 0.85, 1: 0.15}
ESTIMATED_TOTAL_RUNTIME_MINUTES = 120
PARTIALS = {
    "oof": ROOT / ".master_step3b_partial_oof.csv",
    "test": ROOT / ".master_step3b_partial_test.csv",
    "leakage": ROOT / ".master_step3b_partial_leakage.csv",
    "runtime": ROOT / ".master_step3b_partial_runtime.csv",
    "all_nan": ROOT / ".master_step3b_partial_all_nan.csv",
}
OUTPUTS = {
    "split": ROOT / "master_step3b_seed_split_summary.csv",
    "leakage": ROOT / "master_step3b_leakage_audit_by_seed.csv",
    "oof": ROOT / "master_step3b_oof_metrics_by_seed.csv",
    "test": ROOT / "master_step3b_test_metrics_by_seed.csv",
    "summary": ROOT / "master_step3b_repeated_test_summary.csv",
    "all_nan": ROOT / "master_step3b_all_nan_analysis_by_seed.csv",
    "runtime": ROOT / "master_step3b_runtime_report.csv",
    "report": ROOT / "master_step3b_report.md",
    "metadata": ROOT / "master_step3b_metadata.json",
}


def find_input() -> Path:
    if EXPECTED_INPUT.exists():
        return EXPECTED_INPUT.resolve()
    matches = []
    for search_root in SEARCH_ROOTS:
        if search_root.exists():
            matches.extend(search_root.rglob("YARISMA_TRAIN_MASTER.csv"))
    matches = sorted(set(matches))
    if not matches:
        raise FileNotFoundError("YARISMA_TRAIN_MASTER.csv not found")
    return matches[0].resolve()


def optimal_split_counts(n_benign: int, n_pathogenic: int) -> tuple[int, int]:
    matrix = np.array([[0.85, -0.15], [-0.15, 0.85]], dtype=float)
    rhs = np.array([0.0, 0.85 * n_pathogenic - 0.15 * n_benign], dtype=float)
    target_b, target_p = np.linalg.solve(matrix, rhs)
    candidates = []
    for benign in range(max(1, math.floor(target_b) - 5), min(n_benign, math.ceil(target_b) + 6)):
        for pathogenic in range(max(1, math.floor(target_p) - 5), min(n_pathogenic, math.ceil(target_p) + 6)):
            train_share = benign / (benign + pathogenic)
            test_share = (n_benign - benign) / (
                n_benign - benign + n_pathogenic - pathogenic
            )
            ratio_error = (train_share - 0.15) ** 2 + (test_share - 0.85) ** 2
            distance = (benign - target_b) ** 2 + (pathogenic - target_p) ** 2
            candidates.append((ratio_error, distance, benign, pathogenic))
    _, _, benign_train, pathogenic_train = min(candidates)
    return benign_train, pathogenic_train


def make_split(full: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = full["Label"].value_counts().to_dict()
    benign_train, pathogenic_train = optimal_split_counts(int(counts[0]), int(counts[1]))
    rng = np.random.RandomState(seed)
    benign_idx = full.index[full.Label.eq(0)].to_numpy().copy()
    pathogenic_idx = full.index[full.Label.eq(1)].to_numpy().copy()
    rng.shuffle(benign_idx)
    rng.shuffle(pathogenic_idx)
    train_idx = np.r_[benign_idx[:benign_train], pathogenic_idx[:pathogenic_train]]
    test_idx = np.r_[benign_idx[benign_train:], pathogenic_idx[pathogenic_train:]]
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return full.loc[train_idx].reset_index(drop=True), full.loc[test_idx].reset_index(drop=True)


def error_types(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.select(
        [(y == 1) & (pred == 1), (y == 0) & (pred == 0),
         (y == 0) & (pred == 1), (y == 1) & (pred == 0)],
        ["TP", "TN", "FP", "FN"], default="UNKNOWN",
    )


def leakage_audit(full: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame,
                  features: list[str], seed: int, full_duplicate_rows: int) -> tuple[dict, dict]:
    train_ids = set(train.Variant_ID.dropna().astype(str))
    test_ids = set(test.Variant_ID.dropna().astype(str))
    overlap = train_ids & test_ids
    train_dup = int(train.Variant_ID.duplicated(keep=False).sum())
    test_dup = int(test.Variant_ID.duplicated(keep=False).sum())
    train_hash = pd.util.hash_pandas_object(train[features], index=False)
    test_hash = pd.util.hash_pandas_object(test[features], index=False)
    shared_hashes = set(train_hash.tolist()) & set(test_hash.tolist())
    shared_train_mask = train_hash.isin(shared_hashes)
    shared_test_mask = test_hash.isin(shared_hashes)
    shared_all_nan = bool(shared_hashes) and all(
        train.loc[train_hash.eq(value), features].isna().all(axis=1).all()
        and test.loc[test_hash.eq(value), features].isna().all(axis=1).all()
        for value in shared_hashes
    )
    suspicious_tokens = ("label", "target", "outcome", "variant_id", "variantid")
    suspicious = [col for col in features if any(token in col.lower() for token in suspicious_tokens)]
    feature_leak = "Label" in features or "Variant_ID" in features
    failures = []
    if overlap:
        failures.append(f"Variant_ID overlap={len(overlap)}")
    if "Label" in features:
        failures.append("Label is in feature list")
    if "Variant_ID" in features:
        failures.append("Variant_ID is in feature list")
    threshold_selected_on_test = False
    model_selected_on_test = False
    warnings = []
    if shared_hashes:
        warnings.append(
            f"shared feature hashes={len(shared_hashes)}; all-NaN explained={shared_all_nan}"
        )
    if full_duplicate_rows:
        warnings.append(f"full duplicate rows={full_duplicate_rows}")
    if suspicious:
        warnings.append(f"suspicious feature names={suspicious}")
    status = "FAIL" if failures else ("WARNING" if warnings else "PASS")
    shared_details = {
        "train_rows": int(shared_train_mask.sum()),
        "test_rows": int(shared_test_mask.sum()),
        "train_labels": train.loc[shared_train_mask, "Label"].value_counts().sort_index().to_dict(),
        "test_labels": test.loc[shared_test_mask, "Label"].value_counts().sort_index().to_dict(),
    }
    audit = {
        "seed": seed,
        "status": status,
        "variant_id_overlap_count": len(overlap),
        "train_duplicate_variant_id_count": train_dup,
        "test_duplicate_variant_id_count": test_dup,
        "identical_feature_hash_count": len(shared_hashes),
        "identical_hash_all_nan_explained": shared_all_nan,
        "suspicious_feature_columns": json.dumps(suspicious),
        "threshold_selected_on_test": threshold_selected_on_test,
        "model_selected_on_test": model_selected_on_test,
        "details": json.dumps({
            "failures": failures, "warnings": warnings,
            "full_duplicate_rows": full_duplicate_rows,
            "shared_hash_details": shared_details,
            "feature_list_has_label_or_variant_id": feature_leak,
        }),
    }
    split = {
        "seed": seed, "train_n": len(train), "test_n": len(test),
        "train_benign": int(train.Label.eq(0).sum()),
        "train_pathogenic": int(train.Label.eq(1).sum()),
        "test_benign": int(test.Label.eq(0).sum()),
        "test_pathogenic": int(test.Label.eq(1).sum()),
        "train_benign_share": float(train.Label.eq(0).mean()),
        "test_benign_share": float(test.Label.eq(0).mean()),
    }
    return audit, split


def load_partial_rows() -> tuple[dict[str, list[dict]], set[int]]:
    if not all(path.exists() for path in PARTIALS.values()):
        return {name: [] for name in PARTIALS}, set()
    rows = {name: pd.read_csv(path).to_dict("records") for name, path in PARTIALS.items()}
    seed_sets = [{int(row["seed"]) for row in values} for values in rows.values()]
    completed = set.intersection(*seed_sets) if seed_sets else set()
    rows = {
        name: [row for row in values if int(row["seed"]) in completed]
        for name, values in rows.items()
    }
    return rows, completed


def save_partials(rows: dict[str, list[dict]]) -> None:
    for name, path in PARTIALS.items():
        pd.DataFrame(rows[name]).to_csv(path, index=False)


def summary_rows(frame: pd.DataFrame, metrics: list[str]) -> list[dict]:
    result = []
    for metric in metrics:
        values = frame[metric].astype(float)
        result.append({
            "metric": metric, "mean": values.mean(), "std": values.std(ddof=1),
            "min": values.min(), "q25": values.quantile(0.25),
            "median": values.median(), "q75": values.quantile(0.75), "max": values.max(),
        })
    return result


def main() -> None:
    script_start = time.perf_counter()
    print("=== Estimated runtime ===", flush=True)
    print("Input loading + validation: approximately 5-20 seconds", flush=True)
    print("One-seed split + leakage audit: approximately 5-20 seconds", flush=True)
    print("One-seed 5-fold CatBoost OOF: approximately 3-8 minutes on CPU", flush=True)
    print("One-seed full-train CatBoost + test reporting: approximately 30-90 seconds", flush=True)
    print("20-seed total: approximately 60-180 minutes on CPU", flush=True)
    print("A slow machine may take up to 3 hours; fast CatBoost may finish in 45-60 minutes.", flush=True)

    load_start = time.perf_counter()
    input_path = find_input()
    full = pd.read_csv(input_path)
    if not {"Variant_ID", "Label"}.issubset(full.columns):
        raise ValueError("Variant_ID and Label are required")
    full["Label"] = full["Label"].astype(int)
    if set(full.Label.unique()) != {0, 1}:
        raise ValueError("Label must contain exactly 0 and 1")
    features = [col for col in full.columns if col not in ["Variant_ID", "Label"]]
    categorical = full[features].select_dtypes(include=["object", "category"]).columns.tolist()
    full_duplicate_rows = int(full.duplicated().sum())
    input_loading_seconds = time.perf_counter() - load_start
    print(f"Input loaded in {input_loading_seconds:.2f}s: {input_path}", flush=True)

    rows, completed = load_partial_rows()
    if completed:
        print(f"Checkpoint resume; completed seeds: {sorted(completed)}", flush=True)

    for seed in SEEDS:
        if seed in completed:
            continue
        seed_start = time.perf_counter()
        print(f"\nSeed {seed}: starting repeated split", flush=True)

        split_start = time.perf_counter()
        train, test = make_split(full, seed)
        split_seconds = time.perf_counter() - split_start

        leakage_start = time.perf_counter()
        audit, split_row = leakage_audit(
            full, train, test, features, seed, full_duplicate_rows
        )
        leakage_seconds = time.perf_counter() - leakage_start

        x_train = catboost_frame(train[features], categorical)
        x_test = catboost_frame(test[features], categorical)
        y_train = train.Label.to_numpy(dtype=int)
        y_test = test.Label.to_numpy(dtype=int)
        shares = pd.Series(y_train).value_counts(normalize=True).to_dict()
        class_weights = {0: 0.85 / shares[0], 1: 0.15 / shares[1]}
        sample_weights = np.where(y_train == 0, class_weights[0], class_weights[1])

        oof_start = time.perf_counter()
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        oof_prob = np.full(len(train), np.nan, dtype=float)
        for fold, (fit_idx, valid_idx) in enumerate(cv.split(x_train, y_train), 1):
            model = make_model(seed)
            model.fit(x_train.iloc[fit_idx], y_train[fit_idx], cat_features=categorical)
            oof_prob[valid_idx] = model.predict_proba(x_train.iloc[valid_idx])[:, 1]
            print(f"  OOF fold {fold}/{N_SPLITS}", flush=True)
        if np.isnan(oof_prob).any():
            raise RuntimeError(f"NaN OOF predictions for seed {seed}")
        threshold, weighted = select_weighted_f1_threshold(
            y_train, oof_prob, sample_weights
        )
        oof_raw = classification_metrics(y_train, oof_prob, threshold)
        oof_training_seconds = time.perf_counter() - oof_start

        final_start = time.perf_counter()
        final_model = make_model(seed)
        final_model.fit(x_train, y_train, cat_features=categorical)
        test_prob = final_model.predict_proba(x_test)[:, 1]
        test_raw = classification_metrics(y_test, test_prob, threshold)
        final_fit_test_seconds = time.perf_counter() - final_start

        reporting_start = time.perf_counter()
        rows["oof"].append({
            "seed": seed, "selected_threshold": threshold,
            "oof_auc": roc_auc_score(y_train, oof_prob),
            "oof_average_precision": average_precision_score(y_train, oof_prob),
            "oof_f1": oof_raw["f1"], "oof_precision": oof_raw["precision"],
            "oof_recall": oof_raw["recall"], "oof_specificity": oof_raw["specificity"],
            "oof_fpr": oof_raw["fpr"], "oof_balanced_accuracy": oof_raw["balanced_accuracy"],
            "oof_predicted_positive_rate": oof_raw["predicted_positive_rate"],
            **weighted, "oof_tn": oof_raw["tn"], "oof_fp": oof_raw["fp"],
            "oof_fn": oof_raw["fn"], "oof_tp": oof_raw["tp"],
        })
        rows["test"].append({
            "seed": seed, "selected_threshold_from_oof": threshold,
            "threshold_source": "OOF train CV only",
            "test_usage": "repeated split reporting only",
            "test_auc": roc_auc_score(y_test, test_prob),
            "test_average_precision": average_precision_score(y_test, test_prob),
            "test_f1": test_raw["f1"], "test_precision": test_raw["precision"],
            "test_recall": test_raw["recall"], "test_specificity": test_raw["specificity"],
            "test_fpr": test_raw["fpr"], "test_balanced_accuracy": test_raw["balanced_accuracy"],
            "test_predicted_positive_rate": test_raw["predicted_positive_rate"],
            "test_tn": test_raw["tn"], "test_fp": test_raw["fp"],
            "test_fn": test_raw["fn"], "test_tp": test_raw["tp"],
        })
        audit.update({f"split_{key}": value for key, value in split_row.items() if key != "seed"})
        rows["leakage"].append(audit)

        train_all_nan = train[features].isna().all(axis=1).to_numpy()
        test_all_nan = test[features].isna().all(axis=1).to_numpy()
        test_pred = (test_prob >= threshold).astype(int)
        all_nan_errors = pd.Series(
            error_types(y_test[test_all_nan], test_pred[test_all_nan])
        ).value_counts().to_dict()
        rows["all_nan"].append({
            "seed": seed,
            "all_nan_train_count": int(train_all_nan.sum()),
            "all_nan_train_benign": int(((y_train == 0) & train_all_nan).sum()),
            "all_nan_train_pathogenic": int(((y_train == 1) & train_all_nan).sum()),
            "all_nan_test_count": int(test_all_nan.sum()),
            "all_nan_test_benign": int(((y_test == 0) & test_all_nan).sum()),
            "all_nan_test_pathogenic": int(((y_test == 1) & test_all_nan).sum()),
            "all_nan_test_tn": int(all_nan_errors.get("TN", 0)),
            "all_nan_test_fp": int(all_nan_errors.get("FP", 0)),
            "all_nan_test_fn": int(all_nan_errors.get("FN", 0)),
            "all_nan_test_tp": int(all_nan_errors.get("TP", 0)),
        })
        reporting_seconds = time.perf_counter() - reporting_start
        actual_runtime_seconds = time.perf_counter() - seed_start
        rows["runtime"].append({
            "seed": seed,
            "estimated_runtime_minutes": ESTIMATED_TOTAL_RUNTIME_MINUTES / len(SEEDS),
            "actual_runtime_seconds": actual_runtime_seconds,
            "split_seconds": split_seconds, "leakage_seconds": leakage_seconds,
            "oof_training_seconds": oof_training_seconds,
            "final_fit_test_seconds": final_fit_test_seconds,
            "reporting_seconds": reporting_seconds, "status": "COMPLETED",
        })
        save_partials(rows)
        completed.add(seed)
        average_seconds = np.mean([float(row["actual_runtime_seconds"]) for row in rows["runtime"]])
        remaining_minutes = average_seconds * (len(SEEDS) - len(completed)) / 60
        print(
            f"Seed {seed} complete | elapsed={actual_runtime_seconds/60:.2f} min | "
            f"estimated remaining={remaining_minutes:.1f} min | threshold={threshold:.6f} | "
            f"test F1={test_raw['f1']:.6f} | FP/FN={test_raw['fp']}/{test_raw['fn']} | "
            f"leakage={audit['status']}", flush=True,
        )

    order = {seed: index for index, seed in enumerate(SEEDS)}
    for values in rows.values():
        values.sort(key=lambda row: order[int(row["seed"])])
    oof = pd.DataFrame(rows["oof"])
    test_metrics = pd.DataFrame(rows["test"])
    leakage_partial = pd.DataFrame(rows["leakage"])
    all_nan = pd.DataFrame(rows["all_nan"])
    runtime_seeds = pd.DataFrame(rows["runtime"])
    split_columns = [
        "seed", "train_n", "test_n", "train_benign", "train_pathogenic",
        "test_benign", "test_pathogenic", "train_benign_share", "test_benign_share",
    ]
    split = pd.DataFrame({
        "seed": leakage_partial.seed,
        **{column: leakage_partial[f"split_{column}"] for column in split_columns if column != "seed"},
    })[split_columns]
    leakage_columns = [
        "seed", "status", "variant_id_overlap_count", "train_duplicate_variant_id_count",
        "test_duplicate_variant_id_count", "identical_feature_hash_count",
        "identical_hash_all_nan_explained", "suspicious_feature_columns",
        "threshold_selected_on_test", "model_selected_on_test", "details",
    ]
    leakage = leakage_partial[leakage_columns]
    summary = pd.DataFrame(summary_rows(test_metrics, [
        "test_f1", "test_precision", "test_recall", "test_fpr",
        "test_balanced_accuracy", "test_fp", "test_fn", "selected_threshold_from_oof",
    ]))
    split.to_csv(OUTPUTS["split"], index=False)
    leakage.to_csv(OUTPUTS["leakage"], index=False)
    oof.to_csv(OUTPUTS["oof"], index=False)
    test_metrics.to_csv(OUTPUTS["test"], index=False)
    summary.to_csv(OUTPUTS["summary"], index=False)
    all_nan.to_csv(OUTPUTS["all_nan"], index=False)

    component_total = float(runtime_seeds.actual_runtime_seconds.sum()) + input_loading_seconds
    total_row = {
        "seed": "TOTAL", "estimated_runtime_minutes": ESTIMATED_TOTAL_RUNTIME_MINUTES,
        "actual_runtime_seconds": component_total,
        "split_seconds": runtime_seeds.split_seconds.sum(),
        "leakage_seconds": runtime_seeds.leakage_seconds.sum(),
        "oof_training_seconds": runtime_seeds.oof_training_seconds.sum(),
        "final_fit_test_seconds": runtime_seeds.final_fit_test_seconds.sum(),
        "reporting_seconds": runtime_seeds.reporting_seconds.sum(), "status": "COMPLETED",
    }
    runtime_report = pd.concat([runtime_seeds, pd.DataFrame([total_row])], ignore_index=True)
    runtime_report.to_csv(OUTPUTS["runtime"], index=False)

    f1_summary = summary.loc[summary.metric.eq("test_f1")].iloc[0]
    precision_summary = summary.loc[summary.metric.eq("test_precision")].iloc[0]
    recall_summary = summary.loc[summary.metric.eq("test_recall")].iloc[0]
    fp_summary = summary.loc[summary.metric.eq("test_fp")].iloc[0]
    fn_summary = summary.loc[summary.metric.eq("test_fn")].iloc[0]
    threshold_summary = summary.loc[summary.metric.eq("selected_threshold_from_oof")].iloc[0]
    leakage_fail = leakage.status.eq("FAIL").any()
    leakage_warning = leakage.status.eq("WARNING").any()
    f1_stable = f1_summary["std"] <= 0.05
    fp_range_limit = max(20.0, 0.25 * fp_summary["mean"])
    fp_stable = (fp_summary["max"] - fp_summary["min"]) <= fp_range_limit
    same_split_f1 = 0.524
    split_delta = f1_summary["mean"] - same_split_f1
    materially_lower = split_delta < -0.03
    if leakage_fail:
        recommendation = "Resolve leakage FAIL findings before any Step 4 work."
    elif materially_lower:
        recommendation = "Treat v0.2 as split-specific optimistic and revise validation/threshold policy."
    elif not fp_stable:
        recommendation = "Revise threshold policy because repeated-split FP control is volatile."
    elif f1_stable and f1_summary["mean"] >= 0.50:
        recommendation = "Proceed to Step 4 ensemble experiments using repeated-split evaluation."
    else:
        recommendation = "Revise the threshold policy or validation strategy before Step 4."

    metadata = {
        "project_folder": "TeknoPatho",
        "step": "Step 3b repeated train/test split CatBoost leakage audit",
        "seed_list": SEEDS,
        "split_rule": "automatic nearest integer split to train 15% benign/85% pathogenic and test 85% benign/15% pathogenic",
        "target_train_prior": {"benign": 0.15, "pathogenic": 0.85},
        "target_test_prior": {"benign": 0.85, "pathogenic": 0.15},
        "model_family": "CatBoostClassifier",
        "threshold_policy": "weighted_final_like_oof_f1_threshold",
        "test_usage": "repeated split reporting only",
        "no_single_seed_selection": True,
        "estimated_total_runtime_minutes": ESTIMATED_TOTAL_RUNTIME_MINUTES,
        "actual_total_runtime_minutes": component_total / 60,
        "created_files": [
            Path(__file__).name,
            *[path.name for path in OUTPUTS.values()],
            *[path.name for path in PARTIALS.values()],
        ],
    }
    OUTPUTS["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    all_nan_totals = all_nan.drop(columns="seed").agg(["mean", "std", "min", "max"])
    report = f"""# MASTER Step 3b repeated-split CatBoost stability and leakage audit

## Why repeated train/test splitting?

A fixed test partition can make stability look better or worse because only CV/model randomness changes. Step 3b rebuilds the class-prior-controlled train/test partition for every seed, retrains from scratch, and selects each threshold only from that seed's train OOF predictions.

**Single-seed scores are not used as final evidence. Aggregate repeated split metrics are used.**

## Context and protocol

- v0: CatBoost selected by train OOF; fixed-split test F1 approximately 0.348.
- v0.1: final-like prior-weighted OOF threshold policy introduced.
- v0.2: same-split mean test reporting-only F1 approximately {same_split_f1:.3f}.
- Step 3b purpose: measure split-to-split reliability across {len(SEEDS)} seeds: {SEEDS}.
- Input: `{input_path}`; features={len(features)}; excluded=`Variant_ID`, `Label`.
- Every threshold is selected by weighted final-like F1 on train OOF only. Test is repeated-split reporting only.
- Checkpoint files were written after every completed seed and used for safe resume where necessary.

## Estimated versus actual runtime

- Initial estimate: 60-180 minutes; planning value={ESTIMATED_TOTAL_RUNTIME_MINUTES} minutes.
- Accumulated actual seed runtime plus input loading: {component_total/60:.2f} minutes.

{markdown_table(runtime_report)}

## Split distributions

{markdown_table(split)}

## Leakage audit

- FAIL present: {leakage_fail}
- WARNING present: {leakage_warning}
- Status counts: {leakage.status.value_counts().to_dict()}
- Shared all-NaN feature hashes are warnings, not evidence of Variant_ID overlap. Threshold/model selection flags remain false.

{markdown_table(leakage)}

## Repeated-split OOF metrics

{markdown_table(oof)}

## Repeated-split test metrics — reporting only

{markdown_table(test_metrics)}

## Aggregate repeated-split result

- Test F1 mean ± std: **{f1_summary['mean']:.6f} ± {f1_summary['std']:.6f}**
- Test F1 min / median / max: {f1_summary['min']:.6f} / {f1_summary['median']:.6f} / {f1_summary['max']:.6f}
- Precision mean={precision_summary['mean']:.6f}; recall mean={recall_summary['mean']:.6f}.
- FP mean={fp_summary['mean']:.3f}, std={fp_summary['std']:.3f}, range={fp_summary['min']:.0f}-{fp_summary['max']:.0f}; diagnostic stable={fp_stable} under range <= max(20, 25% mean).
- FN mean={fn_summary['mean']:.3f}, std={fn_summary['std']:.3f}, range={fn_summary['min']:.0f}-{fn_summary['max']:.0f}.
- Threshold mean={threshold_summary['mean']:.6f}, std={threshold_summary['std']:.6f}.
- Repeated-split F1 stable by std <= 0.05: {f1_stable}.

{markdown_table(summary)}

## Same-split v0.2 comparison

v0.2 same-split mean F1={same_split_f1:.3f}; Step 3b repeated-split mean={f1_summary['mean']:.6f}; delta={split_delta:+.6f}. A drop larger than 0.03 is treated as evidence that the same-split result was materially optimistic: {materially_lower}.

## All-NaN effect

{markdown_table(all_nan)}

Across-seed all-NaN summary:

{markdown_table(all_nan_totals.reset_index().rename(columns={'index': 'statistic'}))}

## Conclusion and next step

{recommendation}

No seed was selected as final. The highest single-seed test F1 was not used for model or policy selection.

**Single-seed scores are not used as final evidence. Aggregate repeated split metrics are used.**
"""
    OUTPUTS["report"].write_text(report, encoding="utf-8")

    print("\n=== MASTER Step 3b final summary ===")
    print(f"Total seeds: {len(SEEDS)}")
    print(f"Successful seeds: {len(test_metrics)}")
    print(f"Leakage FAIL present: {leakage_fail}")
    print(f"Leakage WARNING present: {leakage_warning}")
    print(f"Repeated-split test F1 mean: {f1_summary['mean']:.6f}")
    print(f"Repeated-split test F1 std: {f1_summary['std']:.6f}")
    print(f"Repeated-split test F1 min/max/median: {f1_summary['min']:.6f}/{f1_summary['max']:.6f}/{f1_summary['median']:.6f}")
    print(f"Mean precision: {precision_summary['mean']:.6f}")
    print(f"Mean recall: {recall_summary['mean']:.6f}")
    print(f"Mean FP: {fp_summary['mean']:.3f}")
    print(f"Mean FN: {fn_summary['mean']:.3f}")
    print(f"Mean selected threshold: {threshold_summary['mean']:.6f}")
    print(f"Threshold std: {threshold_summary['std']:.6f}")
    print(f"Total runtime: {component_total/60:.2f} minutes")
    print(f"Average runtime per seed: {runtime_seeds.actual_runtime_seconds.mean()/60:.2f} minutes")
    print(f"v0.2 same-split F1=0.524 comparison delta: {split_delta:+.6f}")
    print(f"Recommended next step: {recommendation}")


if __name__ == "__main__":
    main()
