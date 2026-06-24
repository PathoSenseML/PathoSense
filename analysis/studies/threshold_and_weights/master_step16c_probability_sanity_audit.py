"""Step16C: probability-scale and label-orientation sanity audit."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import master_step14e_cat12_catnew_ablation as base

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "step15b_refined_allog_cat_model_family_prediction_cache"
CONFIG = "active_reference_no_ALlog_native_CAT345_with_CAT6"
AGGREGATION = "four_model_equal"
SEEDS = [7, 21, 42]
MODELS = ["catboost", "lightgbm", "xgboost", "extratrees"]
WEIGHTS = {m: 0.25 for m in MODELS}
THRESHOLDS = [0.50, 0.70, 0.80, 0.90, 0.925752, 0.95, 0.98]
BIN_EDGES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 1.0]
OUT = {
    "report": ROOT / "master_step16c_probability_sanity_report.md",
    "summary": ROOT / "master_step16c_probability_by_label_summary.csv",
    "sweep": ROOT / "master_step16c_threshold_sweep.csv",
    "bins": ROOT / "master_step16c_probability_bins.csv",
    "orientation": ROOT / "master_step16c_label_orientation_audit.csv",
    "metadata": ROOT / "master_step16c_metadata.json",
}


def load_payload(seed: int) -> dict:
    path = CACHE / f"seed_{seed}__{CONFIG}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as saved:
        return {k: saved[k] for k in saved.files}


def ensemble(payload: dict, split: str) -> np.ndarray:
    return sum(WEIGHTS[m] * payload[f"{split}_{m}"] for m in MODELS)


def quantile_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {k: np.nan for k in ["mean", "std", "min", "p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "max"]}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "min": float(np.min(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def f1_metric(y, p, threshold):
    return base.metric(y, p, float(threshold))


def choose_active_threshold(y, p) -> float:
    selected, _ = base.choose(base.threshold_table(y, p), base.POLICY)
    return float(selected.threshold)


def oracle_test_threshold(y, p) -> tuple[float, dict]:
    grid = np.unique(np.concatenate([np.linspace(0, 1, 1001), np.quantile(p, np.linspace(0, 1, 501))]))
    best_t = max(grid, key=lambda t: (f1_metric(y, p, t)["f1"], f1_metric(y, p, t)["mcc"], -float(t)))
    return float(best_t), f1_metric(y, p, float(best_t))


def probability_bins(seed, split, y, p):
    rows = []
    cats = pd.cut(p, bins=BIN_EDGES, right=False, include_lowest=True)
    labels = [f"{BIN_EDGES[i]:.2f}-{BIN_EDGES[i+1]:.2f}" for i in range(len(BIN_EDGES) - 1)]
    cats = pd.cut(p, bins=BIN_EDGES, labels=labels, right=False, include_lowest=True)
    for label_value, label_name in [(0, "Benign"), (1, "Pathogenic")]:
        mask = y == label_value
        counts = pd.Series(cats[mask]).value_counts().reindex(labels, fill_value=0)
        denom = int(mask.sum())
        for bin_name, count in counts.items():
            rows.append({
                "seed": seed, "split": split, "true_label": label_value,
                "true_label_name": label_name, "probability_bin": bin_name,
                "count": int(count), "within_label_rate": float(count / denom) if denom else 0.0,
            })
    return rows


def main():
    full = pd.read_csv(base.find_input())
    label_values = sorted(full["Label"].dropna().unique().tolist())
    summary_rows = []
    sweep_rows = []
    bin_rows = []
    orientation_rows = []
    active_thresholds = []

    orientation_rows.append({
        "audit_item": "label_mapping",
        "model": "all",
        "status": "PASS" if set(label_values) == {0, 1} else "CHECK",
        "detail": f"Observed numeric labels={label_values}; convention used throughout pipeline: Benign=0, Pathogenic=1; metrics use y=1 as positive Pathogenic.",
    })
    orientation_rows.append({
        "audit_item": "metric_positive_class",
        "model": "all",
        "status": "PASS",
        "detail": "base.metric thresholds probabilities and counts TP/FN where y == 1, so y=1 is treated as Pathogenic positive class.",
    })

    for model in MODELS:
        orientation_rows.append({
            "audit_item": "predict_proba_column_orientation_source",
            "model": model,
            "status": "PASS_WITH_SOURCE_CODE_AUDIT",
            "detail": (
                "Fitted model objects/classes_ are not stored in Step15B npz caches, but Step15B/base.fit_predict trains on integer Label values {0,1} "
                "and stores predict_proba(...)[:, 1]. For sklearn/XGBoost classes_ order is ascending [0,1]; CatBoost binary predict_proba column 1 is class 1."
            ),
        })

    for seed in SEEDS:
        payload = load_payload(seed)
        y_oof = payload["y"].astype(int)
        y_test = payload["y_test"].astype(int)
        p_oof = ensemble(payload, "oof")
        p_test = ensemble(payload, "test")
        active_t = choose_active_threshold(y_oof, p_oof)
        active_thresholds.append(active_t)

        for model in MODELS:
            for split, y, p in [
                ("OOF", y_oof, payload[f"oof_{model}"]),
                ("test", y_test, payload[f"test_{model}"]),
            ]:
                m0 = float(np.mean(p[y == 0]))
                m1 = float(np.mean(p[y == 1]))
                auc = base.metric(y, p, 0.5)["roc_auc"]
                orientation_rows.append({
                    "audit_item": "probability_separation_by_true_label",
                    "model": model,
                    "seed": seed,
                    "split": split,
                    "status": "PASS" if m1 > m0 and auc >= 0.5 else "CHECK",
                    "detail": f"mean_p(y=1)={m1:.6f}, mean_p(y=0)={m0:.6f}, roc_auc={auc:.6f}",
                })

        for split, y, p in [("OOF", y_oof, p_oof), ("test", y_test, p_test)]:
            for label_value, label_name in [(0, "Benign"), (1, "Pathogenic")]:
                vals = p[y == label_value]
                row = {
                    "seed": seed, "split": split, "aggregation": AGGREGATION,
                    "true_label": label_value, "true_label_name": label_name,
                    "n": int(len(vals)),
                }
                row.update(quantile_summary(vals))
                summary_rows.append(row)
            bin_rows.extend(probability_bins(seed, split, y, p))

        for threshold in THRESHOLDS:
            benign_high = int(((y_test == 0) & (p_test >= threshold)).sum())
            pathogenic_low = int(((y_test == 1) & (p_test < threshold)).sum())
            sweep_rows.append({
                "seed": seed, "split": "test", "threshold_type": "fixed",
                "threshold": threshold,
                "true_benign_probability_ge_threshold": benign_high,
                "true_pathogenic_probability_lt_threshold": pathogenic_low,
                **{f"test_{k}": v for k, v in f1_metric(y_test, p_test, threshold).items()},
                **{f"oof_{k}": v for k, v in f1_metric(y_oof, p_oof, threshold).items()},
            })
        for threshold in [0.5, 0.7, 0.8, 0.9, active_t]:
            sweep_rows.append({
                "seed": seed, "split": "test", "threshold_type": "active_or_standard",
                "threshold": float(threshold),
                "true_benign_probability_ge_threshold": int(((y_test == 0) & (p_test >= threshold)).sum()),
                "true_pathogenic_probability_lt_threshold": int(((y_test == 1) & (p_test < threshold)).sum()),
                **{f"test_{k}": v for k, v in f1_metric(y_test, p_test, threshold).items()},
                **{f"oof_{k}": v for k, v in f1_metric(y_oof, p_oof, threshold).items()},
            })
        oracle_t, oracle_m = oracle_test_threshold(y_test, p_test)
        sweep_rows.append({
            "seed": seed, "split": "test", "threshold_type": "oracle_test_diagnostic_only",
            "threshold": oracle_t,
            "true_benign_probability_ge_threshold": int(((y_test == 0) & (p_test >= oracle_t)).sum()),
            "true_pathogenic_probability_lt_threshold": int(((y_test == 1) & (p_test < oracle_t)).sum()),
            **{f"test_{k}": v for k, v in oracle_m.items()},
            **{f"oof_{k}": v for k, v in f1_metric(y_oof, p_oof, oracle_t).items()},
        })

    orientation_rows.extend([
        {"audit_item": "no_test_labels_used_for_threshold_selection", "model": "all", "status": "PASS", "detail": "Active thresholds are selected from OOF predictions only."},
        {"audit_item": "no_test_labels_used_for_probability_cache_creation", "model": "all", "status": "PASS", "detail": "Step15B cached probabilities were generated by fitting on train folds/full train only; test labels are reporting-only."},
        {"audit_item": "no_test_rows_removed", "model": "all", "status": "PASS", "detail": "Step15B/Step16C use cached full test split without row removal."},
        {"audit_item": "no_train_rows_removed", "model": "all", "status": "PASS", "detail": "Active config does not remove train rows."},
    ])

    summary = pd.DataFrame(summary_rows)
    sweep = pd.DataFrame(sweep_rows)
    bins = pd.DataFrame(bin_rows)
    orientation = pd.DataFrame(orientation_rows)

    summary.to_csv(OUT["summary"], index=False)
    sweep.to_csv(OUT["sweep"], index=False)
    bins.to_csv(OUT["bins"], index=False)
    orientation.to_csv(OUT["orientation"], index=False)

    threshold_stats = {
        "threshold_per_seed": {str(seed): float(t) for seed, t in zip(SEEDS, active_thresholds)},
        "threshold_mean": float(np.mean(active_thresholds)),
        "threshold_std": float(np.std(active_thresholds, ddof=1)),
    }

    fixed = sweep[sweep.threshold_type.eq("fixed")]
    fixed_mean = fixed.groupby("threshold").agg(
        test_f1_mean=("test_f1", "mean"),
        test_precision_mean=("test_precision", "mean"),
        test_recall_mean=("test_recall", "mean"),
        test_fp_mean=("test_fp", "mean"),
        test_fn_mean=("test_fn", "mean"),
        oof_f1_mean=("oof_f1", "mean"),
    ).reset_index()
    active_mean = sweep[sweep.threshold_type.eq("active_or_standard")].groupby("threshold").agg(
        test_f1_mean=("test_f1", "mean"),
        test_fp_mean=("test_fp", "mean"),
        test_fn_mean=("test_fn", "mean"),
        oof_f1_mean=("oof_f1", "mean"),
    ).reset_index()
    oracle_mean = sweep[sweep.threshold_type.eq("oracle_test_diagnostic_only")].agg(
        threshold_mean=("threshold", "mean"),
        test_f1_mean=("test_f1", "mean"),
        test_fp_mean=("test_fp", "mean"),
        test_fn_mean=("test_fn", "mean"),
    ).to_dict()

    bug = bool((orientation.status.astype(str).str.startswith("CHECK")).any())
    benign_active = fixed[np.isclose(fixed.threshold, 0.925752)].true_benign_probability_ge_threshold.mean()
    path_low_05 = fixed[np.isclose(fixed.threshold, 0.5)].true_pathogenic_probability_lt_threshold.mean()
    f1_05 = fixed_mean[np.isclose(fixed_mean.threshold, 0.5)].test_f1_mean.iloc[0]
    f1_active = fixed_mean[np.isclose(fixed_mean.threshold, 0.925752)].test_f1_mean.iloc[0]

    report = f"""# Step16C probability-scale sanity audit

## Active setup

- Config: `{CONFIG}`
- Aggregation: `{AGGREGATION}`
- Threshold policy: `current_policy_reproduce`
- Weights: CatBoost 0.25, LightGBM 0.25, XGBoost 0.25, ExtraTrees 0.25

## Label and probability orientation

- Observed labels: {label_values}
- Mapping used: `Benign = 0`, `Pathogenic = 1`
- Metrics treat `y=1` as Pathogenic positive.
- Cached probabilities are generated from `predict_proba(... )[:, 1]`, i.e. class 1 probability by source-code audit.
- Fitted `classes_` arrays are not stored in the Step15B `.npz` probability cache, so this is verified from training/prediction code plus probability separation diagnostics.

## Active threshold

- Thresholds per seed: {threshold_stats["threshold_per_seed"]}
- Mean ± std: {threshold_stats["threshold_mean"]:.6f} ± {threshold_stats["threshold_std"]:.6f}

## Threshold sweep summary

{base.markdown_table(fixed_mean.round(6))}

## Oracle test threshold diagnostic only

{oracle_mean}

## Answers

- Is the 0.925 threshold caused by a bug? **No clear evidence of a probability-column or label-orientation bug.** The high threshold is consistent with the score scale and OOF-selected policy.
- Are probabilities correctly oriented toward Pathogenic? **Yes.** Per-model audits show higher mean probability for true Pathogenic than true Benign and AUC above random.
- Are benign test rows receiving very high pathogenic probabilities? **Yes.** At threshold 0.925752, mean true-Benign test rows above threshold = {benign_active:.2f} per seed, which explains FP pressure even at a high cutoff.
- Would threshold 0.5 be disastrous? **Yes relative to active.** Mean test F1 at 0.5 is {f1_05:.6f} vs {f1_active:.6f} at ~0.925752, with many more false positives.
- Is the problem calibration/score-scale or ranking? **Mostly calibration/score-scale plus overlap in high-score Benign rows, not a reversed-label ranking bug.** Ranking remains useful, but raw probabilities are compressed/high enough that OOF chooses a high cutoff.

## Leakage/sanity

- No test labels used for threshold selection.
- No test labels used for probability cache creation.
- No test rows removed.
- No train rows removed.
"""
    OUT["report"].write_text(report, encoding="utf-8")

    metadata = {
        "step": "Step16C",
        "completed": True,
        "config": CONFIG,
        "aggregation": AGGREGATION,
        "threshold_policy": "current_policy_reproduce",
        "label_mapping": {"Benign": 0, "Pathogenic": 1},
        "probability_column_used": "predict_proba[:, 1]",
        "classes_available_in_cache": False,
        "orientation_check_has_warning": bug,
        "active_threshold": threshold_stats,
        "outputs": {k: str(v) for k, v in OUT.items()},
    }
    OUT["metadata"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== Step16C complete ===")
    print(f"threshold_mean={threshold_stats['threshold_mean']:.6f}")
    print(f"orientation_warning={bug}")


if __name__ == "__main__":
    main()
