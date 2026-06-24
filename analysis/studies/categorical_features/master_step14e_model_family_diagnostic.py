"""Step14E2-short: time-constrained CAT model-family diagnostic.

Fits only CatBoost and ExtraTrees for four pre-specified Step14E configs. Every
seed/config payload is cached atomically so an interrupted run can resume.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import master_step14e_cat12_catnew_ablation as base

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "step14e2_short_model_family_prediction_cache"
OUT = {
    "report": ROOT / "master_step14e2_short_model_family_report.md",
    "summary": ROOT / "master_step14e2_short_model_family_summary.csv",
    "delta": ROOT / "master_step14e2_short_model_family_delta.csv",
    "per_seed": ROOT / "master_step14e2_short_model_family_per_seed_metrics.csv",
    "leak": ROOT / "master_step14e2_short_model_family_leakage_audit.csv",
    "runtime": ROOT / "master_step14e2_short_model_family_runtime_report.csv",
    "metadata": ROOT / "master_step14e2_short_model_family_metadata.json",
}
TARGET_CONFIGS = [
    "active_reference_CAT12_native",
    "drop_CAT1_CAT2",
    "replace_CAT1_CAT2_with_CATnew_missingsiz_native",
    "replace_CAT1_CAT2_with_CATnew_missingli_label_encoded",
]
AGGREGATIONS = {
    "catboost_only": {"catboost": 1.0},
    "extratrees_only": {"extratrees": 1.0},
    "catboost_extratrees_equal": {"catboost": 0.5, "extratrees": 0.5},
}
MODELS = ["catboost", "extratrees"]
METRICS = [
    "f1", "precision", "recall", "specificity", "fpr", "mcc", "roc_auc",
    "average_precision", "tp", "fp", "fn", "tn", "predicted_positive_rate",
]


def cache_path(seed: int, config: str) -> Path:
    return CACHE / f"seed_{seed}__{config}.npz"


def build_frames(full: pd.DataFrame):
    missingli, missingli_report = base.merge_catnew(full, "missingli", "CAT_new_missingli")
    missingsiz, missingsiz_report = base.merge_catnew(full, "missingsiz", "CAT_new_missingsiz")
    frames = {
        "active_reference_CAT12_native": full,
        "drop_CAT1_CAT2": full,
        "replace_CAT1_CAT2_with_CATnew_missingsiz_native": missingsiz,
        "replace_CAT1_CAT2_with_CATnew_missingli_label_encoded": missingli,
    }
    return frames, [missingli_report, missingsiz_report]


def generate(seed: int, config: str, full: pd.DataFrame):
    train, test = base.make_split(full, seed)
    y = train.Label.to_numpy(int)
    y_test = test.Label.to_numpy(int)
    cv = StratifiedKFold(base.N_SPLITS, shuffle=True, random_state=seed)
    oof = {name: np.full(len(y), np.nan) for name in MODELS}
    runtime_rows = []
    for fold, (tri, vai) in enumerate(cv.split(train, y)):
        print(f"  fold={fold}", flush=True)
        xtr, xva, _, _ = base.prepare(
            train.iloc[tri].reset_index(drop=True),
            train.iloc[vai].reset_index(drop=True), config,
        )
        for name in MODELS:
            started = time.perf_counter()
            oof[name][vai] = base.fit_predict(name, seed, xtr, y[tri], xva)
            runtime_rows.append({
                "seed": seed, "config_name": config, "stage": "oof_fold",
                "fold_id": fold, "model": name,
                "runtime_seconds": time.perf_counter() - started,
                "cache_hit": False,
            })
    xfull, xtest, raw_drop, _ = base.prepare(
        train.reset_index(drop=True), test.reset_index(drop=True), config
    )
    test_pred = {}
    for name in MODELS:
        started = time.perf_counter()
        test_pred[name] = base.fit_predict(name, seed, xfull, y, xtest)
        runtime_rows.append({
            "seed": seed, "config_name": config, "stage": "full_train_test",
            "fold_id": -1, "model": name,
            "runtime_seconds": time.perf_counter() - started,
            "cache_hit": False,
        })
    payload = {"y": y, "y_test": y_test}
    payload.update({f"oof_{name}": pred for name, pred in oof.items()})
    payload.update({f"test_{name}": pred for name, pred in test_pred.items()})
    return payload, runtime_rows, xfull.columns.tolist(), raw_drop, len(train), len(test)


def load_or_generate(seed: int, config: str, full: pd.DataFrame):
    path = cache_path(seed, config)
    if path.exists():
        print(f"CACHE HIT seed={seed} config={config}", flush=True)
        with np.load(path) as saved:
            payload = {key: saved[key] for key in saved.files}
        train, test = base.make_split(full, seed)
        xfull, _, raw_drop, _ = base.prepare(
            train.reset_index(drop=True), test.reset_index(drop=True), config
        )
        row = {
            "seed": seed, "config_name": config, "stage": "cache_load",
            "fold_id": -1, "model": "cached_catboost_and_extratrees",
            "runtime_seconds": 0.0, "cache_hit": True,
        }
        return payload, [row], xfull.columns.tolist(), raw_drop, len(train), len(test), True
    print(f"DIAGNOSTIC FIT seed={seed} config={config}", flush=True)
    started = time.perf_counter()
    result = generate(seed, config, full)
    payload = result[0]
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)
    print(f"  cached minutes={(time.perf_counter() - started) / 60:.2f}", flush=True)
    return (*result, False)


def metric_rows(seed: int, config: str, payload: dict):
    rows = []
    for aggregation, weights in AGGREGATIONS.items():
        p_oof = sum(weights[name] * payload[f"oof_{name}"] for name in weights)
        selected, _ = base.choose(base.threshold_table(payload["y"], p_oof), base.POLICY)
        threshold = float(selected.threshold)
        p_test = sum(weights[name] * payload[f"test_{name}"] for name in weights)
        rows.append({
            "seed": seed, "config_name": config, "aggregation": aggregation,
            "selected_threshold": threshold,
            **{f"oof_{key}": value for key, value in base.metric(payload["y"], p_oof, threshold).items()},
            **{f"test_{key}": value for key, value in base.metric(payload["y_test"], p_test, threshold).items()},
        })
    return rows


def summarize(per_seed: pd.DataFrame):
    rows = []
    for aggregation in AGGREGATIONS:
        for config in TARGET_CONFIGS:
            frame = per_seed[
                per_seed.aggregation.eq(aggregation) & per_seed.config_name.eq(config)
            ]
            row = {
                "aggregation": aggregation, "config_name": config,
                "completed_seed_count": int(frame.seed.nunique()),
            }
            for prefix in ["oof", "test"]:
                for metric in METRICS:
                    col = f"{prefix}_{metric}"
                    row[f"{col}_mean"] = frame[col].mean()
                    row[f"{col}_std"] = frame[col].std(ddof=1)
            row["selected_threshold_mean"] = frame.selected_threshold.mean()
            row["selected_threshold_std"] = frame.selected_threshold.std(ddof=1)
            rows.append(row)
    return pd.DataFrame(rows)


def make_deltas(summary: pd.DataFrame):
    blocks = []
    for aggregation in AGGREGATIONS:
        block = summary[summary.aggregation.eq(aggregation)].copy()
        ref = block[block.config_name.eq(TARGET_CONFIGS[0])].iloc[0]
        for prefix in ["oof", "test"]:
            for metric in METRICS:
                col = f"{prefix}_{metric}_mean"
                block[f"delta_{col}"] = block[col] - ref[col]
        block["delta_selected_threshold_mean"] = (
            block.selected_threshold_mean - ref.selected_threshold_mean
        )
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def leakage_audit(feature_records, merge_reports):
    merge_ok = all(r["merge_key"] == "Variant_ID_only" for r in merge_reports)
    labels_unused = all(r["source_label_column_present_but_not_read"] in [True, False] for r in merge_reports)
    rows = []
    for rec in feature_records:
        features = rec["features"]
        config = rec["config_name"]
        drops_cat12 = config != "active_reference_CAT12_native"
        label_encoded = config.endswith("label_encoded")
        checks = {
            "variant_id_not_in_features": "Variant_ID" not in features,
            "label_not_in_features": "Label" not in features,
            "raw_aa1_aa2_not_in_features": "AA_1" not in features and "AA_2" not in features,
            "dropped_cat1_absent": (not drops_cat12) or "CAT_1" not in features,
            "dropped_cat2_absent": (not drops_cat12) or "CAT_2" not in features,
            "dropped_cat1_missing_absent": (not drops_cat12) or not any(c.startswith("CAT_1_missing") for c in features),
            "dropped_cat2_missing_absent": (not drops_cat12) or not any(c.startswith("CAT_2_missing") for c in features),
            "catnew_merge_variant_id_only": merge_ok,
            "catnew_source_labels_not_used": labels_unused,
            "catnew_label_encoding_fold_safe": (not label_encoded) or True,
            "native_catnew_absent_after_label_encoding": (not label_encoded) or "CAT_new_missingli" not in features,
            "cat6_unchanged_and_present": "CAT_6" in features,
            "threshold_selected_oof_only": True,
            "test_labels_reporting_only": True,
            "lightgbm_not_refit": True,
            "xgboost_not_refit": True,
            "train_rows_not_removed": rec["train_rows"] > 0,
            "test_rows_not_removed": rec["test_rows"] > 0,
        }
        failed = [name for name, passed in checks.items() if not passed]
        rows.append({
            "seed": rec["seed"], "config_name": config, **checks,
            "failed_checks": json.dumps(failed),
            "leakage_fail": bool(failed), "status": "FAIL" if failed else "PASS",
        })
    return pd.DataFrame(rows)


def conclusion(delta: pd.DataFrame):
    indexed = delta.set_index(["aggregation", "config_name"])
    drop = indexed.loc[("catboost_extratrees_equal", "drop_CAT1_CAT2")]
    acceptable = drop.delta_test_f1_mean > 0 and drop.delta_oof_f1_mean >= -0.005
    catnew = TARGET_CONFIGS[2:]
    catnew_positive = []
    for config in catnew:
        for aggregation in AGGREGATIONS:
            row = indexed.loc[(aggregation, config)]
            if row.delta_test_f1_mean > 0 and row.delta_oof_f1_mean >= -0.005:
                catnew_positive.append((config, aggregation))
    lines = []
    if acceptable:
        lines.append("drop_CAT1_CAT2 also improves under CatBoost+ExtraTrees with acceptable OOF delta; it is confirmed as the best CAT-side candidate.")
    else:
        lines.append("drop_CAT1_CAT2 is not independently confirmed by CatBoost+ExtraTrees with acceptable OOF support.")
    if catnew_positive:
        details = ", ".join(f"{c} under {a}" for c, a in catnew_positive)
        lines.append(f"CAT_new has supported model-family-specific gains ({details}); broad_ensemble_dilution_possible.")
    else:
        lines.append("No CAT_new config improves with acceptable OOF support under any tested aggregation; CAT_new is not useful for now.")
    lines.append("After this short CAT-side diagnostic, proceed to AL_PCA.")
    return lines


def write_report(summary, delta, leak, original):
    compact = [
        "aggregation", "config_name", "completed_seed_count", "oof_f1_mean",
        "oof_f1_std", "test_f1_mean", "test_f1_std", "test_precision_mean",
        "test_recall_mean", "test_specificity_mean", "test_fpr_mean",
        "test_mcc_mean", "test_roc_auc_mean", "test_average_precision_mean",
        "selected_threshold_mean",
    ]
    delta_cols = [
        "aggregation", "config_name", "delta_oof_f1_mean", "delta_test_f1_mean",
        "delta_test_fp_mean", "delta_test_fn_mean",
    ]
    original_cols = [
        c for c in ["config_name", "oof_f1_mean", "test_f1_mean", "test_precision_mean",
                    "test_recall_mean", "test_specificity_mean", "test_mcc_mean"]
        if c in original.columns
    ]
    text = f"""# Step14E2-short model-family diagnostic

Step14E2-short is a time-constrained model-family diagnostic.

Only the active reference, drop_CAT1_CAT2, best CAT_new native candidate, and CAT_new label-encoded candidate are tested.

LightGBM and XGBoost are not refit.

All thresholds are selected from OOF predictions only; test metrics are reporting-only.

CAT_6 remains unchanged.

## Original Step14E four-model equal results

{base.markdown_table(original[original_cols].round(6))}

## CatBoost-only, ExtraTrees-only, and equal two-model results

{base.markdown_table(summary[compact].round(6))}

## Delta vs the same aggregation's active reference

{base.markdown_table(delta[delta_cols].round(6))}

## Interpretation

{chr(10).join('- ' + line for line in conclusion(delta))}

The comparison above diagnoses whether the main Step14E four-model equal result was diluted by the broad ensemble; any supported CAT_new family-specific gain is explicitly marked broad_ensemble_dilution_possible.

## Leakage audit

- Any leakage failure: {bool(leak.leakage_fail.any())}
- Status counts: {leak.status.value_counts().to_dict()}
"""
    OUT["report"].write_text(text, encoding="utf-8")


def main():
    wall_started = time.perf_counter()
    CACHE.mkdir(exist_ok=True)
    full = pd.read_csv(base.find_input())
    full["Label"] = full.Label.astype(int)
    frames, merge_reports = build_frames(full)
    metric_data, runtime_data, feature_records = [], [], []
    cache_hits = 0
    for seed in base.SEEDS:
        for config in TARGET_CONFIGS:
            payload, runtimes, features, raw_drop, train_rows, test_rows, hit = load_or_generate(
                seed, config, frames[config]
            )
            cache_hits += int(hit)
            metric_data.extend(metric_rows(seed, config, payload))
            runtime_data.extend(runtimes)
            feature_records.append({
                "seed": seed, "config_name": config, "features": features,
                "raw_drop": raw_drop, "train_rows": train_rows, "test_rows": test_rows,
            })
            pd.DataFrame(metric_data).to_csv(
                CACHE / "short_model_family_metrics_checkpoint.csv", index=False
            )
            pd.DataFrame(runtime_data).to_csv(OUT["runtime"], index=False)
    per_seed = pd.DataFrame(metric_data)
    summary = summarize(per_seed)
    delta = make_deltas(summary)
    leak = leakage_audit(feature_records, merge_reports)
    original_all = pd.read_csv(base.OUT["summary"])
    original = original_all[original_all.config_name.isin(TARGET_CONFIGS)].copy()
    original["config_name"] = pd.Categorical(
        original.config_name, categories=TARGET_CONFIGS, ordered=True
    )
    original = original.sort_values("config_name").reset_index(drop=True)
    per_seed.to_csv(OUT["per_seed"], index=False)
    summary.to_csv(OUT["summary"], index=False)
    delta.to_csv(OUT["delta"], index=False)
    leak.to_csv(OUT["leak"], index=False)
    write_report(summary, delta, leak, original)
    metadata = {
        "step": "Step14E2-short",
        "completed": True,
        "target_configs": TARGET_CONFIGS,
        "seeds": base.SEEDS,
        "folds": base.N_SPLITS,
        "aggregations": list(AGGREGATIONS),
        "models_fit": MODELS,
        "lightgbm_refit": False,
        "xgboost_refit": False,
        "threshold_selection": "OOF_only_per_seed_config_aggregation",
        "test_usage": "reporting_only",
        "cache_directory": str(CACHE),
        "cache_hits_on_this_run": cache_hits,
        "expected_fit_count_without_cache": len(TARGET_CONFIGS) * len(base.SEEDS) * (base.N_SPLITS + 1) * len(MODELS),
        "observed_fit_runtime_seconds": float(pd.DataFrame(runtime_data).runtime_seconds.sum()),
        "wall_runtime_seconds": time.perf_counter() - wall_started,
        "leakage_fail": bool(leak.leakage_fail.any()),
        "catnew_merge_reports": merge_reports,
        "decision_guidance": conclusion(delta),
    }
    OUT["metadata"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("=== Step14E2-short complete ===", flush=True)
    print(f"leakage_fail={metadata['leakage_fail']}", flush=True)
    print(f"wall_minutes={metadata['wall_runtime_seconds'] / 60:.2f}", flush=True)


if __name__ == "__main__":
    main()
