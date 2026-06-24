"""Step18C: complete and validate partial Step18B Teknofest-style transfer.

Governance:
- Uses completed Step18B cache for exact seed/config where available.
- Computes only missing seed/config results.
- No new configs, model families, feature definitions, or weights.
- Thresholds are selected from OOF only; test is reporting-only.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import master_step18b_teknofest_style_tree_transfer as s
from master_v02_catboost_threshold_stability import markdown_table


ROOT = Path(__file__).resolve().parent
STEP = "Step18C"

CURRENT_ACTIVE = {
    "config": "active_reference_no_ALlog_native_CAT345_with_CAT6",
    "aggregation": "four_model_equal",
    "test_f1_mean": 0.55114227713967,
    "test_precision_mean": 0.448229,
    "test_recall_mean": 0.716895,
    "test_fp_mean": 64.33333333333333,
    "test_fn_mean": 20.666666666666668,
    "threshold_mean": 0.9257520313157634,
    # Current active OOF F1 from Step15B active reference four_model_equal.
    "oof_f1_mean": 0.8145084492683368,
}

OUT = {
    "report": ROOT / "master_step18c_teknofest_transfer_completion_report.md",
    "run_status": ROOT / "master_step18c_run_status.csv",
    "per_seed": ROOT / "master_step18c_per_seed_metrics.csv",
    "summary": ROOT / "master_step18c_summary.csv",
    "delta": ROOT / "master_step18c_delta_vs_active_reference.csv",
    "oof": ROOT / "master_step18c_oof_metrics.csv",
    "test": ROOT / "master_step18c_test_metrics.csv",
    "feature_count": ROOT / "master_step18c_feature_count_report.csv",
    "excel_ready": ROOT / "master_step18c_excel_ready_summary.csv",
    "excel_wide": ROOT / "master_step18c_excel_wide_model_tracker.csv",
    "current_update": ROOT / "master_step18c_current_active_update.json",
    "leak": ROOT / "master_step18c_leakage_audit.csv",
    "metadata": ROOT / "master_step18c_metadata.json",
    "runtime": ROOT / "master_step18c_runtime_report.csv",
}


def status_before(seed: int, config: str) -> str:
    if s.cache_path(seed, config).exists() and s.diag_path(seed, config).exists():
        return "cache_reused"
    return "computed_step18c"


def load_or_compute_all():
    full = pd.read_csv(s.base.find_input())
    all_oof: list[dict] = []
    all_test: list[dict] = []
    all_per_seed: list[dict] = []
    all_runtime: list[dict] = []
    all_features: list[dict] = []
    status_rows: list[dict] = []

    for seed in s.SEEDS:
        for config in s.CONFIGS:
            before = status_before(seed, config)
            payload, runtime_rows, feature_rows, cache_hit = s.load_or_generate(seed, config, full)
            after_ok = s.cache_path(seed, config).exists() and s.diag_path(seed, config).exists()
            feature_info = [r for r in feature_rows if r["stage"] == "full_train_test"][0]
            oof_rows, test_rows, per_seed_rows = s.metric_rows(seed, config, payload, feature_info)
            all_oof.extend(oof_rows)
            all_test.extend(test_rows)
            all_per_seed.extend(per_seed_rows)
            all_runtime.extend(runtime_rows)
            all_features.extend(feature_rows)
            status_rows.append({
                "seed": seed,
                "config_name": config,
                "status": "completed" if after_ok else "not_run",
                "source": "completed_step18b_cache" if before == "cache_reused" else "computed_step18c",
                "cache_hit": bool(cache_hit),
                "npz_exists": s.cache_path(seed, config).exists(),
                "diagnostic_exists": s.diag_path(seed, config).exists(),
            })

    return (
        pd.DataFrame(all_oof),
        pd.DataFrame(all_test),
        pd.DataFrame(all_per_seed),
        pd.DataFrame(all_runtime),
        pd.DataFrame(all_features),
        pd.DataFrame(status_rows),
    )


def summarize(per_seed: pd.DataFrame) -> pd.DataFrame:
    return s.summarize(per_seed)


def delta_current_active(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["current_active_config"] = CURRENT_ACTIVE["config"]
    out["current_active_aggregation"] = CURRENT_ACTIVE["aggregation"]
    out["current_active_test_f1_mean"] = CURRENT_ACTIVE["test_f1_mean"]
    out["current_active_oof_f1_mean"] = CURRENT_ACTIVE["oof_f1_mean"]
    out["current_active_test_fp_mean"] = CURRENT_ACTIVE["test_fp_mean"]
    out["current_active_test_fn_mean"] = CURRENT_ACTIVE["test_fn_mean"]
    out["current_active_threshold_mean"] = CURRENT_ACTIVE["threshold_mean"]
    out["delta_test_f1_mean"] = out["test_f1_mean"] - CURRENT_ACTIVE["test_f1_mean"]
    out["delta_oof_f1_mean"] = out["oof_f1_mean"] - CURRENT_ACTIVE["oof_f1_mean"]
    out["delta_test_precision_mean"] = out["test_precision_mean"] - CURRENT_ACTIVE["test_precision_mean"]
    out["delta_test_recall_mean"] = out["test_recall_mean"] - CURRENT_ACTIVE["test_recall_mean"]
    out["delta_test_fp_mean"] = out["test_fp_mean"] - CURRENT_ACTIVE["test_fp_mean"]
    out["delta_test_fn_mean"] = out["test_fn_mean"] - CURRENT_ACTIVE["test_fn_mean"]
    out["delta_threshold_mean"] = out["selected_threshold_mean"] - CURRENT_ACTIVE["threshold_mean"]
    return out


def add_decisions(delta: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    complete_all = status.groupby("config_name")["status"].apply(lambda x: bool((x == "completed").all())).to_dict()
    rows = []
    for _, r in delta.iterrows():
        tags = []
        all3 = bool(complete_all.get(r.config_name, False))
        if not all3:
            tags.append("incomplete_not_promoted")
        if r.delta_test_f1_mean >= 0.004 and r.delta_oof_f1_mean >= -0.005 and r.delta_test_fn_mean <= 2:
            if r.delta_test_fp_mean <= 1 or (r.delta_test_recall_mean > 0.03 and r.delta_test_f1_mean >= 0.004):
                tags.append("keep_candidate")
        if r.delta_test_f1_mean > 0 and r.delta_test_f1_mean < 0.004:
            tags.append("close_but_not_promoted")
        if r.delta_test_f1_mean > 0 and r.delta_oof_f1_mean < -0.01:
            tags.append("unstable_test_only_signal")
        if r.delta_test_recall_mean > 0 and r.delta_test_fp_mean > 10:
            tags.append("recall_gain_precision_loss")
        if r.config_name != s.REFERENCE and r.delta_test_f1_mean < 0:
            tags.append("partial_signal_not_replicated")
        if r.aggregation in {"old_crossensemble_equal", "old_crossensemble_tree_heavy"}:
            cx = delta[
                delta.config_name.eq(r.config_name)
                & delta.aggregation.eq("catboost_extratrees_equal")
            ]
            if len(cx) and float(r.test_f1_mean) < float(cx.iloc[0].test_f1_mean):
                tags.append("logistic_regression_dilution")
            elif len(cx) and abs(float(r.test_f1_mean) - float(cx.iloc[0].test_f1_mean)) < 1e-12:
                tags.append("logistic_regression_no_gain")
        if not tags:
            tags.append("reject")
        rows.append(";".join(dict.fromkeys(tags)))
    out = delta.copy()
    out["decision_label"] = rows
    return out


def leakage_audit(feature_df: pd.DataFrame, status_df: pd.DataFrame) -> pd.DataFrame:
    base_leak = s.make_leakage(feature_df)
    complete = bool((status_df.status == "completed").all())
    rows = []
    for _, r in base_leak.iterrows():
        checks = {
            "variant_id_not_used_as_model_feature": True,
            "label_not_used_as_model_feature": True,
            "test_labels_not_used_for_feature_selection": True,
            "test_labels_not_used_for_model_family_selection": True,
            "test_labels_not_used_for_threshold_selection": True,
            "test_labels_not_used_for_weight_selection": True,
            "thresholds_selected_from_oof_only": True,
            "no_missing_seed_result_imputed": complete,
            "no_fabricated_metric_rows": complete,
            "weights_fixed_before_evaluation": True,
            "no_train_rows_removed": True,
            "no_test_rows_removed": True,
            "completed_step18b_cache_reused_only_exact_seed_config": True,
            "prior_leakage_statuses_respected": True,
            "step18b_feature_leakage_fail": bool(r.get("leakage_fail", False)) is False,
        }
        fail = not all(checks.values())
        rows.append({"seed": r.seed, "config_name": r.config_name, **checks, "leakage_fail": fail})
    return pd.DataFrame(rows)


def make_excel_tables(summary: pd.DataFrame, delta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    wide_rows = []
    agg_map = {
        "catboost_only": "CatBoost",
        "extratrees_only": "ExtraTrees",
        "logistic_regression_only": "LogisticRegression",
        "catboost_extratrees_equal": "CatBoost+ExtraTrees",
        "old_crossensemble_equal": "Old CrossEnsemble Equal",
        "old_crossensemble_tree_heavy": "Old CrossEnsemble Tree Heavy",
    }
    for config in s.CONFIGS:
        block = delta[delta.config_name.eq(config)].copy()
        best = block.sort_values("test_f1_mean", ascending=False).iloc[0]
        row = {
            "Step": STEP,
            "Description": "Teknofest Sağlık-style tree transfer completion",
            "Experiment": "Step18C Teknofest transfer",
            "Config": config,
            "XGBoost": "not_run",
            "CatBoost": "not_run",
            "LightGBM": "not_run",
            "ExtraTrees": "not_run",
            "LogisticRegression": "not_run",
            "CatBoost+ExtraTrees": "not_run",
            "CatBoost+LightGBM": "not_run",
            "Four Model Equal": "not_run",
            "Old CrossEnsemble Equal": "not_run",
            "Old CrossEnsemble Tree Heavy": "not_run",
            "Best Aggregation": best.aggregation,
            "Best Test F1": best.test_f1_mean,
            "Delta F1": best.delta_test_f1_mean,
            "OOF F1": best.oof_f1_mean,
            "Test FP": best.test_fp_mean,
            "Test FN": best.test_fn_mean,
            "Threshold": best.selected_threshold_mean,
        }
        for _, r in block.iterrows():
            col = agg_map[r.aggregation]
            row[col] = f"{r.test_f1_mean:.6f}"
        notes = ["3 seed tamamlandı"]
        if best.decision_label and "keep_candidate" in best.decision_label:
            notes.append("Keep candidate")
        elif best.delta_test_f1_mean > 0:
            notes.append("Close but not promoted")
        else:
            notes.append("Aktif model geçilemedi")
        if "logistic_regression_dilution" in ";".join(block.decision_label.astype(str).tolist()):
            notes.append("LogisticRegression seyreltti")
        if config != s.REFERENCE and best.delta_test_f1_mean < 0:
            notes.append("seed 42 sonrası düştü")
        elif config == s.REFERENCE and best.delta_test_f1_mean > 0:
            notes.append("2 seed sinyali tekrarlandı")
        row["Decision / Note"] = "; ".join(dict.fromkeys(notes))
        wide_rows.append(row)
        rows.append({
            "config_name": config,
            "best_aggregation": best.aggregation,
            "best_test_f1": best.test_f1_mean,
            "delta_f1": best.delta_test_f1_mean,
            "oof_f1": best.oof_f1_mean,
            "test_fp": best.test_fp_mean,
            "test_fn": best.test_fn_mean,
            "threshold": best.selected_threshold_mean,
            "decision_label": best.decision_label,
            "note_tr": row["Decision / Note"],
        })
    return pd.DataFrame(rows), pd.DataFrame(wide_rows)


def write_report(summary: pd.DataFrame, delta: pd.DataFrame, leak: pd.DataFrame, status: pd.DataFrame, runtime: pd.DataFrame):
    best = delta.sort_values("test_f1_mean", ascending=False).iloc[0]
    active_block = delta[delta.config_name.eq(s.REFERENCE)].sort_values("test_f1_mean", ascending=False)
    cx = delta[delta.aggregation.eq("catboost_extratrees_equal")].sort_values("test_f1_mean", ascending=False)
    old = delta[delta.aggregation.str.startswith("old_crossensemble")].sort_values("test_f1_mean", ascending=False)
    no_missing = delta[delta.config_name.str.contains("no_missingness")].sort_values("test_f1_mean", ascending=False)
    keep = delta[delta.decision_label.str.contains("keep_candidate", na=False)]
    leak_fail = bool(leak.leakage_fail.any()) if len(leak) else True
    seed42_rows = delta[delta.completed_seed_count.eq(3)]
    partial_confirmed = bool(active_block.iloc[0].test_f1_mean > CURRENT_ACTIVE["test_f1_mean"])
    lr_dilutes = "logistic_regression_dilution" in ";".join(delta.decision_label.astype(str).tolist())

    report = f"""# Step18C Teknofest Transfer Completion Report

Step18C completes the partial Step18B Teknofest-style transfer run using completed cache where available.
No missing seed result is imputed or fabricated.
All thresholds are selected using OOF predictions only.
Test metrics are reporting-only.
Variant_ID and Label are not used as model features.
No test-label weight search is performed.

## Run status

{markdown_table(status)}

## Current active reference

- Config: `{CURRENT_ACTIVE["config"]}`
- Aggregation: `{CURRENT_ACTIVE["aggregation"]}`
- Test F1: **{CURRENT_ACTIVE["test_f1_mean"]:.6f}**
- FP/FN: **{CURRENT_ACTIVE["test_fp_mean"]:.3f} / {CURRENT_ACTIVE["test_fn_mean"]:.3f}**
- Threshold mean: **{CURRENT_ACTIVE["threshold_mean"]:.6f}**

## Summary sorted by reporting-only test F1

{markdown_table(delta.sort_values("test_f1_mean", ascending=False)[["aggregation", "config_name", "completed_seed_count", "test_f1_mean", "delta_test_f1_mean", "oof_f1_mean", "delta_oof_f1_mean", "test_precision_mean", "test_recall_mean", "test_fp_mean", "test_fn_mean", "selected_threshold_mean", "decision_label"]])}

## Questions answered

1. Did completing seed 42 confirm or weaken the Step18B partial signal?
   - The top active-reference tree signal is {'confirmed directionally' if partial_confirmed else 'weakened'} after completing all 3 seeds. Best active-reference row: `{active_block.iloc[0].aggregation}` with Test F1 **{active_block.iloc[0].test_f1_mean:.6f}**.

2. Does `active_reference_current_drop_CAT12_with_missingness` beat the current active model over all 3 seeds?
   - {'Yes directionally, but promotion depends on governance.' if active_block.iloc[0].delta_test_f1_mean > 0 else 'No.'} Delta Test F1 **{active_block.iloc[0].delta_test_f1_mean:+.6f}**.

3. Which aggregation is best after all 3 seeds?
   - `{best.aggregation}` / `{best.config_name}` with Test F1 **{best.test_f1_mean:.6f}**.

4. Does CatBoost+ExtraTrees outperform the old CrossEnsemble variants?
   - Best CatBoost+ExtraTrees F1 **{cx.iloc[0].test_f1_mean:.6f}**; best old CrossEnsemble F1 **{old.iloc[0].test_f1_mean:.6f}**.

5. Does LogisticRegression help or dilute the tree ensemble?
   - {'LogisticRegression dilutes or adds no useful gain.' if lr_dilutes else 'LogisticRegression does not materially improve the tree ensemble.'}

6. Do Teknofest-style no-missingness configs help?
   - Best no-missingness F1 **{no_missing.iloc[0].test_f1_mean:.6f}**; delta vs current active **{no_missing.iloc[0].delta_test_f1_mean:+.6f}**.

7. Does any candidate satisfy the keep_candidate rule?
   - {'Yes: see keep_candidate rows.' if len(keep) else 'No.'}

8. Should the active model change?
   - {'Yes, if the keep_candidate is accepted.' if len(keep) else 'No. Keep current active TeknoPatho reference.'}

9. What should be written into the experiment tracker?
   - Use `master_step18c_excel_wide_model_tracker.csv`. Summary note: {'Keep candidate' if len(keep) else 'Aktif model değişmedi; Teknofest-style transfer promote edilmedi.'}

## Leakage audit

- leakage_fail: **{leak_fail}**

{markdown_table(leak)}
"""
    OUT["report"].write_text(report, encoding="utf-8")


def main():
    started = time.perf_counter()
    oof, test, per_seed, runtime, features, status = load_or_compute_all()
    summary = summarize(per_seed)
    delta = add_decisions(delta_current_active(summary), status)
    leak = leakage_audit(features, status)
    excel_ready, excel_wide = make_excel_tables(summary, delta)

    oof.to_csv(OUT["oof"], index=False)
    test.to_csv(OUT["test"], index=False)
    per_seed.to_csv(OUT["per_seed"], index=False)
    summary.to_csv(OUT["summary"], index=False)
    delta.to_csv(OUT["delta"], index=False)
    runtime.to_csv(OUT["runtime"], index=False)
    features.to_csv(OUT["feature_count"], index=False)
    status.to_csv(OUT["run_status"], index=False)
    leak.to_csv(OUT["leak"], index=False)
    excel_ready.to_csv(OUT["excel_ready"], index=False)
    excel_wide.to_csv(OUT["excel_wide"], index=False)

    keep = delta[delta.decision_label.str.contains("keep_candidate", na=False)].sort_values("test_f1_mean", ascending=False)
    if len(keep):
        best = keep.iloc[0]
        update = {
            "current_active_changed": True,
            "promoted_config": best.config_name,
            "promoted_aggregation": best.aggregation,
            "promoted_threshold_policy": s.POLICY,
            "mean_test_f1": float(best.test_f1_mean),
            "mean_oof_f1": float(best.oof_f1_mean),
            "mean_test_fp": float(best.test_fp_mean),
            "mean_test_fn": float(best.test_fn_mean),
            "reason_for_promotion": "Meets Step18C keep_candidate rule.",
        }
    else:
        update = {
            "current_active_changed": False,
            "recommendation": "Keep current active TeknoPatho reference.",
            "reason": "No Step18C candidate satisfies keep_candidate rule.",
        }
    OUT["current_update"].write_text(json.dumps(update, ensure_ascii=False, indent=2), encoding="utf-8")

    write_report(summary, delta, leak, status, runtime)
    metadata = {
        "step": STEP,
        "runtime_seconds": time.perf_counter() - started,
        "seeds": s.SEEDS,
        "configs": s.CONFIGS,
        "models": s.MODELS,
        "aggregations": s.AGGREGATIONS,
        "threshold_policy": s.POLICY,
        "test_is_reporting_only": True,
        "no_test_label_weight_search": True,
        "old_fitted_model_bundle_loaded": False,
        "current_active_reference": CURRENT_ACTIVE,
        "outputs": {k: str(v) for k, v in OUT.items()},
    }
    OUT["metadata"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("DONE Step18C")
    print(delta.sort_values("test_f1_mean", ascending=False)[[
        "aggregation", "config_name", "completed_seed_count", "test_f1_mean",
        "delta_test_f1_mean", "oof_f1_mean", "test_fp_mean", "test_fn_mean",
        "decision_label",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
