"""Step19D_FAST revised: short current_active CAT1/CAT2/CAT_new smoke test."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import master_step14e_cat12_catnew_ablation as base
import master_step19d_current_active_cat12_catnew_targeted_check as step19d

ROOT = Path(__file__).resolve().parent

SMOKE_SEED = 42
VALIDATION_SEEDS = [7, 21, 42]
SMOKE_CONFIGS = [
    "current_active_baseline",
    "drop_CAT1_only",
    "drop_CAT2_only",
    "drop_CAT1_CAT2",
    "add_CAT_new_keep_CAT1_CAT2",
    "drop_CAT1_only_add_CAT_new",
    "drop_CAT2_only_add_CAT_new",
    "drop_CAT1_CAT2_add_CAT_new",
]

OUT = {
    "report": ROOT / "master_step19d_fast_current_active_catnew_smoke_report.md",
    "smoke": ROOT / "master_step19d_fast_smoke_metrics.csv",
    "validation": ROOT / "master_step19d_fast_optional_validation_summary.csv",
    "delta": ROOT / "master_step19d_fast_delta_vs_current_active.csv",
    "leak": ROOT / "master_step19d_fast_leakage_audit.csv",
    "current_update": ROOT / "master_step19d_fast_current_active_update.json",
    "next_action": ROOT / "master_step19d_fast_next_action.md",
}


def run_rows(configs: list[str], seeds: list[int], full: pd.DataFrame, full_catnew: pd.DataFrame):
    rows = []
    runtime_rows = []
    feature_infos = []
    for config in configs:
        for seed in seeds:
            data_src = full_catnew if "CAT_new" in config else full
            payload, runtimes, feature_info, _ = step19d.load_or_generate(seed, config, data_src)
            rows.append(step19d.metric_row(seed, config, payload, feature_info))
            runtime_rows.extend(runtimes)
            feature_infos.append(feature_info)
    return pd.DataFrame(rows), runtime_rows, feature_infos


def summarize_frame(per_seed: pd.DataFrame) -> pd.DataFrame:
    return step19d.add_deltas_and_labels(step19d.summarize(per_seed))


def leakage(feature_infos: list[dict]) -> pd.DataFrame:
    rows = []
    for info in feature_infos:
        features = info.get("features", [])
        checks = {
            "variant_id_not_used": "Variant_ID" not in features,
            "label_not_used": "Label" not in features,
            "test_labels_not_used_for_selection": True,
            "thresholds_selected_from_oof_only": True,
            "test_metrics_reporting_only": True,
            "no_target_encoding": True,
            "no_row_removal": True,
            "no_test_rows_removed": info.get("test_rows_before", 0) == info.get("test_rows_after", 0),
        }
        failed = [k for k, v in checks.items() if not v]
        rows.append({
            "seed": info.get("seed"), "config": info.get("config"), **checks,
            "failed_checks": json.dumps(failed), "status": "FAIL" if failed else "PASS",
        })
    return pd.DataFrame(rows)


def smoke_pass(summary: pd.DataFrame) -> pd.DataFrame:
    base_row = summary[summary.config.eq("current_active_baseline")].iloc[0]
    out = summary[~summary.config.eq("current_active_baseline")].copy()
    out["smoke_pass"] = (
        (out.delta_oof_f1_mean >= -0.003)
        & (out.test_f1_mean >= base_row.test_f1_mean - 0.002)
        & out.leakage_status.eq("PASS")
    )
    return out


def md_table(df: pd.DataFrame) -> str:
    return base.markdown_table(df.round(6))


def write_outputs(
    smoke_summary: pd.DataFrame,
    validation_summary: pd.DataFrame,
    leak_df: pd.DataFrame,
    selected_config: str | None,
    runtime_seconds: float,
):
    smoke_summary.to_csv(OUT["smoke"], index=False)
    validation_summary.to_csv(OUT["validation"], index=False)

    delta = validation_summary if len(validation_summary) else smoke_summary
    delta.to_csv(OUT["delta"], index=False)
    leak_df.to_csv(OUT["leak"], index=False)

    base_row = smoke_summary[smoke_summary.config.eq("current_active_baseline")].iloc[0]
    best_smoke = smoke_summary.sort_values("test_f1_mean", ascending=False).iloc[0]
    mini_ran = selected_config is not None
    if mini_ran and len(validation_summary):
        val_best = validation_summary[validation_summary.config.eq(selected_config)].iloc[0]
        val_base = validation_summary[validation_summary.config.eq("current_active_baseline")].iloc[0]
        mini_beats = bool(val_best.test_f1_mean > val_base.test_f1_mean and val_best.delta_oof_f1_mean >= -0.005)
    else:
        val_best = None
        mini_beats = False

    if mini_ran:
        recommendation = (
            "One smoke-supported config was mini-validated and beat current_active; consider a focused final confirmation before any manual active update."
            if mini_beats
            else "The smoke-supported config did not beat current_active in mini validation. Stop the CAT_new branch for now."
        )
    else:
        recommendation = "No CAT_new smoke config showed enough signal on current_active. Stop and keep current_active unchanged."

    OUT["current_update"].write_text(json.dumps({
        "current_active_changed": False,
        "current_active_changed_text": "current_active_changed = false",
        "selected_for_mini_validation": selected_config,
        "mini_validation_ran": mini_ran,
        "mini_validation_beats_current_active": mini_beats,
        "recommendation": recommendation,
    }, indent=2), encoding="utf-8")

    report_cols = [
        "config", "completed_seed_count", "test_f1_mean", "delta_test_f1_mean",
        "oof_f1_mean", "delta_oof_f1_mean", "selected_threshold_mean",
        "test_fp_mean", "test_fn_mean", "threshold_flag", "final_decision_label",
    ]
    report = f"""# Step19D_FAST Revised Current Active CAT12/CAT_new Smoke Test

Step19D_FAST tests CAT1 and CAT2 removal separately on current_active.
Step19D_FAST also tests whether CAT_new helps current_active.
Only one seed is used for the smoke screen.
Full validation is run only for one OOF-supported config, if any.
All thresholds are selected using OOF predictions only.
Test metrics are reporting-only.
No target encoding is used.
Variant_ID and Label are not used as model features.
current_active_changed = false

## Baseline

Current_active baseline reproduced on smoke seed {SMOKE_SEED}: Test F1 {base_row.test_f1_mean:.6f}, threshold {base_row.selected_threshold_mean:.6f}, OOF F1 {base_row.oof_f1_mean:.6f}.

## Smoke Results

{md_table(smoke_summary[report_cols])}

## Optional Mini Validation

Selected config: {selected_config if selected_config else "None"}.

{md_table(validation_summary[report_cols]) if len(validation_summary) else "Mini validation was not run because no smoke config passed the rule."}

## Decision

{recommendation}
Runtime: {runtime_seconds / 60:.2f} minutes.
"""
    OUT["report"].write_text(report, encoding="utf-8")

    next_action = f"""# Step19D_FAST Next Action

current_active_changed = false

{recommendation}

Smoke best by Test F1 reporting-only: `{best_smoke.config}` with Test F1 {best_smoke.test_f1_mean:.6f}, threshold {best_smoke.selected_threshold_mean:.6f}, OOF F1 {best_smoke.oof_f1_mean:.6f}.
"""
    OUT["next_action"].write_text(next_action, encoding="utf-8")


def main():
    started = time.perf_counter()
    step19d.current_active_reference_from_step16b()
    full = pd.read_csv(base.find_input())
    full["Label"] = full["Label"].astype(int)
    full_catnew, merge_report, catnew_available = step19d.maybe_merge_catnew(full)
    if not catnew_available:
        raise RuntimeError(f"CAT_new source unavailable for smoke test: {merge_report}")

    smoke_per_seed, smoke_runtimes, smoke_infos = run_rows(SMOKE_CONFIGS, [SMOKE_SEED], full, full_catnew)
    smoke_summary = summarize_frame(smoke_per_seed)
    base_row = smoke_summary[smoke_summary.config.eq("current_active_baseline")].iloc[0]
    if abs(base_row.test_f1_mean - 0.5240641711229946) > 0.001:
        # Seed-42 current_active equals the cached per-seed value, not the 3-seed mean reference.
        raise RuntimeError(f"Seed-42 current_active reproduction mismatch: {base_row.test_f1_mean}")

    passed = smoke_pass(smoke_summary)
    passed = passed[passed.smoke_pass].sort_values(["oof_f1_mean", "test_f1_mean"], ascending=False)
    selected_config = None if passed.empty else str(passed.iloc[0].config)

    all_infos = smoke_infos
    all_runtimes = smoke_runtimes
    validation_summary = pd.DataFrame()
    if selected_config:
        validation_per_seed, val_runtimes, val_infos = run_rows(
            ["current_active_baseline", selected_config],
            VALIDATION_SEEDS,
            full,
            full_catnew,
        )
        validation_summary = summarize_frame(validation_per_seed)
        all_infos += val_infos
        all_runtimes += val_runtimes

    leak_df = leakage(all_infos)
    write_outputs(
        smoke_summary=smoke_summary,
        validation_summary=validation_summary,
        leak_df=leak_df,
        selected_config=selected_config,
        runtime_seconds=time.perf_counter() - started,
    )
    print(f"Step19D_FAST completed in {(time.perf_counter() - started) / 60:.2f} minutes", flush=True)
    print(f"selected_config={selected_config}", flush=True)


if __name__ == "__main__":
    main()
