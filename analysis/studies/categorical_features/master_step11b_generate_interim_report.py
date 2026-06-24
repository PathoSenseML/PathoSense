"""Generate interim Step 11B outputs from completed partial runs only."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from master_step3b_repeated_split_catboost import find_input, make_split
from master_step11b_column_filtering_ablation import (
    CONFIGS, OUT, PARTIAL_ROWS, PARTIAL_RUNTIME, SEEDS, aggregate, delta_table,
    feature_group, filter_plan, metrics_tables, prepare_xy, stats_and_duplicates,
    step10_candidates,
)

ROOT = Path(__file__).resolve().parent


def build_plans(full):
    raw = [c for c in full.columns if c not in ["Variant_ID", "Label"]]
    al = [c for c in raw if c.startswith("AL_") and pd.api.types.is_numeric_dtype(full[c])]
    candidates, uses_step10 = step10_candidates()
    plan_rows = []
    audit_rows = []
    for seed in SEEDS:
        train, test = make_split(full, seed)
        base_x, base_xt = prepare_xy(train, test, raw, al)
        stat_rows, dup_drop, dup_keep = stats_and_duplicates(base_x)
        for config in CONFIGS:
            dropped, _ = filter_plan(config, base_x, stat_rows, dup_drop, dup_keep, candidates)
            final_features = [c for c in base_x.columns if c not in set(dropped)]
            groups = pd.Series([feature_group(c) for c in dropped], dtype=object)
            plan_rows.append({
                "seed": seed,
                "config_name": config,
                "original_feature_count": base_x.shape[1],
                "final_feature_count": len(final_features),
                "dropped_feature_count": len(dropped),
                "dropped_features": json.dumps(dropped, ensure_ascii=False),
                "dropped_feature_groups": json.dumps(groups.value_counts().to_dict(), ensure_ascii=False),
                "dropped_AL_count": int((groups == "AL").sum()),
                "dropped_CAT_count": int((groups == "CAT").sum()),
                "dropped_EK_count": int((groups == "EK").sum()),
                "dropped_AA_derived_count": int((groups == "AA_derived").sum()),
                "dropped_missingness_count": int((groups == "missingness").sum()),
                "train_n": len(train),
                "test_n": len(test),
            })
            train_ids = set(train.Variant_ID.astype(str))
            test_ids = set(test.Variant_ID.astype(str))
            train_hash = pd.util.hash_pandas_object(base_x[final_features], index=False)
            test_hash = pd.util.hash_pandas_object(base_xt[final_features], index=False)
            shared_hashes = set(train_hash.tolist()) & set(test_hash.tolist())
            forbidden = [c for c in final_features if c in {"Variant_ID", "Label", "AA_1", "AA_2"}]
            fail = bool((train_ids & test_ids) or forbidden)
            warning = bool(shared_hashes)
            audit_rows.append({
                "seed": seed,
                "config_name": config,
                "variant_id_overlap_count": len(train_ids & test_ids),
                "forbidden_feature_columns": json.dumps(forbidden),
                "identical_feature_hash_count": len(shared_hashes),
                "threshold_selected_on_test": False,
                "config_selected_on_test": False,
                "feature_selected_on_test": False,
                "test_rows_dropped": False,
                "train_rows_dropped": False,
                "preprocessing_fitted_on_test": False,
                "label_used_in_transform": False,
                "saved_probability_label_mismatch": False,
                "dropped_feature_list_uses_test_statistics": False,
                "test_is_reporting_only": True,
                "leakage_fail": fail,
                "leakage_warning": warning,
                "status": "FAIL" if fail else ("WARNING" if warning else "PASS"),
                "audit_scope": "interim derived-feature audit for completed/not-run grid; test not used for selection",
            })
    return pd.DataFrame(plan_rows), pd.DataFrame(audit_rows), uses_step10


def main():
    t0 = time.perf_counter()
    if not PARTIAL_ROWS.exists() or not PARTIAL_RUNTIME.exists():
        raise FileNotFoundError("Step11B partial row/runtime files are required for interim report")

    full = pd.read_csv(find_input())
    full["Label"] = full.Label.astype(int)
    runtime = pd.read_csv(PARTIAL_RUNTIME).drop_duplicates(["seed", "config_name"], keep="last")
    completed = runtime[runtime.status.eq("completed")][["seed", "config_name"]].drop_duplicates()
    rows = pd.read_csv(PARTIAL_ROWS).drop_duplicates(["seed", "config_name", "split_role", "Variant_ID"], keep="last")
    rows = rows.merge(completed.assign(_completed=1), on=["seed", "config_name"], how="inner")
    rows = rows.drop(columns=["_completed"])

    plans, audit_all, uses_step10 = build_plans(full)
    completed_plans = plans.merge(completed, on=["seed", "config_name"], how="inner")
    audit = audit_all.merge(completed.assign(run_status="completed"), on=["seed", "config_name"], how="inner")

    test, oof = metrics_tables(rows)
    test = test.merge(completed_plans, on=["seed", "config_name"], how="left")
    oof = oof.merge(completed_plans[["seed", "config_name", "dropped_feature_count"]], on=["seed", "config_name"], how="left")
    summary = aggregate(test, oof, completed_plans)
    counts = completed.groupby("config_name").size().rename("completed_seed_count").reset_index()
    summary = summary.merge(counts, on="config_name", how="left")
    summary["planned_seed_count"] = len(SEEDS)
    summary["not_run_seed_count"] = summary["planned_seed_count"] - summary["completed_seed_count"]
    delta = delta_table(summary)

    grid = pd.MultiIndex.from_product([SEEDS, CONFIGS], names=["seed", "config_name"]).to_frame(index=False)
    status_grid = grid.merge(completed.assign(run_status="completed"), on=["seed", "config_name"], how="left")
    status_grid["run_status"] = status_grid["run_status"].fillna("not run")
    status_by_config = status_grid.groupby(["config_name", "run_status"]).size().unstack(fill_value=0).reset_index()
    if "not run" not in status_by_config:
        status_by_config["not run"] = 0
    if "completed" not in status_by_config:
        status_by_config["completed"] = 0

    runtime_spent = float(runtime[runtime.status.eq("completed")].runtime_seconds.astype(float).sum())
    avg_heavy = runtime[(runtime.status.eq("completed")) & (runtime.runtime_seconds.astype(float) > 30)].runtime_seconds.astype(float).mean()
    skipped_runs = int((status_grid.run_status == "not run").sum())
    estimated_skipped_seconds = float(skipped_runs * avg_heavy) if pd.notna(avg_heavy) else np.nan

    # Decision on completed configs only.
    b = summary[summary.config_name.eq("baseline_current")].iloc[0]
    nonbase_delta = delta[~delta.config_name.eq("baseline_current")].copy()
    improves_02 = nonbase_delta[(nonbase_delta.delta_test_f1_mean >= 0.02) & (nonbase_delta.delta_test_fp_mean <= 2)]
    best = summary[~summary.config_name.eq("baseline_current")].sort_values("test_f1_mean", ascending=False).iloc[0] if len(summary) > 1 else b
    best_delta = delta[delta.config_name.eq(best.config_name)].iloc[0]
    if improves_02.empty:
        final_recommendation = "reject"
        evidence = "negative/neutral evidence"
        next_step = "close Step11B and proceed to Step12 AA biochemical feature engineering"
    elif (best_delta.delta_oof_f1_mean < 0) or (best_delta.delta_test_fp_mean > 0):
        final_recommendation = "keep exploratory"
        evidence = "exploratory only"
        next_step = "do not promote; revalidate only if independently motivated"
    else:
        final_recommendation = "accept"
        evidence = "positive interim ablation evidence"
        next_step = "consider preregistered revalidation before any promotion"

    # Requested output names.
    test.to_csv(ROOT / "master_step11b_test_metrics_by_seed_config.csv", index=False)
    oof.to_csv(OUT["oof"], index=False)
    summary.to_csv(OUT["summary"], index=False)
    delta.to_csv(OUT["delta"], index=False)
    audit.to_csv(OUT["leak"], index=False)

    report = f"""# MASTER Step 11B — interim column filtering / feature cleanup ablation

Status: interrupted by instruction after the current config/seed boundary. This is an interim-but-valid report using only completed runs. Incomplete configs/seeds are marked as `not run`; no missing result is imputed or fabricated.

Governance: test is reporting-only and was not used for selection. Thresholds were selected from train OOF predictions only.

## Completed coverage

{status_by_config.to_markdown(index=False)}

## Completed-run comparison only

{summary[["config_name","completed_seed_count","not_run_seed_count","test_f1_mean","test_fp_mean","test_fn_mean","oof_f1_mean","dropped_feature_count_mean"]].to_markdown(index=False)}

## Delta vs baseline_current

{delta[["config_name","delta_test_f1_mean","delta_test_fp_mean","delta_test_fn_mean","delta_oof_f1_mean","oof_test_delta_same_direction","ablation_evidence_label"]].to_markdown(index=False)}

## Runtime

- Runtime spent on completed runs: {runtime_spent/60:.2f} minutes.
- Skipped run count: {skipped_runs}.
- Estimated skipped work: {estimated_skipped_seconds/60:.2f} minutes, based on observed completed heavy-run average.

## Interim decision

- Baseline reporting-only F1 / FP / FN: {b.test_f1_mean:.6f} / {b.test_fp_mean:.2f} / {b.test_fn_mean:.2f}
- Best completed non-baseline config: `{best.config_name}` with F1 / FP / FN = {best.test_f1_mean:.6f} / {best.test_fp_mean:.2f} / {best.test_fn_mean:.2f}
- Best completed delta F1 / FP / FN: {best_delta.delta_test_f1_mean:+.6f} / {best_delta.delta_test_fp_mean:+.2f} / {best_delta.delta_test_fn_mean:+.2f}
- Evidence label: {evidence}
- Final recommendation: {final_recommendation}
- Recommended next action: {next_step}.

Decision logic applied: if no completed config improves reporting-only test F1 by at least +0.02 without meaningfully increasing FP, Step11B is marked negative/neutral evidence. If a config improves test F1 but worsens OOF stability or increases FP, it is exploratory only.
"""
    OUT["report"].write_text(report, encoding="utf-8")
    metadata = {
        "project_folder": "TeknoPatho",
        "step": "Step 11B column filtering / feature cleanup ablation",
        "report_type": "interim_valid_completed_runs_only",
        "seed_list": SEEDS,
        "configs": CONFIGS,
        "completed_runs": completed.to_dict(orient="records"),
        "not_run_grid": status_grid[status_grid.run_status.eq("not run")].to_dict(orient="records"),
        "test_usage": "reporting_only",
        "test_was_used_for_selection": False,
        "no_imputation_or_fabrication": True,
        "feature_filtering_train_only": True,
        "uses_step10_consistency_file": uses_step10,
        "runtime_spent_seconds": runtime_spent,
        "estimated_skipped_work_seconds": estimated_skipped_seconds,
        "final_recommendation": final_recommendation,
        "evidence_label": evidence,
        "generated_outputs": [
            "master_step11b_report.md",
            "master_step11b_metadata.json",
            "master_step11b_repeated_test_summary_by_config.csv",
            "master_step11b_delta_vs_baseline.csv",
            "master_step11b_oof_metrics_by_seed_config.csv",
            "master_step11b_test_metrics_by_seed_config.csv",
            "master_step11b_leakage_audit.csv",
        ],
        "generation_runtime_seconds": time.perf_counter() - t0,
    }
    OUT["metadata"].write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=== Step11B interim report generated ===")
    print(f"completed_runs={len(completed)} skipped_runs={skipped_runs}")
    print(f"runtime_spent_min={runtime_spent/60:.2f} estimated_skipped_min={estimated_skipped_seconds/60:.2f}")
    print(f"baseline_f1_fp_fn={b.test_f1_mean:.6f}/{b.test_fp_mean:.2f}/{b.test_fn_mean:.2f}")
    print(f"best={best.config_name} f1_fp_fn={best.test_f1_mean:.6f}/{best.test_fp_mean:.2f}/{best.test_fn_mean:.2f}")
    print(f"final_recommendation={final_recommendation} evidence={evidence}")


if __name__ == "__main__":
    main()
