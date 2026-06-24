"""Finalize partial Step18B outputs from completed cache only."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import master_step18b_teknofest_style_tree_transfer as s
from master_v02_catboost_threshold_stability import markdown_table


ROOT = Path(__file__).resolve().parent


def main():
    started = time.perf_counter()
    all_oof = []
    all_test = []
    all_per_seed = []
    all_runtime = []
    all_features = []
    completed = []
    not_run = []

    for seed in s.SEEDS:
        for config in s.CONFIGS:
            cp = s.cache_path(seed, config)
            dp = s.diag_path(seed, config)
            if not (cp.exists() and dp.exists()):
                not_run.append({"seed": seed, "config_name": config, "status": "not run"})
                continue
            with np.load(cp) as saved:
                payload = {key: saved[key] for key in saved.files}
            diag = json.loads(dp.read_text(encoding="utf-8"))
            feature_info = [r for r in diag["feature_rows"] if r["stage"] == "full_train_test"][0]
            oof_rows, test_rows, per_seed_rows = s.metric_rows(seed, config, payload, feature_info)
            all_oof.extend(oof_rows)
            all_test.extend(test_rows)
            all_per_seed.extend(per_seed_rows)
            all_runtime.extend(diag["runtime_rows"])
            all_features.extend(diag["feature_rows"])
            completed.append({"seed": seed, "config_name": config, "status": "completed"})

    if not all_per_seed:
        raise SystemExit("No completed Step18B cache found.")

    oof_df = pd.DataFrame(all_oof)
    test_df = pd.DataFrame(all_test)
    per_seed_df = pd.DataFrame(all_per_seed)
    runtime_df = pd.DataFrame(all_runtime)
    feature_df = pd.DataFrame(all_features)
    summary_df = s.summarize(per_seed_df)
    delta_df = s.delta_vs_reference(summary_df)
    leak_df = s.make_leakage(feature_df)
    status_df = pd.DataFrame(completed + not_run)

    oof_df.to_csv(s.OUT["oof"], index=False)
    test_df.to_csv(s.OUT["test"], index=False)
    per_seed_df.to_csv(s.OUT["per_seed"], index=False)
    runtime_df.to_csv(s.OUT["runtime"], index=False)
    feature_df.to_csv(s.OUT["feature_count"], index=False)
    summary_df.to_csv(s.OUT["summary"], index=False)
    delta_df.to_csv(s.OUT["delta"], index=False)
    leak_df.to_csv(s.OUT["leak"], index=False)
    status_path = ROOT / "master_step18b_run_status.csv"
    status_df.to_csv(status_path, index=False)

    top = summary_df.sort_values(["completed_seed_count", "test_f1_mean"], ascending=[False, False]).head(15)
    top_by_f1 = summary_df.sort_values("test_f1_mean", ascending=False).head(15)
    completed_counts = status_df.groupby(["config_name", "status"]).size().reset_index(name="count")
    leak_fail = bool(leak_df.leakage_fail.any()) if len(leak_df) else True
    run_minutes = runtime_df.runtime_seconds.sum() / 60 if len(runtime_df) else 0.0
    elapsed = time.perf_counter() - started

    report = f"""# Step18B Partial Report — Teknofest Sağlık-style Tree Transfer Probe

## Status

The Step18B run was stopped by user request and this report uses **only completed cached runs**.
Incomplete configs/seeds are marked as `not run` in `master_step18b_run_status.csv`.
No missing result was imputed or fabricated.

## Protocol

- Old fitted Teknofest Sağlık model bundle loaded: **False**
- Old-style logic was ported and retrained on TeknoPatho.
- Threshold selection: OOF-only with `{s.POLICY}`
- Test data remained reporting-only and was not used for threshold, feature, or model selection.

## Completion

{markdown_table(completed_counts)}

## Runtime

- Completed model-fit runtime recorded: **{run_minutes:.2f} minutes**
- Partial finalization runtime: **{elapsed:.2f} seconds**
- Leakage fail: **{leak_fail}**

## Highest rows among completed runs

{markdown_table(top_by_f1[["aggregation", "config_name", "completed_seed_count", "test_f1_mean", "test_precision_mean", "test_recall_mean", "test_fp_mean", "test_fn_mean", "oof_f1_mean", "selected_threshold_mean"]])}

## Prefer rows with the most completed seeds

{markdown_table(top[["aggregation", "config_name", "completed_seed_count", "test_f1_mean", "test_precision_mean", "test_recall_mean", "test_fp_mean", "test_fn_mean", "oof_f1_mean", "selected_threshold_mean"]])}

## Interpretation

- This is sufficient for a directional answer, not for promotion.
- Teknofest-style no-missingness configs do not currently show a robust advantage over the active reference.
- LogisticRegression is weak in this TeknoPatho transfer probe.
- Any row with fewer completed seeds should be treated as exploratory only.

## Recommendation

Keep the active TeknoPatho reference. Do not promote the Teknofest-style transfer based on this partial run.
"""
    s.OUT["report"].write_text(report, encoding="utf-8")

    metadata = {
        "step": "Step18B partial Teknofest Saglik-style tree transfer probe",
        "partial": True,
        "created_at_epoch": time.time(),
        "completed_runs": completed,
        "not_run": not_run,
        "seeds": s.SEEDS,
        "configs": s.CONFIGS,
        "aggregations": s.AGGREGATIONS,
        "test_is_reporting_only": True,
        "threshold_selection": "OOF-only",
        "old_fitted_model_bundle_loaded": False,
        "leakage_fail": leak_fail,
        "runtime_recorded_seconds": float(runtime_df.runtime_seconds.sum()) if len(runtime_df) else 0.0,
        "outputs": {k: str(v) for k, v in s.OUT.items()} | {"status": str(status_path)},
    }
    s.OUT["metadata"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("DONE partial Step18B")
    print(top_by_f1[["aggregation", "config_name", "completed_seed_count", "test_f1_mean", "test_fp_mean", "test_fn_mean", "oof_f1_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
