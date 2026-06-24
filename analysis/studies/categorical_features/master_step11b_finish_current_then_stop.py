"""Finish only the interrupted Step 11B current run: seed 7 / EK shift config."""
from __future__ import annotations

import time
import pandas as pd

from master_step3b_repeated_split_catboost import find_input, leakage_audit, make_split
from master_step11b_column_filtering_ablation import (
    ROOT, PARTIAL_ROWS, PARTIAL_RUNTIME, append_csv, filter_plan, prepare_xy,
    stats_and_duplicates, step10_candidates, train_config,
)


def main():
    seed = 7
    config = "drop_EK_consistent_shift_candidates"
    if PARTIAL_RUNTIME.exists():
        done = pd.read_csv(PARTIAL_RUNTIME)
        if ((done.seed == seed) & (done.config_name == config) & (done.status == "completed")).any():
            print("already completed")
            return
    full = pd.read_csv(find_input())
    full["Label"] = full.Label.astype(int)
    raw = [c for c in full.columns if c not in ["Variant_ID", "Label"]]
    al = [c for c in raw if c.startswith("AL_") and pd.api.types.is_numeric_dtype(full[c])]
    train, test = make_split(full, seed)
    base_x, _ = prepare_xy(train, test, raw, al)
    stat_rows, dup_drop, dup_keep = stats_and_duplicates(base_x)
    candidates, _ = step10_candidates()
    dropped, _ = filter_plan(config, base_x, stat_rows, dup_drop, dup_keep, candidates)
    print(f"finishing only seed={seed} config={config} drop={len(dropped)}", flush=True)
    t0 = time.perf_counter()
    rows, features, original, final, oof_seconds, final_seconds, _ = train_config(seed, config, train, test, raw, al, dropped)
    append_csv(PARTIAL_ROWS, rows)
    runtime = {
        "seed": seed,
        "config_name": config,
        "status": "completed",
        "prediction_source": "RETRAINED_FROZEN_FACTORY_SINGLE_FINISH",
        "runtime_seconds": time.perf_counter() - t0,
        "oof_seconds": oof_seconds,
        "final_fit_seconds": final_seconds,
        "dropped_feature_count": len(dropped),
    }
    append_csv(PARTIAL_RUNTIME, pd.DataFrame([runtime]))
    audit, _ = leakage_audit(full, train, test, features, seed, int(full.duplicated().sum()))
    print("completed single current config", runtime, audit["status"], flush=True)


if __name__ == "__main__":
    main()
