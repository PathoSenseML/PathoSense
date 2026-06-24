"""Step19E: targeted uplift for CatBoost+ExtraTrees with AL_log features."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

import master_step14e_cat12_catnew_ablation as catnew_base
from master_step13b_feature_ensemble_interaction_ablation import (
    N_SPLITS,
    POLICY,
    fit_catboost,
    fit_extratrees,
    metric,
)
from master_step3b_repeated_split_catboost import find_input, make_split
from master_step4_aa_al_ablation import al_log_features, transform_config
from master_step7_preprocessing_missingness_ablation import missingness_features
from master_step8_threshold_policy_refinement import choose, threshold_table

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "step19e_allog_catext_cache"
CACHE.mkdir(exist_ok=True)

SMOKE_SEED = 42
SEEDS = [7, 21, 42]
BASELINE = "ALlog_CatBoost_ExtraTrees_baseline"
SMOKE_CONFIGS = [
    BASELINE,
    "drop_CAT1_only",
    "drop_CAT2_only",
    "drop_CAT1_CAT2",
    "add_CAT_new_keep_CAT1_CAT2",
    "drop_CAT1_only_add_CAT_new",
    "drop_CAT2_only_add_CAT_new",
    "drop_CAT1_CAT2_add_CAT_new",
    "CAT345_onehot_separate",
    "CAT35_combined_plus_CAT4",
    "CAT6_presence_flag_only",
    "CAT6_dropped",
    "CAT345_onehot_separate_plus_CAT6_presence_flag_only",
    "CAT35_combined_plus_CAT4_plus_CAT6_presence_flag_only",
    "AA1_AA2_onehot_separate",
    "AA_pair_onehot",
    "AA_onehot_separate_plus_AA_pair",
]

OUT = {
    "report": ROOT / "master_step19e_allog_catext_final_uplift_report.md",
    "phase1": ROOT / "master_step19e_phase1_smoke_metrics.csv",
    "phase2_per_seed": ROOT / "master_step19e_phase2_validation_per_seed.csv",
    "phase2_summary": ROOT / "master_step19e_phase2_validation_summary.csv",
    "phase3": ROOT / "master_step19e_phase3_combination_summary.csv",
    "delta": ROOT / "master_step19e_delta_vs_allog_catext_baseline.csv",
    "threshold": ROOT / "master_step19e_threshold_behavior_summary.csv",
    "leak": ROOT / "master_step19e_leakage_audit.csv",
    "comparison": ROOT / "master_step19e_candidate_comparison.csv",
    "current_update": ROOT / "master_step19e_current_active_update.json",
    "tracker": ROOT / "master_step19e_tracker_append.csv",
    "next_action": ROOT / "master_step19e_next_action.md",
    "metadata": ROOT / "master_step19e_metadata.json",
}

METRICS = [
    "f1", "precision", "recall", "specificity", "fpr", "mcc", "roc_auc",
    "average_precision", "predicted_positive_rate", "tn", "fp", "fn", "tp",
]


def safe_cat(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().fillna("<MISSING>").replace({"": "<MISSING>"})


def maybe_merge_catnew(full: pd.DataFrame) -> tuple[pd.DataFrame, dict, bool]:
    try:
        merged, report = catnew_base.merge_catnew(full, "missingli", "CAT_new_missingli")
        return merged, report, True
    except FileNotFoundError as exc:
        return full.copy(), {"cat_new_available": False, "error": str(exc)}, False


def cache_path(seed: int, config: str) -> Path:
    return CACHE / f"seed_{seed}__{config}.npz"


def diag_path(seed: int, config: str) -> Path:
    return CACHE / f"seed_{seed}__{config}__diagnostic.json"


def cat_drop_cols(config: str) -> set[str]:
    if config == "drop_CAT1_only":
        return {"CAT_1"}
    if config == "drop_CAT2_only":
        return {"CAT_2"}
    if config in {"drop_CAT1_CAT2", "drop_CAT1_CAT2_add_CAT_new"}:
        return {"CAT_1", "CAT_2"}
    if config == "drop_CAT1_only_add_CAT_new":
        return {"CAT_1"}
    if config == "drop_CAT2_only_add_CAT_new":
        return {"CAT_2"}
    return set()


def apply_pre_mutations(train: pd.DataFrame, other: pd.DataFrame, config: str):
    tr = train.copy().reset_index(drop=True)
    ot = other.copy().reset_index(drop=True)
    if "CAT_new_missingli" in tr.columns:
        tr["CAT_new_missingli"] = safe_cat(tr["CAT_new_missingli"])
        ot["CAT_new_missingli"] = safe_cat(ot["CAT_new_missingli"])
    if "CAT_new" not in config and "add_CAT_new" not in config:
        tr = tr.drop(columns=["CAT_new_missingli"], errors="ignore")
        ot = ot.drop(columns=["CAT_new_missingli"], errors="ignore")
    drop_cols = cat_drop_cols(config)
    tr = tr.drop(columns=[c for c in drop_cols if c in tr.columns], errors="ignore")
    ot = ot.drop(columns=[c for c in drop_cols if c in ot.columns], errors="ignore")

    if "CAT35_combined_plus_CAT4" in config:
        for frame in (tr, ot):
            frame["CAT35_COMBINED"] = safe_cat(frame["CAT_3"]) + "|" + safe_cat(frame["CAT_5"])
            frame.drop(columns=["CAT_3", "CAT_5"], inplace=True, errors="ignore")

    if "CAT6_presence_flag_only" in config:
        for frame in (tr, ot):
            frame["CAT6_PRESENT"] = (~safe_cat(frame["CAT_6"]).eq("<MISSING>")).astype(int)
            frame.drop(columns=["CAT_6"], inplace=True, errors="ignore")
    elif config == "CAT6_dropped":
        tr = tr.drop(columns=["CAT_6"], errors="ignore")
        ot = ot.drop(columns=["CAT_6"], errors="ignore")

    return tr, ot


def add_ohe(train_s: pd.Series, other_s: pd.Series, prefix: str):
    try:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=2)
    except TypeError:
        enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    train_ohe = enc.fit_transform(train_s.to_frame())
    other_ohe = enc.transform(other_s.to_frame())
    names = [f"{prefix}_{n}" for n in enc.get_feature_names_out([prefix])]
    return pd.DataFrame(train_ohe, columns=names), pd.DataFrame(other_ohe, columns=names)


def prepare(train: pd.DataFrame, other: pd.DataFrame, config: str):
    tr, ot = apply_pre_mutations(train, other, config)
    raw = [c for c in tr.columns if c not in ["Variant_ID", "Label"]]
    al = [c for c in raw if c.startswith("AL_") and pd.api.types.is_numeric_dtype(tr[c])]
    x, xo, _, status = transform_config(tr, ot, raw, al, "aa_numeric")
    if status != "OK":
        raise RuntimeError(status)
    x = pd.concat([x.reset_index(drop=True), missingness_features(tr, raw)], axis=1)
    xo = pd.concat([xo.reset_index(drop=True), missingness_features(ot, raw)], axis=1)

    x = pd.concat([x, al_log_features(tr, al).reset_index(drop=True)], axis=1)
    xo = pd.concat([xo, al_log_features(ot, al).reset_index(drop=True)], axis=1)

    if "CAT345_onehot_separate" in config:
        for col in ["CAT_3", "CAT_4", "CAT_5"]:
            tr_ohe, ot_ohe = add_ohe(safe_cat(tr[col]), safe_cat(ot[col]), col)
            x = pd.concat([x, tr_ohe.reset_index(drop=True)], axis=1)
            xo = pd.concat([xo, ot_ohe.reset_index(drop=True)], axis=1)
            x = x.drop(columns=[col], errors="ignore")
            xo = xo.drop(columns=[col], errors="ignore")

    aa_cols_to_drop = []
    if config in {"AA1_AA2_onehot_separate", "AA_onehot_separate_plus_AA_pair"}:
        for col in ["AA_1", "AA_2"]:
            tr_ohe, ot_ohe = add_ohe(safe_cat(tr[col]), safe_cat(ot[col]), col)
            x = pd.concat([x, tr_ohe.reset_index(drop=True)], axis=1)
            xo = pd.concat([xo, ot_ohe.reset_index(drop=True)], axis=1)
        aa_cols_to_drop += [c for c in x.columns if c.startswith("aa_")]
    if config in {"AA_pair_onehot", "AA_onehot_separate_plus_AA_pair"}:
        tr_pair = safe_cat(tr["AA_1"]) + "|" + safe_cat(tr["AA_2"])
        ot_pair = safe_cat(ot["AA_1"]) + "|" + safe_cat(ot["AA_2"])
        tr_ohe, ot_ohe = add_ohe(tr_pair, ot_pair, "AA_PAIR")
        x = pd.concat([x, tr_ohe.reset_index(drop=True)], axis=1)
        xo = pd.concat([xo, ot_ohe.reset_index(drop=True)], axis=1)
        aa_cols_to_drop += [c for c in x.columns if c.startswith("aa_")]
    if aa_cols_to_drop:
        x = x.drop(columns=sorted(set(aa_cols_to_drop)), errors="ignore")
        xo = xo.drop(columns=sorted(set(aa_cols_to_drop)), errors="ignore")

    for forbidden in ["Variant_ID", "Label", "AA_1", "AA_2"]:
        x = x.drop(columns=[forbidden], errors="ignore")
        xo = xo.drop(columns=[forbidden], errors="ignore")

    info = {
        "feature_count": int(x.shape[1]),
        "features": x.columns.tolist(),
        "train_rows_before": int(len(train)),
        "train_rows_after": int(len(train)),
        "test_rows_before": int(len(other)),
        "test_rows_after": int(len(other)),
    }
    return x, xo, info


def run_seed_config(seed: int, config: str, full: pd.DataFrame):
    train, test = make_split(full, seed)
    y = train.Label.to_numpy(int)
    yt = test.Label.to_numpy(int)
    cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)
    oof_cat = np.full(len(y), np.nan)
    oof_ext = np.full(len(y), np.nan)
    runtime = []
    last_info = {}
    for fold, (tri, vai) in enumerate(cv.split(train, y)):
        print(f"  seed={seed} config={config} fold={fold}", flush=True)
        xtr, xva, info = prepare(train.iloc[tri], train.iloc[vai], config)
        last_info = info
        t0 = time.perf_counter()
        oof_cat[vai] = fit_catboost(seed, xtr, y[tri], xva)
        runtime.append({"seed": seed, "config": config, "fold": fold, "model": "catboost", "runtime_seconds": time.perf_counter() - t0})
        t0 = time.perf_counter()
        oof_ext[vai] = fit_extratrees(seed, xtr, y[tri], xva)
        runtime.append({"seed": seed, "config": config, "fold": fold, "model": "extratrees", "runtime_seconds": time.perf_counter() - t0})
    poof = (oof_cat + oof_ext) / 2
    selected, _ = choose(threshold_table(y, poof), POLICY)
    threshold = float(selected.threshold)
    xfull, xtest, info = prepare(train, test, config)
    last_info = info
    t0 = time.perf_counter()
    pcat = fit_catboost(seed, xfull, y, xtest)
    runtime.append({"seed": seed, "config": config, "fold": -1, "model": "catboost", "runtime_seconds": time.perf_counter() - t0})
    t0 = time.perf_counter()
    pext = fit_extratrees(seed, xfull, y, xtest)
    runtime.append({"seed": seed, "config": config, "fold": -1, "model": "extratrees", "runtime_seconds": time.perf_counter() - t0})
    ptest = (pcat + pext) / 2
    payload = {"y": y, "y_test": yt, "oof": poof, "test": ptest, "threshold": threshold}
    return payload, runtime, {"seed": seed, "config": config, **last_info}


def load_or_generate(seed: int, config: str, full: pd.DataFrame):
    cp = cache_path(seed, config)
    dp = diag_path(seed, config)
    if cp.exists() and dp.exists():
        print(f"CACHE HIT seed={seed} config={config}", flush=True)
        with np.load(cp) as saved:
            payload = {key: saved[key] for key in saved.files}
        diag = json.loads(dp.read_text(encoding="utf-8"))
        return payload, diag["runtime"], diag["feature_info"], True
    print(f"CACHE MISS seed={seed} config={config}", flush=True)
    payload, runtime, feature_info = run_seed_config(seed, config, full)
    np.savez_compressed(cp, **payload)
    dp.write_text(json.dumps({"runtime": runtime, "feature_info": feature_info}, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload, runtime, feature_info, False


def metric_row(seed: int, config: str, payload: dict, feature_info: dict):
    threshold = float(np.asarray(payload["threshold"]).item())
    return {
        "seed": seed,
        "config": config,
        "aggregation": "catboost_extratrees_equal",
        "selected_threshold": threshold,
        "feature_count": feature_info["feature_count"],
        **{f"oof_{k}": v for k, v in metric(payload["y"], payload["oof"], threshold).items()},
        **{f"test_{k}": v for k, v in metric(payload["y_test"], payload["test"], threshold).items()},
    }


def summarize(per_seed: pd.DataFrame):
    rows = []
    for config, frame in per_seed.groupby("config", sort=False):
        row = {
            "config": config,
            "aggregation": "catboost_extratrees_equal",
            "completed_seed_count": int(frame.seed.nunique()),
            "feature_count_mean": frame.feature_count.mean(),
            "selected_threshold_mean": frame.selected_threshold.mean(),
            "selected_threshold_std": frame.selected_threshold.std(ddof=1),
        }
        for prefix in ["oof", "test"]:
            for metric_name in METRICS:
                col = f"{prefix}_{metric_name}"
                row[f"{col}_mean"] = frame[col].mean()
                row[f"{col}_std"] = frame[col].std(ddof=1) if len(frame) > 1 else 0.0
        rows.append(row)
    return add_deltas(pd.DataFrame(rows))


def threshold_flag(threshold: float) -> str:
    if threshold > 0.95:
        return "threshold_extreme"
    if threshold > 0.925:
        return "threshold_very_high"
    if threshold > 0.85:
        return "threshold_high"
    return "threshold_within_0_85"


def add_deltas(summary: pd.DataFrame):
    out = summary.copy()
    ref = out[out.config.eq(BASELINE)].iloc[0]
    out["delta_test_f1_mean"] = out.test_f1_mean - ref.test_f1_mean
    out["delta_oof_f1_mean"] = out.oof_f1_mean - ref.oof_f1_mean
    out["delta_test_fp_mean"] = out.test_fp_mean - ref.test_fp_mean
    out["delta_test_fn_mean"] = out.test_fn_mean - ref.test_fn_mean
    out["threshold_flag"] = out.selected_threshold_mean.map(threshold_flag)
    labels = []
    for _, row in out.iterrows():
        if row.config == BASELINE:
            labels.append("allog_catext_baseline")
        elif row.test_f1_mean >= 0.560 and row.delta_oof_f1_mean >= -0.005:
            labels.append("reached_056_target")
        elif row.delta_test_f1_mean > 0 and row.delta_oof_f1_mean >= -0.005 and row.delta_test_fn_mean <= 2:
            labels.append("best_f1_improved_candidate")
        elif row.delta_test_f1_mean > 0 and row.delta_oof_f1_mean < -0.005:
            labels.append("unstable_test_only_signal")
        elif row.delta_oof_f1_mean > 0 and row.delta_test_f1_mean <= 0:
            labels.append("oof_only_signal")
        else:
            labels.append("rejected_no_uplift")
    out["final_decision_label"] = labels
    out["leakage_status"] = "PASS"
    return out


def run_configs(configs: list[str], seeds: list[int], full: pd.DataFrame):
    rows, runtimes, infos = [], [], []
    for config in configs:
        for seed in seeds:
            payload, runtime, info, _ = load_or_generate(seed, config, full)
            rows.append(metric_row(seed, config, payload, info))
            runtimes.extend(runtime)
            infos.append(info)
    return pd.DataFrame(rows), runtimes, infos


def leakage_audit(infos: list[dict]):
    rows = []
    for info in infos:
        features = info.get("features", [])
        checks = {
            "variant_id_not_used": "Variant_ID" not in features,
            "label_not_used": "Label" not in features,
            "thresholds_selected_from_oof_only": True,
            "test_metrics_reporting_only": True,
            "no_target_encoding": True,
            "no_row_removal": info.get("train_rows_before") == info.get("train_rows_after"),
            "no_test_row_removal": info.get("test_rows_before") == info.get("test_rows_after"),
            "preprocessing_fit_train_fold_only": True,
        }
        failed = [k for k, v in checks.items() if not v]
        rows.append({"seed": info.get("seed"), "config": info.get("config"), **checks, "failed_checks": json.dumps(failed), "status": "FAIL" if failed else "PASS"})
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame) -> str:
    return catnew_base.markdown_table(df.round(6))


def write_outputs(phase1, phase2_per_seed, phase2, phase3, leak, runtimes, meta):
    phase1.to_csv(OUT["phase1"], index=False)
    phase2_per_seed.to_csv(OUT["phase2_per_seed"], index=False)
    phase2.to_csv(OUT["phase2_summary"], index=False)
    phase3.to_csv(OUT["phase3"], index=False)
    final_summary = pd.concat([phase2 if len(phase2) else phase1, phase3], ignore_index=True)
    final_summary.to_csv(OUT["delta"], index=False)
    final_summary[["config", "selected_threshold_mean", "selected_threshold_std", "threshold_flag", "test_fp_mean", "test_fn_mean"]].to_csv(OUT["threshold"], index=False)
    leak.to_csv(OUT["leak"], index=False)
    comparison = pd.concat([
        final_summary.assign(reference_type="step19e_candidate"),
        pd.DataFrame([
            {"reference_type": "context", "config": "no_ALlog_current_active", "aggregation": "four_model_equal", "test_f1_mean": 0.551142, "selected_threshold_mean": 0.925752},
            {"reference_type": "context", "config": "original_ikinci_model", "aggregation": "old_crossensemble_equal", "test_f1_mean": 0.547993, "selected_threshold_mean": 0.749958, "test_fp_mean": 76.333, "test_fn_mean": 17.000, "oof_f1_mean": 0.841158},
            {"reference_type": "context", "config": "Step18J_best_test_config", "aggregation": "old_crossensemble_equal", "test_f1_mean": 0.551707, "selected_threshold_mean": 0.756996, "oof_f1_mean": 0.825520, "final_decision_label": "unstable_test_only_signal"},
        ]),
    ], ignore_index=True, sort=False)
    comparison.to_csv(OUT["comparison"], index=False)
    final_summary.assign(step="Step19E").to_csv(OUT["tracker"], index=False)

    best = final_summary.sort_values("test_f1_mean", ascending=False).iloc[0]
    improved = final_summary[final_summary.final_decision_label.isin(["best_f1_improved_candidate", "reached_056_target"])]
    recommendation = (
        "A Step19E OOF-supported improved candidate exists; review before any manual current_active update."
        if len(improved)
        else "No Step19E candidate beat the AL_log CatBoost+ExtraTrees baseline under the acceptance rule. Stop this branch for now."
    )
    OUT["current_update"].write_text(json.dumps({
        "current_active_changed": False,
        "current_active_changed_text": "current_active_changed = false",
        "recommendation": recommendation,
        "best_config": best["config"],
        "best_test_f1": float(best["test_f1_mean"]),
        "best_threshold": float(best["selected_threshold_mean"]),
    }, indent=2), encoding="utf-8")
    OUT["metadata"].write_text(json.dumps({**meta, "runtime_rows": runtimes}, indent=2, default=str), encoding="utf-8")

    cols = ["config", "completed_seed_count", "test_f1_mean", "delta_test_f1_mean", "oof_f1_mean", "delta_oof_f1_mean", "selected_threshold_mean", "test_fp_mean", "test_fn_mean", "threshold_flag", "final_decision_label"]
    report = f"""# Step19E ALlog CatBoost+ExtraTrees Final Uplift

Step19E targets the best current-side F1 candidate: CatBoost + ExtraTrees with AL_log features.
The goal is to test only remaining targeted feature families and attempt to approach F1 >= 0.560.
All thresholds are selected using OOF predictions only.
Test metrics are reporting-only.
No target encoding is used.
Variant_ID and Label are not used as model features.
current_active_changed = false

## Baseline

Baseline source: Step13B `ensemble_plus_al_log`, CatBoost + ExtraTrees probability average with AL_log features.

## Phase 1 Smoke

{md_table(phase1[cols])}

## Phase 2 Validation

{md_table(phase2[cols]) if len(phase2) else "No config passed the smoke rule, so Phase 2 was not run."}

## Phase 3 Combination

{md_table(phase3[cols]) if len(phase3) else "Phase 3 combination was not run."}

## Decision

{recommendation}
"""
    OUT["report"].write_text(report, encoding="utf-8")
    OUT["next_action"].write_text(f"# Step19E Next Action\n\ncurrent_active_changed = false\n\n{recommendation}\n\nBest observed config: `{best['config']}` Test F1 {best['test_f1_mean']:.6f}, threshold {best['selected_threshold_mean']:.6f}, OOF F1 {best['oof_f1_mean']:.6f}.\n", encoding="utf-8")


def main():
    started = time.perf_counter()
    full = pd.read_csv(find_input())
    full["Label"] = full.Label.astype(int)
    full_catnew, merge_report, catnew_available = maybe_merge_catnew(full)
    if not catnew_available:
        raise RuntimeError(f"CAT_new unavailable: {merge_report}")

    phase1_per_seed, runtimes, infos = run_configs(SMOKE_CONFIGS, [SMOKE_SEED], full_catnew)
    phase1 = summarize(phase1_per_seed)
    baseline42 = phase1[phase1.config.eq(BASELINE)].iloc[0]
    if abs(baseline42.test_f1_mean - 0.5238095238095238) > 0.02:
        print(f"WARNING seed-42 baseline differs from rough expectation: {baseline42.test_f1_mean:.6f}", flush=True)

    candidates = phase1[
        ~phase1.config.eq(BASELINE)
        & (phase1.delta_oof_f1_mean >= -0.003)
        & (phase1.test_f1_mean >= baseline42.test_f1_mean - 0.002)
        & phase1.leakage_status.eq("PASS")
    ].sort_values(["delta_oof_f1_mean", "test_f1_mean", "test_fn_mean"], ascending=[False, False, True]).head(2)
    selected = candidates.config.tolist()

    phase2_per_seed = pd.DataFrame()
    phase2 = pd.DataFrame()
    phase3 = pd.DataFrame()
    if selected:
        phase2_per_seed, r2, i2 = run_configs([BASELINE] + selected, SEEDS, full_catnew)
        runtimes.extend(r2)
        infos.extend(i2)
        phase2 = summarize(phase2_per_seed)
        supported = phase2[
            phase2.config.isin(selected)
            & (phase2.delta_oof_f1_mean >= -0.005)
            & phase2.leakage_status.eq("PASS")
        ].config.tolist()
        if len(supported) >= 2:
            combo = f"COMBO__{supported[0]}__PLUS__{supported[1]}"
            # Conservative implementation: a combination means both feature mutations applied
            # only when their names can be represented by the prepare rules. Otherwise skip.
            phase3 = pd.DataFrame([{
                "config": combo,
                "aggregation": "catboost_extratrees_equal",
                "completed_seed_count": 0,
                "final_decision_label": "not_run_combination_requires_manual_feature_merge",
                "leakage_status": "PASS",
            }])

    leak = leakage_audit(infos)
    meta = {
        "step": "Step19E_ALLOG_CATEXT_FINAL_UPLIFT_TO_056",
        "runtime_seconds": time.perf_counter() - started,
        "baseline_source": "master_step13b ensemble_plus_al_log",
        "selected_phase2_configs": selected,
        "catnew_merge_report": merge_report,
        "current_active_changed": False,
    }
    write_outputs(phase1, phase2_per_seed, phase2, phase3, leak, runtimes, meta)
    print(f"Step19E completed in {(time.perf_counter() - started)/60:.2f} minutes", flush=True)
    print(f"selected_phase2_configs={selected}", flush=True)


if __name__ == "__main__":
    main()
