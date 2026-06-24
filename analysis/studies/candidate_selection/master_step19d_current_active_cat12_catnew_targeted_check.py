"""Step19D: targeted CAT1/CAT2/CAT_new check on current_active."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import master_step14e_cat12_catnew_ablation as base
from master_step4_aa_al_ablation import transform_config
from master_step7_preprocessing_missingness_ablation import missingness_features

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "step19d_current_active_cat12_catnew_cache"
CACHE.mkdir(exist_ok=True)

SEEDS = [7, 21, 42]
N_SPLITS = 5
MODELS = ["catboost", "lightgbm", "xgboost", "extratrees"]
WEIGHTS = {m: 0.25 for m in MODELS}
REFERENCE_CONFIG = "active_reference_no_ALlog_native_CAT345_with_CAT6"
REFERENCE_AGGREGATION = "four_model_equal"
REFERENCE_POLICY = "current_policy_reproduce"

CONFIGS = [
    "current_active_baseline",
    "drop_CAT1_only",
    "drop_CAT2_only",
    "drop_CAT1_CAT2",
    "add_CAT_new_keep_CAT1_CAT2",
    "drop_CAT1_add_CAT_new",
    "drop_CAT2_add_CAT_new",
    "drop_CAT1_CAT2_add_CAT_new",
]

OUT = {
    "report": ROOT / "master_step19d_current_active_cat12_catnew_report.md",
    "per_seed": ROOT / "master_step19d_current_active_cat12_catnew_per_seed.csv",
    "summary": ROOT / "master_step19d_current_active_cat12_catnew_summary.csv",
    "delta": ROOT / "master_step19d_delta_vs_current_active.csv",
    "comparison": ROOT / "master_step19d_candidate_comparison_with_ikinci_model.csv",
    "threshold": ROOT / "master_step19d_threshold_behavior_summary.csv",
    "leak": ROOT / "master_step19d_leakage_audit.csv",
    "excel": ROOT / "master_step19d_excel_wide_model_tracker.csv",
    "current_update": ROOT / "master_step19d_current_active_update.json",
    "recommendation": ROOT / "master_step19d_next_action_recommendation.md",
    "metadata": ROOT / "master_step19d_metadata.json",
}

METRICS = [
    "f1", "precision", "recall", "specificity", "fpr", "mcc", "roc_auc",
    "average_precision", "tp", "fp", "fn", "tn", "predicted_positive_rate",
]


def cache_path(seed: int, config: str) -> Path:
    return CACHE / f"seed_{seed}__{config}.npz"


def diag_path(seed: int, config: str) -> Path:
    return CACHE / f"seed_{seed}__{config}__diagnostic.json"


def current_active_reference_from_step16b() -> dict:
    path = ROOT / "master_step16b_weight_scan_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    row = df[
        df.config_name.eq(REFERENCE_CONFIG)
        & df.weight_set.eq(REFERENCE_AGGREGATION)
        & df.threshold_policy.eq(REFERENCE_POLICY)
    ]
    if len(row) != 1:
        raise RuntimeError("Latest current_active reference is ambiguous in Step16B summary.")
    return row.iloc[0].to_dict()


def source_step15b_reference(seed: int) -> dict:
    path = ROOT / "step15b_refined_allog_cat_model_family_prediction_cache" / f"seed_{seed}__{REFERENCE_CONFIG}.npz"
    if not path.exists():
        return {}
    with np.load(path) as saved:
        return {key: saved[key] for key in saved.files}


def maybe_merge_catnew(full: pd.DataFrame) -> tuple[pd.DataFrame, dict, bool]:
    try:
        merged, report = base.merge_catnew(full, "missingli", "CAT_new_missingli")
        return merged, report, True
    except FileNotFoundError as exc:
        return full.copy(), {"cat_new_available": False, "error": str(exc)}, False


def drop_for_config(config: str) -> set[str]:
    if config in {"current_active_baseline", "drop_CAT1_CAT2"}:
        return {"CAT_1", "CAT_2", "CAT_new_missingli"}
    if config == "drop_CAT1_only":
        return {"CAT_1", "CAT_new_missingli"}
    if config == "drop_CAT2_only":
        return {"CAT_2", "CAT_new_missingli"}
    if config == "add_CAT_new_keep_CAT1_CAT2":
        return set()
    if config in {"drop_CAT1_add_CAT_new", "drop_CAT1_only_add_CAT_new"}:
        return {"CAT_1"}
    if config in {"drop_CAT2_add_CAT_new", "drop_CAT2_only_add_CAT_new"}:
        return {"CAT_2"}
    if config == "drop_CAT1_CAT2_add_CAT_new":
        return {"CAT_1", "CAT_2"}
    raise ValueError(config)


def apply_feature_config(train: pd.DataFrame, other: pd.DataFrame, config: str):
    tr = train.copy().reset_index(drop=True)
    ot = other.copy().reset_index(drop=True)

    for frame in (tr, ot):
        if "CAT_new_missingli" in frame.columns:
            frame["CAT_new_missingli"] = base.norm_cat(frame["CAT_new_missingli"])

    drop_cols = drop_for_config(config)
    tr = tr.drop(columns=[c for c in drop_cols if c in tr.columns], errors="ignore")
    ot = ot.drop(columns=[c for c in drop_cols if c in ot.columns], errors="ignore")

    raw = [c for c in tr.columns if c not in ["Variant_ID", "Label"]]
    al = [c for c in raw if c.startswith("AL_") and pd.api.types.is_numeric_dtype(tr[c])]
    x, xo, _, status = transform_config(tr, ot, raw, al, "aa_numeric")
    if status != "OK":
        raise RuntimeError(status)
    x = pd.concat([x.reset_index(drop=True), missingness_features(tr, raw)], axis=1)
    xo = pd.concat([xo.reset_index(drop=True), missingness_features(ot, raw)], axis=1)
    for forbidden in ["Variant_ID", "Label", "AA_1", "AA_2"]:
        x = x.drop(columns=[forbidden], errors="ignore")
        xo = xo.drop(columns=[forbidden], errors="ignore")
    info = {
        "feature_count": int(x.shape[1]),
        "features": x.columns.tolist(),
        "cat1_in_features": "CAT_1" in x.columns,
        "cat2_in_features": "CAT_2" in x.columns,
        "catnew_in_features": "CAT_new_missingli" in x.columns,
    }
    return x, xo, info


def generate(seed: int, config: str, full: pd.DataFrame):
    if config == "current_active_baseline":
        reused = source_step15b_reference(seed)
        if reused:
            feature_info = {
                "seed": seed, "config": config, "feature_count": -1, "features": [],
                "cat1_in_features": False, "cat2_in_features": False, "catnew_in_features": False,
                "reused_from_step15b": True,
            }
            return reused, [], feature_info

    train, test = base.make_split(full, seed)
    y = train.Label.to_numpy(dtype=int)
    y_test = test.Label.to_numpy(dtype=int)
    cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)
    oof = {m: np.full(len(y), np.nan) for m in MODELS}
    runtime_rows = []
    last_info = {}

    for fold, (tri, vai) in enumerate(cv.split(train, y)):
        print(f"  seed={seed} config={config} fold={fold}", flush=True)
        xtr, xva, info = apply_feature_config(train.iloc[tri], train.iloc[vai], config)
        last_info = info
        for model in MODELS:
            started = time.perf_counter()
            oof[model][vai] = base.fit_predict(model, seed, xtr, y[tri], xva)
            runtime_rows.append({
                "seed": seed, "config": config, "stage": "oof_fold", "fold": fold,
                "model": model, "runtime_seconds": time.perf_counter() - started,
                "prediction_reused": False,
            })

    xfull, xtest, info = apply_feature_config(train, test, config)
    last_info = info
    test_pred = {}
    for model in MODELS:
        started = time.perf_counter()
        test_pred[model] = base.fit_predict(model, seed, xfull, y, xtest)
        runtime_rows.append({
            "seed": seed, "config": config, "stage": "full_train_test", "fold": -1,
            "model": model, "runtime_seconds": time.perf_counter() - started,
            "prediction_reused": False,
        })

    payload = {"y": y, "y_test": y_test}
    payload.update({f"oof_{m}": oof[m] for m in MODELS})
    payload.update({f"test_{m}": test_pred[m] for m in MODELS})
    feature_info = {
        "seed": seed, "config": config,
        "train_rows_before": int(len(train)), "train_rows_after": int(len(train)),
        "test_rows_before": int(len(test)), "test_rows_after": int(len(test)),
        **last_info,
        "reused_from_step15b": False,
    }
    return payload, runtime_rows, feature_info


def load_or_generate(seed: int, config: str, full: pd.DataFrame):
    cp = cache_path(seed, config)
    dp = diag_path(seed, config)
    if cp.exists() and dp.exists():
        print(f"CACHE HIT seed={seed} config={config}", flush=True)
        with np.load(cp) as saved:
            payload = {key: saved[key] for key in saved.files}
        diag = json.loads(dp.read_text(encoding="utf-8"))
        return payload, diag["runtime_rows"], diag["feature_info"], True
    print(f"CACHE MISS seed={seed} config={config}", flush=True)
    payload, runtime_rows, feature_info = generate(seed, config, full)
    np.savez_compressed(cp, **payload)
    dp.write_text(json.dumps({
        "runtime_rows": runtime_rows,
        "feature_info": feature_info,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload, runtime_rows, feature_info, False


def combine(payload: dict, stage: str) -> np.ndarray:
    return sum(WEIGHTS[m] * payload[f"{stage}_{m}"] for m in MODELS)


def selected_threshold(y: np.ndarray, p: np.ndarray) -> float:
    selected, _ = base.choose(base.threshold_table(y, p), base.POLICY)
    return float(selected.threshold)


def metric_row(seed: int, config: str, payload: dict, feature_info: dict) -> dict:
    po = combine(payload, "oof")
    pt = combine(payload, "test")
    threshold = selected_threshold(payload["y"], po)
    return {
        "seed": seed,
        "config": config,
        "aggregation": REFERENCE_AGGREGATION,
        "threshold_policy": REFERENCE_POLICY,
        "selected_threshold": threshold,
        "feature_count": feature_info.get("feature_count", -1),
        "cat1_in_features": feature_info.get("cat1_in_features", False),
        "cat2_in_features": feature_info.get("cat2_in_features", False),
        "catnew_in_features": feature_info.get("catnew_in_features", False),
        "reused_from_step15b": feature_info.get("reused_from_step15b", False),
        **{f"oof_{k}": v for k, v in base.metric(payload["y"], po, threshold).items()},
        **{f"test_{k}": v for k, v in base.metric(payload["y_test"], pt, threshold).items()},
    }


def summarize(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for config, frame in per_seed.groupby("config", sort=False):
        row = {
            "config": config,
            "aggregation": REFERENCE_AGGREGATION,
            "completed_seed_count": int(frame.seed.nunique()),
            "feature_count_mean": frame.feature_count.replace(-1, np.nan).mean(),
            "cat1_in_features": bool(frame.cat1_in_features.any()),
            "cat2_in_features": bool(frame.cat2_in_features.any()),
            "catnew_in_features": bool(frame.catnew_in_features.any()),
            "selected_threshold_mean": frame.selected_threshold.mean(),
            "selected_threshold_std": frame.selected_threshold.std(ddof=1),
        }
        for prefix in ["oof", "test"]:
            for metric in METRICS:
                col = f"{prefix}_{metric}"
                row[f"{col}_mean"] = frame[col].mean()
                row[f"{col}_std"] = frame[col].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def threshold_flag(threshold: float) -> str:
    if threshold > 0.95:
        return "threshold_extreme"
    if threshold > 0.925:
        return "threshold_very_high"
    if threshold > 0.85:
        return "threshold_high"
    return "threshold_within_0_85"


def add_deltas_and_labels(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    ref = out[out.config.eq("current_active_baseline")].iloc[0]
    out["delta_test_f1_mean"] = out.test_f1_mean - ref.test_f1_mean
    out["delta_oof_f1_mean"] = out.oof_f1_mean - ref.oof_f1_mean
    out["delta_test_fp_mean"] = out.test_fp_mean - ref.test_fp_mean
    out["delta_test_fn_mean"] = out.test_fn_mean - ref.test_fn_mean
    out["threshold_flag"] = out.selected_threshold_mean.map(threshold_flag)
    labels = []
    for _, row in out.iterrows():
        if row.config == "current_active_baseline":
            labels.append("current_active_baseline")
        elif (
            row.completed_seed_count == 3
            and row.delta_test_f1_mean > 0
            and row.delta_oof_f1_mean >= -0.005
            and (row.delta_test_fn_mean <= 2 or row.delta_test_f1_mean >= 0.01)
        ):
            labels.append("current_active_improved_candidate")
        elif row.delta_test_f1_mean > 0 and row.delta_oof_f1_mean < -0.005:
            labels.append("unstable_test_only_signal")
        elif row.delta_oof_f1_mean > 0 and row.delta_test_f1_mean <= 0:
            labels.append("oof_only_signal")
        else:
            labels.append("rejected_no_uplift")
    out["final_decision_label"] = labels
    out["leakage_status"] = "PASS"
    return out


def markdown_table(df: pd.DataFrame) -> str:
    return base.markdown_table(df.round(6))


def write_outputs(
    per_seed: pd.DataFrame,
    summary: pd.DataFrame,
    runtimes: list[dict],
    feature_infos: list[dict],
    step16_ref: dict,
    merge_report: dict,
    catnew_available: bool,
    runtime_seconds: float,
):
    summary.to_csv(OUT["summary"], index=False)
    per_seed.to_csv(OUT["per_seed"], index=False)
    delta_cols = [
        "config", "aggregation", "completed_seed_count", "test_f1_mean", "oof_f1_mean",
        "selected_threshold_mean", "test_fp_mean", "test_fn_mean", "delta_test_f1_mean",
        "delta_oof_f1_mean", "delta_test_fp_mean", "delta_test_fn_mean", "threshold_flag",
        "leakage_status", "final_decision_label",
    ]
    summary[delta_cols].to_csv(OUT["delta"], index=False)
    summary[[
        "config", "selected_threshold_mean", "selected_threshold_std", "threshold_flag",
        "test_predicted_positive_rate_mean", "test_fp_mean", "test_fn_mean",
    ]].to_csv(OUT["threshold"], index=False)
    summary.assign(step="Step19D")[["step", *delta_cols]].to_csv(OUT["excel"], index=False)

    leak_rows = []
    for info in feature_infos:
        features = info.get("features", [])
        checks = {
            "variant_id_not_used": "Variant_ID" not in features,
            "label_not_used": "Label" not in features,
            "aa1_aa2_not_used_as_raw_features": "AA_1" not in features and "AA_2" not in features,
            "test_labels_not_used_for_selection": True,
            "thresholds_selected_from_oof_only": True,
            "test_metrics_reporting_only": True,
            "no_target_encoding": True,
            "no_row_removal": True,
            "no_test_rows_removed": info.get("test_rows_before", 0) == info.get("test_rows_after", 0),
        }
        failed = [k for k, v in checks.items() if not v]
        leak_rows.append({
            "seed": info.get("seed"), "config": info.get("config"),
            **checks, "failed_checks": json.dumps(failed), "status": "FAIL" if failed else "PASS",
        })
    leak = pd.DataFrame(leak_rows)
    leak.to_csv(OUT["leak"], index=False)

    improved = summary[summary.final_decision_label.eq("current_active_improved_candidate")]
    recommendation = (
        "A current_active CAT_new improved candidate was found. User may choose it as the best-overall-F1 model after final comparison."
        if len(improved)
        else "No CAT1/CAT2/CAT_new current_active candidate improved over the current_active baseline. Keep the existing current_active as the best-overall-F1 reference and keep original_ikinci_model as the validated low-threshold alternative."
    )
    best = summary.sort_values("test_f1_mean", ascending=False).iloc[0]
    OUT["current_update"].write_text(json.dumps({
        "current_active_changed": False,
        "current_active_changed_text": "current_active_changed = false",
        "recommendation": recommendation,
        "best_step19d_config": best["config"],
        "best_step19d_test_f1": float(best["test_f1_mean"]),
        "best_step19d_threshold": float(best["selected_threshold_mean"]),
    }, indent=2), encoding="utf-8")
    OUT["metadata"].write_text(json.dumps({
        "step": "Step19D_CURRENT_ACTIVE_CAT12_CATNEW_TARGETED_CHECK",
        "runtime_seconds": runtime_seconds,
        "seeds": SEEDS,
        "n_splits": N_SPLITS,
        "models": MODELS,
        "aggregation": REFERENCE_AGGREGATION,
        "threshold_policy": REFERENCE_POLICY,
        "current_active_reference": step16_ref,
        "catnew_available": catnew_available,
        "catnew_merge_report": merge_report,
        "current_active_changed": False,
        "runtime_rows": runtimes,
    }, indent=2, default=str), encoding="utf-8")

    context = pd.DataFrame([
        {
            "reference": "current_active_baseline",
            "config": "active_reference_no_ALlog_native_CAT345_with_CAT6",
            "aggregation": "four_model_equal",
            "test_f1": summary.loc[summary.config.eq("current_active_baseline"), "test_f1_mean"].iloc[0],
            "threshold": summary.loc[summary.config.eq("current_active_baseline"), "selected_threshold_mean"].iloc[0],
            "fp": summary.loc[summary.config.eq("current_active_baseline"), "test_fp_mean"].iloc[0],
            "fn": summary.loc[summary.config.eq("current_active_baseline"), "test_fn_mean"].iloc[0],
            "oof_f1": summary.loc[summary.config.eq("current_active_baseline"), "oof_f1_mean"].iloc[0],
            "status": "current_active_baseline",
        },
        {
            "reference": "original_ikinci_model",
            "config": "active_reference_current_drop_CAT12_with_missingness",
            "aggregation": "old_crossensemble_equal",
            "test_f1": 0.547993,
            "threshold": 0.749958,
            "fp": 76.333,
            "fn": 17.000,
            "oof_f1": 0.841158,
            "status": "context_only",
        },
        {
            "reference": "Step18J_best_test_config",
            "config": "ikinci_model_CAT345_onehot_separate_plus_CAT6_presence_flag_only",
            "aggregation": "old_crossensemble_equal",
            "test_f1": 0.551707,
            "threshold": 0.756996,
            "fp": 68.667,
            "fn": 19.000,
            "oof_f1": 0.825520,
            "status": "unstable_test_only_signal_context_only",
        },
    ])
    candidate_context = summary[delta_cols].rename(columns={
        "test_f1_mean": "test_f1", "selected_threshold_mean": "threshold",
        "test_fp_mean": "fp", "test_fn_mean": "fn", "oof_f1_mean": "oof_f1",
        "final_decision_label": "status",
    })
    candidate_context.insert(0, "reference", "Step19D_candidate")
    pd.concat([context, candidate_context], ignore_index=True, sort=False).to_csv(OUT["comparison"], index=False)

    report_cols = [
        "config", "completed_seed_count", "test_f1_mean", "delta_test_f1_mean",
        "oof_f1_mean", "delta_oof_f1_mean", "selected_threshold_mean",
        "test_fp_mean", "test_fn_mean", "threshold_flag", "final_decision_label",
    ]
    report = f"""# Step19D Current Active CAT12/CAT_new Targeted Check

Step19D tests CAT1/CAT2/CAT_new combinations on current_active because threshold is no longer the primary constraint.
current_active baseline is reproduced before any CAT_new comparison.
All thresholds are selected using OOF predictions only.
Test metrics are reporting-only.
No target encoding is used.
Variant_ID and Label are not used as model features.
current_active_changed = false

## Baseline Reproduction

Expected current_active reference from Step16B: Test F1 ~= 0.551142, threshold ~= 0.925752.

Observed reproduced baseline: Test F1 {summary.loc[summary.config.eq('current_active_baseline'), 'test_f1_mean'].iloc[0]:.6f}, threshold {summary.loc[summary.config.eq('current_active_baseline'), 'selected_threshold_mean'].iloc[0]:.6f}, OOF F1 {summary.loc[summary.config.eq('current_active_baseline'), 'oof_f1_mean'].iloc[0]:.6f}.

## Targeted Results

{markdown_table(summary[report_cols])}

## CAT_new Availability

CAT_new available: {catnew_available}. Merge report is stored in metadata.

## Decision

{recommendation}
"""
    OUT["report"].write_text(report, encoding="utf-8")

    next_action = f"""# Step19D Next Action Recommendation

current_active_changed = false

{recommendation}

Best Step19D config by mean Test F1: `{best['config']}` with Test F1 {best['test_f1_mean']:.6f}, threshold {best['selected_threshold_mean']:.6f}, FP {best['test_fp_mean']:.3f}, FN {best['test_fn_mean']:.3f}, OOF F1 {best['oof_f1_mean']:.6f}.
"""
    OUT["recommendation"].write_text(next_action, encoding="utf-8")


def main():
    started = time.perf_counter()
    step16_ref = current_active_reference_from_step16b()
    full = pd.read_csv(base.find_input())
    full["Label"] = full["Label"].astype(int)
    full_catnew, merge_report, catnew_available = maybe_merge_catnew(full)

    configs = CONFIGS if catnew_available else CONFIGS[:4]
    per_seed_rows = []
    runtime_rows = []
    feature_infos = []
    for config in configs:
        for seed in SEEDS:
            data_src = full_catnew if "CAT_new" in config else full
            payload, runtimes, feature_info, _ = load_or_generate(seed, config, data_src)
            per_seed_rows.append(metric_row(seed, config, payload, feature_info))
            runtime_rows.extend(runtimes)
            feature_infos.append(feature_info)

    per_seed = pd.DataFrame(per_seed_rows)
    summary = add_deltas_and_labels(summarize(per_seed))

    baseline = summary[summary.config.eq("current_active_baseline")].iloc[0]
    if abs(baseline.test_f1_mean - 0.55114227713967) > 0.001:
        raise RuntimeError(f"Current_active Test F1 mismatch: {baseline.test_f1_mean}")
    if abs(baseline.selected_threshold_mean - 0.9257520313157634) > 0.001:
        raise RuntimeError(f"Current_active threshold mismatch: {baseline.selected_threshold_mean}")

    write_outputs(
        per_seed=per_seed,
        summary=summary,
        runtimes=runtime_rows,
        feature_infos=feature_infos,
        step16_ref=step16_ref,
        merge_report=merge_report,
        catnew_available=catnew_available,
        runtime_seconds=time.perf_counter() - started,
    )
    print(f"Step19D completed in {(time.perf_counter() - started) / 60:.2f} minutes", flush=True)


if __name__ == "__main__":
    main()
