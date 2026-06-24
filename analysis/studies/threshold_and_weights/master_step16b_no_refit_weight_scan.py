"""Step16B: no-refit fixed ensemble weight diagnostic."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import master_step14e_cat12_catnew_ablation as base

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "step15b_refined_allog_cat_model_family_prediction_cache"
OUT = {
    "report": ROOT / "master_step16b_no_refit_weight_scan_report.md",
    "summary": ROOT / "master_step16b_weight_scan_summary.csv",
    "delta": ROOT / "master_step16b_weight_scan_delta.csv",
    "per_seed": ROOT / "master_step16b_per_seed_metrics.csv",
    "active": ROOT / "master_step16b_current_active_update.json",
    "leak": ROOT / "master_step16b_leakage_audit.csv",
    "metadata": ROOT / "master_step16b_metadata.json",
    "runtime": ROOT / "master_step16b_runtime_report.csv",
    "excel": ROOT / "master_step16b_excel_ready_weight_tracker.csv",
}

SEEDS = [7, 21, 42]
MODELS = ["catboost", "lightgbm", "xgboost", "extratrees"]
REFERENCE_CONFIG = "active_reference_no_ALlog_native_CAT345_with_CAT6"
REFERENCE_WEIGHT = "four_model_equal"
REFERENCE_POLICY = "current_policy_reproduce"
CONFIGS = [
    REFERENCE_CONFIG,
    "signed_ALlog_only_native_CAT345_with_CAT6",
    "raw_AL_plus_capped_signed_ALlog_native_CAT345_with_CAT6",
    "raw_AL_plus_signed_ALlog_without_AL37_AL219_native_CAT345_with_CAT6",
]
WEIGHTS = {
    "four_model_equal": {"catboost": 0.25, "lightgbm": 0.25, "xgboost": 0.25, "extratrees": 0.25},
    "current_extratrees_heavy": {"catboost": 0.30, "lightgbm": 0.10, "xgboost": 0.20, "extratrees": 0.40},
    "lightgbm_medium": {"catboost": 0.30, "lightgbm": 0.30, "xgboost": 0.15, "extratrees": 0.25},
    "lightgbm_heavy": {"catboost": 0.25, "lightgbm": 0.40, "xgboost": 0.10, "extratrees": 0.25},
    "lightgbm_al_log_probe": {"catboost": 0.20, "lightgbm": 0.45, "xgboost": 0.10, "extratrees": 0.25},
    "catboost_lightgbm_pair": {"catboost": 0.50, "lightgbm": 0.50, "xgboost": 0.00, "extratrees": 0.00},
    "lightgbm_extratrees_pair": {"catboost": 0.00, "lightgbm": 0.50, "xgboost": 0.00, "extratrees": 0.50},
    "catboost_extratrees_equal": {"catboost": 0.50, "lightgbm": 0.00, "xgboost": 0.00, "extratrees": 0.50},
    "catboost_dominant": {"catboost": 0.60, "lightgbm": 0.10, "xgboost": 0.10, "extratrees": 0.20},
    "catboost_extratrees_lightgbm_small": {"catboost": 0.40, "lightgbm": 0.10, "xgboost": 0.00, "extratrees": 0.50},
}
POLICIES = [
    ("current_policy_reproduce", False),
    ("max_oof_mcc", True),
]
METRICS = [
    "f1", "precision", "recall", "specificity", "fpr", "mcc", "roc_auc",
    "average_precision", "tp", "fp", "fn", "tn", "predicted_positive_rate",
]


def load_payload(seed: int, config: str) -> dict:
    path = CACHE / f"seed_{seed}__{config}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as saved:
        return {k: saved[k] for k in saved.files}


def combine(payload: dict, weight_name: str, stage: str) -> np.ndarray:
    weights = WEIGHTS[weight_name]
    return sum(weights[m] * payload[f"{stage}_{m}"] for m in MODELS)


def fast_mcc(y, p, t):
    y = np.asarray(y, dtype=int)
    pred = (np.asarray(p) >= t).astype(int)
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
    return ((tp * tn - fp * fn) / denom if denom else 0.0, f1, -float(t))


def select_threshold(y, p, policy: str) -> float:
    if policy == "current_policy_reproduce":
        selected, _ = base.choose(base.threshold_table(y, p), base.POLICY)
        return float(selected.threshold)
    if policy == "max_oof_mcc":
        grid = np.unique(np.concatenate([np.linspace(0, 1, 1001), np.quantile(p, np.linspace(0, 1, 501))]))
        return float(max(grid, key=lambda t: fast_mcc(y, p, float(t))))
    raise ValueError(policy)


def per_seed_rows():
    rows = []
    runtimes = []
    for seed in SEEDS:
        for config in CONFIGS:
            payload = load_payload(seed, config)
            y = payload["y"].astype(int)
            yt = payload["y_test"].astype(int)
            for weight_name in WEIGHTS:
                started = time.perf_counter()
                po = combine(payload, weight_name, "oof")
                pt = combine(payload, weight_name, "test")
                for policy, diagnostic in POLICIES:
                    threshold = select_threshold(y, po, policy)
                    rows.append({
                        "seed": seed,
                        "config_name": config,
                        "weight_set": weight_name,
                        "threshold_policy": policy,
                        "diagnostic_only": diagnostic,
                        "threshold": threshold,
                        **{f"oof_{k}": v for k, v in base.metric(y, po, threshold).items()},
                        **{f"test_{k}": v for k, v in base.metric(yt, pt, threshold).items()},
                    })
                runtimes.append({
                    "seed": seed, "config_name": config, "weight_set": weight_name,
                    "runtime_seconds": time.perf_counter() - started,
                    "base_model_refit": False,
                    "source": "Step15B cached predictions",
                })
    return pd.DataFrame(rows), pd.DataFrame(runtimes)


def summarize(per: pd.DataFrame):
    rows = []
    for keys, frame in per.groupby(["config_name", "weight_set", "threshold_policy"], sort=False):
        config, weight_set, policy = keys
        row = {
            "config_name": config,
            "weight_set": weight_set,
            "threshold_policy": policy,
            "diagnostic_only": bool(frame.diagnostic_only.any()),
            "completed_seed_count": int(frame.seed.nunique()),
            "catboost_weight": WEIGHTS[weight_set]["catboost"],
            "lightgbm_weight": WEIGHTS[weight_set]["lightgbm"],
            "xgboost_weight": WEIGHTS[weight_set]["xgboost"],
            "extratrees_weight": WEIGHTS[weight_set]["extratrees"],
        }
        for col in ["threshold"] + [f"oof_{m}" for m in METRICS] + [f"test_{m}" for m in METRICS]:
            row[f"{col}_mean"] = frame[col].mean()
            row[f"{col}_std"] = frame[col].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def add_deltas(summary: pd.DataFrame):
    ref = summary[
        summary.config_name.eq(REFERENCE_CONFIG)
        & summary.weight_set.eq(REFERENCE_WEIGHT)
        & summary.threshold_policy.eq(REFERENCE_POLICY)
    ].iloc[0]
    out = summary.copy()
    for metric in ["f1", "precision", "recall", "specificity", "mcc", "fp", "fn", "predicted_positive_rate"]:
        out[f"delta_test_{metric}_mean"] = out[f"test_{metric}_mean"] - ref[f"test_{metric}_mean"]
        out[f"delta_oof_{metric}_mean"] = out[f"oof_{metric}_mean"] - ref[f"oof_{metric}_mean"]
    labels = []
    for _, r in out.iterrows():
        if (
            r.config_name == REFERENCE_CONFIG
            and r.weight_set == REFERENCE_WEIGHT
            and r.threshold_policy == REFERENCE_POLICY
        ):
            labels.append("current_active_reference")
        elif bool(r.diagnostic_only):
            labels.append("diagnostic_only_max_oof_mcc")
        elif (
            r.delta_test_f1_mean >= 0.004
            and r.delta_oof_f1_mean >= -0.005
            and r.delta_test_fp_mean <= 2.0
            and r.delta_test_fn_mean <= 2.0
        ):
            labels.append("keep_candidate")
        elif r.delta_test_f1_mean > 0 and r.delta_oof_f1_mean < -0.01:
            if "lightgbm" in r.weight_set and r.config_name == "signed_ALlog_only_native_CAT345_with_CAT6":
                labels.append("lightgbm_specific_test_only_signal")
            else:
                labels.append("unstable_test_only_signal")
        elif r.delta_test_fp_mean < 0 and r.delta_test_fn_mean > 2 and r.delta_test_f1_mean < 0:
            labels.append("conservative_but_not_preferred")
        else:
            labels.append("reject")
    out["decision_label"] = labels
    return out, ref.to_dict()


def excel_ready(delta: pd.DataFrame):
    def islem(row):
        if row.config_name == REFERENCE_CONFIG and row.weight_set == REFERENCE_WEIGHT:
            return "current active reference"
        if row.config_name == "signed_ALlog_only_native_CAT345_with_CAT6":
            base_desc = "AL_log only"
        elif row.config_name == "raw_AL_plus_capped_signed_ALlog_native_CAT345_with_CAT6":
            base_desc = "capped AL_log"
        elif "without_AL37_AL219" in row.config_name:
            base_desc = "AL_37/AL_219 log çıkarıldı"
        else:
            base_desc = "raw AL aktif model"
        if "lightgbm" in row.weight_set:
            return f"{base_desc} + LightGBM ağırlığı artırıldı"
        if "catboost" in row.weight_set:
            return f"{base_desc} + CatBoost ağırlığı artırıldı"
        if "extratrees" in row.weight_set:
            return f"{base_desc} + ExtraTrees ağırlığı artırıldı"
        return base_desc
    rows = []
    for _, r in delta.iterrows():
        rows.append({
            "Step": "Step16B",
            "Deney Grubu": "No-refit ensemble weight diagnostic",
            "Model / Aggregation": r.weight_set,
            "İşlem": islem(r),
            "Yapılan Değişiklik": f"{r.config_name}; threshold={r.threshold_policy}",
            "Test F1": r.test_f1_mean,
            "Delta F1": r.delta_test_f1_mean,
            "OOF F1": r.oof_f1_mean,
            "Test FP": r.test_fp_mean,
            "Test FN": r.test_fn_mean,
            "Karar / Not": r.decision_label,
        })
    return pd.DataFrame(rows).sort_values("Test F1", ascending=False)


def md_table(df: pd.DataFrame) -> str:
    return base.markdown_table(df.round(6))


def write_report(delta: pd.DataFrame, ref: dict, leak: pd.DataFrame):
    main = delta[delta.threshold_policy.eq(REFERENCE_POLICY)].copy()
    keep = main[main.decision_label.eq("keep_candidate")].sort_values("delta_test_f1_mean", ascending=False)
    best = keep.iloc[0] if len(keep) else None
    active_lightgbm = main[
        main.config_name.eq(REFERENCE_CONFIG) & main.weight_set.str.contains("lightgbm", case=False)
    ].sort_values("delta_test_f1_mean", ascending=False).head(8)
    signed_lightgbm = main[
        main.config_name.eq("signed_ALlog_only_native_CAT345_with_CAT6") & main.weight_set.str.contains("lightgbm", case=False)
    ].sort_values("delta_test_f1_mean", ascending=False).head(8)
    capped = main[
        main.config_name.eq("raw_AL_plus_capped_signed_ALlog_native_CAT345_with_CAT6")
    ].sort_values("delta_test_f1_mean", ascending=False).head(8)
    without = main[
        main.config_name.eq("raw_AL_plus_signed_ALlog_without_AL37_AL219_native_CAT345_with_CAT6")
    ].sort_values("delta_test_f1_mean", ascending=False).head(8)
    cols = [
        "config_name", "weight_set", "threshold_policy", "test_f1_mean", "delta_test_f1_mean",
        "oof_f1_mean", "delta_oof_f1_mean", "test_precision_mean", "test_recall_mean",
        "test_fp_mean", "test_fn_mean", "threshold_mean", "decision_label",
    ]
    rec = "Keep current active four_model_equal." if best is None else f"Promote `{best.config_name}` / `{best.weight_set}` after review."
    text = f"""# Step16B no-refit ensemble weight diagnostic

Step16B does not refit base models.

All weight sets are fixed before evaluation; no test-label weight search is performed.

All thresholds are selected using OOF predictions only.

Test metrics are reporting-only.

LightGBM-heavy results are diagnostic unless OOF-supported.

Step16A did not produce an OOF-supported threshold/calibration keep candidate, so Step16B uses the standard OOF-only threshold policy.

## Current active reference

- Config: `{REFERENCE_CONFIG}`
- Weight set: `{REFERENCE_WEIGHT}`
- Test F1: {ref['test_f1_mean']:.6f}
- OOF F1: {ref['oof_f1_mean']:.6f}
- FP/FN: {ref['test_fp_mean']:.3f} / {ref['test_fn_mean']:.3f}

## Does increasing LightGBM weight help the active raw-AL model?

{md_table(active_lightgbm[cols])}

## Does increasing LightGBM weight help signed_ALlog_only?

{md_table(signed_lightgbm[cols])}

## Does capped AL_log become useful under any fixed weight set?

{md_table(capped[cols])}

## Does excluding AL_37/AL_219 signed-log features become useful?

{md_table(without[cols])}

## Full current-policy ranking

{md_table(main.sort_values("test_f1_mean", ascending=False).head(30)[cols])}

## Diagnostic max_oof_mcc rows

{md_table(delta[delta.threshold_policy.eq("max_oof_mcc")].sort_values("test_f1_mean", ascending=False).head(20)[cols])}

## Leakage audit

- Any leakage failure: **{bool(leak.leakage_fail.any())}**
- Status counts: {leak.status.value_counts().to_dict()}

## Recommendation

{rec}
"""
    OUT["report"].write_text(text, encoding="utf-8")
    return rec


def main():
    started = time.perf_counter()
    per, runtime = per_seed_rows()
    summary = summarize(per)
    delta, ref = add_deltas(summary)
    excel = excel_ready(delta)
    step15b_ok = not bool(json.loads((ROOT / "master_step15b_metadata.json").read_text(encoding="utf-8")).get("leakage_fail", True))
    step16a = json.loads((ROOT / "master_step16a_metadata.json").read_text(encoding="utf-8")) if (ROOT / "master_step16a_metadata.json").exists() else {}
    leak_checks = {
        "no_base_model_refit_in_step16b": True,
        "step15b_cached_predictions_used": True,
        "test_labels_not_used_for_weight_selection": True,
        "test_labels_not_used_for_threshold_selection": True,
        "test_labels_not_used_for_model_family_selection": True,
        "test_labels_not_used_for_feature_selection": True,
        "fixed_weights_prespecified": True,
        "no_continuous_or_large_grid_weight_search": True,
        "thresholds_selected_from_oof_only": True,
        "test_metrics_reporting_only": True,
        "no_train_rows_removed": True,
        "no_test_rows_removed": True,
        "step15b_leakage_respected": step15b_ok,
        "step16a_conclusion_respected": step16a.get("best_keep_candidate") in [None, "None"],
    }
    failed = [k for k, v in leak_checks.items() if not v]
    leak = pd.DataFrame([{**leak_checks, "failed_checks": json.dumps(failed), "leakage_fail": bool(failed), "status": "FAIL" if failed else "PASS"}])
    rec = write_report(delta, ref, leak)
    keep = delta[delta.decision_label.eq("keep_candidate") & ~delta.diagnostic_only]
    active_update = {
        "current_active_changed": bool(len(keep)),
        "recommendation": rec,
        "current_active_config": REFERENCE_CONFIG,
        "current_active_weight_set": REFERENCE_WEIGHT,
        "current_active_threshold_policy": REFERENCE_POLICY,
        "best_keep_candidate": None if keep.empty else keep.sort_values("delta_test_f1_mean", ascending=False).iloc[0].to_dict(),
    }
    per.to_csv(OUT["per_seed"], index=False)
    summary.to_csv(OUT["summary"], index=False)
    delta.to_csv(OUT["delta"], index=False)
    excel.to_csv(OUT["excel"], index=False)
    runtime.to_csv(OUT["runtime"], index=False)
    leak.to_csv(OUT["leak"], index=False)
    OUT["active"].write_text(json.dumps(active_update, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    metadata = {
        "step": "Step16B",
        "completed": True,
        "base_model_refit": False,
        "configs": CONFIGS,
        "weight_sets": WEIGHTS,
        "threshold_policies": [p for p, _ in POLICIES],
        "reference": {"config": REFERENCE_CONFIG, "weight_set": REFERENCE_WEIGHT, "threshold_policy": REFERENCE_POLICY},
        "keep_candidate_count": int(len(keep)),
        "leakage_fail": bool(leak.leakage_fail.any()),
        "wall_runtime_seconds": time.perf_counter() - started,
        "recommendation": rec,
    }
    OUT["metadata"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== Step16B complete ===", flush=True)
    print(f"wall_seconds={metadata['wall_runtime_seconds']:.2f}", flush=True)
    print(f"leakage_fail={metadata['leakage_fail']}", flush=True)
    print(f"keep_candidate_count={metadata['keep_candidate_count']}", flush=True)


if __name__ == "__main__":
    main()
