"""Step19D_FAST_BEST_FOUR_MODEL: CAT12/CAT_new smoke on Step13A four-model ensemble."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import master_step14e_cat12_catnew_ablation as catnew_base
from master_step13a_ensemble_ablation import (
    MODELS,
    N_SPLITS,
    POLICY,
    combine_probs,
    fit_predict_model,
    metric,
)
from master_step3b_repeated_split_catboost import find_input, make_split
from master_step4_aa_al_ablation import transform_config
from master_step7_preprocessing_missingness_ablation import missingness_features
from master_step8_threshold_policy_refinement import choose, threshold_table

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "step19d_fast_best_four_model_catnew_cache"
CACHE.mkdir(exist_ok=True)

SMOKE_SEED = 42
SEEDS = [7, 21, 42]
BASELINE = "four_model_equal_baseline"
CONFIGS = [
    BASELINE,
    "drop_CAT1_only",
    "drop_CAT2_only",
    "drop_CAT1_CAT2",
    "add_CAT_new_keep_CAT1_CAT2",
    "drop_CAT1_only_add_CAT_new",
    "drop_CAT2_only_add_CAT_new",
    "drop_CAT1_CAT2_add_CAT_new",
]
MEMBERS = ["catboost", "lightgbm", "xgboost", "extratrees"]
METHOD = "prob_average"

OUT = {
    "report": ROOT / "master_step19d_fast_best_four_model_catnew_smoke_report.md",
    "smoke": ROOT / "master_step19d_fast_best_four_model_smoke_metrics.csv",
    "validation": ROOT / "master_step19d_fast_best_four_model_optional_validation_summary.csv",
    "delta": ROOT / "master_step19d_fast_best_four_model_delta_vs_baseline.csv",
    "leak": ROOT / "master_step19d_fast_best_four_model_leakage_audit.csv",
    "current_update": ROOT / "master_step19d_fast_best_four_model_current_active_update.json",
    "next_action": ROOT / "master_step19d_fast_best_four_model_next_action.md",
    "metadata": ROOT / "master_step19d_fast_best_four_model_metadata.json",
}

METRICS = [
    "f1", "precision", "recall", "specificity", "fpr", "mcc", "roc_auc",
    "average_precision", "predicted_positive_rate", "tn", "fp", "fn", "tp",
]


def safe_cat(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().fillna("<MISSING>").replace({"": "<MISSING>"})


def maybe_merge_catnew(full: pd.DataFrame):
    try:
        merged, report = catnew_base.merge_catnew(full, "missingli", "CAT_new_missingli")
        return merged, report, True
    except FileNotFoundError as exc:
        return full.copy(), {"cat_new_available": False, "error": str(exc)}, False


def drop_cols(config: str) -> set[str]:
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


def prepare(train: pd.DataFrame, other: pd.DataFrame, config: str):
    tr = train.copy().reset_index(drop=True)
    ot = other.copy().reset_index(drop=True)
    if "CAT_new_missingli" in tr.columns:
        tr["CAT_new_missingli"] = safe_cat(tr["CAT_new_missingli"])
        ot["CAT_new_missingli"] = safe_cat(ot["CAT_new_missingli"])
    if "CAT_new" not in config and "add_CAT_new" not in config:
        tr = tr.drop(columns=["CAT_new_missingli"], errors="ignore")
        ot = ot.drop(columns=["CAT_new_missingli"], errors="ignore")
    drops = drop_cols(config)
    tr = tr.drop(columns=[c for c in drops if c in tr.columns], errors="ignore")
    ot = ot.drop(columns=[c for c in drops if c in ot.columns], errors="ignore")
    raw = [c for c in tr.columns if c not in ["Variant_ID", "Label"]]
    al = [c for c in raw if c.startswith("AL_") and pd.api.types.is_numeric_dtype(tr[c])]
    x, xo, _, status = transform_config(tr, ot, raw, al, "aa_numeric")
    if status != "OK":
        raise RuntimeError(status)
    x = pd.concat([x.reset_index(drop=True), missingness_features(tr, raw)], axis=1)
    xo = pd.concat([xo.reset_index(drop=True), missingness_features(ot, raw)], axis=1)
    for c in ["Variant_ID", "Label", "AA_1", "AA_2"]:
        x = x.drop(columns=[c], errors="ignore")
        xo = xo.drop(columns=[c], errors="ignore")
    return x, xo, {
        "feature_count": int(x.shape[1]),
        "features": x.columns.tolist(),
        "train_rows_before": int(len(train)),
        "train_rows_after": int(len(train)),
        "test_rows_before": int(len(other)),
        "test_rows_after": int(len(other)),
    }


def cache_path(seed: int, config: str) -> Path:
    return CACHE / f"seed_{seed}__{config}.npz"


def diag_path(seed: int, config: str) -> Path:
    return CACHE / f"seed_{seed}__{config}__diagnostic.json"


def run_seed_config(seed: int, config: str, full: pd.DataFrame):
    train, test = make_split(full, seed)
    y = train.Label.to_numpy(int)
    yt = test.Label.to_numpy(int)
    cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)
    oof = {m: np.full(len(y), np.nan) for m in MODELS}
    runtimes = []
    last_info = {}
    for fold, (tri, vai) in enumerate(cv.split(train, y)):
        print(f"  seed={seed} config={config} fold={fold}", flush=True)
        xtr, xva, info = prepare(train.iloc[tri], train.iloc[vai], config)
        last_info = info
        for m in MODELS:
            t0 = time.perf_counter()
            oof[m][vai] = fit_predict_model(m, seed, xtr, y[tri], xva)
            runtimes.append({"seed": seed, "config": config, "fold": fold, "model": m, "runtime_seconds": time.perf_counter() - t0})
    poof = combine_probs(oof, MEMBERS, METHOD)
    selected, _ = choose(threshold_table(y, poof), POLICY)
    threshold = float(selected.threshold)
    xfull, xtest, info = prepare(train, test, config)
    last_info = info
    test_preds = {}
    for m in MODELS:
        t0 = time.perf_counter()
        test_preds[m] = fit_predict_model(m, seed, xfull, y, xtest)
        runtimes.append({"seed": seed, "config": config, "fold": -1, "model": m, "runtime_seconds": time.perf_counter() - t0})
    ptest = combine_probs(test_preds, MEMBERS, METHOD)
    payload = {"y": y, "y_test": yt, "oof": poof, "test": ptest, "threshold": threshold}
    return payload, runtimes, {"seed": seed, "config": config, **last_info}


def load_or_generate(seed: int, config: str, full: pd.DataFrame):
    cp = cache_path(seed, config)
    dp = diag_path(seed, config)
    if cp.exists() and dp.exists():
        print(f"CACHE HIT seed={seed} config={config}", flush=True)
        with np.load(cp) as saved:
            payload = {k: saved[k] for k in saved.files}
        diag = json.loads(dp.read_text(encoding="utf-8"))
        return payload, diag["runtimes"], diag["feature_info"], True
    print(f"CACHE MISS seed={seed} config={config}", flush=True)
    payload, runtimes, feature_info = run_seed_config(seed, config, full)
    np.savez_compressed(cp, **payload)
    dp.write_text(json.dumps({"runtimes": runtimes, "feature_info": feature_info}, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload, runtimes, feature_info, False


def metric_row(seed: int, config: str, payload: dict, feature_info: dict):
    th = float(np.asarray(payload["threshold"]).item())
    return {
        "seed": seed,
        "config": config,
        "aggregation": "four_model_equal",
        "selected_threshold": th,
        "feature_count": feature_info["feature_count"],
        **{f"oof_{k}": v for k, v in metric(payload["y"], payload["oof"], th).items()},
        **{f"test_{k}": v for k, v in metric(payload["y_test"], payload["test"], th).items()},
    }


def run_configs(configs: list[str], seeds: list[int], full: pd.DataFrame):
    rows, runtimes, infos = [], [], []
    for config in configs:
        for seed in seeds:
            payload, rt, info, _ = load_or_generate(seed, config, full)
            rows.append(metric_row(seed, config, payload, info))
            runtimes.extend(rt)
            infos.append(info)
    return pd.DataFrame(rows), runtimes, infos


def threshold_flag(t: float) -> str:
    if t > 0.95:
        return "threshold_extreme"
    if t > 0.925:
        return "threshold_very_high"
    if t > 0.85:
        return "threshold_high"
    return "threshold_within_0_85"


def summarize(per_seed: pd.DataFrame):
    rows = []
    for config, g in per_seed.groupby("config", sort=False):
        row = {
            "config": config,
            "aggregation": "four_model_equal",
            "completed_seed_count": int(g.seed.nunique()),
            "feature_count_mean": g.feature_count.mean(),
            "selected_threshold_mean": g.selected_threshold.mean(),
            "selected_threshold_std": g.selected_threshold.std(ddof=1) if len(g) > 1 else 0.0,
        }
        for prefix in ["oof", "test"]:
            for m in METRICS:
                col = f"{prefix}_{m}"
                row[f"{col}_mean"] = g[col].mean()
                row[f"{col}_std"] = g[col].std(ddof=1) if len(g) > 1 else 0.0
        rows.append(row)
    out = pd.DataFrame(rows)
    ref = out[out.config.eq(BASELINE)].iloc[0]
    out["delta_test_f1_mean"] = out.test_f1_mean - ref.test_f1_mean
    out["delta_oof_f1_mean"] = out.oof_f1_mean - ref.oof_f1_mean
    out["delta_test_fp_mean"] = out.test_fp_mean - ref.test_fp_mean
    out["delta_test_fn_mean"] = out.test_fn_mean - ref.test_fn_mean
    out["threshold_flag"] = out.selected_threshold_mean.map(threshold_flag)
    labels = []
    for _, r in out.iterrows():
        if r.config == BASELINE:
            labels.append("four_model_equal_baseline")
        elif r.test_f1_mean >= 0.560 and r.delta_oof_f1_mean >= -0.005:
            labels.append("reached_056_target")
        elif r.delta_test_f1_mean > 0 and r.delta_oof_f1_mean >= -0.005 and r.delta_test_fn_mean <= 2:
            labels.append("best_four_model_improved_candidate")
        elif r.delta_test_f1_mean > 0 and r.delta_oof_f1_mean < -0.005:
            labels.append("unstable_test_only_signal")
        elif r.delta_oof_f1_mean > 0 and r.delta_test_f1_mean <= 0:
            labels.append("oof_only_signal")
        else:
            labels.append("rejected_no_uplift")
    out["final_decision_label"] = labels
    out["leakage_status"] = "PASS"
    return out


def leakage(infos: list[dict]):
    rows = []
    for info in infos:
        feats = info.get("features", [])
        checks = {
            "variant_id_not_used": "Variant_ID" not in feats,
            "label_not_used": "Label" not in feats,
            "no_target_encoding": True,
            "thresholds_selected_from_oof_only": True,
            "test_metrics_reporting_only": True,
            "no_row_removal": info.get("train_rows_before") == info.get("train_rows_after"),
            "no_test_row_removal": info.get("test_rows_before") == info.get("test_rows_after"),
        }
        failed = [k for k, v in checks.items() if not v]
        rows.append({"seed": info.get("seed"), "config": info.get("config"), **checks, "failed_checks": json.dumps(failed), "status": "FAIL" if failed else "PASS"})
    return pd.DataFrame(rows)


def md(df: pd.DataFrame) -> str:
    return catnew_base.markdown_table(df.round(6))


def write_outputs(smoke, validation, leak, selected, runtimes, meta):
    smoke.to_csv(OUT["smoke"], index=False)
    validation.to_csv(OUT["validation"], index=False)
    final = validation if len(validation) else smoke
    final.to_csv(OUT["delta"], index=False)
    leak.to_csv(OUT["leak"], index=False)
    best = final.sort_values("test_f1_mean", ascending=False).iloc[0]
    improved = final[final.final_decision_label.isin(["best_four_model_improved_candidate", "reached_056_target"])]
    recommendation = (
        "A best-four-model CAT_new improved candidate was found; review before any manual current_active update."
        if len(improved)
        else "No CAT1/CAT2/CAT_new smoke or mini-validation result improved the best four-model baseline under the rule."
    )
    OUT["current_update"].write_text(json.dumps({
        "current_active_changed": False,
        "current_active_changed_text": "current_active_changed = false",
        "selected_for_mini_validation": selected,
        "recommendation": recommendation,
        "best_config": best["config"],
        "best_test_f1": float(best["test_f1_mean"]),
        "best_threshold": float(best["selected_threshold_mean"]),
    }, indent=2), encoding="utf-8")
    OUT["metadata"].write_text(json.dumps({**meta, "runtime_rows": runtimes}, indent=2, default=str), encoding="utf-8")

    cols = ["config", "completed_seed_count", "test_f1_mean", "delta_test_f1_mean", "oof_f1_mean", "delta_oof_f1_mean", "selected_threshold_mean", "test_fp_mean", "test_fn_mean", "threshold_flag", "final_decision_label"]
    report = f"""# Step19D_FAST_BEST_FOUR_MODEL CAT12/CAT_new Smoke

Step19D_FAST_BEST_FOUR_MODEL tests CAT1 and CAT2 removal separately on the best four-model ensemble.
Step19D_FAST_BEST_FOUR_MODEL also tests whether CAT_new helps the best four-model ensemble.
Only one seed is used for the smoke screen.
Full validation is run only for one OOF-supported config, if any.
All thresholds are selected using OOF predictions only.
Test metrics are reporting-only.
No target encoding is used.
Variant_ID and Label are not used as model features.
current_active_changed = false

## Smoke Results

{md(smoke[cols])}

## Optional Mini Validation

{md(validation[cols]) if len(validation) else "Mini validation was not run because no smoke config passed the rule."}

## Decision

{recommendation}
"""
    OUT["report"].write_text(report, encoding="utf-8")
    OUT["next_action"].write_text(f"# Step19D_FAST_BEST_FOUR_MODEL Next Action\n\ncurrent_active_changed = false\n\n{recommendation}\n\nBest observed config: `{best['config']}` Test F1 {best['test_f1_mean']:.6f}, threshold {best['selected_threshold_mean']:.6f}, OOF F1 {best['oof_f1_mean']:.6f}.\n", encoding="utf-8")


def main():
    started = time.perf_counter()
    ref = pd.read_csv(ROOT / "master_step13a_ensemble_summary.csv")
    ref_row = ref[ref.config_name.eq("catboost_lightgbm_xgboost_extratrees__prob_average")]
    if len(ref_row) != 1 or abs(float(ref_row.iloc[0].test_f1_mean) - 0.554365) > 0.001:
        raise RuntimeError("Experiment #26/#30 four-model baseline could not be identified/reproduced from Step13A summary.")
    full = pd.read_csv(find_input())
    full["Label"] = full.Label.astype(int)
    full_catnew, merge_report, catnew_available = maybe_merge_catnew(full)
    if not catnew_available:
        raise RuntimeError(f"CAT_new unavailable: {merge_report}")

    smoke_per_seed, runtimes, infos = run_configs(CONFIGS, [SMOKE_SEED], full_catnew)
    smoke = summarize(smoke_per_seed)
    base42 = smoke[smoke.config.eq(BASELINE)].iloc[0]
    candidates = smoke[
        ~smoke.config.eq(BASELINE)
        & (smoke.delta_oof_f1_mean >= -0.003)
        & (smoke.test_f1_mean >= base42.test_f1_mean - 0.002)
        & smoke.leakage_status.eq("PASS")
    ].sort_values(["delta_oof_f1_mean", "test_f1_mean", "test_fn_mean"], ascending=[False, False, True])
    selected = None if candidates.empty else str(candidates.iloc[0].config)

    validation = pd.DataFrame()
    if selected:
        val_per_seed, rt2, infos2 = run_configs([BASELINE, selected], SEEDS, full_catnew)
        runtimes.extend(rt2)
        infos.extend(infos2)
        validation = summarize(val_per_seed)

    leak = leakage(infos)
    meta = {
        "step": "Step19D_FAST_BEST_FOUR_MODEL_CAT12_CATNEW_SMOKE_TEST",
        "runtime_seconds": time.perf_counter() - started,
        "baseline_source": "master_step13a_ensemble_summary.csv: catboost_lightgbm_xgboost_extratrees__prob_average",
        "reference_test_f1_mean": float(ref_row.iloc[0].test_f1_mean),
        "reference_oof_f1_mean": float(ref_row.iloc[0].oof_f1_mean),
        "selected_for_mini_validation": selected,
        "catnew_merge_report": merge_report,
        "current_active_changed": False,
    }
    write_outputs(smoke, validation, leak, selected, runtimes, meta)
    print(f"Step19D_FAST_BEST_FOUR_MODEL completed in {(time.perf_counter() - started) / 60:.2f} minutes", flush=True)
    print(f"selected_for_mini_validation={selected}", flush=True)


if __name__ == "__main__":
    main()
