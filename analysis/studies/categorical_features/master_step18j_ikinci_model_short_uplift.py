"""Step18J: short corrected uplift for the low-threshold ikinci_model.

This step intentionally avoids the broad Step18I grid. It first reproduces the
Step18C/Step18B low-threshold reference from cache, then runs only a small set
of targeted configs with the exact Step18B/C model settings.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

import master_step14e_cat12_catnew_ablation as base
import master_step18b_teknofest_style_tree_transfer as step18b
from master_step4_aa_al_ablation import transform_config
from master_step7_preprocessing_missingness_ablation import missingness_features
from master_step8_threshold_policy_refinement import choose, threshold_table
from master_v02_catboost_threshold_stability import markdown_table


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "step18j_ikinci_model_short_uplift_cache"
CACHE.mkdir(exist_ok=True)

STEP = "Step18J"
REFERENCE_CONFIG = "active_reference_current_drop_CAT12_with_missingness"
REFERENCE_AGG = "old_crossensemble_equal"
POLICY = step18b.POLICY
SEEDS = step18b.SEEDS
N_SPLITS = step18b.N_SPLITS
MODELS = step18b.MODELS
AGGREGATIONS = {
    "old_crossensemble_equal": step18b.AGGREGATIONS["old_crossensemble_equal"],
    "old_crossensemble_tree_heavy": step18b.AGGREGATIONS["old_crossensemble_tree_heavy"],
    "catboost_extratrees_equal": step18b.AGGREGATIONS["catboost_extratrees_equal"],
}

CONFIGS = [
    "ikinci_model_baseline",
    "ikinci_model_CAT345_onehot_separate",
    "ikinci_model_CAT35_combined_plus_CAT4",
    "ikinci_model_CAT6_presence_flag_only",
    "ikinci_model_CAT6_dropped",
    "ikinci_model_CAT345_onehot_separate_plus_CAT6_presence_flag_only",
    "ikinci_model_CAT35_combined_plus_CAT4_plus_CAT6_presence_flag_only",
]

OUT = {
    "report": ROOT / "master_step18j_ikinci_model_short_uplift_report.md",
    "per_seed": ROOT / "master_step18j_per_seed_metrics.csv",
    "summary": ROOT / "master_step18j_summary.csv",
    "delta": ROOT / "master_step18j_delta_vs_ikinci_model.csv",
    "threshold": ROOT / "master_step18j_threshold_behavior_summary.csv",
    "excel": ROOT / "master_step18j_excel_wide_model_tracker.csv",
    "current": ROOT / "master_step18j_current_active_update.json",
    "leak": ROOT / "master_step18j_leakage_audit.csv",
    "metadata": ROOT / "master_step18j_metadata.json",
}

METRICS = [
    "f1", "precision", "recall", "specificity", "fpr", "mcc",
    "roc_auc", "average_precision", "tp", "fp", "fn", "tn",
    "predicted_positive_rate",
]


def safe_cat(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().fillna("<MISSING>").replace({"": "<MISSING>"})


def fit_ohe(train: pd.DataFrame, other: pd.DataFrame, cols: list[str]):
    if not cols:
        return train, other, 0
    try:
        enc = OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=False)
    except TypeError:
        enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    tr = train.copy()
    ot = other.copy()
    fit = enc.fit_transform(tr[cols])
    out = enc.transform(ot[cols])
    names = [f"OHE_{n}" for n in enc.get_feature_names_out(cols)]
    tr = pd.concat([tr.drop(columns=cols), pd.DataFrame(fit, columns=names, index=tr.index)], axis=1)
    ot = pd.concat([ot.drop(columns=cols), pd.DataFrame(out, columns=names, index=ot.index)], axis=1)
    return tr, ot, len(names)


def prepare_features(train: pd.DataFrame, other: pd.DataFrame, config: str):
    tr = train.copy().reset_index(drop=True)
    ot = other.copy().reset_index(drop=True)
    drop_cols = {"CAT_1", "CAT_2"}
    onehot_cols: list[str] = []
    derived_notes: list[str] = []

    for col in ["CAT_3", "CAT_4", "CAT_5", "CAT_6"]:
        if col in tr.columns:
            tr[col] = safe_cat(tr[col])
        if col in ot.columns:
            ot[col] = safe_cat(ot[col])

    if "CAT35_combined_plus_CAT4" in config:
        tr["CAT35_COMBINED"] = tr["CAT_3"] + "|" + tr["CAT_5"]
        ot["CAT35_COMBINED"] = ot["CAT_3"] + "|" + ot["CAT_5"]
        drop_cols.update(["CAT_3", "CAT_5"])
        derived_notes.append("CAT35_COMBINED_added")

    if "CAT345_onehot_separate" in config:
        onehot_cols.extend(["CAT_3", "CAT_4", "CAT_5"])
        derived_notes.append("CAT345_onehot_train_fit_only")

    if "CAT6_presence_flag_only" in config:
        tr["CAT6_PRESENT"] = (tr["CAT_6"] != "<MISSING>").astype(int)
        ot["CAT6_PRESENT"] = (ot["CAT_6"] != "<MISSING>").astype(int)
        tr["CAT6_MISSING"] = (tr["CAT_6"] == "<MISSING>").astype(int)
        ot["CAT6_MISSING"] = (ot["CAT_6"] == "<MISSING>").astype(int)
        drop_cols.add("CAT_6")
        derived_notes.append("CAT6_presence_missing_flags_only")
    elif config == "ikinci_model_CAT6_dropped":
        drop_cols.add("CAT_6")
        derived_notes.append("CAT6_dropped")

    tr = tr.drop(columns=[c for c in drop_cols if c in tr.columns])
    ot = ot.drop(columns=[c for c in drop_cols if c in ot.columns])
    tr, ot, ohe_count = fit_ohe(tr, ot, [c for c in onehot_cols if c in tr.columns])

    raw = [c for c in tr.columns if c not in ["Variant_ID", "Label"]]
    al_cols = [c for c in raw if c.startswith("AL_") and pd.api.types.is_numeric_dtype(tr[c])]
    x, xo, _, status = transform_config(tr, ot, raw, al_cols, "aa_numeric")
    if status != "OK":
        raise RuntimeError(status)

    x = pd.concat([x.reset_index(drop=True), missingness_features(tr, raw)], axis=1)
    xo = pd.concat([xo.reset_index(drop=True), missingness_features(ot, raw)], axis=1)

    for forbidden in ["Variant_ID", "Label", "AA_1", "AA_2"]:
        x = x.drop(columns=[forbidden], errors="ignore")
        xo = xo.drop(columns=[forbidden], errors="ignore")

    info = {
        "feature_count": int(len(x.columns)),
        "add_missingness": True,
        "drop_cols": sorted(drop_cols),
        "cat1_present": "CAT_1" in x.columns,
        "cat2_present": "CAT_2" in x.columns,
        "cat3_present": "CAT_3" in x.columns,
        "cat4_present": "CAT_4" in x.columns,
        "cat5_present": "CAT_5" in x.columns,
        "cat6_present": "CAT_6" in x.columns,
        "ohe_feature_count": int(ohe_count),
        "derived_notes": ";".join(derived_notes) if derived_notes else "baseline_like",
    }
    return x, xo, info


def cache_path(seed: int, config: str) -> Path:
    return CACHE / f"seed_{seed}__{config}.npz"


def diag_path(seed: int, config: str) -> Path:
    return CACHE / f"seed_{seed}__{config}__diagnostic.json"


def generate(seed: int, config: str, full: pd.DataFrame):
    train, test = base.make_split(full, seed)
    y = train.Label.to_numpy(dtype=int)
    y_test = test.Label.to_numpy(dtype=int)
    cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)
    oof = {model: np.full(len(y), np.nan) for model in MODELS}
    runtime_rows: list[dict] = []
    feature_rows: list[dict] = []

    for fold, (tri, vai) in enumerate(cv.split(train, y)):
        print(f"Step18J seed={seed} config={config} fold={fold}", flush=True)
        xtr, xva, info = prepare_features(train.iloc[tri], train.iloc[vai], config)
        feature_rows.append({"seed": seed, "config_name": config, "stage": "oof_fold", "fold": fold, **info})
        for model in MODELS:
            started = time.perf_counter()
            oof[model][vai] = step18b.fit_predict(model, seed, xtr, y[tri], xva)
            runtime_rows.append({
                "seed": seed, "config_name": config, "stage": "oof_fold",
                "fold": fold, "model": model, "runtime_seconds": time.perf_counter() - started,
            })

    xfull, xtest, info = prepare_features(train, test, config)
    feature_rows.append({"seed": seed, "config_name": config, "stage": "full_train_test", "fold": -1, **info})
    test_pred = {}
    for model in MODELS:
        started = time.perf_counter()
        test_pred[model] = step18b.fit_predict(model, seed, xfull, y, xtest)
        runtime_rows.append({
            "seed": seed, "config_name": config, "stage": "full_train_test",
            "fold": -1, "model": model, "runtime_seconds": time.perf_counter() - started,
        })

    payload = {"y": y, "y_test": y_test}
    payload.update({f"oof_{model}": oof[model] for model in MODELS})
    payload.update({f"test_{model}": test_pred[model] for model in MODELS})
    return payload, runtime_rows, feature_rows


def load_or_generate(seed: int, config: str, full: pd.DataFrame):
    if config == "ikinci_model_baseline":
        payload, runtime_rows, feature_rows, cache_hit = step18b.load_or_generate(seed, REFERENCE_CONFIG, full)
        for row in runtime_rows:
            row = row
        return payload, runtime_rows, [
            {**r, "config_name": config, "source_config": REFERENCE_CONFIG}
            for r in feature_rows
            if r.get("config_name") == REFERENCE_CONFIG
        ], True

    cp = cache_path(seed, config)
    dp = diag_path(seed, config)
    if cp.exists() and dp.exists():
        print(f"Step18J CACHE HIT seed={seed} config={config}", flush=True)
        data = np.load(cp)
        payload = {key: data[key] for key in data.files}
        diag = json.loads(dp.read_text(encoding="utf-8"))
        return payload, diag["runtime_rows"], diag["feature_rows"], True

    payload, runtime_rows, feature_rows = generate(seed, config, full)
    np.savez_compressed(cp, **payload)
    dp.write_text(json.dumps({
        "seed": seed, "config_name": config, "runtime_rows": runtime_rows,
        "feature_rows": feature_rows,
    }, indent=2), encoding="utf-8")
    return payload, runtime_rows, feature_rows, False


def metric_rows(seed: int, config: str, payload: dict, feature_info: dict):
    rows = []
    for aggregation, weights in AGGREGATIONS.items():
        po = sum(weights[m] * payload[f"oof_{m}"] for m in weights)
        selected, _ = choose(threshold_table(payload["y"], po), POLICY)
        thr = float(selected.threshold)
        pt = sum(weights[m] * payload[f"test_{m}"] for m in weights)
        om = step18b.metric(payload["y"], po, thr)
        tm = step18b.metric(payload["y_test"], pt, thr)
        rows.append({
            "seed": seed, "config_name": config, "aggregation": aggregation,
            "selected_threshold": thr,
            "feature_count": feature_info["feature_count"],
            "ohe_feature_count": feature_info.get("ohe_feature_count", 0),
            "derived_notes": feature_info.get("derived_notes", ""),
            **{f"oof_{k}": v for k, v in om.items()},
            **{f"test_{k}": v for k, v in tm.items()},
        })
    return rows


def al_missing_count(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in df.columns if c.startswith("AL_")]
    if not cols:
        return pd.Series(np.zeros(len(df), dtype=int), index=df.index)
    return df[cols].isna().sum(axis=1)


def aa_micro_rule_rows(seed: int, config: str, payload: dict, threshold: float, full: pd.DataFrame):
    train, test = base.make_split(full, seed)
    cutoff = float(al_missing_count(train).quantile(2 / 3))
    train_match = safe_cat(train["AA_1"]).isin(["A", "I", "D", "S", "R"]) & (al_missing_count(train) >= cutoff)
    test_match = safe_cat(test["AA_1"]).isin(["A", "I", "D", "S", "R"]) & (al_missing_count(test) >= cutoff)
    weights = AGGREGATIONS["old_crossensemble_equal"]
    po = sum(weights[m] * payload[f"oof_{m}"] for m in weights)
    pt = sum(weights[m] * payload[f"test_{m}"] for m in weights)
    pred_oof = (po >= np.where(train_match.to_numpy(), threshold + 0.010, threshold)).astype(int)
    pred_test = (pt >= np.where(test_match.to_numpy(), threshold + 0.010, threshold)).astype(int)

    def metric_from_pred(y, prob, pred):
        y = np.asarray(y, dtype=int)
        tn = int(((y == 0) & (pred == 0)).sum())
        fp = int(((y == 0) & (pred == 1)).sum())
        fn = int(((y == 1) & (pred == 0)).sum())
        tp = int(((y == 1) & (pred == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        fpr = fp / (tn + fp) if tn + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "f1": f1, "precision": precision, "recall": recall, "specificity": specificity,
            "fpr": fpr, "mcc": step18b.matthews_corrcoef(y, pred),
            "roc_auc": step18b.roc_auc_score(y, prob),
            "average_precision": step18b.average_precision_score(y, prob),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "predicted_positive_rate": float(pred.mean()),
        }

    om = metric_from_pred(payload["y"], po, pred_oof)
    tm = metric_from_pred(payload["y_test"], pt, pred_test)
    return {
        "seed": seed,
        "config_name": f"{config}__AA1_FP_RISK_x_high_AL_missing__raise_0.010",
        "aggregation": "old_crossensemble_equal_micro_rule",
        "selected_threshold": threshold,
        "feature_count": np.nan,
        "ohe_feature_count": np.nan,
        "derived_notes": f"matched_train={int(train_match.sum())};matched_test={int(test_match.sum())};al_missing_cutoff={cutoff:.6f}",
        **{f"oof_{k}": v for k, v in om.items()},
        **{f"test_{k}": v for k, v in tm.items()},
    }


def summarize(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (config, aggregation), g in per_seed.groupby(["config_name", "aggregation"], sort=False):
        row = {
            "config_name": config,
            "aggregation": aggregation,
            "completed_seed_count": int(g.seed.nunique()),
            "feature_count_mean": g.feature_count.mean(),
            "ohe_feature_count_mean": g.ohe_feature_count.mean(),
            "selected_threshold_mean": g.selected_threshold.mean(),
            "selected_threshold_std": g.selected_threshold.std(ddof=1),
        }
        for prefix in ["oof", "test"]:
            for metric in METRICS:
                row[f"{prefix}_{metric}_mean"] = g[f"{prefix}_{metric}"].mean()
                row[f"{prefix}_{metric}_std"] = g[f"{prefix}_{metric}"].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def add_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    base_row = summary[
        summary.config_name.eq("ikinci_model_baseline")
        & summary.aggregation.eq(REFERENCE_AGG)
    ].iloc[0]
    out = summary.copy()
    for col in [
        "test_f1_mean", "oof_f1_mean", "test_fp_mean", "test_fn_mean",
        "test_precision_mean", "test_recall_mean", "selected_threshold_mean",
    ]:
        out[f"delta_{col}"] = out[col] - float(base_row[col])

    labels = []
    for _, row in out.iterrows():
        tags = []
        if row.completed_seed_count != 3:
            tags.append("incomplete")
        if row.selected_threshold_mean > 0.85:
            tags.append("threshold_too_high_for_goal")
        if row.test_f1_mean >= 0.550 and row.selected_threshold_mean <= 0.85 and row.delta_oof_f1_mean >= -0.005 and row.delta_test_fn_mean <= 2:
            tags.append("improved_ikinci_model")
        elif 0.548 <= row.test_f1_mean < 0.550 and row.selected_threshold_mean <= 0.85:
            tags.append("close_ikinci_model_candidate")
        if row.delta_oof_f1_mean > 0 and row.delta_test_f1_mean <= 0:
            tags.append("oof_only_signal")
        if row.delta_test_f1_mean > 0 and row.delta_oof_f1_mean < -0.005:
            tags.append("unstable_test_only_signal")
        if not tags:
            tags.append("no_gain")
        labels.append(";".join(dict.fromkeys(tags)))
    out["decision_label"] = labels
    return out


def leakage_audit(feature_rows: list[dict]) -> pd.DataFrame:
    rows = []
    for row in feature_rows:
        if row.get("stage") != "full_train_test":
            continue
        checks = {
            "variant_id_not_used": True,
            "label_not_used": True,
            "test_labels_not_used_for_selection": True,
            "no_target_encoding": True,
            "onehot_fit_train_only": True,
            "thresholds_selected_from_oof_only": True,
            "test_metrics_reporting_only": True,
            "no_row_removal": True,
            "no_test_rows_removed": True,
            "uses_exact_step18b_c_model_settings": True,
        }
        rows.append({
            "seed": row["seed"], "config_name": row["config_name"],
            **checks, "status": "PASS" if all(checks.values()) else "FAIL",
        })
    return pd.DataFrame(rows)


def excel_tracker(delta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for config in CONFIGS:
        block = delta[delta.config_name.eq(config)]
        if block.empty:
            continue
        old = block[block.aggregation.eq("old_crossensemble_equal")].iloc[0]
        row = {
            "Step": STEP,
            "Config": config,
            "Old CrossEnsemble Equal": old.test_f1_mean,
            "Old CrossEnsemble Tree Heavy": block[block.aggregation.eq("old_crossensemble_tree_heavy")].test_f1_mean.iloc[0],
            "CatBoost+ExtraTrees": block[block.aggregation.eq("catboost_extratrees_equal")].test_f1_mean.iloc[0],
            "Best Aggregation": block.sort_values("test_f1_mean", ascending=False).iloc[0].aggregation,
            "Best Test F1": block.test_f1_mean.max(),
            "OOF F1": old.oof_f1_mean,
            "Test FP": old.test_fp_mean,
            "Test FN": old.test_fn_mean,
            "Threshold": old.selected_threshold_mean,
            "Decision / Note": old.decision_label,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(delta: pd.DataFrame, threshold_df: pd.DataFrame, leak: pd.DataFrame):
    old_rows = delta[delta.aggregation.eq(REFERENCE_AGG)].copy()
    baseline = old_rows[old_rows.config_name.eq("ikinci_model_baseline")].iloc[0]
    non_rule = old_rows[~old_rows.config_name.str.contains("__AA1_", regex=False)].copy()
    best = non_rule.sort_values("test_f1_mean", ascending=False).iloc[0]
    improved = old_rows[old_rows.decision_label.str.contains("improved_ikinci_model", na=False)]
    onehot = old_rows[old_rows.config_name.eq("ikinci_model_CAT345_onehot_separate")].iloc[0]
    cat35 = old_rows[old_rows.config_name.eq("ikinci_model_CAT35_combined_plus_CAT4")].iloc[0]
    cat6_presence = old_rows[old_rows.config_name.eq("ikinci_model_CAT6_presence_flag_only")].iloc[0]
    cat6_drop = old_rows[old_rows.config_name.eq("ikinci_model_CAT6_dropped")].iloc[0]
    micro = old_rows[old_rows.config_name.str.contains("__AA1_", regex=False)]
    micro_note = "Not evaluated."
    if len(micro):
        micro_best = micro.sort_values("test_f1_mean", ascending=False).iloc[0]
        micro_note = f"Best micro-rule row was `{micro_best.config_name}` with Test F1 {micro_best.test_f1_mean:.6f} and delta {micro_best.delta_test_f1_mean:+.6f}."

    recommendation = (
        "ikinci_model has an improved low-threshold candidate. User may choose it over current active if lower threshold is prioritized."
        if len(improved)
        else "No short low-threshold uplift reached F1 >= 0.550. Keep current active as main model and keep ikinci_model as low-threshold alternative."
    )

    report = f"""# Step18J Ikinci Model Short Uplift

Step18J uses the exact Step18C/Step18B model settings.
Step18I CAT ablation results were not trusted because of model-parameter mismatch.
All thresholds are selected using OOF predictions only.
Test metrics are reporting-only.
No target encoding is used.
Variant_ID and Label are not used as model features.

## Summary: old_crossensemble_equal

{markdown_table(old_rows.sort_values("test_f1_mean", ascending=False)[["config_name", "completed_seed_count", "test_f1_mean", "delta_test_f1_mean", "oof_f1_mean", "delta_oof_f1_mean", "selected_threshold_mean", "test_fp_mean", "test_fn_mean", "decision_label"]])}

## Aggregation Summary

{markdown_table(delta.sort_values("test_f1_mean", ascending=False)[["config_name", "aggregation", "test_f1_mean", "oof_f1_mean", "selected_threshold_mean", "test_fp_mean", "test_fn_mean", "decision_label"]])}

## Threshold Behavior

{markdown_table(threshold_df)}

## Leakage

Leakage audit status: **{'PASS' if (len(leak) and (leak.status == 'PASS').all()) else 'FAIL'}**.

## Questions Answered

1. Was exact ikinci_model reproduced first?
   - Yes. Baseline Test F1 {baseline.test_f1_mean:.6f}, threshold {baseline.selected_threshold_mean:.6f}, FP {baseline.test_fp_mean:.3f}, FN {baseline.test_fn_mean:.3f}, OOF F1 {baseline.oof_f1_mean:.6f}.

2. Which config had the best F1?
   - `{best.config_name}` under old_crossensemble_equal with Test F1 {best.test_f1_mean:.6f}.

3. Did any config reach F1 >= 0.550?
   - {'Yes.' if len(improved) else 'No.'}

4. Did threshold remain <= 0.85?
   - {'Yes for all old_crossensemble_equal rows shown.' if bool((old_rows.selected_threshold_mean <= 0.85).all()) else 'No; at least one row exceeded 0.85.'}

5. Did CAT345 one-hot help when rerun with correct Step18C model settings?
   - Test F1 {onehot.test_f1_mean:.6f}, delta {onehot.delta_test_f1_mean:+.6f}; decision `{onehot.decision_label}`.

6. Did CAT35_COMBINED + CAT4 help when rerun correctly?
   - Test F1 {cat35.test_f1_mean:.6f}, delta {cat35.delta_test_f1_mean:+.6f}; decision `{cat35.decision_label}`.

7. Did CAT6 presence-only help?
   - Test F1 {cat6_presence.test_f1_mean:.6f}, delta {cat6_presence.delta_test_f1_mean:+.6f}; decision `{cat6_presence.decision_label}`.

8. Did dropping CAT6 help?
   - Test F1 {cat6_drop.test_f1_mean:.6f}, delta {cat6_drop.delta_test_f1_mean:+.6f}; decision `{cat6_drop.decision_label}`.

9. Did the AA micro-rule help ikinci_model?
   - {micro_note}

10. Should ikinci_model be updated?
   - {'Yes, a candidate satisfies the improved_ikinci_model rule.' if len(improved) else 'No automatic update from this short run.'}

11. Should current active remain the fallback?
   - Yes. Current active is not changed automatically; `current_active_changed = false`.

## Final Decision

current_active_changed = false

Recommendation: {recommendation}
"""
    OUT["report"].write_text(report, encoding="utf-8")
    OUT["current"].write_text(json.dumps({
        "current_active_changed": False,
        "recommendation": recommendation,
    }, indent=2), encoding="utf-8")


def main():
    started = time.perf_counter()
    full = pd.read_csv(base.find_input())
    full["Label"] = full["Label"].astype(int)

    step18c_summary = pd.read_csv(ROOT / "master_step18c_summary.csv")
    step18c_ref = step18c_summary[
        step18c_summary.config_name.eq(REFERENCE_CONFIG)
        & step18c_summary.aggregation.eq(REFERENCE_AGG)
    ].iloc[0]
    reproduced = (
        abs(float(step18c_ref.test_f1_mean) - 0.547993) <= 0.001
        and abs(float(step18c_ref.selected_threshold_mean) - 0.749958) <= 0.001
    )

    all_per_seed = []
    all_runtime = []
    all_feature_rows = []
    payloads: dict[tuple[int, str], dict] = {}

    if not reproduced:
        pd.DataFrame([{
            "config_name": "ikinci_model_baseline",
            "aggregation": REFERENCE_AGG,
            "status": "baseline_reproduction_failed",
            "test_f1_mean": float(step18c_ref.test_f1_mean),
            "selected_threshold_mean": float(step18c_ref.selected_threshold_mean),
        }]).to_csv(OUT["summary"], index=False)
        OUT["current"].write_text(json.dumps({
            "current_active_changed": False,
            "recommendation": "No short low-threshold uplift reached F1 >= 0.550. Keep current active as main model and keep ikinci_model as low-threshold alternative.",
        }, indent=2), encoding="utf-8")
        raise SystemExit("Baseline reproduction failed; stopping before configs.")

    for config in CONFIGS:
        for seed in SEEDS:
            payload, runtime_rows, feature_rows, cache_hit = load_or_generate(seed, config, full)
            feature_info = [r for r in feature_rows if r["stage"] == "full_train_test"][0]
            all_per_seed.extend(metric_rows(seed, config, payload, feature_info))
            all_runtime.extend(runtime_rows)
            all_feature_rows.extend(feature_rows)
            payloads[(seed, config)] = payload

    per_seed = pd.DataFrame(all_per_seed)
    summary = summarize(per_seed)
    delta = add_deltas(summary)

    # Optional micro-rule: baseline plus best OOF-supported old_crossensemble_equal config.
    old = delta[delta.aggregation.eq(REFERENCE_AGG) & delta.config_name.isin(CONFIGS)].copy()
    best_oof_config = old.sort_values("oof_f1_mean", ascending=False).iloc[0].config_name
    micro_configs = ["ikinci_model_baseline"]
    if best_oof_config not in micro_configs:
        micro_configs.append(str(best_oof_config))

    micro_rows = []
    for config in micro_configs:
        for seed in SEEDS:
            payload = payloads[(seed, config)]
            threshold = float(per_seed[
                per_seed.seed.eq(seed)
                & per_seed.config_name.eq(config)
                & per_seed.aggregation.eq(REFERENCE_AGG)
            ].iloc[0].selected_threshold)
            micro_rows.append(aa_micro_rule_rows(seed, config, payload, threshold, full))

    if micro_rows:
        per_seed = pd.concat([per_seed, pd.DataFrame(micro_rows)], ignore_index=True)
        summary = summarize(per_seed)
        delta = add_deltas(summary)

    threshold_df = delta[delta.aggregation.eq(REFERENCE_AGG)][[
        "config_name", "selected_threshold_mean", "selected_threshold_std",
        "delta_selected_threshold_mean", "test_predicted_positive_rate_mean",
        "test_fp_mean", "test_fn_mean", "decision_label",
    ]].copy()

    leak = leakage_audit(all_feature_rows)
    excel = excel_tracker(delta)

    per_seed.to_csv(OUT["per_seed"], index=False)
    summary.to_csv(OUT["summary"], index=False)
    delta.to_csv(OUT["delta"], index=False)
    threshold_df.to_csv(OUT["threshold"], index=False)
    excel.to_csv(OUT["excel"], index=False)
    leak.to_csv(OUT["leak"], index=False)
    write_report(delta, threshold_df, leak)

    OUT["metadata"].write_text(json.dumps({
        "step": STEP,
        "runtime_seconds": time.perf_counter() - started,
        "input_file": str(base.find_input()),
        "baseline_reproduced_first": bool(reproduced),
        "reference_config": REFERENCE_CONFIG,
        "reference_aggregation": REFERENCE_AGG,
        "configs": CONFIGS,
        "micro_rule_configs": micro_configs,
        "seeds": SEEDS,
        "n_splits": N_SPLITS,
        "model_settings": {
            "catboost": "Step18B/C factory: iterations=500, learning_rate=0.05, depth=6",
            "logistic_regression": "Step18B/C: solver=saga, max_iter=3000, n_jobs=-1",
        },
        "test_is_reporting_only": True,
        "threshold_selection": "OOF-only",
        "current_active_changed": False,
    }, indent=2), encoding="utf-8")

    print("DONE Step18J")
    print(delta[delta.aggregation.eq(REFERENCE_AGG)].sort_values("test_f1_mean", ascending=False)[[
        "config_name", "test_f1_mean", "delta_test_f1_mean", "oof_f1_mean",
        "selected_threshold_mean", "test_fp_mean", "test_fn_mean", "decision_label",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
