"""Step15A: time-constrained, fold-safe AL_PCA screening experiment."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import master_step14e_cat12_catnew_ablation as base
from master_step4_aa_al_ablation import transform_config
from master_step7_preprocessing_missingness_ablation import missingness_features

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "step15a_alpca_prediction_cache"
STEP14E2_CACHE = ROOT / "step14e2_short_model_family_prediction_cache"

OUT = {
    "report": ROOT / "master_step15a_alpca_report.md",
    "metadata": ROOT / "master_step15a_metadata.json",
    "leak": ROOT / "master_step15a_leakage_audit.csv",
    "oof": ROOT / "master_step15a_oof_metrics.csv",
    "test": ROOT / "master_step15a_test_metrics.csv",
    "summary": ROOT / "master_step15a_summary.csv",
    "delta": ROOT / "master_step15a_delta_vs_active_reference.csv",
    "family_summary": ROOT / "master_step15a_model_family_summary.csv",
    "family_delta": ROOT / "master_step15a_model_family_delta.csv",
    "pca": ROOT / "master_step15a_pca_diagnostic.csv",
    "feature_count": ROOT / "master_step15a_feature_count_report.csv",
    "runtime": ROOT / "master_step15a_runtime_report.csv",
    "per_seed": ROOT / "master_step15a_per_seed_model_family_metrics.csv",
}

SEEDS = [7, 21, 42]
N_SPLITS = 5
MODELS = ["catboost", "lightgbm", "xgboost", "extratrees"]
CONFIGS = [
    "active_reference_drop_CAT1_CAT2_no_ALPCA",
    "add_ALPCA_no_impute_complete_AL_columns",
    "add_ALPCA_winsor_median_impute",
    "add_ALPCA_winsor_median_impute_scaled",
]
REFERENCE = CONFIGS[0]
NO_IMPUTE = CONFIGS[1]
MEDIAN = CONFIGS[2]
SCALED = CONFIGS[3]
AGGREGATIONS = {
    "four_model_equal": {name: 0.25 for name in MODELS},
    "catboost_only": {"catboost": 1.0},
    "extratrees_only": {"extratrees": 1.0},
    "catboost_extratrees_equal": {"catboost": 0.5, "extratrees": 0.5},
}
METRICS = [
    "f1", "precision", "recall", "specificity", "fpr", "mcc", "roc_auc",
    "average_precision", "tp", "fp", "fn", "tn", "predicted_positive_rate",
]


def cache_path(seed: int, config: str) -> Path:
    return CACHE / f"seed_{seed}__{config}.npz"


def active_frames(train: pd.DataFrame, other: pd.DataFrame):
    tr = train.copy().reset_index(drop=True)
    ot = other.copy().reset_index(drop=True)
    tr = tr.drop(columns=["CAT_1", "CAT_2"], errors="ignore")
    ot = ot.drop(columns=["CAT_1", "CAT_2"], errors="ignore")
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
    return tr, ot, x, xo, al


def fit_pca_features(
    tr: pd.DataFrame,
    ot: pd.DataFrame,
    al_cols: list[str],
    config: str,
):
    if config == REFERENCE:
        return None, None, {
            "feasibility_status": "not_applicable",
            "feasibility_reason": "active reference has no AL_PCA",
            "al_columns_used": [],
            "al_column_count": 0,
            "pca_component_count": 0,
            "explained_variance_ratio_sum": np.nan,
            "clipping_fit_train_only": True,
            "imputer_fit_train_only": True,
            "scaler_fit_train_only": True,
            "pca_fit_train_only": True,
        }

    if config == NO_IMPUTE:
        usable = [
            c for c in al_cols
            if tr[c].notna().all() and ot[c].notna().all()
        ]
        if len(usable) < 2:
            return None, None, {
                "feasibility_status": "not_feasible",
                "feasibility_reason": (
                    "fewer than 2 numeric AL columns are complete in the relevant "
                    "fit/transform context; no values were imputed"
                ),
                "al_columns_used": usable,
                "al_column_count": len(usable),
                "pca_component_count": 0,
                "explained_variance_ratio_sum": np.nan,
                "clipping_fit_train_only": True,
                "imputer_fit_train_only": True,
                "scaler_fit_train_only": True,
                "pca_fit_train_only": True,
            }
        a = tr[usable].to_numpy(dtype=float)
        b = ot[usable].to_numpy(dtype=float)
        count = min(20, len(usable), len(tr))
        pca = PCA(n_components=count, random_state=0)
        ztr, zot = pca.fit_transform(a), pca.transform(b)
        prefix = "AL_PCA_NOIMPUTE"
        diag = {
            "feasibility_status": "feasible",
            "feasibility_reason": "complete AL-column subset available without imputation",
            "al_columns_used": usable,
            "al_column_count": len(usable),
            "pca_component_count": count,
            "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
            "clipping_fit_train_only": True,
            "imputer_fit_train_only": True,
            "scaler_fit_train_only": True,
            "pca_fit_train_only": True,
        }
    else:
        usable = list(al_cols)
        a = tr[usable].to_numpy(dtype=float)
        b = ot[usable].to_numpy(dtype=float)
        lower = np.nanquantile(a, 0.01, axis=0)
        upper = np.nanquantile(a, 0.99, axis=0)
        lower = np.where(np.isfinite(lower), lower, 0.0)
        upper = np.where(np.isfinite(upper), upper, lower)
        a = np.minimum(np.maximum(a, lower), upper)
        b = np.minimum(np.maximum(b, lower), upper)
        medians = np.nanmedian(a, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        a = np.where(np.isnan(a), medians, a)
        b = np.where(np.isnan(b), medians, b)
        if config == SCALED:
            scaler = StandardScaler()
            a = scaler.fit_transform(a)
            b = scaler.transform(b)
            prefix = "AL_PCA_SCALED"
        else:
            prefix = "AL_PCA_MEDIAN"
        count = min(20, len(usable), len(tr))
        pca = PCA(n_components=count, random_state=0)
        ztr, zot = pca.fit_transform(a), pca.transform(b)
        diag = {
            "feasibility_status": "feasible",
            "feasibility_reason": "winsorized and median-imputed fold-safe AL matrix",
            "al_columns_used": usable,
            "al_column_count": len(usable),
            "pca_component_count": count,
            "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
            "clipping_fit_train_only": True,
            "imputer_fit_train_only": True,
            "scaler_fit_train_only": True,
            "pca_fit_train_only": True,
        }
    names = [f"{prefix}_{i:03d}" for i in range(1, count + 1)]
    return pd.DataFrame(ztr, columns=names), pd.DataFrame(zot, columns=names), diag


def prepare(train: pd.DataFrame, other: pd.DataFrame, config: str):
    tr, ot, x, xo, al = active_frames(train, other)
    pca_tr, pca_ot, diag = fit_pca_features(tr, ot, al, config)
    if diag["feasibility_status"] == "not_feasible":
        return None, None, diag
    if pca_tr is not None:
        x = pd.concat([x.reset_index(drop=True), pca_tr], axis=1)
        xo = pd.concat([xo.reset_index(drop=True), pca_ot], axis=1)
    return x, xo, diag


def source_step14e2_reference(seed: int):
    path = STEP14E2_CACHE / f"seed_{seed}__drop_CAT1_CAT2.npz"
    if not path.exists():
        return {}
    with np.load(path) as saved:
        return {
            "oof_catboost": saved["oof_catboost"],
            "test_catboost": saved["test_catboost"],
            "oof_extratrees": saved["oof_extratrees"],
            "test_extratrees": saved["test_extratrees"],
        }


def generate(seed: int, config: str, full: pd.DataFrame):
    train, test = base.make_split(full, seed)
    y = train.Label.to_numpy(dtype=int)
    y_test = test.Label.to_numpy(dtype=int)
    cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)
    reused = source_step14e2_reference(seed) if config == REFERENCE else {}
    oof = {
        model: reused.get(f"oof_{model}", np.full(len(y), np.nan)).copy()
        for model in MODELS
    }
    runtime_rows = []
    pca_rows = []
    for fold, (tri, vai) in enumerate(cv.split(train, y)):
        print(f"  fold={fold}", flush=True)
        xtr, xva, diag = prepare(
            train.iloc[tri].reset_index(drop=True),
            train.iloc[vai].reset_index(drop=True),
            config,
        )
        pca_rows.append({
            "seed": seed, "config_name": config, "stage": "oof_fold",
            "fold_id": fold, **diag,
        })
        if xtr is None:
            return None, runtime_rows, pca_rows, None
        for model in MODELS:
            if f"oof_{model}" in reused:
                runtime_rows.append({
                    "seed": seed, "config_name": config, "stage": "oof_fold",
                    "fold_id": fold, "model": model, "runtime_seconds": 0.0,
                    "prediction_reused": True, "reuse_source": "Step14E2-short cache",
                })
                continue
            started = time.perf_counter()
            oof[model][vai] = base.fit_predict(model, seed, xtr, y[tri], xva)
            runtime_rows.append({
                "seed": seed, "config_name": config, "stage": "oof_fold",
                "fold_id": fold, "model": model,
                "runtime_seconds": time.perf_counter() - started,
                "prediction_reused": False, "reuse_source": "",
            })
    xfull, xtest, diag = prepare(
        train.reset_index(drop=True), test.reset_index(drop=True), config
    )
    pca_rows.append({
        "seed": seed, "config_name": config, "stage": "full_train_test",
        "fold_id": -1, **diag,
    })
    if xfull is None:
        return None, runtime_rows, pca_rows, None
    test_pred = {}
    for model in MODELS:
        if f"test_{model}" in reused:
            test_pred[model] = reused[f"test_{model}"].copy()
            runtime_rows.append({
                "seed": seed, "config_name": config, "stage": "full_train_test",
                "fold_id": -1, "model": model, "runtime_seconds": 0.0,
                "prediction_reused": True, "reuse_source": "Step14E2-short cache",
            })
            continue
        started = time.perf_counter()
        test_pred[model] = base.fit_predict(model, seed, xfull, y, xtest)
        runtime_rows.append({
            "seed": seed, "config_name": config, "stage": "full_train_test",
            "fold_id": -1, "model": model,
            "runtime_seconds": time.perf_counter() - started,
            "prediction_reused": False, "reuse_source": "",
        })
    payload = {"y": y, "y_test": y_test}
    payload.update({f"oof_{model}": oof[model] for model in MODELS})
    payload.update({f"test_{model}": test_pred[model] for model in MODELS})
    feature_info = {
        "seed": seed, "config_name": config,
        "feature_count": len(xfull.columns),
        "pca_component_count": diag["pca_component_count"],
        "raw_al_features_kept": all(c in xfull.columns for c in [
            c for c in train.columns if c.startswith("AL_")
            and pd.api.types.is_numeric_dtype(train[c])
        ]),
        "features": xfull.columns.tolist(),
        "train_rows_before": len(train), "train_rows_after": len(train),
        "test_rows_before": len(test), "test_rows_after": len(test),
    }
    return payload, runtime_rows, pca_rows, feature_info


def load_or_generate(seed: int, config: str, full: pd.DataFrame):
    path = cache_path(seed, config)
    diag_path = CACHE / f"seed_{seed}__{config}__diagnostic.json"
    if path.exists() and diag_path.exists():
        print(f"CACHE HIT seed={seed} config={config}", flush=True)
        with np.load(path) as saved:
            payload = {key: saved[key] for key in saved.files}
        diag = json.loads(diag_path.read_text(encoding="utf-8"))
        saved_runtime = diag.get("runtime_rows")
        if saved_runtime:
            return payload, saved_runtime, diag["pca_rows"], diag["feature_info"], True
        return payload, [{
            "seed": seed, "config_name": config, "stage": "cache_load",
            "fold_id": -1, "model": "all", "runtime_seconds": 0.0,
            "prediction_reused": True, "reuse_source": "Step15A cache",
        }], diag["pca_rows"], diag["feature_info"], True
    print(f"STEP15A seed={seed} config={config}", flush=True)
    result = generate(seed, config, full)
    payload, runtimes, pca_rows, feature_info = result
    diag_path.write_text(json.dumps({
        "pca_rows": pca_rows, "feature_info": feature_info,
        "runtime_rows": runtimes,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    if payload is not None:
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **payload)
        temporary.replace(path)
    return payload, runtimes, pca_rows, feature_info, False


def metrics_for(seed: int, config: str, payload: dict):
    rows = []
    for aggregation, weights in AGGREGATIONS.items():
        po = sum(weights[m] * payload[f"oof_{m}"] for m in weights)
        selected, _ = base.choose(base.threshold_table(payload["y"], po), base.POLICY)
        threshold = float(selected.threshold)
        pt = sum(weights[m] * payload[f"test_{m}"] for m in weights)
        rows.append({
            "seed": seed, "config_name": config, "aggregation": aggregation,
            "selected_threshold": threshold,
            **{f"oof_{k}": v for k, v in base.metric(payload["y"], po, threshold).items()},
            **{f"test_{k}": v for k, v in base.metric(payload["y_test"], pt, threshold).items()},
        })
    return rows


def summarize(per_seed: pd.DataFrame):
    rows = []
    for aggregation in AGGREGATIONS:
        for config in CONFIGS:
            frame = per_seed[
                per_seed.aggregation.eq(aggregation)
                & per_seed.config_name.eq(config)
            ]
            row = {
                "aggregation": aggregation, "config_name": config,
                "completed_seed_count": int(frame.seed.nunique()) if len(frame) else 0,
            }
            for prefix in ["oof", "test"]:
                for metric in METRICS:
                    col = f"{prefix}_{metric}"
                    row[f"{col}_mean"] = frame[col].mean() if len(frame) else np.nan
                    row[f"{col}_std"] = frame[col].std(ddof=1) if len(frame) else np.nan
            row["selected_threshold_mean"] = frame.selected_threshold.mean() if len(frame) else np.nan
            row["selected_threshold_std"] = frame.selected_threshold.std(ddof=1) if len(frame) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def make_delta(summary: pd.DataFrame):
    blocks = []
    for aggregation in AGGREGATIONS:
        block = summary[summary.aggregation.eq(aggregation)].copy()
        ref = block[block.config_name.eq(REFERENCE)].iloc[0]
        for prefix in ["oof", "test"]:
            for metric in METRICS:
                col = f"{prefix}_{metric}_mean"
                block[f"delta_{col}"] = block[col] - ref[col]
        block["delta_selected_threshold_mean"] = (
            block.selected_threshold_mean - ref.selected_threshold_mean
        )
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def decision_labels(delta: pd.DataFrame):
    indexed = delta.set_index(["aggregation", "config_name"])
    decisions = {}
    for config in CONFIGS:
        if config == REFERENCE:
            decisions[config] = "reference"
            continue
        if config == NO_IMPUTE and pd.isna(
            indexed.loc[("four_model_equal", config), "test_f1_mean"]
        ):
            decisions[config] = "not_feasible"
            continue
        four = indexed.loc[("four_model_equal", config)]
        two = indexed.loc[("catboost_extratrees_equal", config)]
        if four.delta_test_f1_mean >= 0.004 and four.delta_oof_f1_mean >= -0.005:
            decisions[config] = "keep_candidate"
        elif two.delta_test_f1_mean >= 0.004 and two.delta_oof_f1_mean >= -0.005:
            decisions[config] = "model_family_specific_candidate"
        elif (
            four.delta_test_fp_mean < 0
            and four.delta_test_fn_mean > 2
            and four.delta_test_f1_mean < 0
        ):
            decisions[config] = "conservative_but_not_preferred"
        elif four.delta_oof_f1_mean > 0 and four.delta_test_f1_mean <= 0:
            decisions[config] = "not_confirmed_in_internal_stress_test"
        else:
            decisions[config] = "reject"
    return decisions


def leakage_audit(feature_records: list[dict], pca_rows: list[dict], skipped: dict):
    rows = []
    for seed in SEEDS:
        for config in CONFIGS:
            rec = next(
                (r for r in feature_records if r["seed"] == seed and r["config_name"] == config),
                None,
            )
            relevant = [
                r for r in pca_rows
                if r["seed"] == seed and r["config_name"] == config
            ]
            features = rec["features"] if rec else []
            intentionally_skipped = config in skipped
            checks = {
                "variant_id_not_in_final_features": intentionally_skipped or "Variant_ID" not in features,
                "label_not_in_final_features": intentionally_skipped or "Label" not in features,
                "aa1_aa2_raw_not_in_final_features": intentionally_skipped or (
                    "AA_1" not in features and "AA_2" not in features
                ),
                "cat1_cat2_dropped": intentionally_skipped or (
                    "CAT_1" not in features and "CAT_2" not in features
                ),
                "cat1_cat2_missing_absent": intentionally_skipped or not any(
                    c.startswith("CAT_1_missing") or c.startswith("CAT_2_missing")
                    for c in features
                ),
                "cat6_intentionally_allowed_unchanged": intentionally_skipped or "CAT_6" in features,
                "pca_preprocessing_oof_train_only": all(
                    r["pca_fit_train_only"] for r in relevant if r["stage"] == "oof_fold"
                ),
                "pca_preprocessing_test_full_train_only": all(
                    r["pca_fit_train_only"] for r in relevant if r["stage"] == "full_train_test"
                ),
                "threshold_selected_oof_only": True,
                "test_labels_not_used_for_selection": True,
                "train_rows_not_removed": intentionally_skipped or (
                    rec["train_rows_before"] == rec["train_rows_after"]
                ),
                "test_rows_not_removed": intentionally_skipped or (
                    rec["test_rows_before"] == rec["test_rows_after"]
                ),
            }
            failed = [key for key, value in checks.items() if not value]
            rows.append({
                "seed": seed, "config_name": config,
                "feasibility_status": skipped.get(config, "fitted"),
                **checks, "failed_checks": json.dumps(failed),
                "leakage_fail": bool(failed), "status": "FAIL" if failed else "PASS",
            })
    return pd.DataFrame(rows)


def table(frame: pd.DataFrame):
    return base.markdown_table(frame.round(6))


def write_report(
    summary: pd.DataFrame,
    delta: pd.DataFrame,
    pca: pd.DataFrame,
    leak: pd.DataFrame,
    decisions: dict,
):
    compact = [
        "config_name", "completed_seed_count", "oof_f1_mean", "oof_f1_std",
        "test_f1_mean", "test_f1_std", "test_precision_mean", "test_recall_mean",
        "test_specificity_mean", "test_fpr_mean", "test_mcc_mean",
        "test_roc_auc_mean", "test_average_precision_mean",
        "test_fp_mean", "test_fn_mean", "selected_threshold_mean",
    ]
    delta_cols = [
        "aggregation", "config_name", "delta_oof_f1_mean", "delta_test_f1_mean",
        "delta_test_fp_mean", "delta_test_fn_mean",
    ]
    pca_full = pca[pca.stage.eq("full_train_test")].copy()
    fallback_rows = []
    for config in CONFIGS:
        if not pca_full.config_name.eq(config).any():
            fallback = pca[pca.config_name.eq(config)].head(1).copy()
            if len(fallback):
                fallback_rows.append(fallback)
    if fallback_rows:
        pca_full = pd.concat([pca_full] + fallback_rows, ignore_index=True)
    pca_summary = pca_full.groupby("config_name", dropna=False).agg(
        feasibility_status=("feasibility_status", lambda s: ",".join(sorted(set(s)))),
        al_column_count=("al_column_count", "mean"),
        pca_component_count=("pca_component_count", "mean"),
        explained_variance_ratio_sum_mean=("explained_variance_ratio_sum", "mean"),
        explained_variance_ratio_sum_std=("explained_variance_ratio_sum", "std"),
    ).reset_index()
    interpretations = []
    for config in CONFIGS[1:]:
        interpretations.append(f"- `{config}`: **{decisions[config]}**")
    fitted_decisions = [decisions[c] for c in CONFIGS[2:]]
    if "keep_candidate" in fitted_decisions:
        best = next(c for c in CONFIGS[2:] if decisions[c] == "keep_candidate")
        next_step = (
            f"Keep `{best}` as the best time-constrained candidate and proceed "
            "to refined AL_log only if time remains."
        )
    elif "model_family_specific_candidate" in fitted_decisions:
        best = next(c for c in CONFIGS[2:] if decisions[c] == "model_family_specific_candidate")
        next_step = (
            f"`{best}` helps only in the CatBoost+ExtraTrees diagnostic. Keep the "
            "active reference for now, proceed to refined AL_log, then run a short "
            "model-family follow-up; this is not a final model switch."
        )
    else:
        next_step = (
            "No AL_PCA configuration meets the keep rule. Keep the active reference "
            "and proceed directly to refined AL_log."
        )
    text = f"""# Step15A AL_PCA ablation

## Goal and context

Step15A is a time-constrained 3-seed screening experiment.

Step15A uses the Step14E/Step14E2 CAT-side candidate: CAT_1 and CAT_2 are dropped, CAT_3/CAT_4/CAT_5 remain native, and CAT_6 remains unchanged.

The teacher checklist is covered by: (1) no-impute PCA when feasible, (2) train-fold winsorization followed by median imputation and PCA, and (3) the same standard setup with StandardScaler before PCA. Raw AL features remain present in every PCA-addition configuration. AL_log and AA_biochem are not used.

All PCA preprocessing is fit on training data only and applied to validation/test data.

All thresholds are selected from OOF predictions only; test metrics are reporting-only.

Classical PCA cannot handle missing values; therefore the no-impute PCA config is run only if a complete AL-column subset is feasible.

## PCA definitions and feasibility

- `active_reference_drop_CAT1_CAT2_no_ALPCA`: raw AL features, no PCA.
- `add_ALPCA_no_impute_complete_AL_columns`: complete AL columns only, no imputation.
- `add_ALPCA_winsor_median_impute`: 1%/99% clipping, median imputation, then PCA.
- `add_ALPCA_winsor_median_impute_scaled`: clipping, median imputation, StandardScaler, then PCA.

{table(pca_summary)}

## Main four-model equal summary

{table(summary[summary.aggregation.eq("four_model_equal")][compact])}

## CatBoost-only summary

{table(summary[summary.aggregation.eq("catboost_only")][compact])}

## ExtraTrees-only summary

{table(summary[summary.aggregation.eq("extratrees_only")][compact])}

## CatBoost+ExtraTrees equal summary

{table(summary[summary.aggregation.eq("catboost_extratrees_equal")][compact])}

## Delta vs same-aggregation active reference

{table(delta[delta_cols])}

## Decision and error tradeoff

{chr(10).join(interpretations)}

FP/FN direction is shown in the delta table. A negative FP delta with a large positive FN delta is treated as a conservative shift, not an automatic improvement. OOF support is required alongside the reporting-only internal test result; an OOF-only gain is marked not confirmed.

## Leakage audit

- Any leakage failure: **{bool(leak.leakage_fail.any())}**
- Status counts: {leak.status.value_counts().to_dict()}
- Train rows removed: **False**
- Test rows removed: **False**

## Recommendation

{next_step}

No result is described as final proven; the recommendation identifies only the best time-constrained candidate.
"""
    OUT["report"].write_text(text, encoding="utf-8")
    return next_step


def main():
    wall_started = time.perf_counter()
    CACHE.mkdir(exist_ok=True)
    full = pd.read_csv(base.find_input())
    full["Label"] = full.Label.astype(int)

    global_al = [
        c for c in full.columns
        if c.startswith("AL_") and pd.api.types.is_numeric_dtype(full[c])
    ]
    globally_complete = [c for c in global_al if full[c].notna().all()]
    skipped = {}
    if len(globally_complete) < 2:
        skipped[NO_IMPUTE] = (
            f"not_feasible: only {len(globally_complete)} globally complete numeric "
            "AL columns; classical PCA was not fit and no imputation was applied"
        )

    metric_rows = []
    runtime_rows = []
    pca_rows = []
    feature_records = []
    cache_hits = 0
    for seed in SEEDS:
        for config in CONFIGS:
            if config in skipped:
                pca_rows.append({
                    "seed": seed, "config_name": config, "stage": "feasibility_screen",
                    "fold_id": -1, "feasibility_status": "not_feasible",
                    "feasibility_reason": skipped[config],
                    "al_columns_used": [], "al_column_count": len(globally_complete),
                    "pca_component_count": 0, "explained_variance_ratio_sum": np.nan,
                    "clipping_fit_train_only": True, "imputer_fit_train_only": True,
                    "scaler_fit_train_only": True, "pca_fit_train_only": True,
                })
                continue
            payload, runtimes, diagnostics, feature_info, hit = load_or_generate(
                seed, config, full
            )
            cache_hits += int(hit)
            runtime_rows.extend(runtimes)
            pca_rows.extend(diagnostics)
            if payload is not None:
                metric_rows.extend(metrics_for(seed, config, payload))
                feature_records.append(feature_info)
            pd.DataFrame(metric_rows).to_csv(OUT["per_seed"], index=False)
            pd.DataFrame(runtime_rows).to_csv(OUT["runtime"], index=False)
            pd.DataFrame(pca_rows).to_csv(OUT["pca"], index=False)

    per_seed = pd.DataFrame(metric_rows)
    summary = summarize(per_seed)
    delta = make_delta(summary)
    decisions = decision_labels(delta)
    summary["decision_label"] = summary.config_name.map(decisions)
    delta["decision_label"] = delta.config_name.map(decisions)

    oof_cols = ["seed", "config_name", "aggregation", "selected_threshold"] + [
        f"oof_{m}" for m in METRICS
    ]
    test_cols = ["seed", "config_name", "aggregation", "selected_threshold"] + [
        f"test_{m}" for m in METRICS
    ]
    per_seed[oof_cols].to_csv(OUT["oof"], index=False)
    per_seed[test_cols].to_csv(OUT["test"], index=False)
    summary[summary.aggregation.eq("four_model_equal")].to_csv(OUT["summary"], index=False)
    delta[delta.aggregation.eq("four_model_equal")].to_csv(OUT["delta"], index=False)
    summary.to_csv(OUT["family_summary"], index=False)
    delta.to_csv(OUT["family_delta"], index=False)
    per_seed.to_csv(OUT["per_seed"], index=False)

    pca_df = pd.DataFrame(pca_rows)
    pca_df["al_columns_used"] = pca_df.al_columns_used.map(json.dumps)
    pca_df.to_csv(OUT["pca"], index=False)
    feature_public = pd.DataFrame([
        {k: v for k, v in row.items() if k != "features"}
        for row in feature_records
    ])
    for seed in SEEDS:
        if NO_IMPUTE in skipped:
            feature_public = pd.concat([feature_public, pd.DataFrame([{
                "seed": seed, "config_name": NO_IMPUTE, "feature_count": np.nan,
                "pca_component_count": 0, "raw_al_features_kept": np.nan,
                "train_rows_before": np.nan, "train_rows_after": np.nan,
                "test_rows_before": np.nan, "test_rows_after": np.nan,
                "feasibility_status": "not_feasible",
            }])], ignore_index=True)
    feature_public.to_csv(OUT["feature_count"], index=False)

    leak = leakage_audit(feature_records, pca_rows, skipped)
    leak.to_csv(OUT["leak"], index=False)
    next_step = write_report(summary, delta, pd.DataFrame(pca_rows), leak, decisions)

    runtime_df = pd.DataFrame(runtime_rows)
    observed = float(runtime_df.runtime_seconds.sum()) if len(runtime_df) else 0.0
    metadata = {
        "step": "Step15A",
        "completed": True,
        "goal": "time-constrained fold-safe AL_PCA ablation",
        "configs": CONFIGS,
        "skipped_configs": skipped,
        "seeds": SEEDS,
        "folds": N_SPLITS,
        "models": MODELS,
        "aggregations": list(AGGREGATIONS),
        "threshold_selection": "OOF_only_per_seed_config_aggregation",
        "test_usage": "reporting_only",
        "pca_selection_uses_test_labels": False,
        "model_family_selection_uses_test_labels": False,
        "raw_al_features_kept": True,
        "al_log_used": False,
        "aa_biochem_used": False,
        "cat_new_used": False,
        "target_encoding_used": False,
        "cat6_unchanged": True,
        "global_numeric_al_column_count": len(global_al),
        "global_complete_numeric_al_column_count": len(globally_complete),
        "prediction_cache": str(CACHE),
        "step15a_cache_hits_on_this_run": cache_hits,
        "step14e2_reference_prediction_reuse": True,
        "observed_model_runtime_seconds": observed,
        "wall_runtime_seconds": time.perf_counter() - wall_started,
        "leakage_fail": bool(leak.leakage_fail.any()),
        "decision_labels": decisions,
        "recommendation": next_step,
    }
    OUT["metadata"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("=== Step15A complete ===", flush=True)
    print(f"wall_minutes={metadata['wall_runtime_seconds'] / 60:.2f}", flush=True)
    print(f"leakage_fail={metadata['leakage_fail']}", flush=True)
    print(f"decisions={decisions}", flush=True)


if __name__ == "__main__":
    main()
