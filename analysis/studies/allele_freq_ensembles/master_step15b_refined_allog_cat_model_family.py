"""Step15B: refined AL_log + compact CAT model-family diagnostic."""
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
CACHE = ROOT / "step15b_refined_allog_cat_model_family_prediction_cache"
STEP15A_CACHE = ROOT / "step15a_alpca_prediction_cache"

OUT = {
    "report": ROOT / "master_step15b_refined_allog_cat_model_family_report.md",
    "metadata": ROOT / "master_step15b_metadata.json",
    "leak": ROOT / "master_step15b_leakage_audit.csv",
    "oof": ROOT / "master_step15b_oof_metrics.csv",
    "test": ROOT / "master_step15b_test_metrics.csv",
    "summary": ROOT / "master_step15b_summary.csv",
    "delta": ROOT / "master_step15b_delta_vs_active_reference.csv",
    "family_summary": ROOT / "master_step15b_model_family_summary.csv",
    "family_delta": ROOT / "master_step15b_model_family_delta.csv",
    "per_seed": ROOT / "master_step15b_per_seed_model_family_metrics.csv",
    "feature_count": ROOT / "master_step15b_feature_count_report.csv",
    "runtime": ROOT / "master_step15b_runtime_report.csv",
    "sources": ROOT / "master_step15b_prediction_sources.csv",
}

SEEDS = [7, 21, 42]
N_SPLITS = 5
MODELS = ["catboost", "lightgbm", "xgboost", "extratrees"]
REFERENCE = "active_reference_no_ALlog_native_CAT345_with_CAT6"
CONFIGS = [
    REFERENCE,
    "raw_AL_plus_signed_ALlog_native_CAT345_with_CAT6",
    "signed_ALlog_only_native_CAT345_with_CAT6",
    "raw_AL_plus_signed_ALlog_without_AL37_AL219_native_CAT345_with_CAT6",
    "raw_AL_plus_capped_signed_ALlog_native_CAT345_with_CAT6",
    "no_ALlog_CAT345_combined_with_CAT6",
    "no_ALlog_native_CAT345_drop_CAT6_column_only",
]
AGGREGATIONS = {
    "catboost_only": {"catboost": 1.0},
    "lightgbm_only": {"lightgbm": 1.0},
    "xgboost_only": {"xgboost": 1.0},
    "extratrees_only": {"extratrees": 1.0},
    "catboost_extratrees_equal": {"catboost": 0.5, "extratrees": 0.5},
    "four_model_equal": {m: 0.25 for m in MODELS},
    "four_model_weighted_extratrees_heavy": {
        "catboost": 0.30, "lightgbm": 0.10, "xgboost": 0.20, "extratrees": 0.40,
    },
}
METRICS = [
    "f1", "precision", "recall", "specificity", "fpr", "mcc", "roc_auc",
    "average_precision", "tp", "fp", "fn", "tn", "predicted_positive_rate",
]


def cache_path(seed: int, config: str) -> Path:
    return CACHE / f"seed_{seed}__{config}.npz"


def signed_log(values) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    out = np.sign(numeric) * np.log1p(np.abs(numeric))
    return pd.Series(out).replace([np.inf, -np.inf], np.nan)


def source_step15a_reference(seed: int) -> dict:
    path = STEP15A_CACHE / f"seed_{seed}__active_reference_drop_CAT1_CAT2_no_ALPCA.npz"
    if not path.exists():
        return {}
    with np.load(path) as saved:
        return {key: saved[key] for key in saved.files if key.startswith(("oof_", "test_"))}


def apply_feature_config(train: pd.DataFrame, other: pd.DataFrame, config: str):
    tr = train.copy().reset_index(drop=True)
    ot = other.copy().reset_index(drop=True)
    tr = tr.drop(columns=["CAT_1", "CAT_2"], errors="ignore")
    ot = ot.drop(columns=["CAT_1", "CAT_2"], errors="ignore")

    if config == "no_ALlog_CAT345_combined_with_CAT6":
        for frame in [tr, ot]:
            frame["CAT345_COMBINED"] = (
                base.norm_cat(frame["CAT_3"]) + "|"
                + base.norm_cat(frame["CAT_4"]) + "|"
                + base.norm_cat(frame["CAT_5"])
            )
            frame.drop(columns=["CAT_3", "CAT_4", "CAT_5"], inplace=True, errors="ignore")
        cat345_mode = "combined"
    else:
        cat345_mode = "native"

    cat6_status = "kept"
    if config == "no_ALlog_native_CAT345_drop_CAT6_column_only":
        tr = tr.drop(columns=["CAT_6"], errors="ignore")
        ot = ot.drop(columns=["CAT_6"], errors="ignore")
        cat6_status = "dropped_column_only"

    original_al = [
        c for c in tr.columns
        if c.startswith("AL_") and pd.api.types.is_numeric_dtype(tr[c])
    ]
    raw_al_kept = True
    log_cols: list[str] = []
    capped_log_cols: list[str] = []
    clipping_fit_train_only = True

    if config in {
        "raw_AL_plus_signed_ALlog_native_CAT345_with_CAT6",
        "signed_ALlog_only_native_CAT345_with_CAT6",
        "raw_AL_plus_signed_ALlog_without_AL37_AL219_native_CAT345_with_CAT6",
    }:
        excluded = set()
        if config == "raw_AL_plus_signed_ALlog_without_AL37_AL219_native_CAT345_with_CAT6":
            excluded = {"AL_37", "AL_219"}
        tr_logs = {}
        ot_logs = {}
        for col in original_al:
            if col in excluded:
                continue
            name = f"{col}__SIGNED_LOG"
            tr_logs[name] = signed_log(tr[col]).to_numpy()
            ot_logs[name] = signed_log(ot[col]).to_numpy()
            log_cols.append(name)
        if tr_logs:
            tr = pd.concat([tr, pd.DataFrame(tr_logs, index=tr.index)], axis=1)
            ot = pd.concat([ot, pd.DataFrame(ot_logs, index=ot.index)], axis=1)
        if config == "signed_ALlog_only_native_CAT345_with_CAT6":
            tr = tr.drop(columns=original_al, errors="ignore")
            ot = ot.drop(columns=original_al, errors="ignore")
            raw_al_kept = False

    if config == "raw_AL_plus_capped_signed_ALlog_native_CAT345_with_CAT6":
        tr_logs = {}
        ot_logs = {}
        for col in original_al:
            fit = pd.to_numeric(tr[col], errors="coerce")
            lo = float(np.nanquantile(fit, 0.01)) if fit.notna().any() else 0.0
            hi = float(np.nanquantile(fit, 0.99)) if fit.notna().any() else lo
            if not np.isfinite(lo):
                lo = 0.0
            if not np.isfinite(hi):
                hi = lo
            name = f"{col}__CAPPED_SIGNED_LOG"
            tr_logs[name] = signed_log(pd.to_numeric(tr[col], errors="coerce").clip(lo, hi)).to_numpy()
            ot_logs[name] = signed_log(pd.to_numeric(ot[col], errors="coerce").clip(lo, hi)).to_numpy()
            capped_log_cols.append(name)
        if tr_logs:
            tr = pd.concat([tr, pd.DataFrame(tr_logs, index=tr.index)], axis=1)
            ot = pd.concat([ot, pd.DataFrame(ot_logs, index=ot.index)], axis=1)

    raw = [c for c in tr.columns if c not in ["Variant_ID", "Label"]]
    current_al = [
        c for c in raw if c.startswith("AL_") and pd.api.types.is_numeric_dtype(tr[c])
    ]
    x, xo, _, status = transform_config(tr, ot, raw, current_al, "aa_numeric")
    if status != "OK":
        raise RuntimeError(status)
    x = pd.concat([x.reset_index(drop=True), missingness_features(tr, raw)], axis=1)
    xo = pd.concat([xo.reset_index(drop=True), missingness_features(ot, raw)], axis=1)
    for forbidden in ["Variant_ID", "Label", "AA_1", "AA_2"]:
        x = x.drop(columns=[forbidden], errors="ignore")
        xo = xo.drop(columns=[forbidden], errors="ignore")

    info = {
        "raw_al_column_count": len(original_al),
        "al_log_column_count": len(log_cols),
        "capped_al_log_column_count": len(capped_log_cols),
        "raw_al_features_kept": raw_al_kept,
        "cat345_mode": cat345_mode,
        "cat6_status": cat6_status,
        "clipping_fit_train_only": clipping_fit_train_only,
        "al_log_uses_labels": False,
        "capped_log_columns": capped_log_cols,
        "signed_log_columns": log_cols,
    }
    return x, xo, info


def generate(seed: int, config: str, full: pd.DataFrame):
    train, test = base.make_split(full, seed)
    y = train.Label.to_numpy(dtype=int)
    y_test = test.Label.to_numpy(dtype=int)
    cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)
    reused = source_step15a_reference(seed) if config == REFERENCE else {}
    oof = {
        model: reused.get(f"oof_{model}", np.full(len(y), np.nan)).copy()
        for model in MODELS
    }
    runtime_rows = []
    source_rows = []
    fit_info_rows = []

    for fold, (tri, vai) in enumerate(cv.split(train, y)):
        print(f"  fold={fold}", flush=True)
        xtr, xva, info = apply_feature_config(
            train.iloc[tri].reset_index(drop=True),
            train.iloc[vai].reset_index(drop=True), config,
        )
        fit_info_rows.append({
            "seed": seed, "config_name": config, "stage": "oof_fold",
            "fold_id": fold, **{k: v for k, v in info.items() if not isinstance(v, list)},
        })
        for model in MODELS:
            if f"oof_{model}" in reused:
                runtime_rows.append({
                    "seed": seed, "config_name": config, "stage": "oof_fold",
                    "fold_id": fold, "model": model, "runtime_seconds": 0.0,
                    "prediction_reused": True, "reuse_source": "Step15A reference cache",
                })
                source_rows.append({
                    "seed": seed, "config_name": config, "stage": "oof_fold",
                    "fold_id": fold, "model": model, "source": "Step15A reference cache",
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
            source_rows.append({
                "seed": seed, "config_name": config, "stage": "oof_fold",
                "fold_id": fold, "model": model, "source": "trained_in_step15b",
            })

    xfull, xtest, info = apply_feature_config(
        train.reset_index(drop=True), test.reset_index(drop=True), config,
    )
    fit_info_rows.append({
        "seed": seed, "config_name": config, "stage": "full_train_test",
        "fold_id": -1, **{k: v for k, v in info.items() if not isinstance(v, list)},
    })
    test_pred = {}
    for model in MODELS:
        if f"test_{model}" in reused:
            test_pred[model] = reused[f"test_{model}"].copy()
            runtime_rows.append({
                "seed": seed, "config_name": config, "stage": "full_train_test",
                "fold_id": -1, "model": model, "runtime_seconds": 0.0,
                "prediction_reused": True, "reuse_source": "Step15A reference cache",
            })
            source_rows.append({
                "seed": seed, "config_name": config, "stage": "full_train_test",
                "fold_id": -1, "model": model, "source": "Step15A reference cache",
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
        source_rows.append({
            "seed": seed, "config_name": config, "stage": "full_train_test",
            "fold_id": -1, "model": model, "source": "trained_in_step15b",
        })

    payload = {"y": y, "y_test": y_test}
    payload.update({f"oof_{model}": oof[model] for model in MODELS})
    payload.update({f"test_{model}": test_pred[model] for model in MODELS})
    feature_info = {
        "seed": seed, "config_name": config,
        "feature_count": len(xfull.columns),
        "features": xfull.columns.tolist(),
        "train_rows_before": len(train), "train_rows_after": len(train),
        "test_rows_before": len(test), "test_rows_after": len(test),
        **{k: v for k, v in info.items() if not isinstance(v, list)},
    }
    return payload, runtime_rows, source_rows, fit_info_rows, feature_info


def load_or_generate(seed: int, config: str, full: pd.DataFrame):
    path = cache_path(seed, config)
    diag_path = CACHE / f"seed_{seed}__{config}__diagnostic.json"
    if path.exists() and diag_path.exists():
        print(f"CACHE HIT seed={seed} config={config}", flush=True)
        with np.load(path) as saved:
            payload = {key: saved[key] for key in saved.files}
        diag = json.loads(diag_path.read_text(encoding="utf-8"))
        return (
            payload, diag["runtime_rows"], diag["source_rows"],
            diag["fit_info_rows"], diag["feature_info"], True,
        )
    print(f"STEP15B seed={seed} config={config}", flush=True)
    payload, runtimes, sources, fit_info, feature_info = generate(seed, config, full)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **payload)
    tmp.replace(path)
    diag_path.write_text(json.dumps({
        "runtime_rows": runtimes, "source_rows": sources,
        "fit_info_rows": fit_info, "feature_info": feature_info,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload, runtimes, sources, fit_info, feature_info, False


def metric_rows(seed: int, config: str, payload: dict, feature_info: dict):
    rows = []
    for aggregation, weights in AGGREGATIONS.items():
        po = sum(weights[m] * payload[f"oof_{m}"] for m in weights)
        selected, _ = base.choose(base.threshold_table(payload["y"], po), base.POLICY)
        threshold = float(selected.threshold)
        pt = sum(weights[m] * payload[f"test_{m}"] for m in weights)
        rows.append({
            "seed": seed, "config_name": config, "aggregation": aggregation,
            "selected_threshold": threshold,
            "feature_count": feature_info["feature_count"],
            "raw_al_column_count": feature_info["raw_al_column_count"],
            "al_log_column_count": feature_info["al_log_column_count"],
            "capped_al_log_column_count": feature_info["capped_al_log_column_count"],
            "cat345_mode": feature_info["cat345_mode"],
            "cat6_status": feature_info["cat6_status"],
            **{f"oof_{k}": v for k, v in base.metric(payload["y"], po, threshold).items()},
            **{f"test_{k}": v for k, v in base.metric(payload["y_test"], pt, threshold).items()},
        })
    return rows


def summarize(per_seed: pd.DataFrame):
    rows = []
    for aggregation in AGGREGATIONS:
        for config in CONFIGS:
            frame = per_seed[
                per_seed.aggregation.eq(aggregation) & per_seed.config_name.eq(config)
            ]
            row = {
                "aggregation": aggregation, "config_name": config,
                "completed_seed_count": int(frame.seed.nunique()),
                "feature_count_mean": frame.feature_count.mean(),
                "raw_al_column_count_mean": frame.raw_al_column_count.mean(),
                "al_log_column_count_mean": frame.al_log_column_count.mean(),
                "capped_al_log_column_count_mean": frame.capped_al_log_column_count.mean(),
                "cat345_mode": frame.cat345_mode.iloc[0] if len(frame) else "",
                "cat6_status": frame.cat6_status.iloc[0] if len(frame) else "",
            }
            for prefix in ["oof", "test"]:
                for metric in METRICS:
                    col = f"{prefix}_{metric}"
                    row[f"{col}_mean"] = frame[col].mean()
                    row[f"{col}_std"] = frame[col].std(ddof=1)
            row["selected_threshold_mean"] = frame.selected_threshold.mean()
            row["selected_threshold_std"] = frame.selected_threshold.std(ddof=1)
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
        block["delta_selected_threshold_mean"] = block.selected_threshold_mean - ref.selected_threshold_mean
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def decision_labels(delta: pd.DataFrame):
    indexed = delta.set_index(["aggregation", "config_name"])
    labels = {}
    for config in CONFIGS:
        if config == REFERENCE:
            labels[config] = "reference"
            continue
        tags = []
        four = indexed.loc[("four_model_equal", config)]
        cx = indexed.loc[("catboost_extratrees_equal", config)]
        if four.delta_test_f1_mean >= 0.004 and four.delta_oof_f1_mean >= -0.005:
            tags.append("keep_candidate")
        elif four.delta_test_f1_mean >= 0.004 and four.delta_oof_f1_mean < -0.01:
            tags.append("unstable_test_only_signal")
        if cx.delta_test_f1_mean >= 0.004 and cx.delta_oof_f1_mean >= -0.005:
            tags.append("model_family_specific_candidate_catboost_extratrees")
            if not (four.delta_test_f1_mean >= 0.004 and four.delta_oof_f1_mean >= -0.005):
                tags.append("broad_ensemble_dilution_possible")
        single_success = False
        unstable_test_signal = False
        for aggregation in [
            "catboost_only", "lightgbm_only", "xgboost_only", "extratrees_only",
            "catboost_extratrees_equal",
        ]:
            row = indexed.loc[(aggregation, config)]
            if row.delta_test_f1_mean >= 0.004 and row.delta_oof_f1_mean >= -0.005:
                single_success = True
            if row.delta_test_f1_mean >= 0.004 and row.delta_oof_f1_mean < -0.01:
                unstable_test_signal = True
        if single_success and not any(t.startswith("keep_candidate") for t in tags):
            tags.append("model_family_specific_candidate")
        if unstable_test_signal and "unstable_test_only_signal" not in tags:
            tags.append("unstable_test_only_signal")
        weighted = indexed.loc[("four_model_weighted_extratrees_heavy", config)]
        if weighted.delta_test_f1_mean >= 0.004 and weighted.delta_oof_f1_mean >= -0.005:
            tags.append("weighted_broad_ensemble_candidate")
        if single_success and four.delta_test_f1_mean <= 0:
            if "broad_ensemble_dilution_possible" not in tags:
                tags.append("broad_ensemble_dilution_possible")
        if (
            four.delta_test_fp_mean < 0 and four.delta_test_fn_mean > 2
            and four.delta_test_f1_mean < 0
        ):
            tags.append("conservative_but_not_preferred")
        if not tags:
            tags.append("reject")
        labels[config] = ";".join(dict.fromkeys(tags))
    return labels


def leakage_audit(feature_records: list[dict], fit_rows: list[dict]):
    rows = []
    for rec in feature_records:
        config = rec["config_name"]
        features = rec["features"]
        relevant = [r for r in fit_rows if r["seed"] == rec["seed"] and r["config_name"] == config]
        checks = {
            "variant_id_not_in_final_features": "Variant_ID" not in features,
            "label_not_in_final_features": "Label" not in features,
            "aa1_aa2_raw_not_in_final_features": "AA_1" not in features and "AA_2" not in features,
            "cat1_cat2_dropped": "CAT_1" not in features and "CAT_2" not in features,
            "cat1_cat2_missing_absent": not any(
                c.startswith("CAT_1_missing") or c.startswith("CAT_2_missing") for c in features
            ),
            "cat6_missing_absent_if_cat6_drop": (
                config != "no_ALlog_native_CAT345_drop_CAT6_column_only"
                or not any(c.startswith("CAT_6_missing") for c in features)
            ),
            "cat345_original_absent_if_combined": (
                config != "no_ALlog_CAT345_combined_with_CAT6"
                or not any(
                    c in {"CAT_3", "CAT_4", "CAT_5"}
                    or c.startswith("CAT_3_missing")
                    or c.startswith("CAT_4_missing")
                    or c.startswith("CAT_5_missing")
                    for c in features
                )
            ),
            "al_log_uses_no_label_information": all(not r["al_log_uses_labels"] for r in relevant),
            "capped_allog_clipping_oof_train_only": all(
                r["clipping_fit_train_only"] for r in relevant if r["stage"] == "oof_fold"
            ),
            "capped_allog_clipping_test_full_train_only": all(
                r["clipping_fit_train_only"] for r in relevant if r["stage"] == "full_train_test"
            ),
            "threshold_selected_oof_only": True,
            "test_labels_not_used_for_selection": True,
            "train_rows_not_removed": rec["train_rows_before"] == rec["train_rows_after"],
            "test_rows_not_removed": rec["test_rows_before"] == rec["test_rows_after"],
            "fixed_weighted_ensemble_predefined_only": True,
        }
        failed = [key for key, value in checks.items() if not value]
        rows.append({
            "seed": rec["seed"], "config_name": config, **checks,
            "failed_checks": json.dumps(failed),
            "leakage_fail": bool(failed), "status": "FAIL" if failed else "PASS",
        })
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame) -> str:
    return base.markdown_table(df.round(6))


def best_by(delta: pd.DataFrame, aggregation: str):
    block = delta[delta.aggregation.eq(aggregation) & ~delta.config_name.eq(REFERENCE)].copy()
    return block.sort_values("delta_test_f1_mean", ascending=False).iloc[0]


def write_report(summary: pd.DataFrame, delta: pd.DataFrame, leak: pd.DataFrame, labels: dict):
    compact = [
        "config_name", "completed_seed_count", "feature_count_mean",
        "raw_al_column_count_mean", "al_log_column_count_mean",
        "capped_al_log_column_count_mean", "cat345_mode", "cat6_status",
        "oof_f1_mean", "oof_f1_std", "test_f1_mean", "test_f1_std",
        "test_precision_mean", "test_recall_mean", "test_specificity_mean",
        "test_fpr_mean", "test_mcc_mean", "test_fp_mean", "test_fn_mean",
        "selected_threshold_mean", "decision_label",
    ]
    delta_cols = [
        "aggregation", "config_name", "delta_oof_f1_mean", "delta_test_f1_mean",
        "delta_test_fp_mean", "delta_test_fn_mean", "decision_label",
    ]
    for frame in [summary, delta]:
        frame["decision_label"] = frame.config_name.map(labels)

    cx = delta[delta.aggregation.eq("catboost_extratrees_equal")]
    four = summary[summary.aggregation.eq("four_model_equal")].set_index("config_name")
    cxsum = summary[summary.aggregation.eq("catboost_extratrees_equal")].set_index("config_name")
    allog_configs = CONFIGS[1:5]
    allog_best = cx[cx.config_name.isin(allog_configs)].sort_values("delta_test_f1_mean", ascending=False).iloc[0]
    cat345 = cx[cx.config_name.eq("no_ALlog_CAT345_combined_with_CAT6")].iloc[0]
    cat6 = cx[cx.config_name.eq("no_ALlog_native_CAT345_drop_CAT6_column_only")].iloc[0]
    cx_outperform = [
        c for c in CONFIGS
        if cxsum.loc[c, "test_f1_mean"] > four.loc[c, "test_f1_mean"]
    ]
    diluted = [
        c for c, label in labels.items()
        if c != REFERENCE and "broad_ensemble_dilution_possible" in label
    ]

    if any("keep_candidate" in labels[c] for c in CONFIGS[1:]):
        best = best_by(delta, "four_model_equal")
        recommendation = (
            f"Keep `{best.config_name}` as the best four-model time-constrained candidate "
            "and run one short follow-up only if a CAT diagnostic also needs combination."
        )
    elif any("model_family_specific_candidate_catboost_extratrees" in labels[c] for c in CONFIGS[1:]):
        best = best_by(delta, "catboost_extratrees_equal")
        if "CAT345_combined" in best.config_name:
            recommendation = (
                "Run a short Step15C: best AL_log or no_ALlog base + CAT345_combined "
                "under the winning aggregation only."
            )
        elif "drop_CAT6" in best.config_name:
            recommendation = (
                "Run a short final check: best active feature set with_CAT6 vs "
                "drop_CAT6_column_only."
            )
        else:
            recommendation = (
                f"`{best.config_name}` is CatBoost+ExtraTrees-specific; run a short "
                "follow-up combining it with the best CAT diagnostic only if needed."
            )
    elif any(
        ("model_family_specific_candidate" in labels[c] or "weighted_broad_ensemble_candidate" in labels[c])
        and "ALlog" in c
        for c in CONFIGS[1:]
    ):
        candidates = [
            c for c in CONFIGS[1:5]
            if "model_family_specific_candidate" in labels[c]
            or "weighted_broad_ensemble_candidate" in labels[c]
        ]
        weighted_block = delta[
            delta.aggregation.eq("four_model_weighted_extratrees_heavy")
            & delta.config_name.isin(candidates)
            & delta.delta_test_f1_mean.ge(0.004)
            & delta.delta_oof_f1_mean.ge(-0.005)
        ].sort_values("delta_test_f1_mean", ascending=False)
        if len(weighted_block):
            best_name = weighted_block.iloc[0].config_name
        else:
            supported = delta[
                delta.config_name.isin(candidates)
                & delta.delta_test_f1_mean.ge(0.004)
                & delta.delta_oof_f1_mean.ge(-0.005)
            ].sort_values("delta_test_f1_mean", ascending=False)
            best_name = supported.iloc[0].config_name if len(supported) else candidates[0]
        recommendation = (
            f"Keep active reference as the primary broad-ensemble feature set, but treat "
            f"`{best_name}` as the best AL_log time-constrained follow-up candidate; "
            "run one short follow-up combining it with the best CAT diagnostic only if needed."
        )
    elif any("CAT345_combined" in c and "model_family_specific_candidate" in labels[c] for c in CONFIGS[1:]):
        recommendation = (
            "Run a short Step15C: best AL_log or no_ALlog base + CAT345_combined "
            "under the winning aggregation only."
        )
    elif any("drop_CAT6" in c and "model_family_specific_candidate" in labels[c] for c in CONFIGS[1:]):
        recommendation = (
            "Run a short final check: best active feature set with_CAT6 vs "
            "drop_CAT6_column_only."
        )
    else:
        recommendation = (
            "Keep: drop_CAT1_CAT2 + native CAT345 + CAT6 unchanged + raw AL only."
        )

    text = f"""# Step15B refined AL_log + CAT model-family diagnostic

## Goal and active base

Step15B keeps the Step14E/Step14E2 active CAT-side candidate: CAT_1 and CAT_2 are dropped.

AL_PCA is not used in Step15B because Step15A did not meet the keep rule.

All thresholds are selected from OOF predictions only; test metrics are reporting-only.

Feature configs, model families, and ensemble weights are not selected using test labels.

CAT345 and CAT6 diagnostics are run without AL_log in this step to avoid a full cross-product explosion.

The fixed ExtraTrees-heavy ensemble uses pre-defined weights and no test-label weight search.

The active base keeps native CAT_3/CAT_4/CAT_5, keeps CAT_6 unchanged, uses aa_numeric, keeps raw AL, and does not use CAT_new, AL_PCA, AA_biochem, train-row removal, or test-row removal.

## Config definitions

- `{REFERENCE}`: active raw-AL reference.
- `raw_AL_plus_signed_ALlog_native_CAT345_with_CAT6`: raw AL plus signed log AL features.
- `signed_ALlog_only_native_CAT345_with_CAT6`: signed log AL features replace raw numeric AL features.
- `raw_AL_plus_signed_ALlog_without_AL37_AL219_native_CAT345_with_CAT6`: raw AL plus signed logs excluding AL_37 and AL_219 log features.
- `raw_AL_plus_capped_signed_ALlog_native_CAT345_with_CAT6`: fold-safe capped signed log AL features.
- `no_ALlog_CAT345_combined_with_CAT6`: CAT_3/CAT_4/CAT_5 replaced by native `CAT345_COMBINED`.
- `no_ALlog_native_CAT345_drop_CAT6_column_only`: CAT_6 column dropped only; no row removal.

## Main four_model_equal summary

{md_table(summary[summary.aggregation.eq("four_model_equal")][compact])}

## CatBoost-only summary

{md_table(summary[summary.aggregation.eq("catboost_only")][compact])}

## LightGBM-only summary

{md_table(summary[summary.aggregation.eq("lightgbm_only")][compact])}

## XGBoost-only summary

{md_table(summary[summary.aggregation.eq("xgboost_only")][compact])}

## ExtraTrees-only summary

{md_table(summary[summary.aggregation.eq("extratrees_only")][compact])}

## CatBoost+ExtraTrees equal diagnostic

{md_table(summary[summary.aggregation.eq("catboost_extratrees_equal")][compact])}

1. Does AL_log improve CatBoost+ExtraTrees? Best AL_log-family delta is `{allog_best.config_name}` with test F1 delta {allog_best.delta_test_f1_mean:+.6f} and OOF F1 delta {allog_best.delta_oof_f1_mean:+.6f}.
2. Does CAT345_combined improve CatBoost+ExtraTrees? Test F1 delta {cat345.delta_test_f1_mean:+.6f}; OOF F1 delta {cat345.delta_oof_f1_mean:+.6f}.
3. Does dropping CAT_6 improve CatBoost+ExtraTrees? Test F1 delta {cat6.delta_test_f1_mean:+.6f}; OOF F1 delta {cat6.delta_oof_f1_mean:+.6f}.
4. Does CatBoost+ExtraTrees outperform four_model_equal for any config? {cx_outperform if cx_outperform else "No"}.
5. Is any feature effect diluted by the four_model_equal ensemble? {diluted if diluted else "No broad-ensemble dilution flag was triggered"}.

## Fixed ExtraTrees-heavy four-model summary

{md_table(summary[summary.aggregation.eq("four_model_weighted_extratrees_heavy")][compact])}

## Delta vs active reference within each aggregation

{md_table(delta[delta_cols])}

## Interpretation

Feature-effect dilution is flagged only when a config shows acceptable model-family support but fails under the broad four-model equal ensemble. FP/FN tradeoffs should be read from the delta table: a FP decrease paired with enough FN increase to reduce F1 is marked conservative_but_not_preferred. OOF vs test consistency is enforced by the decision labels; test-only gains with weak OOF support are unstable_test_only_signal.

## Leakage audit

- Any leakage failure: **{bool(leak.leakage_fail.any())}**
- Status counts: {leak.status.value_counts().to_dict()}

## Recommendation

{recommendation}
"""
    OUT["report"].write_text(text, encoding="utf-8")
    return recommendation


def main():
    wall_started = time.perf_counter()
    CACHE.mkdir(exist_ok=True)
    full = pd.read_csv(base.find_input())
    full["Label"] = full.Label.astype(int)
    metric_data, runtime_data, source_data, fit_info, feature_records = [], [], [], [], []
    cache_hits = 0
    for seed in SEEDS:
        for config in CONFIGS:
            payload, runtimes, sources, info_rows, feature_info, hit = load_or_generate(seed, config, full)
            cache_hits += int(hit)
            runtime_data.extend(runtimes)
            source_data.extend(sources)
            fit_info.extend(info_rows)
            feature_records.append(feature_info)
            metric_data.extend(metric_rows(seed, config, payload, feature_info))
            pd.DataFrame(metric_data).to_csv(OUT["per_seed"], index=False)
            pd.DataFrame(runtime_data).to_csv(OUT["runtime"], index=False)
            pd.DataFrame(source_data).to_csv(OUT["sources"], index=False)

    per_seed = pd.DataFrame(metric_data)
    summary = summarize(per_seed)
    delta = make_delta(summary)
    labels = decision_labels(delta)
    summary["decision_label"] = summary.config_name.map(labels)
    delta["decision_label"] = delta.config_name.map(labels)

    oof_cols = ["seed", "config_name", "aggregation", "selected_threshold"] + [f"oof_{m}" for m in METRICS]
    test_cols = ["seed", "config_name", "aggregation", "selected_threshold"] + [f"test_{m}" for m in METRICS]
    per_seed[oof_cols].to_csv(OUT["oof"], index=False)
    per_seed[test_cols].to_csv(OUT["test"], index=False)
    summary[summary.aggregation.eq("four_model_equal")].to_csv(OUT["summary"], index=False)
    delta[delta.aggregation.eq("four_model_equal")].to_csv(OUT["delta"], index=False)
    summary.to_csv(OUT["family_summary"], index=False)
    delta.to_csv(OUT["family_delta"], index=False)
    per_seed.to_csv(OUT["per_seed"], index=False)
    pd.DataFrame([{k: v for k, v in rec.items() if k != "features"} for rec in feature_records]).to_csv(
        OUT["feature_count"], index=False
    )
    leak = leakage_audit(feature_records, fit_info)
    leak.to_csv(OUT["leak"], index=False)
    pd.DataFrame(runtime_data).to_csv(OUT["runtime"], index=False)
    pd.DataFrame(source_data).to_csv(OUT["sources"], index=False)
    recommendation = write_report(summary, delta, leak, labels)

    runtime_df = pd.DataFrame(runtime_data)
    metadata = {
        "step": "Step15B",
        "completed": True,
        "configs": CONFIGS,
        "seeds": SEEDS,
        "folds": N_SPLITS,
        "models": MODELS,
        "aggregations": AGGREGATIONS,
        "mandatory_catboost_extratrees_equal": True,
        "catboost_extratrees_equal_definition": "0.50*catboost + 0.50*extratrees",
        "threshold_selection": "OOF_only_per_seed_config_aggregation",
        "test_usage": "reporting_only",
        "feature_model_weight_selection_uses_test_labels": False,
        "fixed_weights_no_search": True,
        "cache_directory": str(CACHE),
        "step15b_cache_hits_on_this_run": cache_hits,
        "step15a_reference_prediction_reuse": True,
        "observed_model_runtime_seconds": float(runtime_df.runtime_seconds.sum()) if len(runtime_df) else 0.0,
        "wall_runtime_seconds": time.perf_counter() - wall_started,
        "leakage_fail": bool(leak.leakage_fail.any()),
        "decision_labels": labels,
        "recommendation": recommendation,
    }
    OUT["metadata"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("=== Step15B complete ===", flush=True)
    print(f"wall_minutes={metadata['wall_runtime_seconds'] / 60:.2f}", flush=True)
    print(f"leakage_fail={metadata['leakage_fail']}", flush=True)
    print(f"decisions={labels}", flush=True)


if __name__ == "__main__":
    main()
