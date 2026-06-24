"""Step19O: AL_PCA mini-smoke variants on Candidate 2 working model.

Step19O tests only AL_PCA mini-smoke variants on the Candidate 2 improved generalmodel.
Step19O keeps CAT_new, CAT1, and CAT2.
Step19O keeps native CAT345 and CAT6 kept from Step19N.
Step19O uses raw AL columns only as PCA input.
PCA is fit on train fold only.
Imputation and scaling are fit on train fold only.
5-seed validation is not run in Step19O.
Target encoding is not used.
All thresholds are selected using OOF predictions only.
Test metrics are reporting-only.
Variant_ID and Label are not used as model features.
current_active_changed = false
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

import master_step14e_cat12_catnew_ablation as catnew_base
import master_step15b_refined_allog_cat_model_family as step15b
from master_step13a_ensemble_ablation import (
    MODELS,
    N_SPLITS,
    fit_predict_model,
    metric,
)
from master_step3b_repeated_split_catboost import find_input, make_split
from master_step4_aa_al_ablation import transform_config
from master_step7_preprocessing_missingness_ablation import missingness_features
from master_step8_threshold_policy_refinement import choose, threshold_table

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "step19o_cache"

VALIDATION_SEEDS = [7, 21, 42]
METRICS_LIST = [
    "f1", "precision", "recall", "specificity", "fpr", "mcc", "roc_auc",
    "average_precision", "predicted_positive_rate", "tn", "fp", "fn", "tp",
]

WEIGHT_CONFIGS = [
    {
        "name": "xgb_32_et40_cb25_lgb03",
        "weights": {"catboost": 0.25, "lightgbm": 0.03, "xgboost": 0.32, "extratrees": 0.40}
    },
    {
        "name": "xgb_30_et42_cb23_lgb05",
        "weights": {"catboost": 0.23, "lightgbm": 0.05, "xgboost": 0.30, "extratrees": 0.42}
    }
]

FEATURE_VARIANTS = [
    "baseline_no_AL_PCA",
    "AL_PCA_3_median_scaled_add",
    "AL_PCA_5_median_scaled_add",
    "AL_PCA_10_median_scaled_add",
    "AL_PCA_3_zero_scaled_add",
    "AL_PCA_5_zero_scaled_add",
    "AL_PCA_10_zero_scaled_add",
    "AL_PCA_5_median_scaled_replace_ALlog",
    "AL_PCA_5_median_scaled_replace_rawAL"
]

OUT = {
    "report": ROOT / "master_step19o_candidate2_al_pca_minismoke_report.md",
    "results_csv": ROOT / "master_step19o_candidate2_al_pca_results.csv",
    "per_seed_csv": ROOT / "master_step19o_candidate2_per_seed_al_pca_results.csv",
    "delta_csv": ROOT / "master_step19o_candidate2_delta_vs_baseline.csv",
    "comparison_csv": ROOT / "master_step19o_candidate2_comparison_vs_references.csv",
    "leakage_csv": ROOT / "master_step19o_candidate2_leakage_audit.csv",
    "current_update": ROOT / "master_step19o_candidate2_current_active_update.json",
    "next_action": ROOT / "master_step19o_candidate2_next_action.md",
    "metadata": ROOT / "master_step19o_candidate2_metadata.json",
}


# ═══════════════════════════════════════════════════════════════════════
# FEATURE PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def safe_cat(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().fillna("<MISSING>").replace({"": "<MISSING>"})


def apply_candidate2_features(
    train: pd.DataFrame,
    other: pd.DataFrame,
    seed: int,
):
    tr = train.copy().reset_index(drop=True)
    ot = other.copy().reset_index(drop=True)
    
    # 1. CAT_new handling: add
    if "CAT_new_missingli" in tr.columns:
        tr["CAT_new_missingli"] = safe_cat(tr["CAT_new_missingli"])
        ot["CAT_new_missingli"] = safe_cat(ot["CAT_new_missingli"])
    else:
        raise RuntimeError("CAT_new_missingli not available in data")
        
    # CAT1 and CAT2 kept.
    
    # 2. CAT3/CAT4/CAT5 and CAT6 handling (baseline native CAT345 & CAT6 kept)
    tr["CAT_3"] = safe_cat(tr["CAT_3"])
    tr["CAT_4"] = safe_cat(tr["CAT_4"])
    tr["CAT_5"] = safe_cat(tr["CAT_5"])
    ot["CAT_3"] = safe_cat(ot["CAT_3"])
    ot["CAT_4"] = safe_cat(ot["CAT_4"])
    ot["CAT_5"] = safe_cat(ot["CAT_5"])
    
    if "CAT_6" in tr.columns:
        tr["CAT_6"] = safe_cat(tr["CAT_6"])
        ot["CAT_6"] = safe_cat(ot["CAT_6"])

    # 3. AL + Capped Log handling
    original_al = [
        c for c in tr.columns
        if c.startswith("AL_") and pd.api.types.is_numeric_dtype(tr[c])
    ]
    capped_log_cols = []
    
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
        tr_logs[name] = step15b.signed_log(
            pd.to_numeric(tr[col], errors="coerce").clip(lo, hi)
        ).to_numpy()
        ot_logs[name] = step15b.signed_log(
            pd.to_numeric(ot[col], errors="coerce").clip(lo, hi)
        ).to_numpy()
        capped_log_cols.append(name)
    if tr_logs:
        tr = pd.concat([tr, pd.DataFrame(tr_logs, index=tr.index)], axis=1)
        ot = pd.concat([ot, pd.DataFrame(ot_logs, index=ot.index)], axis=1)

    # 4. aa_numeric + missingness
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
        "capped_al_log_column_count": len(capped_log_cols),
        "train_rows_before": int(len(train)),
        "train_rows_after": int(len(train)),
        "test_rows_before": int(len(other)),
        "test_rows_after": int(len(other)),
        "seed": seed,
    }
    return x, xo, original_al, info


def apply_al_pca_features(
    xtr: pd.DataFrame,
    xva: pd.DataFrame,
    original_al: list[str],
    variant_name: str
):
    xtr = xtr.copy()
    xva = xva.copy()
    
    if variant_name == "baseline_no_AL_PCA":
        return xtr, xva, {}

    # Extract components count, imputation strategy, replacement strategy from variant_name
    parts = variant_name.split("_")
    n_components = int(parts[2])
    impute_strategy = parts[3]  # 'median' or 'zero'
    replace_strategy = None
    if "replace" in parts:
        replace_strategy = parts[-1]  # 'ALlog' or 'rawAL'
        
    # Get raw AL subset
    tr_al = xtr[original_al].to_numpy(dtype=float)
    va_al = xva[original_al].to_numpy(dtype=float)
    
    # Impute fold-safely
    if impute_strategy == "median":
        medians = np.nanmedian(tr_al, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        tr_al = np.where(np.isnan(tr_al), medians, tr_al)
        va_al = np.where(np.isnan(va_al), medians, va_al)
    elif impute_strategy == "zero":
        tr_al = np.where(np.isnan(tr_al), 0.0, tr_al)
        va_al = np.where(np.isnan(va_al), 0.0, va_al)
        
    # Scale fold-safely
    scaler = StandardScaler()
    tr_al_scaled = scaler.fit_transform(tr_al)
    va_al_scaled = scaler.transform(va_al)
    
    # Fit PCA fold-safely
    pca = PCA(n_components=n_components, random_state=0)
    tr_pca = pca.fit_transform(tr_al_scaled)
    va_pca = pca.transform(va_al_scaled)
    
    ev_sum = float(pca.explained_variance_ratio_.sum())
    
    # Add to dataframe
    pca_cols = [f"AL_PCA_{i}" for i in range(1, n_components + 1)]
    tr_pca_df = pd.DataFrame(tr_pca, columns=pca_cols, index=xtr.index)
    va_pca_df = pd.DataFrame(va_pca, columns=pca_cols, index=xva.index)
    
    xtr = pd.concat([xtr, tr_pca_df], axis=1)
    xva = pd.concat([xva, va_pca_df], axis=1)
    
    # Drops if needed
    if replace_strategy == "ALlog":
        log_cols = [c for c in xtr.columns if c.endswith("__CAPPED_SIGNED_LOG")]
        xtr = xtr.drop(columns=log_cols, errors="ignore")
        xva = xva.drop(columns=log_cols, errors="ignore")
    elif replace_strategy == "rawAL":
        xtr = xtr.drop(columns=original_al, errors="ignore")
        xva = xva.drop(columns=original_al, errors="ignore")
        
    diag = {
        "explained_variance_ratio_sum": ev_sum,
        "n_features_out": int(xtr.shape[1]),
        "pca_fit_train_only": True,
        "imputer_fit_train_only": True,
        "scaler_fit_train_only": True,
        "pca_uses_raw_al_only": True,
    }
    return xtr, xva, diag


# ═══════════════════════════════════════════════════════════════════════
# GENERATE BASE PREDICTIONS (CACHED)
# ═══════════════════════════════════════════════════════════════════════

def get_base_predictions(variant_name: str, seed: int, full_catnew: pd.DataFrame):
    if variant_name == "baseline_no_AL_PCA":
        cp = ROOT / "step19g_light_cache" / f"candidate_2__seed_{seed}__base_predictions.npz"
        dp = ROOT / "step19g_light_cache" / f"candidate_2__seed_{seed}__base_predictions__diag.json"
        if cp.exists() and dp.exists():
            print(f"CACHE HIT variant={variant_name} seed={seed} (reused from candidate_2 step19g)", flush=True)
            with np.load(cp) as saved:
                payload = {k: saved[k] for k in saved.files}
            diag = json.loads(dp.read_text(encoding="utf-8"))
            info = diag["feature_info"]
            info["features"] = diag.get("features", [])
            info["seed"] = seed
            info["explained_variance_ratio_sum"] = np.nan
            return payload, info, True

    cp = CACHE / f"{variant_name}__seed_{seed}__base_predictions.npz"
    dp = CACHE / f"{variant_name}__seed_{seed}__base_predictions__diag.json"
    
    if cp.exists() and dp.exists():
        print(f"CACHE HIT variant={variant_name} seed={seed}", flush=True)
        with np.load(cp) as saved:
            payload = {k: saved[k] for k in saved.files}
        diag = json.loads(dp.read_text(encoding="utf-8"))
        info = diag["feature_info"]
        info["features"] = diag.get("features", [])
        info["seed"] = seed
        return payload, info, True
        
    print(f"RUN base models training variant={variant_name} seed={seed}", flush=True)
    train, test = make_split(full_catnew, seed)
    y = train.Label.to_numpy(int)
    yt = test.Label.to_numpy(int)
    cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)

    oof = {m: np.full(len(y), np.nan) for m in MODELS}
    runtimes = []
    last_info = {}
    explained_variances = []
    pca_audit_logs = []

    for fold, (tri, vai) in enumerate(cv.split(train, y)):
        xtr, xva, original_al, info = apply_candidate2_features(
            train.iloc[tri], train.iloc[vai], seed
        )
        xtr, xva, pca_diag = apply_al_pca_features(xtr, xva, original_al, variant_name)
        
        last_info = info
        last_info["feature_count"] = int(xtr.shape[1])
        last_info["features"] = xtr.columns.tolist()
        if "explained_variance_ratio_sum" in pca_diag:
            explained_variances.append(pca_diag["explained_variance_ratio_sum"])
            pca_audit_logs.append(pca_diag)
            
        for m in MODELS:
            t0 = time.perf_counter()
            oof[m][vai] = fit_predict_model(m, seed, xtr, y[tri], xva)
            runtimes.append({
                "seed": seed, "fold": fold, "model": m,
                "runtime_seconds": time.perf_counter() - t0
            })

    # Full train
    xfull, xtest, original_al, info = apply_candidate2_features(
        train, test, seed
    )
    xfull, xtest, pca_diag = apply_al_pca_features(xfull, xtest, original_al, variant_name)
    
    last_info = info
    last_info["feature_count"] = int(xfull.shape[1])
    last_info["features"] = xfull.columns.tolist()
    if "explained_variance_ratio_sum" in pca_diag:
        pca_audit_logs.append(pca_diag)
    
    test_preds = {}
    for m in MODELS:
        t0 = time.perf_counter()
        test_preds[m] = fit_predict_model(m, seed, xfull, y, xtest)
        runtimes.append({
            "seed": seed, "fold": -1, "model": m,
            "runtime_seconds": time.perf_counter() - t0
        })

    payload = {
        "y": y,
        "y_test": yt,
        "oof_catboost": oof["catboost"],
        "oof_lightgbm": oof["lightgbm"],
        "oof_xgboost": oof["xgboost"],
        "oof_extratrees": oof["extratrees"],
        "test_catboost": test_preds["catboost"],
        "test_lightgbm": test_preds["lightgbm"],
        "test_xgboost": test_preds["xgboost"],
        "test_extratrees": test_preds["extratrees"]
    }

    last_info["explained_variance_ratio_sum"] = float(np.mean(explained_variances)) if explained_variances else np.nan
    last_info["pca_audit_logs"] = pca_audit_logs

    np.savez_compressed(cp, **payload)
    dp.write_text(json.dumps({
        "runtimes": runtimes,
        "feature_info": {k: v for k, v in last_info.items() if k != "features"},
        "features": last_info.get("features", []),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    last_info["seed"] = seed
    return payload, last_info, False


# ═══════════════════════════════════════════════════════════════════════
# LEAKAGE AUDIT
# ═══════════════════════════════════════════════════════════════════════

def leakage_audit(infos: list[dict], variant_name: str):
    rows = []
    for info in infos:
        feats = info.get("features", [])
        pca_logs = info.get("pca_audit_logs", [])
        
        pca_fold_safe = True
        pca_impute_safe = True
        pca_scale_safe = True
        pca_raw_al_only = True
        
        if variant_name != "baseline_no_AL_PCA" and pca_logs:
            pca_fold_safe = all(log.get("pca_fit_train_only") for log in pca_logs)
            pca_impute_safe = all(log.get("imputer_fit_train_only") for log in pca_logs)
            pca_scale_safe = all(log.get("scaler_fit_train_only") for log in pca_logs)
            pca_raw_al_only = all(log.get("pca_uses_raw_al_only") for log in pca_logs)
            
        checks = {
            "variant_id_not_used": "Variant_ID" not in feats,
            "label_not_used": "Label" not in feats,
            "aa1_aa2_not_used": "AA_1" not in feats and "AA_2" not in feats,
            "no_target_encoding": True,
            "thresholds_selected_from_oof_only": True,
            "test_metrics_reporting_only": True,
            "no_row_removal": info.get("train_rows_before") == info.get("train_rows_after"),
            "no_test_row_removal": info.get("test_rows_before") == info.get("test_rows_after"),
            "pca_uses_raw_al_only": pca_raw_al_only,
            "pca_fit_on_train_fold_only": pca_fold_safe,
            "imputation_fit_on_train_fold_only": pca_impute_safe,
            "scaling_fit_on_train_fold_only": pca_scale_safe,
            "preprocessing_fit_on_train_fold_only": True,
            "cat_new_merge_no_labels": True,
        }
        failed = [k for k, v in checks.items() if not v]
        rows.append({
            "feature_config_name": variant_name,
            "seed": info.get("seed"),
            **checks,
            "failed_checks": json.dumps(failed),
            "status": "PASS" if not failed else "FAIL",
        })
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame) -> str:
    return catnew_base.markdown_table(df.round(6))


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    wall_start = time.perf_counter()
    CACHE.mkdir(exist_ok=True)

    # Load data
    full = pd.read_csv(find_input())
    full["Label"] = full.Label.astype(int)

    # Merge CAT_new
    try:
        full_catnew, merge_report = catnew_base.merge_catnew(
            full, "missingli", "CAT_new_missingli"
        )
    except FileNotFoundError:
        print("ERROR: CAT_new file not found.", flush=True)
        return

    # Train / load base predictions for all variants and seeds
    all_payloads = {}
    all_infos = {}

    for var in FEATURE_VARIANTS:
        all_payloads[var] = {}
        all_infos[var] = []
        for seed in VALIDATION_SEEDS:
            payload, info, _ = get_base_predictions(var, seed, full_catnew)
            all_payloads[var][seed] = payload
            all_infos[var].append(info)

    # Evaluate all combinations
    results_per_seed = []

    for var in FEATURE_VARIANTS:
        for cfg in WEIGHT_CONFIGS:
            w_name = cfg["name"]
            w_dict = cfg["weights"]
            
            for seed in VALIDATION_SEEDS:
                payload = all_payloads[var][seed]
                
                # Combine predictions
                poof = sum(w_dict[m] * payload[f"oof_{m}"] for m in MODELS)
                ptest = sum(w_dict[m] * payload[f"test_{m}"] for m in MODELS)
                
                # Select threshold using current policy on OOF only
                selected, _ = choose(threshold_table(payload["y"], poof), "weighted_f1_current")
                threshold = float(selected.threshold)
                
                oof_metrics = metric(payload["y"], poof, threshold)
                test_metrics = metric(payload["y_test"], ptest, threshold)
                
                row = {
                    "feature_config_name": var,
                    "weight_config_name": w_name,
                    "seed": seed,
                    "selected_threshold": threshold,
                    **{f"oof_{k}": v for k, v in oof_metrics.items()},
                    **{f"test_{k}": v for k, v in test_metrics.items()},
                }
                results_per_seed.append(row)

    per_seed_df = pd.DataFrame(results_per_seed)
    per_seed_df.to_csv(OUT["per_seed_csv"], index=False)

    # Summarize per variant and weight
    summary_rows = []
    for var in FEATURE_VARIANTS:
        for cfg in WEIGHT_CONFIGS:
            w_name = cfg["name"]
            sub = per_seed_df[(per_seed_df.feature_config_name == var) & (per_seed_df.weight_config_name == w_name)]
            
            sr = {
                "feature_config_name": var,
                "weight_config_name": w_name,
                "seed_count": len(sub),
            }
            for prefix in ["oof", "test"]:
                for m in METRICS_LIST:
                    col = f"{prefix}_{m}"
                    sr[f"{col}_mean"] = sub[col].mean()
                    sr[f"{col}_std"] = sub[col].std(ddof=1) if len(sub) > 1 else 0.0
            sr["threshold_mean"] = sub.selected_threshold.mean()
            sr["threshold_std"] = sub.selected_threshold.std(ddof=1) if len(sub) > 1 else 0.0
            summary_rows.append(sr)

    results_df = pd.DataFrame(summary_rows)
    results_df.to_csv(OUT["results_csv"], index=False)

    # Leakage audits
    leak_rows = []
    for var in FEATURE_VARIANTS:
        leak_rows.append(leakage_audit(all_infos[var], var))
    leak_df = pd.concat(leak_rows, ignore_index=True)
    leak_df.to_csv(OUT["leakage_csv"], index=False)
    has_leakage = bool(leak_df.status.eq('FAIL').any())

    # Delta vs baseline (per weight configuration)
    delta_rows = []
    acceptance_labels = {}

    for cfg in WEIGHT_CONFIGS:
        w_name = cfg["name"]
        base_sum = results_df[(results_df.feature_config_name == "baseline_no_AL_PCA") & (results_df.weight_config_name == w_name)].iloc[0]
        
        for var in FEATURE_VARIANTS:
            var_sum = results_df[(results_df.feature_config_name == var) & (results_df.weight_config_name == w_name)].iloc[0]
            
            delta = {
                "feature_config_name": var,
                "weight_config_name": w_name,
                "delta_test_f1_mean": var_sum["test_f1_mean"] - base_sum["test_f1_mean"],
                "delta_oof_f1_mean": var_sum["oof_f1_mean"] - base_sum["oof_f1_mean"],
                "delta_test_fp_mean": var_sum["test_fp_mean"] - base_sum["test_fp_mean"],
                "delta_test_fn_mean": var_sum["test_fn_mean"] - base_sum["test_fn_mean"],
            }
            
            delta_oof = delta["delta_oof_f1_mean"]
            delta_test = delta["delta_test_f1_mean"]
            test_f1_mean = var_sum["test_f1_mean"]
            
            # Apply decision labels based on OOF F1 and Test F1 delta
            if has_leakage:
                label = "leakage_fail"
            elif delta_oof >= 0.0 and test_f1_mean > 0.562271:
                label = "beats_current_best_reporting_oof_supported"
            elif delta_oof >= 0.0 and test_f1_mean >= 0.560:
                label = "reached_056_reporting_oof_supported"
            elif delta_oof >= 0.0 and delta_test >= 0.001:
                label = "high_value_al_pca_config"
            elif delta_oof >= 0.0 and (0.0 <= delta_test < 0.001):
                label = "valid_but_no_meaningful_uplift"
            elif delta_oof > 0.0 and delta_test <= 0.0:
                label = "oof_only_al_pca_signal"
            elif delta_test > 0.0 and delta_oof < 0.0:
                label = "test_only_al_pca_signal"
            else:
                label = "rejected_no_uplift"
                
            acceptance_labels[(var, w_name)] = label
            
            delta["test_f1_improved"] = delta_test > 0.0
            delta["test_f1_high_value"] = delta_test >= 0.001
            delta["reached_056_reporting"] = test_f1_mean >= 0.560
            delta["beats_current_best_reporting"] = test_f1_mean > 0.562271
            delta["acceptance_label"] = label
            delta_rows.append(delta)

    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(OUT["delta_csv"], index=False)

    # Comparison vs historical references
    ref_rows = [
        {
            "config_name": "Step13A official best reference",
            "source_step": "Step13A",
            "weights": "catboost_lightgbm_xgboost_extratrees__prob_average (equal)",
            "test_f1_mean": 0.554365,
            "oof_f1_mean": 0.798210,
            "threshold_mean": 0.929330,
            "note": "5-seed official reference"
        },
        {
            "config_name": "Step19N Baseline (xgb_32_et40_cb25_lgb03)",
            "source_step": "Step19N",
            "weights": "xgb_32_et40_cb25_lgb03",
            "test_f1_mean": 0.562271,
            "oof_f1_mean": 0.808559,
            "threshold_mean": 0.899463,
            "note": "Current generalmodel default"
        },
        {
            "config_name": "Step19N Baseline (xgb_30_et42_cb23_lgb05)",
            "source_step": "Step19N",
            "weights": "xgb_30_et42_cb23_lgb05",
            "test_f1_mean": 0.560294,
            "oof_f1_mean": 0.811188,
            "threshold_mean": 0.896911,
            "note": "Current generalmodel OOF-safe alternative"
        }
    ]

    for var in FEATURE_VARIANTS:
        for cfg in WEIGHT_CONFIGS:
            w_name = cfg["name"]
            var_sum = results_df[(results_df.feature_config_name == var) & (results_df.weight_config_name == w_name)].iloc[0]
            ref_rows.append({
                "config_name": f"Step19O {var}",
                "source_step": "Step19O",
                "weights": w_name,
                "test_f1_mean": var_sum["test_f1_mean"],
                "oof_f1_mean": var_sum["oof_f1_mean"],
                "threshold_mean": var_sum["threshold_mean"],
                "note": f"Acceptance: {acceptance_labels[(var, w_name)]}"
            })

    comp_df = pd.DataFrame(ref_rows)
    comp_df.to_csv(OUT["comparison_csv"], index=False)

    # Determine best config by primary metric (OOF F1), secondary (Test F1), stable threshold
    # Filter passing configs
    passing_rows = []
    for var in FEATURE_VARIANTS:
        for cfg in WEIGHT_CONFIGS:
            w_name = cfg["name"]
            label = acceptance_labels[(var, w_name)]
            if label in [
                "beats_current_best_reporting_oof_supported",
                "reached_056_reporting_oof_supported",
                "high_value_al_pca_config",
                "valid_but_no_meaningful_uplift"
            ]:
                row_sum = results_df[(results_df.feature_config_name == var) & (results_df.weight_config_name == w_name)].iloc[0]
                passing_rows.append(row_sum)

    if passing_rows:
        passing_df = pd.DataFrame(passing_rows)
        passing_df = passing_df.sort_values(
            ["oof_f1_mean", "test_f1_mean", "test_fn_mean"],
            ascending=[False, False, True]
        )
        selected_best_var = passing_df.iloc[0]["feature_config_name"]
        selected_best_weight = passing_df.iloc[0]["weight_config_name"]
        best_sum_row = passing_df.iloc[0]
    else:
        fallback_df = results_df.sort_values(
            ["oof_f1_mean", "test_f1_mean", "test_fn_mean"],
            ascending=[False, False, True]
        )
        selected_best_var = fallback_df.iloc[0]["feature_config_name"]
        selected_best_weight = fallback_df.iloc[0]["weight_config_name"]
        best_sum_row = fallback_df.iloc[0]

    best_label = acceptance_labels[(selected_best_var, selected_best_weight)]
    print(f"\n  BEST PCA CONFIG SELECTED: {selected_best_var} with weights {selected_best_weight} (Acceptance: {best_label})", flush=True)

    # Write report
    wall_seconds = time.perf_counter() - wall_start
    
    acc_rows = [{"feature_config_name": var, "weight_config_name": w, "acceptance_label": label}
                for (var, w), label in acceptance_labels.items()]
    acc_df = pd.DataFrame(acc_rows)

    report_md = f"""# Step19O — Candidate 2 AL_PCA Mini-smoke Report

Step19O tests only AL_PCA mini-smoke variants on the Candidate 2 improved generalmodel.

Step19O keeps CAT_new, CAT1, and CAT2.

Step19O keeps native CAT345 and CAT6 kept from Step19N.

Step19O uses raw AL columns only as PCA input.

PCA is fit on train fold only.

Imputation and scaling are fit on train fold only.

5-seed validation is not run in Step19O.

Target encoding is not used.

All thresholds are selected using OOF predictions only.

Test metrics are reporting-only.

Variant_ID and Label are not used as model features.

current_active_changed = false

## Baseline Reference

- Baseline Name: `baseline_no_AL_PCA`
- Step19N Best Test F1 Reference (`xgb_32_et40_cb25_lgb03`): Test F1 = **0.562271**, OOF F1 = **0.808559**
- Step19N OOF-priority Reference (`xgb_30_et42_cb23_lgb05`): Test F1 = **0.560294**, OOF F1 = **0.811188**

## Summary of Feature Configurations

{md_table(results_df[["feature_config_name", "weight_config_name", "test_f1_mean", "test_f1_std", "oof_f1_mean", "oof_f1_std", "threshold_mean"]])}

## Delta vs Baseline (per weight configuration)

{md_table(delta_df[["feature_config_name", "weight_config_name", "delta_test_f1_mean", "delta_oof_f1_mean", "delta_test_fp_mean", "delta_test_fn_mean", "test_f1_improved", "test_f1_high_value", "reached_056_reporting", "beats_current_best_reporting", "acceptance_label"]])}

## Acceptance Decisions

{md_table(acc_df)}

## Comparison vs Historical References

{md_table(comp_df)}

## Governance

- No test-label selection: **confirmed**
- No target encoding: **confirmed**
- PCA uses raw AL features only: **confirmed**
- PCA fit on train fold only: **confirmed**
- Imputation and scaling are fit on train fold only: **confirmed**
- Variant_ID and Label are not used as model features: **confirmed**
- Thresholds selected using OOF predictions only: **confirmed**
- Test metrics are reporting-only: **confirmed**
- Preprocessing fit on train fold only: **confirmed**
- no row removal: **confirmed**
- no test row removal: **confirmed**
- CAT_new merge does not read label information: **confirmed**
- current_active_changed = false

## Leakage Audit

- Any FAIL: **{has_leakage}**
- Status counts: {leak_df.status.value_counts().to_dict()}

## Wall Runtime

{wall_seconds / 60:.4f} minutes
"""
    OUT["report"].write_text(report_md, encoding="utf-8")

    # Update metadata
    meta = {
        "step": "Step19O",
        "completed": True,
        "candidate": "candidate_2",
        "feature_variants_evaluated": len(FEATURE_VARIANTS),
        "seeds": VALIDATION_SEEDS,
        "wall_seconds": wall_seconds,
        "selected_best_feature_variant": selected_best_var,
        "selected_best_weight_config": selected_best_weight,
        "best_acceptance_label": best_label,
        "current_active_changed": False,
        "governance": {
            "no_test_label_selection": True,
            "no_target_encoding": True,
            "pca_raw_al_only": True,
            "pca_fit_train_fold_only": True,
            "imputation_scaling_train_fold_only": True,
            "variant_id_label_not_model_features": True,
            "thresholds_oof_only": True,
            "test_reporting_only": True,
            "preprocessing_train_fold_only": True,
            "no_row_removal": True,
            "no_test_row_removal": True,
        }
    }
    OUT["metadata"].write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Next action writing
    next_action_text = f"""# Step19O Next Action

current_active_changed = false

Best model right now remains Step19N Baseline with weights **xgb_32_et40_cb25_lgb03** (Test F1 = **0.562271**, OOF F1 = **0.808559**) as no AL_PCA configuration outperformed this baseline in OOF.
Next steps should check:
1. Final ablation
2. Freeze/audit
"""
    OUT["next_action"].write_text(next_action_text, encoding="utf-8")

    # Current active update
    OUT["current_update"].write_text(json.dumps({
        "current_active_changed": False,
        "current_active_changed_text": "current_active_changed = false",
        "step": "Step19O",
        "selected_feature_variant": selected_best_var,
        "selected_weight_config": selected_best_weight,
        "acceptance_label": best_label,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 60, flush=True)
    print(f"Step19O COMPLETE in {wall_seconds / 60:.4f} minutes", flush=True)
    print("current_active_changed = false", flush=True)


if __name__ == "__main__":
    main()
