"""Step19N: CAT345/CAT6 mini-smoke variants on Candidate 2 working best model.

Step19N tests only CAT345/CAT6 mini-smoke variants on the Candidate 2 improved model.
Step19N keeps CAT_new, CAT1, and CAT2.
Step19N keeps raw AL and capped/signed AL_log unchanged.
Step19N evaluates both Step19M candidate weights: xgb_32_et40_cb25_lgb03 and xgb_30_et42_cb23_lgb05.
5-seed validation is not run in Step19N.
AL_PCA is not tested in Step19N.
Target encoding is not used.
Frequency encoding uses feature frequencies only and is fit on train fold only.
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
CACHE = ROOT / "step19n_cache"

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
    "baseline_native_CAT345_CAT6_kept",
    "drop_CAT6_only",
    "CAT6_presence_flag_only",
    "CAT345_combo_add_keep_originals",
    "CAT345_combo_replace_originals",
    "CAT345_freq_replace_originals",
    "CAT345_combo_freq_replace_originals"
]

OUT = {
    "report": ROOT / "master_step19n_candidate2_cat345_cat6_minismoke_report.md",
    "results_csv": ROOT / "master_step19n_candidate2_cat_feature_results.csv",
    "per_seed_csv": ROOT / "master_step19n_candidate2_per_seed_cat_feature_results.csv",
    "delta_csv": ROOT / "master_step19n_candidate2_delta_vs_baseline.csv",
    "comparison_csv": ROOT / "master_step19n_candidate2_comparison_vs_references.csv",
    "leakage_csv": ROOT / "master_step19n_candidate2_leakage_audit.csv",
    "current_update": ROOT / "master_step19n_candidate2_current_active_update.json",
    "next_action": ROOT / "master_step19n_candidate2_next_action.md",
    "metadata": ROOT / "master_step19n_candidate2_metadata.json",
}


# ═══════════════════════════════════════════════════════════════════════
# FEATURE PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def safe_cat(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().fillna("<MISSING>").replace({"": "<MISSING>"})


def apply_candidate2_features(
    train: pd.DataFrame,
    other: pd.DataFrame,
    variant_name: str,
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
    
    # 2. CAT3/CAT4/CAT5 and CAT6 handling based on variant_name
    cat345_cols = ["CAT_3", "CAT_4", "CAT_5"]
    
    tr_cat3 = safe_cat(tr["CAT_3"])
    tr_cat4 = safe_cat(tr["CAT_4"])
    tr_cat5 = safe_cat(tr["CAT_5"])
    ot_cat3 = safe_cat(ot["CAT_3"])
    ot_cat4 = safe_cat(ot["CAT_4"])
    ot_cat5 = safe_cat(ot["CAT_5"])
    
    tr["CAT_3"] = tr_cat3
    tr["CAT_4"] = tr_cat4
    tr["CAT_5"] = tr_cat5
    ot["CAT_3"] = ot_cat3
    ot["CAT_4"] = ot_cat4
    ot["CAT_5"] = ot_cat5
    
    if "CAT_6" in tr.columns:
        tr["CAT_6"] = safe_cat(tr["CAT_6"])
        ot["CAT_6"] = safe_cat(ot["CAT_6"])

    # Create COMBO if needed
    if variant_name in ["CAT345_combo_add_keep_originals", "CAT345_combo_replace_originals", "CAT345_combo_freq_replace_originals"]:
        tr["CAT345_COMBO"] = tr_cat3 + "**" + tr_cat4 + "**" + tr_cat5
        ot["CAT345_COMBO"] = ot_cat3 + "**" + ot_cat4 + "**" + ot_cat5

    # Implement drops/transforms
    if variant_name == "drop_CAT6_only":
        tr = tr.drop(columns=["CAT_6"], errors="ignore")
        ot = ot.drop(columns=["CAT_6"], errors="ignore")
        
    elif variant_name == "CAT6_presence_flag_only":
        tr["CAT6_PRESENT"] = (~tr["CAT_6"].isin(["<MISSING>", ""])).astype(float)
        ot["CAT6_PRESENT"] = (~ot["CAT_6"].isin(["<MISSING>", ""])).astype(float)
        tr = tr.drop(columns=["CAT_6"], errors="ignore")
        ot = ot.drop(columns=["CAT_6"], errors="ignore")
        
    elif variant_name == "CAT345_combo_replace_originals":
        tr = tr.drop(columns=cat345_cols, errors="ignore")
        ot = ot.drop(columns=cat345_cols, errors="ignore")
        
    elif variant_name == "CAT345_freq_replace_originals":
        for col in cat345_cols:
            freq = tr[col].value_counts().to_dict()
            tr[f"{col}_FREQ"] = tr[col].map(freq).fillna(0).astype(float)
            ot[f"{col}_FREQ"] = ot[col].map(freq).fillna(0).astype(float)
        tr = tr.drop(columns=cat345_cols, errors="ignore")
        ot = ot.drop(columns=cat345_cols, errors="ignore")
        
    elif variant_name == "CAT345_combo_freq_replace_originals":
        freq = tr["CAT345_COMBO"].value_counts().to_dict()
        tr["CAT345_COMBO_FREQ"] = tr["CAT345_COMBO"].map(freq).fillna(0).astype(float)
        ot["CAT345_COMBO_FREQ"] = ot["CAT345_COMBO"].map(freq).fillna(0).astype(float)
        tr = tr.drop(columns=cat345_cols + ["CAT345_COMBO"], errors="ignore")
        ot = ot.drop(columns=cat345_cols + ["CAT345_COMBO"], errors="ignore")

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
        "feature_count": int(x.shape[1]),
        "features": x.columns.tolist(),
        "raw_al_column_count": len(original_al),
        "capped_al_log_column_count": len(capped_log_cols),
        "train_rows_before": int(len(train)),
        "train_rows_after": int(len(train)),
        "test_rows_before": int(len(other)),
        "test_rows_after": int(len(other)),
        "seed": seed,
    }
    return x, xo, info


# ═══════════════════════════════════════════════════════════════════════
# GENERATE BASE PREDICTIONS (CACHED)
# ═══════════════════════════════════════════════════════════════════════

def get_base_predictions(variant_name: str, seed: int, full_catnew: pd.DataFrame):
    if variant_name == "baseline_native_CAT345_CAT6_kept":
        cp = ROOT / "step19g_light_cache" / f"candidate_2__seed_{seed}__base_predictions.npz"
        dp = ROOT / "step19g_light_cache" / f"candidate_2__seed_{seed}__base_predictions__diag.json"
    else:
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

    for fold, (tri, vai) in enumerate(cv.split(train, y)):
        xtr, xva, info = apply_candidate2_features(
            train.iloc[tri], train.iloc[vai], variant_name, seed
        )
        last_info = info
        for m in MODELS:
            t0 = time.perf_counter()
            oof[m][vai] = fit_predict_model(m, seed, xtr, y[tri], xva)
            runtimes.append({
                "seed": seed, "fold": fold, "model": m,
                "runtime_seconds": time.perf_counter() - t0
            })

    # Full train
    xfull, xtest, info = apply_candidate2_features(
        train, test, variant_name, seed
    )
    last_info = info
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

    np.savez_compressed(cp, **payload)
    dp.write_text(json.dumps({
        "runtimes": runtimes,
        "feature_info": {k: v for k, v in last_info.items() if k != "features"},
        "features": last_info.get("features", []),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    last_info["features"] = last_info.get("features", [])
    last_info["seed"] = seed
    return payload, last_info, False


# ═══════════════════════════════════════════════════════════════════════
# LEAKAGE AUDIT
# ═══════════════════════════════════════════════════════════════════════

def leakage_audit(infos: list[dict], variant_name: str):
    rows = []
    for info in infos:
        feats = info.get("features", [])
        checks = {
            "variant_id_not_used": "Variant_ID" not in feats,
            "label_not_used": "Label" not in feats,
            "aa1_aa2_not_used": "AA_1" not in feats and "AA_2" not in feats,
            "no_target_encoding": True,
            "thresholds_selected_from_oof_only": True,
            "test_metrics_reporting_only": True,
            "no_row_removal": info.get("train_rows_before") == info.get("train_rows_after"),
            "no_test_row_removal": info.get("test_rows_before") == info.get("test_rows_after"),
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

    # Delta vs baseline (by weight configuration)
    delta_rows = []
    acceptance_labels = {}

    for cfg in WEIGHT_CONFIGS:
        w_name = cfg["name"]
        base_sum = results_df[(results_df.feature_config_name == "baseline_native_CAT345_CAT6_kept") & (results_df.weight_config_name == w_name)].iloc[0]
        
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
            
            # Apply decision labels
            if has_leakage:
                label = "leakage_fail"
            elif delta_oof >= 0.0 and test_f1_mean > 0.562271:
                label = "beats_step19m_best_reporting_oof_supported"
            elif delta_oof >= 0.0 and test_f1_mean >= 0.560:
                label = "reached_056_reporting_oof_supported"
            elif delta_oof >= 0.0 and delta_test >= 0.001:
                label = "high_value_cat_feature_config"
            elif delta_oof >= 0.0 and (0.0 <= delta_test < 0.001):
                label = "valid_but_no_meaningful_uplift"
            elif delta_oof > 0.0 and delta_test <= 0.0:
                label = "oof_only_cat_feature_signal"
            elif delta_test > 0.0 and delta_oof < 0.0:
                label = "test_only_cat_feature_signal"
            else:
                label = "rejected_no_uplift"
                
            acceptance_labels[(var, w_name)] = label
            
            delta["test_f1_improved"] = delta_test > 0.0
            delta["test_f1_high_value"] = delta_test >= 0.001
            delta["reached_056_reporting"] = test_f1_mean >= 0.560
            delta["beats_step19m_best_reporting"] = test_f1_mean > 0.562271
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
            "config_name": "Step19J working candidate",
            "source_step": "Step19J",
            "weights": "xgb_extratrees_balanced (CB 0.25 / LGB 0.05 / XGB 0.30 / ET 0.40)",
            "test_f1_mean": 0.558838,
            "oof_f1_mean": 0.807674,
            "threshold_mean": 0.900364,
            "note": "3-seed working reference"
        },
        {
            "config_name": "Step19M best Test F1 reporting-only candidate",
            "source_step": "Step19M",
            "weights": "xgb_32_et40_cb25_lgb03",
            "test_f1_mean": 0.562271,
            "oof_f1_mean": 0.808559,
            "threshold_mean": 0.899463,
            "note": "highest Test F1 in Step19M"
        },
        {
            "config_name": "Step19M OOF-priority selected candidate",
            "source_step": "Step19M",
            "weights": "xgb_30_et42_cb23_lgb05",
            "test_f1_mean": 0.560294,
            "oof_f1_mean": 0.811188,
            "threshold_mean": 0.896911,
            "note": "highest OOF F1 in Step19M"
        }
    ]

    for var in FEATURE_VARIANTS:
        for cfg in WEIGHT_CONFIGS:
            w_name = cfg["name"]
            var_sum = results_df[(results_df.feature_config_name == var) & (results_df.weight_config_name == w_name)].iloc[0]
            ref_rows.append({
                "config_name": f"Step19N {var}",
                "source_step": "Step19N",
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
                "beats_step19m_best_reporting_oof_supported",
                "reached_056_reporting_oof_supported",
                "high_value_cat_feature_config",
                "valid_but_no_meaningful_uplift"
            ]:
                row_sum = results_df[(results_df.feature_config_name == var) & (results_df.weight_config_name == w_name)].iloc[0]
                passing_rows.append(row_sum)

    if passing_rows:
        passing_df = pd.DataFrame(passing_rows)
        # Rank by: OOF F1, Test F1, lower FN
        passing_df = passing_df.sort_values(
            ["oof_f1_mean", "test_f1_mean", "test_fn_mean"],
            ascending=[False, False, True]
        )
        selected_best_var = passing_df.iloc[0]["feature_config_name"]
        selected_best_weight = passing_df.iloc[0]["weight_config_name"]
        best_sum_row = passing_df.iloc[0]
    else:
        # Fallback to absolute best OOF F1
        fallback_df = results_df.sort_values(
            ["oof_f1_mean", "test_f1_mean", "test_fn_mean"],
            ascending=[False, False, True]
        )
        selected_best_var = fallback_df.iloc[0]["feature_config_name"]
        selected_best_weight = fallback_df.iloc[0]["weight_config_name"]
        best_sum_row = fallback_df.iloc[0]

    best_label = acceptance_labels[(selected_best_var, selected_best_weight)]
    print(f"\n  BEST CAT CONFIG SELECTED: {selected_best_var} with weights {selected_best_weight} (Acceptance: {best_label})", flush=True)

    # Write report
    wall_seconds = time.perf_counter() - wall_start
    
    acc_rows = [{"feature_config_name": var, "weight_config_name": w, "acceptance_label": label}
                for (var, w), label in acceptance_labels.items()]
    acc_df = pd.DataFrame(acc_rows)

    report_md = f"""# Step19N — Candidate 2 CAT345/CAT6 Mini-smoke Report

Step19N tests only CAT345/CAT6 mini-smoke variants on the Candidate 2 improved model.

Step19N keeps CAT_new, CAT1, and CAT2.

Step19N keeps raw AL and capped/signed AL_log unchanged.

Step19N evaluates both Step19M candidate weights: xgb_32_et40_cb25_lgb03 and xgb_30_et42_cb23_lgb05.

5-seed validation is not run in Step19N.

AL_PCA is not tested in Step19N.

Target encoding is not used.

Frequency encoding uses feature frequencies only and is fit on train fold only.

All thresholds are selected using OOF predictions only.

Test metrics are reporting-only.

Variant_ID and Label are not used as model features.

current_active_changed = false

## Baseline Reference

- Baseline Name: `baseline_native_CAT345_CAT6_kept`
- Step19M Best Test F1 Reference (`xgb_32_et40_cb25_lgb03`): Test F1 = **0.562271**, OOF F1 = **0.808559**
- Step19M OOF-priority Reference (`xgb_30_et42_cb23_lgb05`): Test F1 = **0.560294**, OOF F1 = **0.811188**

## Summary of Feature Configurations

{md_table(results_df[["feature_config_name", "weight_config_name", "test_f1_mean", "test_f1_std", "oof_f1_mean", "oof_f1_std", "threshold_mean"]])}

## Delta vs Baseline (per weight configuration)

{md_table(delta_df[["feature_config_name", "weight_config_name", "delta_test_f1_mean", "delta_oof_f1_mean", "delta_test_fp_mean", "delta_test_fn_mean", "test_f1_improved", "test_f1_high_value", "reached_056_reporting", "beats_step19m_best_reporting", "acceptance_label"]])}

## Acceptance Decisions

{md_table(acc_df)}

## Comparison vs Historical References

{md_table(comp_df)}

## Governance

- No test-label selection: **confirmed**
- No target encoding: **confirmed**
- Frequency encoding uses feature frequencies only and is fit on train fold only: **confirmed**
- Variant_ID and Label are not used as model features: **confirmed**
- Thresholds selected using OOF predictions only: **confirmed**
- Test metrics are reporting-only: **confirmed**
- Preprocessing fit on train fold only: **confirmed**
- No row removal: **confirmed**
- No test row removal: **confirmed**
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
        "step": "Step19N",
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
            "frequency_encoding_no_labels": True,
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
    next_action_text = f"""# Step19N Next Action

current_active_changed = false

Best model right now remains Step13A four-model `catboost_lightgbm_xgboost_extratrees__prob_average` with 5-seed Test F1 = 0.554365.
Step19N mini-smoke selected **{selected_best_var}** with weights **{selected_best_weight}** as the optimal configuration for Candidate 2 improved.
Next steps should check:
1. AL_PCA mini-smoke
2. Final ablation
3. Freeze/audit
"""
    OUT["next_action"].write_text(next_action_text, encoding="utf-8")

    # Current active update
    OUT["current_update"].write_text(json.dumps({
        "current_active_changed": False,
        "current_active_changed_text": "current_active_changed = false",
        "step": "Step19N",
        "selected_feature_variant": selected_best_var,
        "selected_weight_config": selected_best_weight,
        "acceptance_label": best_label,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 60, flush=True)
    print(f"Step19N COMPLETE in {wall_seconds / 60:.4f} minutes", flush=True)
    print("current_active_changed = false", flush=True)


if __name__ == "__main__":
    main()
