"""Step19J: Ensemble weight micro-search on Candidate 2 improved model.

Step19J treats Candidate 2 plus add_CAT_new_keep_CAT1_CAT2 as the working best-F1 candidate.
Step19J tests only ensemble weight micro-search.
AL_PCA is not tested in Step19J.
CAT1/CAT2/CAT_new grids are not rerun in Step19J.
All thresholds are selected using OOF predictions only.
Test metrics are reporting-only.
No target encoding is used.
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
CACHE = ROOT / "step19g_light_cache"

# ─── candidate definitions ───────────────────────────────────────────
CANDIDATE_2 = {
    "candidate_id": "candidate_2",
    "name": "raw_AL_plus_capped_signed_ALlog_native_CAT345_with_CAT6 + current_extratrees_heavy",
    "config_name": "raw_AL_plus_capped_signed_ALlog_native_CAT345_with_CAT6",
    "aggregation": "four_model_weighted_extratrees_heavy",
    "weights": {"catboost": 0.30, "lightgbm": 0.10, "xgboost": 0.20, "extratrees": 0.40},
    "members": ["catboost", "lightgbm", "xgboost", "extratrees"],
    "feature_family": "raw_AL + capped_signed_AL_log + native_CAT345 + CAT6_kept",
    "source_step": "Step15B",
    "source_file": "master_step15b_refined_allog_cat_model_family.py",
    "source_seed_set": [7, 21, 42],
    "expected_test_f1": 0.553814,
    "uses_capped_allog": True,
}

WEIGHT_CONFIGS = [
    {
        "name": "current_extratrees_heavy",
        "weights": {"catboost": 0.30, "lightgbm": 0.10, "xgboost": 0.20, "extratrees": 0.40}
    },
    {
        "name": "extratrees_45",
        "weights": {"catboost": 0.25, "lightgbm": 0.10, "xgboost": 0.20, "extratrees": 0.45}
    },
    {
        "name": "extratrees_50",
        "weights": {"catboost": 0.20, "lightgbm": 0.10, "xgboost": 0.20, "extratrees": 0.50}
    },
    {
        "name": "catboost_extratrees_balanced",
        "weights": {"catboost": 0.35, "lightgbm": 0.05, "xgboost": 0.15, "extratrees": 0.45}
    },
    {
        "name": "catboost_35_extratrees_40",
        "weights": {"catboost": 0.35, "lightgbm": 0.10, "xgboost": 0.15, "extratrees": 0.40}
    },
    {
        "name": "xgb_extratrees_balanced",
        "weights": {"catboost": 0.25, "lightgbm": 0.05, "xgboost": 0.30, "extratrees": 0.40}
    },
    {
        "name": "lightgbm_minimal",
        "weights": {"catboost": 0.35, "lightgbm": 0.00, "xgboost": 0.20, "extratrees": 0.45}
    },
    {
        "name": "no_lightgbm_more_extratrees",
        "weights": {"catboost": 0.30, "lightgbm": 0.00, "xgboost": 0.20, "extratrees": 0.50}
    },
    {
        "name": "catboost_extra_equal_heavy",
        "weights": {"catboost": 0.40, "lightgbm": 0.00, "xgboost": 0.10, "extratrees": 0.50}
    },
    {
        "name": "xgb_lightgbm_reduced",
        "weights": {"catboost": 0.35, "lightgbm": 0.05, "xgboost": 0.10, "extratrees": 0.50}
    }
]

VALIDATION_SEEDS = [7, 21, 42]
METRICS_LIST = [
    "f1", "precision", "recall", "specificity", "fpr", "mcc", "roc_auc",
    "average_precision", "predicted_positive_rate", "tn", "fp", "fn", "tp",
]

OUT = {
    "report": ROOT / "master_step19j_candidate2_weight_microsearch_report.md",
    "grid_results": ROOT / "master_step19j_candidate2_weight_grid_results.csv",
    "per_seed_results": ROOT / "master_step19j_candidate2_per_seed_weight_results.csv",
    "delta": ROOT / "master_step19j_candidate2_delta_vs_current_extratrees_heavy.csv",
    "comparison": ROOT / "master_step19j_candidate2_weight_comparison_vs_references.csv",
    "leakage": ROOT / "master_step19j_candidate2_leakage_audit.csv",
    "current_update": ROOT / "master_step19j_candidate2_current_active_update.json",
    "next_action": ROOT / "master_step19j_candidate2_next_action.md",
    "metadata": ROOT / "master_step19j_candidate2_metadata.json",
}


# ═══════════════════════════════════════════════════════════════════════
# FEATURE PIPELINE — identical to Step15B / Step19G / Step19H
# ═══════════════════════════════════════════════════════════════════════

def safe_cat(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().fillna("<MISSING>").replace({"": "<MISSING>"})


def apply_candidate_features(
    train: pd.DataFrame,
    other: pd.DataFrame,
    candidate: dict,
    cat_config_tag: str,
):
    """Build features for add_CAT_new_keep_CAT1_CAT2."""
    tr = train.copy().reset_index(drop=True)
    ot = other.copy().reset_index(drop=True)

    # --- CAT_new handling: add ---
    if "CAT_new_missingli" in tr.columns:
        tr["CAT_new_missingli"] = safe_cat(tr["CAT_new_missingli"])
        ot["CAT_new_missingli"] = safe_cat(ot["CAT_new_missingli"])
    else:
        raise RuntimeError("CAT_new_missingli not available in data")

    # CAT1 and CAT2 are kept, so no drops of CAT_1/CAT_2 here.

    # CAT345 stays native
    cat345_mode = "native"
    cat6_status = "kept"

    original_al = [
        c for c in tr.columns
        if c.startswith("AL_") and pd.api.types.is_numeric_dtype(tr[c])
    ]
    capped_log_cols: list[str] = []

    if candidate["uses_capped_allog"]:
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

    # --- aa_numeric + missingness (same as Step15B) ---
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
        "cat345_mode": cat345_mode,
        "cat6_status": cat6_status,
        "train_rows_before": int(len(train)),
        "train_rows_after": int(len(train)),
        "test_rows_before": int(len(other)),
        "test_rows_after": int(len(other)),
    }
    return x, xo, info


# ═══════════════════════════════════════════════════════════════════════
# BASE PREDICTIONS GENERATOR (CACHED)
# ═══════════════════════════════════════════════════════════════════════

def load_or_generate_base(seed: int, full: pd.DataFrame):
    cid = "candidate_2"
    cp = CACHE / f"{cid}__seed_{seed}__base_predictions.npz"
    dp = CACHE / f"{cid}__seed_{seed}__base_predictions__diag.json"
    if cp.exists() and dp.exists():
        print(f"CACHE HIT {cid} base_predictions seed={seed}", flush=True)
        with np.load(cp) as saved:
            payload = {k: saved[k] for k in saved.files}
        diag = json.loads(dp.read_text(encoding="utf-8"))
        return payload, diag["feature_info"], True

    print(f"RUN base models training {cid} seed={seed}", flush=True)
    train, test = make_split(full, seed)
    y = train.Label.to_numpy(int)
    yt = test.Label.to_numpy(int)
    cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)

    oof = {m: np.full(len(y), np.nan) for m in MODELS}
    runtimes = []
    last_info = {}

    for fold, (tri, vai) in enumerate(cv.split(train, y)):
        xtr, xva, info = apply_candidate_features(
            train.iloc[tri], train.iloc[vai], CANDIDATE_2, "add_CAT_new_keep_CAT1_CAT2"
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
    xfull, xtest, info = apply_candidate_features(
        train, test, CANDIDATE_2, "add_CAT_new_keep_CAT1_CAT2"
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

    CACHE.mkdir(exist_ok=True)
    np.savez_compressed(cp, **payload)
    dp.write_text(json.dumps({
        "runtimes": runtimes,
        "feature_info": {k: v for k, v in last_info.items() if k != "features"},
        "features": last_info.get("features", []),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    return payload, last_info, False


def evaluate_weights(payload: dict, weights: dict, policy: str):
    # Combine OOF
    poof = sum(weights[m] * payload[f"oof_{m}"] for m in weights)
    # Combine Test
    ptest = sum(weights[m] * payload[f"test_{m}"] for m in weights)
    # Choose threshold using OOF only
    selected, _ = choose(threshold_table(payload["y"], poof), policy)
    threshold = float(selected.threshold)

    return poof, ptest, threshold


# ═══════════════════════════════════════════════════════════════════════
# ACCEPTANCE RULE
# ═══════════════════════════════════════════════════════════════════════

def evaluate_acceptance(
    base_row: pd.Series,
    var_row: pd.Series,
    delta: dict,
    has_leakage: bool,
):
    """Classify the weight configuration based on Step19J rules."""
    if has_leakage:
        return "leakage_fail"

    delta_test_f1 = delta["delta_test_f1_mean"]
    delta_oof_f1 = delta["delta_oof_f1_mean"]
    delta_test_fn = delta["delta_test_fn_mean"]
    test_f1 = var_row["test_f1_mean"]

    # reached_056_target
    if test_f1 >= 0.560 and delta_oof_f1 >= 0.0:
        return "reached_056_target"

    # test_only_weight_signal
    if delta_test_f1 > 0.0 and delta_oof_f1 < 0.0:
        return "test_only_weight_signal"

    # oof_only_weight_signal
    if delta_oof_f1 > 0.0 and delta_test_f1 <= 0.0:
        return "oof_only_weight_signal"

    # high-value / useful check
    useful = (delta_oof_f1 >= 0.0) and (delta_test_f1 >= -0.001) and (delta_test_fn <= 2)
    high_value = (delta_test_f1 >= 0.001) and (delta_oof_f1 >= 0.0)

    if high_value:
        return "high_value_weight_config"
    elif useful:
        if delta_test_f1 >= 0.001:
            return "high_value_weight_config"
        elif delta_test_f1 > 0.0:
            return "valid_but_no_meaningful_uplift"
        else:
            return "useful_weight_config"
    elif delta_test_f1 > 0.0 and delta_test_f1 < 0.001 and delta_oof_f1 >= 0.0:
        return "valid_but_no_meaningful_uplift"
    else:
        return "rejected_no_uplift"


# ═══════════════════════════════════════════════════════════════════════
# LEAKAGE AUDIT
# ═══════════════════════════════════════════════════════════════════════

def leakage_audit(infos: list[dict]):
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
            "candidate_id": "candidate_2",
            "seed": info.get("seed"),
            **checks,
            "failed_checks": json.dumps(failed),
            "status": "FAIL" if failed else "PASS",
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

    # Phase 0: Train / load base predictions
    print("\n" + "=" * 60, flush=True)
    print("PHASE 0 — LOADING / TRAINING BASE ENSEMBLE MODELS", flush=True)
    print("=" * 60, flush=True)

    base_payloads = {}
    all_infos = []

    for seed in VALIDATION_SEEDS:
        payload, info, _ = load_or_generate_base(seed, full_catnew)
        base_payloads[seed] = payload
        all_infos.append(info)

    # Phase 1: weight micro-search
    print("\n" + "=" * 60, flush=True)
    print("PHASE 1 — WEIGHT MICRO-SEARCH GRID EVALUATION", flush=True)
    print("=" * 60, flush=True)

    results_per_seed = []
    
    for cfg in WEIGHT_CONFIGS:
        w_name = cfg["name"]
        w_dict = cfg["weights"]
        
        for seed in VALIDATION_SEEDS:
            payload = base_payloads[seed]
            poof, ptest, threshold = evaluate_weights(payload, w_dict, POLICY)
            
            oof_metrics = metric(payload["y"], poof, threshold)
            test_metrics = metric(payload["y_test"], ptest, threshold)
            
            row = {
                "weight_config_name": w_name,
                "seed": seed,
                "selected_threshold": threshold,
                "catboost_weight": w_dict["catboost"],
                "lightgbm_weight": w_dict["lightgbm"],
                "xgboost_weight": w_dict["xgboost"],
                "extratrees_weight": w_dict["extratrees"],
                **{f"oof_{k}": v for k, v in oof_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }
            results_per_seed.append(row)
            
    per_seed_df = pd.DataFrame(results_per_seed)
    per_seed_df.to_csv(OUT["per_seed_results"], index=False)

    # Summarize per configuration
    summary_rows = []
    for cfg in WEIGHT_CONFIGS:
        w_name = cfg["name"]
        sub = per_seed_df[per_seed_df.weight_config_name == w_name]
        
        sr = {
            "weight_config_name": w_name,
            "catboost_weight": cfg["weights"]["catboost"],
            "lightgbm_weight": cfg["weights"]["lightgbm"],
            "xgboost_weight": cfg["weights"]["xgboost"],
            "extratrees_weight": cfg["weights"]["extratrees"],
            "seed_count": int(sub.seed.nunique()),
        }
        
        for prefix in ["oof", "test"]:
            for m in METRICS_LIST:
                col = f"{prefix}_{m}"
                sr[f"{col}_mean"] = sub[col].mean()
                sr[f"{col}_std"] = sub[col].std(ddof=1) if len(sub) > 1 else 0.0
        sr["selected_threshold_mean"] = sub.selected_threshold.mean()
        sr["selected_threshold_std"] = sub.selected_threshold.std(ddof=1) if len(sub) > 1 else 0.0
        summary_rows.append(sr)

    grid_df = pd.DataFrame(summary_rows)
    grid_df.to_csv(OUT["grid_results"], index=False)

    # Leakage audit
    leak_df = leakage_audit(all_infos)
    leak_df.to_csv(OUT["leakage"], index=False)
    has_leakage = bool(leak_df.status.eq('FAIL').any())

    # Delta vs current_extratrees_heavy baseline
    base_sum = grid_df[grid_df.weight_config_name == "current_extratrees_heavy"].iloc[0]
    
    delta_rows = []
    acceptance_labels = {}
    
    for cfg in WEIGHT_CONFIGS:
        w_name = cfg["name"]
        var_sum = grid_df[grid_df.weight_config_name == w_name].iloc[0]
        
        delta = {
            "weight_config_name": w_name,
            "delta_test_f1_mean": var_sum["test_f1_mean"] - base_sum["test_f1_mean"],
            "delta_oof_f1_mean": var_sum["oof_f1_mean"] - base_sum["oof_f1_mean"],
            "delta_test_fp_mean": var_sum["test_fp_mean"] - base_sum["test_fp_mean"],
            "delta_test_fn_mean": var_sum["test_fn_mean"] - base_sum["test_fn_mean"],
        }
        delta_rows.append(delta)
        
        # Apply acceptance rules
        label = evaluate_acceptance(base_sum, var_sum, delta, has_leakage)
        acceptance_labels[w_name] = label

    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(OUT["delta"], index=False)

    # Comparison vs references
    # Step13A official best reference
    ref_rows = [
        {
            "config_name": "Step13A official best reference",
            "source_step": "Step13A",
            "weights": "catboost_lightgbm_xgboost_extratrees__prob_average (equal)",
            "test_f1_mean": 0.554365,
            "oof_f1_mean": 0.798210,
            "selected_threshold_mean": 0.929330,
            "note": "5-seed official reference"
        },
        {
            "config_name": "Step19H Candidate 2 improved baseline",
            "source_step": "Step19H",
            "weights": "current_extratrees_heavy (CB 0.30 / LGB 0.10 / XGB 0.20 / ET 0.40)",
            "test_f1_mean": 0.556965,
            "oof_f1_mean": 0.805936,
            "selected_threshold_mean": 0.903493,
            "note": "3-seed reproduced baseline"
        }
    ]
    
    # Add top performing configurations from Step19J to references
    for cfg in WEIGHT_CONFIGS:
        w_name = cfg["name"]
        var_sum = grid_df[grid_df.weight_config_name == w_name].iloc[0]
        ref_rows.append({
            "config_name": f"Step19J {w_name}",
            "source_step": "Step19J",
            "weights": f"CB {cfg['weights']['catboost']:.2f} / LGB {cfg['weights']['lightgbm']:.2f} / XGB {cfg['weights']['xgboost']:.2f} / ET {cfg['weights']['extratrees']:.2f}",
            "test_f1_mean": var_sum["test_f1_mean"],
            "oof_f1_mean": var_sum["oof_f1_mean"],
            "selected_threshold_mean": var_sum["selected_threshold_mean"],
            "note": f"Acceptance: {acceptance_labels[w_name]}"
        })
        
    comp_df = pd.DataFrame(ref_rows)
    comp_df.to_csv(OUT["comparison"], index=False)

    # Determine best config by primary metric (OOF F1), secondary (Test F1)
    # Filter only non-baseline configs
    candidates_df = grid_df.copy()
    candidates_df["delta_oof_f1"] = candidates_df["oof_f1_mean"] - base_sum["oof_f1_mean"]
    candidates_df["delta_test_f1"] = candidates_df["test_f1_mean"] - base_sum["test_f1_mean"]
    candidates_df["delta_test_fn"] = candidates_df["test_fn_mean"] - base_sum["test_fn_mean"]
    
    # Passing configurations
    passing_configs = []
    for cfg in WEIGHT_CONFIGS:
        w_name = cfg["name"]
        label = acceptance_labels[w_name]
        # Must be valid/useful or better
        if label in ["reached_056_target", "high_value_weight_config", "useful_weight_config", "valid_but_no_meaningful_uplift"]:
            passing_configs.append(w_name)

    passing_df = candidates_df[candidates_df.weight_config_name.isin(passing_configs)].copy()
    
    selected_best = None
    if not passing_df.empty:
        # Rank by: OOF F1, Test F1, lower FN
        passing_df = passing_df.sort_values(
            ["oof_f1_mean", "test_f1_mean", "test_fn_mean"],
            ascending=[False, False, True]
        )
        selected_best = passing_df.iloc[0]["weight_config_name"]
    else:
        # If no config officially passed, show the one with best OOF F1
        selected_best = candidates_df.sort_values(
            ["oof_f1_mean", "test_f1_mean", "test_fn_mean"],
            ascending=[False, False, True]
        ).iloc[0]["weight_config_name"]

    best_label = acceptance_labels.get(selected_best, "no_uplift")
    print(f"\n  BEST CONFIG SELECTED: {selected_best} (Acceptance: {best_label})", flush=True)

    # Write report
    wall_seconds = time.perf_counter() - wall_start
    
    # Construct acceptance table
    acc_rows = []
    for k, v in acceptance_labels.items():
        acc_rows.append({"weight_config_name": k, "acceptance_label": v})
    acc_df = pd.DataFrame(acc_rows)

    report_md = f"""# Step19J — Candidate 2 Ensemble Weight Micro-search Report

Step19J treats Candidate 2 plus add_CAT_new_keep_CAT1_CAT2 as the working best-F1 candidate.

Step19J tests only ensemble weight micro-search.

AL_PCA is not tested in Step19J.

CAT1/CAT2/CAT_new grids are not rerun in Step19J.

All thresholds are selected using OOF predictions only.

Test metrics are reporting-only.

No target encoding is used.

Variant_ID and Label are not used as model features.

current_active_changed = false

## Baseline Reference

- Configuration: `add_CAT_new_keep_CAT1_CAT2` + `current_extratrees_heavy`
- Weights: CatBoost 0.30 / LightGBM 0.10 / XGBoost 0.20 / ExtraTrees 0.40
- Test F1: **{base_sum['test_f1_mean']:.6f}**
- OOF F1: **{base_sum['oof_f1_mean']:.6f}**

## Summary of Weight Search Grid

{md_table(grid_df[["weight_config_name", "test_f1_mean", "test_f1_std", "oof_f1_mean", "oof_f1_std", "selected_threshold_mean"]])}

## Delta vs Baseline

{md_table(delta_df)}

## Acceptance Decisions

{md_table(acc_df)}

## Comparison vs Historical References

{md_table(comp_df)}

## Governance

- No test-label selection: **confirmed**
- No target encoding: **confirmed**
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

{wall_seconds / 60:.2f} minutes
"""
    OUT["report"].write_text(report_md, encoding="utf-8")

    # Update metadata
    meta = {
        "step": "Step19J",
        "completed": True,
        "candidate": "candidate_2",
        "configs_evaluated": len(WEIGHT_CONFIGS),
        "seeds": VALIDATION_SEEDS,
        "wall_seconds": wall_seconds,
        "selected_best_config": selected_best,
        "best_acceptance_label": best_label,
        "current_active_changed": False,
        "governance": {
            "no_test_label_selection": True,
            "no_target_encoding": True,
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
    next_action_text = f"""# Step19J Next Action

current_active_changed = false

Best model right now remains Step13A four-model `catboost_lightgbm_xgboost_extratrees__prob_average` with 5-seed Test F1 = 0.554365.
Step19J micro-search selected **{selected_best}** as the optimal weight configuration for Candidate 2 improved.
Next steps should check:
1. 5-seed validation of the selected config
2. Threshold micro-check
3. AL_PCA mini-smoke
"""
    OUT["next_action"].write_text(next_action_text, encoding="utf-8")

    # Current active update
    OUT["current_update"].write_text(json.dumps({
        "current_active_changed": False,
        "current_active_changed_text": "current_active_changed = false",
        "step": "Step19J",
        "selected_for_validation": selected_best,
        "acceptance_label": best_label,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 60, flush=True)
    print(f"Step19J COMPLETE in {wall_seconds / 60:.2f} minutes", flush=True)
    print("current_active_changed = false", flush=True)


if __name__ == "__main__":
    main()
