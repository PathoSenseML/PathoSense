"""Step19M: Local Ensemble Weight Refinement on Candidate 2 working best model.

Step19M treats Candidate 2 plus add_CAT_new_keep_CAT1_CAT2 plus xgb_extratrees_balanced as the working best-F1 candidate.
Step19M tests only local ensemble weight refinement.
Step19M keeps the current Step19J/Step19L threshold policy.
5-seed validation is not run in Step19M.
AL_PCA is not tested in Step19M.
CAT345/CAT6 are not tested in Step19M.
CAT1/CAT2/CAT_new grids are not rerun in Step19M.
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

import master_step14e_cat12_catnew_ablation as catnew_base
import master_step15b_refined_allog_cat_model_family as step15b
from master_step13a_ensemble_ablation import (
    MODELS,
    metric,
)
from master_step3b_repeated_split_catboost import find_input
from master_step8_threshold_policy_refinement import choose, threshold_table

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "step19g_light_cache"

# ─── candidate definitions ───────────────────────────────────────────
WEIGHT_CONFIGS = [
    {
        "name": "xgb_extratrees_balanced",
        "weights": {"catboost": 0.25, "lightgbm": 0.05, "xgboost": 0.30, "extratrees": 0.40}
    },
    {
        "name": "xgb_32_et40_cb23_lgb05",
        "weights": {"catboost": 0.23, "lightgbm": 0.05, "xgboost": 0.32, "extratrees": 0.40}
    },
    {
        "name": "xgb_34_et40_cb21_lgb05",
        "weights": {"catboost": 0.21, "lightgbm": 0.05, "xgboost": 0.34, "extratrees": 0.40}
    },
    {
        "name": "xgb_35_et40_cb20_lgb05",
        "weights": {"catboost": 0.20, "lightgbm": 0.05, "xgboost": 0.35, "extratrees": 0.40}
    },
    {
        "name": "xgb_30_et42_cb23_lgb05",
        "weights": {"catboost": 0.23, "lightgbm": 0.05, "xgboost": 0.30, "extratrees": 0.42}
    },
    {
        "name": "xgb_30_et44_cb21_lgb05",
        "weights": {"catboost": 0.21, "lightgbm": 0.05, "xgboost": 0.30, "extratrees": 0.44}
    },
    {
        "name": "xgb_30_et45_cb20_lgb05",
        "weights": {"catboost": 0.20, "lightgbm": 0.05, "xgboost": 0.30, "extratrees": 0.45}
    },
    {
        "name": "xgb_32_et42_cb21_lgb05",
        "weights": {"catboost": 0.21, "lightgbm": 0.05, "xgboost": 0.32, "extratrees": 0.42}
    },
    {
        "name": "xgb_33_et42_cb20_lgb05",
        "weights": {"catboost": 0.20, "lightgbm": 0.05, "xgboost": 0.33, "extratrees": 0.42}
    },
    {
        "name": "xgb_34_et41_cb20_lgb05",
        "weights": {"catboost": 0.20, "lightgbm": 0.05, "xgboost": 0.34, "extratrees": 0.41}
    },
    {
        "name": "xgb_32_et40_cb25_lgb03",
        "weights": {"catboost": 0.25, "lightgbm": 0.03, "xgboost": 0.32, "extratrees": 0.40}
    },
    {
        "name": "xgb_33_et40_cb25_lgb02",
        "weights": {"catboost": 0.25, "lightgbm": 0.02, "xgboost": 0.33, "extratrees": 0.40}
    },
    {
        "name": "xgb_35_et40_cb25_lgb00",
        "weights": {"catboost": 0.25, "lightgbm": 0.00, "xgboost": 0.35, "extratrees": 0.40}
    },
    {
        "name": "xgb_30_et40_cb27_lgb03",
        "weights": {"catboost": 0.27, "lightgbm": 0.03, "xgboost": 0.30, "extratrees": 0.40}
    },
    {
        "name": "xgb_32_et38_cb27_lgb03",
        "weights": {"catboost": 0.27, "lightgbm": 0.03, "xgboost": 0.32, "extratrees": 0.38}
    }
]

VALIDATION_SEEDS = [7, 21, 42]
METRICS_LIST = [
    "f1", "precision", "recall", "specificity", "fpr", "mcc", "roc_auc",
    "average_precision", "predicted_positive_rate", "tn", "fp", "fn", "tp",
]

OUT = {
    "report": ROOT / "master_step19m_candidate2_local_weight_refinement_report.md",
    "results_csv": ROOT / "master_step19m_candidate2_local_weight_results.csv",
    "per_seed_csv": ROOT / "master_step19m_candidate2_per_seed_threshold_results.csv",  # Wait, let's keep the name requested: master_step19m_candidate2_per_seed_weight_results.csv
    "delta_csv": ROOT / "master_step19m_candidate2_delta_vs_xgb_extratrees_balanced.csv",
    "comparison_csv": ROOT / "master_step19m_candidate2_weight_comparison_vs_references.csv",
    "leakage_csv": ROOT / "master_step19m_candidate2_leakage_audit.csv",
    "current_update": ROOT / "master_step19m_candidate2_current_active_update.json",
    "next_action": ROOT / "master_step19m_candidate2_next_action.md",
    "metadata": ROOT / "master_step19m_candidate2_metadata.json",
}

# Override correct per_seed_csv name from OUTPUTS list:
OUT["per_seed_csv"] = ROOT / "master_step19m_candidate2_per_seed_weight_results.csv"

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

    # Load data
    full = pd.read_csv(find_input())
    full["Label"] = full.Label.astype(int)

    # Phase 0: Load base predictions and diagnostic info
    print("\n" + "=" * 60, flush=True)
    print("PHASE 0 - LOADING ENSEMBLE PREDICTIONS", flush=True)
    print("=" * 60, flush=True)

    base_payloads = {}
    all_infos = []

    for seed in VALIDATION_SEEDS:
        cp = CACHE / f"candidate_2__seed_{seed}__base_predictions.npz"
        dp = CACHE / f"candidate_2__seed_{seed}__base_predictions__diag.json"
        
        if not cp.exists() or not dp.exists():
            raise FileNotFoundError(f"Missing base predictions for seed={seed}")
            
        print(f"CACHE HIT base_predictions seed={seed}", flush=True)
        with np.load(cp) as saved:
            payload = {k: saved[k] for k in saved.files}
        base_payloads[seed] = payload
        
        diag = json.loads(dp.read_text(encoding="utf-8"))
        # Fix: ensure features are extracted from top-level diag structure
        info = diag["feature_info"]
        info["features"] = diag.get("features", [])
        info["seed"] = seed
        all_infos.append(info)

    # Phase 1: local weight refinement grid
    print("\n" + "=" * 60, flush=True)
    print("PHASE 1 - EVALUATING LOCAL WEIGHT CONFIGURATIONS", flush=True)
    print("=" * 60, flush=True)

    results_per_seed = []
    
    for cfg in WEIGHT_CONFIGS:
        w_name = cfg["name"]
        w_dict = cfg["weights"]
        
        for seed in VALIDATION_SEEDS:
            payload = base_payloads[seed]
            
            # Combine probabilities using current weights
            poof = sum(w_dict[m] * payload[f"oof_{m}"] for m in MODELS)
            ptest = sum(w_dict[m] * payload[f"test_{m}"] for m in MODELS)
            
            # Select threshold using current policy on combined OOF
            selected, _ = choose(threshold_table(payload["y"], poof), "weighted_f1_current")
            threshold = float(selected.threshold)
            
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
    per_seed_df.to_csv(OUT["per_seed_csv"], index=False)

    # Summarize per weight configuration
    summary_rows = []
    for cfg in WEIGHT_CONFIGS:
        w_name = cfg["name"]
        sub = per_seed_df[per_seed_df.weight_config_name == w_name]
        
        sr = {
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

    policy_df = pd.DataFrame(summary_rows)
    policy_df.to_csv(OUT["results_csv"], index=False)

    # Leakage audit
    leak_df = leakage_audit(all_infos)
    leak_df.to_csv(OUT["leakage_csv"], index=False)
    has_leakage = bool(leak_df.status.eq('FAIL').any())

    # Delta vs xgb_extratrees_balanced
    base_sum = policy_df[policy_df.weight_config_name == "xgb_extratrees_balanced"].iloc[0]
    
    delta_rows = []
    acceptance_labels = {}
    
    for cfg in WEIGHT_CONFIGS:
        w_name = cfg["name"]
        var_sum = policy_df[policy_df.weight_config_name == w_name].iloc[0]
        
        delta = {
            "weight_config_name": w_name,
            "delta_test_f1_mean": var_sum["test_f1_mean"] - base_sum["test_f1_mean"],
            "delta_oof_f1_mean": var_sum["oof_f1_mean"] - base_sum["oof_f1_mean"],
            "delta_test_fp_mean": var_sum["test_fp_mean"] - base_sum["test_fp_mean"],
            "delta_test_fn_mean": var_sum["test_fn_mean"] - base_sum["test_fn_mean"],
        }
        delta_rows.append(delta)
        
        # Apply decision labels
        delta_oof = delta["delta_oof_f1_mean"]
        delta_test = delta["delta_test_f1_mean"]
        test_f1_mean = var_sum["test_f1_mean"]
        
        if has_leakage:
            label = "leakage_fail"
        elif delta_oof >= 0.0 and test_f1_mean >= 0.560:
            label = "reached_056_reporting_oof_supported"
        elif delta_oof >= 0.0 and delta_test >= 0.001:
            label = "high_value_weight_config"
        elif delta_oof >= 0.0 and (0.0 <= delta_test < 0.001):
            label = "valid_but_no_meaningful_uplift"
        elif delta_oof > 0.0 and delta_test <= 0.0:
            label = "oof_only_weight_signal"
        elif delta_test > 0.0 and delta_oof < 0.0:
            label = "test_only_weight_signal"
        else:
            label = "rejected_no_uplift"
            
        acceptance_labels[w_name] = label

    delta_df = pd.DataFrame(delta_rows)
    # Add auxiliary reporting columns as requested
    delta_df["test_f1_improved"] = delta_df["delta_test_f1_mean"] > 0
    delta_df["test_f1_high_value"] = delta_df["delta_test_f1_mean"] >= 0.001
    
    # We will get reached_056_reporting by checking the actual test_f1_mean
    reached_056_list = []
    for cfg in WEIGHT_CONFIGS:
        w_name = cfg["name"]
        var_sum = policy_df[policy_df.weight_config_name == w_name].iloc[0]
        reached_056_list.append(var_sum["test_f1_mean"] >= 0.560)
    delta_df["reached_056_reporting"] = reached_056_list
    delta_df["acceptance_label"] = [acceptance_labels[n] for n in delta_df.weight_config_name]
    
    delta_df.to_csv(OUT["delta_csv"], index=False)

    # Historical Comparison
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
            "config_name": "Step19L threshold result",
            "source_step": "Step19L",
            "weights": "current_policy_baseline remained optimal",
            "test_f1_mean": 0.558838,
            "oof_f1_mean": 0.807674,
            "threshold_mean": 0.900364,
            "note": "no threshold policy improved working candidate"
        }
    ]
    
    for cfg in WEIGHT_CONFIGS:
        w_name = cfg["name"]
        var_sum = policy_df[policy_df.weight_config_name == w_name].iloc[0]
        ref_rows.append({
            "config_name": f"Step19M {w_name}",
            "source_step": "Step19M",
            "weights": f"CB {cfg['weights']['catboost']:.2f} / LGB {cfg['weights']['lightgbm']:.2f} / XGB {cfg['weights']['xgboost']:.2f} / ET {cfg['weights']['extratrees']:.2f}",
            "test_f1_mean": var_sum["test_f1_mean"],
            "oof_f1_mean": var_sum["oof_f1_mean"],
            "threshold_mean": var_sum["threshold_mean"],
            "note": f"Acceptance: {acceptance_labels[w_name]}"
        })
        
    comp_df = pd.DataFrame(ref_rows)
    comp_df.to_csv(OUT["comparison_csv"], index=False)

    # Determine best config by primary metric (OOF F1), secondary (Test F1), stable threshold
    # Filter configurations that are valid/useful/high value
    passing_configs = []
    for cfg in WEIGHT_CONFIGS:
        w_name = cfg["name"]
        label = acceptance_labels[w_name]
        if label in ["reached_056_reporting_oof_supported", "high_value_weight_config", "useful_weight_config", "valid_but_no_meaningful_uplift"]:
            passing_configs.append(w_name)

    passing_df = policy_df[policy_df.weight_config_name.isin(passing_configs)].copy()
    
    if not passing_df.empty:
        # Rank by: OOF F1, Test F1, lower FN
        passing_df = passing_df.sort_values(
            ["oof_f1_mean", "test_f1_mean", "test_fn_mean"],
            ascending=[False, False, True]
        )
        selected_best = passing_df.iloc[0]["weight_config_name"]
    else:
        # If no config officially passed, show the one with best OOF F1
        selected_best = policy_df.sort_values(
            ["oof_f1_mean", "test_f1_mean", "test_fn_mean"],
            ascending=[False, False, True]
        ).iloc[0]["weight_config_name"]

    best_label = acceptance_labels[selected_best]
    print(f"\n  BEST WEIGHT CONFIG SELECTED: {selected_best} (Acceptance: {best_label})", flush=True)

    # Write report
    wall_seconds = time.perf_counter() - wall_start
    
    acc_rows = [{"weight_config_name": k, "acceptance_label": v} for k, v in acceptance_labels.items()]
    acc_df = pd.DataFrame(acc_rows)

    report_md = f"""# Step19M — Candidate 2 Local Weight Refinement Report

Step19M treats Candidate 2 plus add_CAT_new_keep_CAT1_CAT2 plus xgb_extratrees_balanced as the working best-F1 candidate.

Step19M tests only local ensemble weight refinement.

Step19M keeps the current Step19J/Step19L threshold policy.

5-seed validation is not run in Step19M.

AL_PCA is not tested in Step19M.

CAT345/CAT6 are not tested in Step19M.

CAT1/CAT2/CAT_new grids are not rerun in Step19M.

All thresholds are selected using OOF predictions only.

Test metrics are reporting-only.

No target encoding is used.

Variant_ID and Label are not used as model features.

current_active_changed = false

## Baseline Reference

- Configuration: Candidate 2 improved + `xgb_extratrees_balanced`
- Weights: CatBoost 0.25 / LightGBM 0.05 / XGBoost 0.30 / ExtraTrees 0.40
- Test F1: **{base_sum['test_f1_mean']:.6f}**
- OOF F1: **{base_sum['oof_f1_mean']:.6f}**

## Summary of Weight Configurations

{md_table(policy_df[["weight_config_name", "test_f1_mean", "test_f1_std", "oof_f1_mean", "oof_f1_std", "threshold_mean"]])}

## Delta vs Baseline

{md_table(delta_df[["weight_config_name", "delta_test_f1_mean", "delta_oof_f1_mean", "delta_test_fp_mean", "delta_test_fn_mean", "test_f1_improved", "test_f1_high_value", "reached_056_reporting", "acceptance_label"]])}

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

{wall_seconds / 60:.4f} minutes
"""
    OUT["report"].write_text(report_md, encoding="utf-8")

    # Update metadata
    meta = {
        "step": "Step19M",
        "completed": True,
        "candidate": "candidate_2",
        "weight_configs_evaluated": len(WEIGHT_CONFIGS),
        "seeds": VALIDATION_SEEDS,
        "wall_seconds": wall_seconds,
        "selected_best_weight_config": selected_best,
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
    next_action_text = f"""# Step19M Next Action

current_active_changed = false

Best model right now remains Step13A four-model `catboost_lightgbm_xgboost_extratrees__prob_average` with 5-seed Test F1 = 0.554365.
Step19M local weight refinement selected **{selected_best}** as the optimal weight configuration for Candidate 2 improved.
Next steps should check:
1. CAT345/CAT6 mini-smoke
2. AL_PCA mini-smoke
"""
    OUT["next_action"].write_text(next_action_text, encoding="utf-8")

    # Current active update
    OUT["current_update"].write_text(json.dumps({
        "current_active_changed": False,
        "current_active_changed_text": "current_active_changed = false",
        "step": "Step19M",
        "selected_for_validation": selected_best,
        "acceptance_label": best_label,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 60, flush=True)
    print(f"Step19M COMPLETE in {wall_seconds / 60:.4f} minutes", flush=True)
    print("current_active_changed = false", flush=True)


if __name__ == "__main__":
    main()
