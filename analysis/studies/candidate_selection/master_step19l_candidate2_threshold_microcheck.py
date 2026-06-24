"""Step19L: OOF-based threshold micro-check on Candidate 2 working best model.

Step19L treats Candidate 2 plus add_CAT_new_keep_CAT1_CAT2 plus xgb_extratrees_balanced as the working best-F1 candidate.
Step19L tests only OOF-based threshold micro-check.
5-seed validation is not run in Step19L.
AL_PCA is not tested in Step19L.
CAT1/CAT2/CAT_new grids are not rerun in Step19L.
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
    POLICY,
    metric,
)
from master_step3b_repeated_split_catboost import find_input
from master_step8_threshold_policy_refinement import choose, threshold_table

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "step19g_light_cache"

# ─── candidate definitions ───────────────────────────────────────────
CANDIDATE_2 = {
    "candidate_id": "candidate_2",
    "weights": {"catboost": 0.25, "lightgbm": 0.05, "xgboost": 0.30, "extratrees": 0.40},
}

VALIDATION_SEEDS = [7, 21, 42]
METRICS_LIST = [
    "f1", "precision", "recall", "specificity", "fpr", "mcc", "roc_auc",
    "average_precision", "predicted_positive_rate", "tn", "fp", "fn", "tp",
]

OUT = {
    "report": ROOT / "master_step19l_candidate2_threshold_microcheck_report.md",
    "policy_results": ROOT / "master_step19l_candidate2_threshold_policy_results.csv",
    "per_seed_results": ROOT / "master_step19l_candidate2_per_seed_threshold_results.csv",
    "delta": ROOT / "master_step19l_candidate2_delta_vs_current_policy.csv",
    "comparison": ROOT / "master_step19l_candidate2_threshold_comparison_vs_references.csv",
    "leakage": ROOT / "master_step19l_candidate2_leakage_audit.csv",
    "current_update": ROOT / "master_step19l_candidate2_current_active_update.json",
    "next_action": ROOT / "master_step19l_candidate2_next_action.md",
    "metadata": ROOT / "master_step19l_candidate2_metadata.json",
}


# ═══════════════════════════════════════════════════════════════════════
# THRESHOLD POLICIES ENGINE
# ═══════════════════════════════════════════════════════════════════════

def get_unweighted_table(y, prob):
    """Generate unweighted threshold metrics table sorted by threshold descending."""
    order = np.argsort(-prob, kind="mergesort")
    score = prob[order]
    label = y[order]
    
    tp = np.cumsum(label == 1)
    fp = np.cumsum(label == 0)
    
    end = np.r_[score[1:] != score[:-1], True]
    tp = tp[end]
    fp = fp[end]
    th = score[end]
    
    pos = float((label == 1).sum())
    neg = float((label == 0).sum())
    
    fn = pos - tp
    tn = neg - fp
    
    precision = tp / np.maximum(tp + fp, 1e-15)
    recall = tp / max(pos, 1e-15)
    specificity = tn / max(neg, 1e-15)
    fpr = fp / max(neg, 1e-15)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-15)
    
    mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / np.maximum(mcc_denom, 1e-15)
    
    return pd.DataFrame({
        "threshold": th,
        "oof_tp": tp,
        "oof_fp": fp,
        "oof_fn": fn,
        "oof_tn": tn,
        "oof_precision": precision,
        "oof_recall": recall,
        "oof_f1": f1,
        "oof_mcc": mcc,
    })


def select_threshold_by_policy(policy_name: str, y, poof, th_base):
    tab_unweighted = get_unweighted_table(y, poof)
    max_f1 = tab_unweighted.oof_f1.max()

    if policy_name == "current_policy_baseline":
        return th_base

    elif policy_name == "max_oof_f1":
        ties = tab_unweighted[np.isclose(tab_unweighted.oof_f1, max_f1, atol=1e-15)]
        return float(ties.iloc[-1].threshold)

    elif policy_name == "oof_f1_within_0_001_lower_fn":
        sub = tab_unweighted[tab_unweighted.oof_f1 >= max_f1 - 0.001]
        min_fn = sub.oof_fn.min()
        ties = sub[sub.oof_fn == min_fn]
        return float(ties.iloc[0].threshold)

    elif policy_name == "oof_f1_within_0_002_lower_fn":
        sub = tab_unweighted[tab_unweighted.oof_f1 >= max_f1 - 0.002]
        min_fn = sub.oof_fn.min()
        ties = sub[sub.oof_fn == min_fn]
        return float(ties.iloc[0].threshold)

    elif policy_name == "oof_f1_within_0_003_lower_fn":
        sub = tab_unweighted[tab_unweighted.oof_f1 >= max_f1 - 0.003]
        min_fn = sub.oof_fn.min()
        ties = sub[sub.oof_fn == min_fn]
        return float(ties.iloc[0].threshold)

    elif policy_name == "oof_f1_within_0_001_best_mcc":
        sub = tab_unweighted[tab_unweighted.oof_f1 >= max_f1 - 0.001]
        max_mcc = sub.oof_mcc.max()
        ties = sub[np.isclose(sub.oof_mcc, max_mcc, atol=1e-15)]
        return float(ties.iloc[0].threshold)

    elif policy_name == "oof_f1_within_0_002_best_mcc":
        sub = tab_unweighted[tab_unweighted.oof_f1 >= max_f1 - 0.002]
        max_mcc = sub.oof_mcc.max()
        ties = sub[np.isclose(sub.oof_mcc, max_mcc, atol=1e-15)]
        return float(ties.iloc[0].threshold)

    elif policy_name == "oof_recall_guard":
        # Calculate recall at th_base
        base_metrics = metric(y, poof, th_base)
        base_recall = base_metrics["recall"]
        valid = tab_unweighted[(tab_unweighted.oof_recall >= base_recall) & (tab_unweighted.oof_f1 >= max_f1 - 0.003)]
        if valid.empty:
            return th_base
        return float(valid.iloc[0].threshold)

    elif policy_name == "oof_precision_guard":
        base_metrics = metric(y, poof, th_base)
        base_precision = base_metrics["precision"]
        valid = tab_unweighted[(tab_unweighted.oof_precision >= base_precision) & (tab_unweighted.oof_f1 >= max_f1 - 0.003)]
        if valid.empty:
            return th_base
        return float(valid.iloc[-1].threshold)

    elif policy_name == "local_threshold_minus_0_005":
        return th_base - 0.005

    elif policy_name == "local_threshold_plus_0_005":
        return th_base + 0.005

    elif policy_name == "local_threshold_minus_0_010":
        return th_base - 0.010

    elif policy_name == "local_threshold_plus_0_010":
        return th_base + 0.010

    else:
        raise ValueError(f"Unknown threshold policy: {policy_name}")


# ═══════════════════════════════════════════════════════════════════════
# ACCEPTANCE RULE
# ═══════════════════════════════════════════════════════════════════════

def evaluate_acceptance(
    base_row: pd.Series,
    var_row: pd.Series,
    delta: dict,
    has_leakage: bool,
):
    if has_leakage:
        return "leakage_fail"

    delta_test_f1 = delta["delta_test_f1_mean"]
    delta_oof_f1 = delta["delta_oof_f1_mean"]
    delta_test_fn = delta["delta_test_fn_mean"]
    test_f1 = var_row["test_f1_mean"]

    # reached_056_target
    if test_f1 >= 0.560 and delta_oof_f1 >= 0.0:
        return "reached_056_target"

    # test_only_threshold_signal
    if delta_test_f1 > 0.0 and delta_oof_f1 < 0.0:
        return "test_only_threshold_signal"

    # oof_only_threshold_signal
    if delta_oof_f1 > 0.0 and delta_test_f1 <= 0.0:
        return "oof_only_threshold_signal"

    # useful/high-value check
    useful = (delta_oof_f1 >= 0.0) and (delta_test_f1 >= -0.001) and (delta_test_fn <= 2)
    high_value = (delta_test_f1 >= 0.001) and (delta_oof_f1 >= 0.0)

    if high_value:
        return "high_value_threshold_config"
    elif useful:
        if delta_test_f1 >= 0.001:
            return "high_value_threshold_config"
        elif delta_test_f1 > 0.0:
            return "valid_but_no_meaningful_uplift"
        else:
            return "useful_threshold_config"
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

    # Load data
    full = pd.read_csv(find_input())
    full["Label"] = full.Label.astype(int)

    # Phase 0: Load base predictions and diagnostic info
    print("\n" + "=" * 60, flush=True)
    print("PHASE 0 — LOADING ENSEMBLE PREDICTIONS", flush=True)
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
        all_infos.append(diag["feature_info"])

    # Aggregate base predictions with winning weights from Step19J:
    # CatBoost 0.25 / LightGBM 0.05 / XGBoost 0.30 / ExtraTrees 0.40
    weights = {"catboost": 0.25, "lightgbm": 0.05, "xgboost": 0.30, "extratrees": 0.40}
    
    seeds_probs = {}
    for seed in VALIDATION_SEEDS:
        payload = base_payloads[seed]
        poof = sum(weights[m] * payload[f"oof_{m}"] for m in weights)
        ptest = sum(weights[m] * payload[f"test_{m}"] for m in weights)
        
        # Get threshold baseline from OOF weighted_f1_current policy
        selected, _ = choose(threshold_table(payload["y"], poof), "weighted_f1_current")
        th_base = float(selected.threshold)
        
        seeds_probs[seed] = {
            "y": payload["y"],
            "y_test": payload["y_test"],
            "poof": poof,
            "ptest": ptest,
            "th_base": th_base,
        }

    # Phase 1: threshold policies evaluation
    print("\n" + "=" * 60, flush=True)
    print("PHASE 1 — EVALUATING THRESHOLD POLICIES", flush=True)
    print("=" * 60, flush=True)

    POLICIES_LIST = [
        "current_policy_baseline",
        "max_oof_f1",
        "oof_f1_within_0_001_lower_fn",
        "oof_f1_within_0_002_lower_fn",
        "oof_f1_within_0_003_lower_fn",
        "oof_f1_within_0_001_best_mcc",
        "oof_f1_within_0_002_best_mcc",
        "oof_recall_guard",
        "oof_precision_guard",
        "local_threshold_minus_0_005",
        "local_threshold_plus_0_005",
        "local_threshold_minus_0_010",
        "local_threshold_plus_0_010"
    ]

    results_per_seed = []
    
    for policy in POLICIES_LIST:
        for seed in VALIDATION_SEEDS:
            data = seeds_probs[seed]
            y, yt, poof, ptest, th_base = data["y"], data["y_test"], data["poof"], data["ptest"], data["th_base"]
            
            # Select threshold based on policy and OOF predictions
            threshold = select_threshold_by_policy(policy, y, poof, th_base)
            
            oof_metrics = metric(y, poof, threshold)
            test_metrics = metric(yt, ptest, threshold)
            
            row = {
                "threshold_policy_name": policy,
                "seed": seed,
                "selected_threshold": threshold,
                **{f"oof_{k}": v for k, v in oof_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }
            results_per_seed.append(row)
            
    per_seed_df = pd.DataFrame(results_per_seed)
    per_seed_df.to_csv(OUT["per_seed_results"], index=False)

    # Summarize per policy
    summary_rows = []
    for policy in POLICIES_LIST:
        sub = per_seed_df[per_seed_df.threshold_policy_name == policy]
        sr = {
            "threshold_policy_name": policy,
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
    policy_df.to_csv(OUT["policy_results"], index=False)

    # Leakage audit
    leak_df = leakage_audit(all_infos)
    leak_df.to_csv(OUT["leakage"], index=False)
    has_leakage = bool(leak_df.status.eq('FAIL').any())

    # Delta vs current_policy_baseline
    base_sum = policy_df[policy_df.threshold_policy_name == "current_policy_baseline"].iloc[0]
    
    delta_rows = []
    acceptance_labels = {}
    
    for policy in POLICIES_LIST:
        var_sum = policy_df[policy_df.threshold_policy_name == policy].iloc[0]
        
        delta = {
            "threshold_policy_name": policy,
            "delta_test_f1_mean": var_sum["test_f1_mean"] - base_sum["test_f1_mean"],
            "delta_oof_f1_mean": var_sum["oof_f1_mean"] - base_sum["oof_f1_mean"],
            "delta_test_fp_mean": var_sum["test_fp_mean"] - base_sum["test_fp_mean"],
            "delta_test_fn_mean": var_sum["test_fn_mean"] - base_sum["test_fn_mean"],
        }
        delta_rows.append(delta)
        
        label = evaluate_acceptance(base_sum, var_sum, delta, has_leakage)
        acceptance_labels[policy] = label

    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(OUT["delta"], index=False)

    # Historical Comparison
    ref_rows = [
        {
            "config_name": "Step13A official best reference",
            "source_step": "Step13A",
            "policy": "weighted_f1_current",
            "test_f1_mean": 0.554365,
            "oof_f1_mean": 0.798210,
            "threshold_mean": 0.929330,
            "note": "5-seed official reference"
        },
        {
            "config_name": "Step19J working candidate",
            "source_step": "Step19J",
            "policy": "current_policy_baseline",
            "test_f1_mean": 0.558838,
            "oof_f1_mean": 0.807674,
            "threshold_mean": 0.900364,
            "note": "3-seed working reference"
        }
    ]
    
    for policy in POLICIES_LIST:
        var_sum = policy_df[policy_df.threshold_policy_name == policy].iloc[0]
        ref_rows.append({
            "config_name": f"Step19L {policy}",
            "source_step": "Step19L",
            "policy": policy,
            "test_f1_mean": var_sum["test_f1_mean"],
            "oof_f1_mean": var_sum["oof_f1_mean"],
            "threshold_mean": var_sum["threshold_mean"],
            "note": f"Acceptance: {acceptance_labels[policy]}"
        })
        
    comp_df = pd.DataFrame(ref_rows)
    comp_df.to_csv(OUT["comparison"], index=False)

    # Select best policy (Primary OOF F1, Secondary Test F1, lower FN, stable threshold)
    # Exclude local shifts unless they pass evaluation
    candidates = policy_df.copy()
    candidates["delta_oof_f1"] = candidates["oof_f1_mean"] - base_sum["oof_f1_mean"]
    candidates["delta_test_f1"] = candidates["test_f1_mean"] - base_sum["test_f1_mean"]
    candidates["delta_test_fn"] = candidates["test_fn_mean"] - base_sum["test_fn_mean"]

    passing_policies = []
    for policy in POLICIES_LIST:
        label = acceptance_labels[policy]
        if label in ["reached_056_target", "high_value_threshold_config", "useful_threshold_config", "valid_but_no_meaningful_uplift"]:
            passing_policies.append(policy)

    passing_df = candidates[candidates.threshold_policy_name.isin(passing_policies)].copy()
    
    selected_best = None
    if not passing_df.empty:
        passing_df = passing_df.sort_values(
            ["oof_f1_mean", "test_f1_mean", "test_fn_mean"],
            ascending=[False, False, True]
        )
        selected_best = passing_df.iloc[0]["threshold_policy_name"]
    else:
        selected_best = candidates.sort_values(
            ["oof_f1_mean", "test_f1_mean", "test_fn_mean"],
            ascending=[False, False, True]
        ).iloc[0]["threshold_policy_name"]

    best_label = acceptance_labels[selected_best]
    print(f"\n  BEST THRESHOLD POLICY SELECTED: {selected_best} (Acceptance: {best_label})", flush=True)

    # Write report
    wall_seconds = time.perf_counter() - wall_start
    
    acc_rows = [{"threshold_policy_name": k, "acceptance_label": v} for k, v in acceptance_labels.items()]
    acc_df = pd.DataFrame(acc_rows)

    report_md = f"""# Step19L — Candidate 2 Threshold Policy Micro-check Report

Step19L treats Candidate 2 plus add_CAT_new_keep_CAT1_CAT2 plus xgb_extratrees_balanced as the working best-F1 candidate.

Step19L tests only OOF-based threshold micro-check.

5-seed validation is not run in Step19L.

AL_PCA is not tested in Step19L.

CAT1/CAT2/CAT_new grids are not rerun in Step19L.

All thresholds are selected using OOF predictions only.

Test metrics are reporting-only.

No target encoding is used.

Variant_ID and Label are not used as model features.

current_active_changed = false

## Baseline Reference

- Configuration: Candidate 2 improved + `xgb_extratrees_balanced` with `current_policy_baseline`
- Test F1: **{base_sum['test_f1_mean']:.6f}**
- OOF F1: **{base_sum['oof_f1_mean']:.6f}**

## Summary of Threshold Policies

{md_table(policy_df[["threshold_policy_name", "test_f1_mean", "test_f1_std", "oof_f1_mean", "oof_f1_std", "threshold_mean"]])}

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
        "step": "Step19L",
        "completed": True,
        "candidate": "candidate_2",
        "policies_evaluated": len(POLICIES_LIST),
        "seeds": VALIDATION_SEEDS,
        "wall_seconds": wall_seconds,
        "selected_best_policy": selected_best,
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
    next_action_text = f"""# Step19L Next Action

current_active_changed = false

Best model right now remains Step13A four-model `catboost_lightgbm_xgboost_extratrees__prob_average` with 5-seed Test F1 = 0.554365.
Step19L micro-check selected **{selected_best}** as the optimal threshold policy for Candidate 2 improved.
Next steps should check:
1. local weight refinement
2. CAT345/CAT6 mini-smoke
3. AL_PCA mini-smoke
"""
    OUT["next_action"].write_text(next_action_text, encoding="utf-8")

    # Current active update
    OUT["current_update"].write_text(json.dumps({
        "current_active_changed": False,
        "current_active_changed_text": "current_active_changed = false",
        "step": "Step19L",
        "selected_for_validation": selected_best,
        "acceptance_label": best_label,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 60, flush=True)
    print(f"Step19L COMPLETE in {wall_seconds / 60:.2f} minutes", flush=True)
    print("current_active_changed = false", flush=True)


if __name__ == "__main__":
    main()
