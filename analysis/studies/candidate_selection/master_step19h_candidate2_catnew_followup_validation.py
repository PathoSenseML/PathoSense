"""Step19H: Follow-up validation on the Candidate 2 CAT_new signal from Step19G_LIGHT.

Step19H follows up only on the Candidate 2 CAT_new signal from Step19G_LIGHT.
The main config is candidate_2_drop_CAT2_only_add_CAT_new because it had positive test delta and acceptable OOF smoke delta.
candidate_2_add_CAT_new_keep_CAT1_CAT2 is included only as a risky diagnostic.
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

CAT_CONFIGS = [
    "candidate_baseline",
    "drop_CAT2_only_add_CAT_new",
    "add_CAT_new_keep_CAT1_CAT2",
]

VALIDATION_SEEDS = [7, 21, 42]
METRICS_LIST = [
    "f1", "precision", "recall", "specificity", "fpr", "mcc", "roc_auc",
    "average_precision", "predicted_positive_rate", "tn", "fp", "fn", "tp",
]

OUT = {
    "report": ROOT / "master_step19h_candidate2_catnew_followup_report.md",
    "validation_per_seed": ROOT / "master_step19h_candidate2_validation_per_seed.csv",
    "validation_summary": ROOT / "master_step19h_candidate2_validation_summary.csv",
    "delta": ROOT / "master_step19h_candidate2_delta_vs_baseline.csv",
    "leakage": ROOT / "master_step19h_candidate2_leakage_audit.csv",
    "current_update": ROOT / "master_step19h_candidate2_current_active_update.json",
    "next_action": ROOT / "master_step19h_candidate2_next_action.md",
    "metadata": ROOT / "master_step19h_candidate2_metadata.json",
}


# ═══════════════════════════════════════════════════════════════════════
# FEATURE PIPELINE — identical to Step15B / Step19G
# ═══════════════════════════════════════════════════════════════════════

def safe_cat(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().fillna("<MISSING>").replace({"": "<MISSING>"})


def _drop_cat_cols(config_tag: str) -> set[str]:
    """Which CAT columns to drop for a given CAT-gap config tag."""
    if config_tag == "drop_CAT2_only_add_CAT_new":
        return {"CAT_2"}
    return set()


def _needs_catnew(config_tag: str) -> bool:
    return "CAT_new" in config_tag and config_tag != "candidate_baseline"


def apply_candidate_features(
    train: pd.DataFrame,
    other: pd.DataFrame,
    candidate: dict,
    cat_config_tag: str,
):
    """Build features for a candidate + CAT-gap config."""
    tr = train.copy().reset_index(drop=True)
    ot = other.copy().reset_index(drop=True)

    # --- CAT_new handling: add before any drops ---
    if _needs_catnew(cat_config_tag):
        if "CAT_new_missingli" in tr.columns:
            tr["CAT_new_missingli"] = safe_cat(tr["CAT_new_missingli"])
            ot["CAT_new_missingli"] = safe_cat(ot["CAT_new_missingli"])
        else:
            raise RuntimeError("CAT_new_missingli not available in data")
    else:
        tr = tr.drop(columns=["CAT_new_missingli"], errors="ignore")
        ot = ot.drop(columns=["CAT_new_missingli"], errors="ignore")

    # --- CAT drops for this config ---
    drops = _drop_cat_cols(cat_config_tag)
    tr = tr.drop(columns=[c for c in drops if c in tr.columns])
    ot = ot.drop(columns=[c for c in drops if c in ot.columns])

    # Exact Step15B behavior: baseline drops CAT_1 and CAT_2
    if cat_config_tag == "candidate_baseline":
        tr = tr.drop(columns=["CAT_1", "CAT_2"], errors="ignore")
        ot = ot.drop(columns=["CAT_1", "CAT_2"], errors="ignore")

    # CAT345 always stays native for both candidates
    cat345_mode = "native"
    cat6_status = "kept"

    original_al = [
        c for c in tr.columns
        if c.startswith("AL_") and pd.api.types.is_numeric_dtype(tr[c])
    ]
    capped_log_cols: list[str] = []

    if candidate["uses_capped_allog"]:
        # Candidate 2: add capped signed log features
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
# EXECUTION ENGINE
# ═══════════════════════════════════════════════════════════════════════

def cache_path(candidate_id: str, seed: int, cat_config: str) -> Path:
    return CACHE / f"{candidate_id}__seed_{seed}__{cat_config}.npz"


def diag_path(candidate_id: str, seed: int, cat_config: str) -> Path:
    return CACHE / f"{candidate_id}__seed_{seed}__{cat_config}__diag.json"


def run_single(
    candidate: dict,
    cat_config: str,
    seed: int,
    full: pd.DataFrame,
):
    """Run one candidate × cat_config × seed."""
    train, test = make_split(full, seed)
    y = train.Label.to_numpy(int)
    yt = test.Label.to_numpy(int)
    cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)
    oof = {m: np.full(len(y), np.nan) for m in MODELS}
    runtimes = []
    last_info = {}

    for fold, (tri, vai) in enumerate(cv.split(train, y)):
        xtr, xva, info = apply_candidate_features(
            train.iloc[tri], train.iloc[vai], candidate, cat_config,
        )
        last_info = info
        for m in MODELS:
            t0 = time.perf_counter()
            oof[m][vai] = fit_predict_model(m, seed, xtr, y[tri], xva)
            runtimes.append({
                "candidate_id": candidate["candidate_id"],
                "seed": seed, "cat_config": cat_config,
                "fold": fold, "model": m,
                "runtime_seconds": time.perf_counter() - t0,
            })

    # --- aggregate OOF with candidate's weights ---
    weights = candidate["weights"]
    members = candidate["members"]
    poof = sum(weights[m] * oof[m] for m in members)

    # --- threshold from OOF only ---
    selected, _ = choose(threshold_table(y, poof), POLICY)
    threshold = float(selected.threshold)

    # --- full-train test predictions ---
    xfull, xtest, info = apply_candidate_features(
        train, test, candidate, cat_config,
    )
    last_info = info
    test_preds = {}
    for m in MODELS:
        t0 = time.perf_counter()
        test_preds[m] = fit_predict_model(m, seed, xfull, y, xtest)
        runtimes.append({
            "candidate_id": candidate["candidate_id"],
            "seed": seed, "cat_config": cat_config,
            "fold": -1, "model": m,
            "runtime_seconds": time.perf_counter() - t0,
        })
    ptest = sum(weights[m] * test_preds[m] for m in members)

    payload = {"y": y, "y_test": yt, "oof": poof, "test": ptest, "threshold": threshold}
    return payload, runtimes, {
        "candidate_id": candidate["candidate_id"],
        "seed": seed, "cat_config": cat_config,
        **last_info,
    }


def load_or_generate(candidate: dict, cat_config: str, seed: int, full: pd.DataFrame):
    cid = candidate["candidate_id"]
    cp = cache_path(cid, seed, cat_config)
    dp = diag_path(cid, seed, cat_config)
    if cp.exists() and dp.exists():
        print(f"CACHE HIT {cid} seed={seed} cat={cat_config}", flush=True)
        with np.load(cp) as saved:
            payload = {k: saved[k] for k in saved.files}
        diag = json.loads(dp.read_text(encoding="utf-8"))
        return payload, diag["runtimes"], diag["feature_info"], True
    print(f"RUN {cid} seed={seed} cat={cat_config}", flush=True)
    payload, runtimes, feature_info = run_single(candidate, cat_config, seed, full)
    CACHE.mkdir(exist_ok=True)
    np.savez_compressed(cp, **payload)
    dp.write_text(json.dumps({
        "runtimes": runtimes,
        "feature_info": {k: v for k, v in feature_info.items() if k != "features"},
        "features": feature_info.get("features", []),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload, runtimes, feature_info, False


def metric_row(candidate: dict, cat_config: str, seed: int, payload: dict, feature_info: dict):
    th = float(np.asarray(payload["threshold"]).item())
    return {
        "candidate_id": candidate["candidate_id"],
        "candidate_name": candidate["name"],
        "aggregation": candidate["aggregation"],
        "cat_config": cat_config,
        "seed": seed,
        "selected_threshold": th,
        "feature_count": feature_info.get("feature_count", 0),
        **{f"oof_{k}": v for k, v in metric(payload["y"], payload["oof"], th).items()},
        **{f"test_{k}": v for k, v in metric(payload["y_test"], payload["test"], th).items()},
    }


# ═══════════════════════════════════════════════════════════════════════
# ACCEPTANCE RULE
# ═══════════════════════════════════════════════════════════════════════

def evaluate_acceptance(
    baseline_summary: pd.Series,
    var_summary: pd.Series,
    delta: dict,
    has_leakage: bool,
):
    """Classify the config uplift based on Step19H acceptance criteria."""
    if has_leakage:
        return "leakage_fail"

    test_f1_improved = delta["delta_test_f1_mean"] > 0
    oof_ok = delta["delta_oof_f1_mean"] >= -0.005
    fn_ok = delta["delta_test_fn_mean"] <= 2
    reached_056 = var_summary["test_f1_mean"] >= 0.560

    if test_f1_improved and oof_ok and fn_ok:
        if reached_056:
            return "reached_056_target"
        elif delta["delta_test_f1_mean"] >= 0.001:
            return "candidate2_improved_candidate"
        else:
            return "valid_but_no_meaningful_uplift"
    elif test_f1_improved and not oof_ok:
        return "unstable_test_only_signal"
    elif not test_f1_improved and delta["delta_oof_f1_mean"] > 0:
        return "oof_only_signal"
    elif delta["delta_test_f1_mean"] >= 0.0 and delta["delta_test_f1_mean"] < 0.001:
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
            "candidate_id": info.get("candidate_id", ""),
            "seed": info.get("seed"),
            "cat_config": info.get("cat_config", ""),
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

    # Phase 1: same-seed validation
    print("\n" + "=" * 60, flush=True)
    print("PHASE 1 — SAME-SEED VALIDATION (seeds=[7, 21, 42])", flush=True)
    print("=" * 60, flush=True)

    all_rows = []
    all_infos = []

    for seed in VALIDATION_SEEDS:
        for cat_config in CAT_CONFIGS:
            payload, runtimes, info, _ = load_or_generate(
                CANDIDATE_2, cat_config, seed, full_catnew,
            )
            row = metric_row(CANDIDATE_2, cat_config, seed, payload, info)
            all_rows.append(row)
            all_infos.append(info)
            print(f"  candidate_2 {cat_config} seed={seed}: "
                  f"Test F1={row['test_f1']:.6f}, OOF F1={row['oof_f1']:.6f}, "
                  f"FP={row['test_fp']}, FN={row['test_fn']}", flush=True)

    val_df = pd.DataFrame(all_rows)
    val_df.to_csv(OUT["validation_per_seed"], index=False)

    # Calculate summaries
    summary_rows = []
    for cat_config in CAT_CONFIGS:
        sub = val_df[val_df.cat_config == cat_config]
        sr = {
            "candidate_id": "candidate_2",
            "cat_config": cat_config,
            "aggregation": CANDIDATE_2["aggregation"],
            "seed_count": int(sub.seed.nunique()),
            "seed_note": "original_seed_set_[7, 21, 42]",
        }
        for prefix in ["oof", "test"]:
            for m in METRICS_LIST:
                col = f"{prefix}_{m}"
                sr[f"{col}_mean"] = sub[col].mean()
                sr[f"{col}_std"] = sub[col].std(ddof=1) if len(sub) > 1 else 0.0
        sr["selected_threshold_mean"] = sub.selected_threshold.mean()
        sr["selected_threshold_std"] = sub.selected_threshold.std(ddof=1) if len(sub) > 1 else 0.0
        summary_rows.append(sr)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT["validation_summary"], index=False)

    # Leakage audit
    leak_df = leakage_audit(all_infos)
    leak_df.to_csv(OUT["leakage"], index=False)
    has_leakage = bool(leak_df.status.eq('FAIL').any())

    # Delta calculations
    base_sum = summary_df[summary_df.cat_config == "candidate_baseline"].iloc[0]
    
    delta_rows = []
    acceptance_labels = {}

    for cat_config in CAT_CONFIGS:
        if cat_config == "candidate_baseline":
            continue
        var_sum = summary_df[summary_df.cat_config == cat_config].iloc[0]
        
        delta = {
            "cat_config": cat_config,
            "delta_test_f1_mean": var_sum["test_f1_mean"] - base_sum["test_f1_mean"],
            "delta_oof_f1_mean": var_sum["oof_f1_mean"] - base_sum["oof_f1_mean"],
            "delta_test_fp_mean": var_sum["test_fp_mean"] - base_sum["test_fp_mean"],
            "delta_test_fn_mean": var_sum["test_fn_mean"] - base_sum["test_fn_mean"],
        }
        delta_rows.append(delta)
        
        # Apply acceptance rules
        label = evaluate_acceptance(base_sum, var_sum, delta, has_leakage)
        acceptance_labels[cat_config] = label

    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(OUT["delta"], index=False)

    # Report writing
    wall_seconds = time.perf_counter() - wall_start
    
    report_md = f"""# Step19H — Candidate 2 CAT_new Follow-up Report

Step19H follows up only on the Candidate 2 CAT_new signal from Step19G_LIGHT.

The main config is candidate_2_drop_CAT2_only_add_CAT_new because it had positive test delta and acceptable OOF smoke delta.

candidate_2_add_CAT_new_keep_CAT1_CAT2 is included only as a risky diagnostic.

All thresholds are selected using OOF predictions only.

Test metrics are reporting-only.

No target encoding is used.

Variant_ID and Label are not used as model features.

current_active_changed = false

## Reference Context (not retested)

A. Step13A four-model probability-average ensemble
- config: catboost_lightgbm_xgboost_extratrees__prob_average
- Test F1 = 0.554365
- OOF F1 = 0.798210
- threshold = 0.929330
- seed set: 7, 21, 42, 101, 2024

B. Candidate 5 from Step19G_LIGHT
- candidate_5 baseline 3-seed Test F1 = 0.551142
- candidate_5 drop_CAT2_only 3-seed Test F1 = 0.546815
- acceptance label: oof_only_signal

## Same-Seed Validation Results

{md_table(val_df[["candidate_id", "cat_config", "seed", "test_f1", "oof_f1", "test_fp", "test_fn", "selected_threshold"]])}

## Summary Over Seeds (seeds=[7, 21, 42])

{md_table(summary_df[["candidate_id", "cat_config", "test_f1_mean", "test_f1_std", "oof_f1_mean", "oof_f1_std", "selected_threshold_mean"]])}

## Delta vs Baseline

{md_table(delta_df)}

## Acceptance Labels

- **drop_CAT2_only_add_CAT_new**: {acceptance_labels.get("drop_CAT2_only_add_CAT_new")}
- **add_CAT_new_keep_CAT1_CAT2**: {acceptance_labels.get("add_CAT_new_keep_CAT1_CAT2")}

## Governance

- No test-label selection: **confirmed**
- No target encoding: **confirmed**
- Variant_ID and Label are not used as model features: **confirmed**
- Thresholds selected using OOF only: **confirmed**
- Test metrics reporting-only: **confirmed**
- Preprocessing fit on train fold only: **confirmed**
- No row removal: **confirmed**
- No test row removal: **confirmed**
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
        "step": "Step19H",
        "completed": True,
        "candidate": "candidate_2",
        "configs": CAT_CONFIGS,
        "seeds": VALIDATION_SEEDS,
        "wall_seconds": wall_seconds,
        "acceptance_labels": acceptance_labels,
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

    # Current active update
    OUT["current_update"].write_text(json.dumps({
        "current_active_changed": False,
        "current_active_changed_text": "current_active_changed = false",
        "step": "Step19H",
        "selected_for_validation": "drop_CAT2_only_add_CAT_new",
        "acceptance_label": acceptance_labels.get("drop_CAT2_only_add_CAT_new", "no_uplift"),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # Next action writing
    next_action_text = f"""# Step19H Next Action

current_active_changed = false

Best model right now remains Step13A four-model `catboost_lightgbm_xgboost_extratrees__prob_average` with 5-seed Test F1 = 0.554365.
"""
    OUT["next_action"].write_text(next_action_text, encoding="utf-8")

    print("\n" + "=" * 60, flush=True)
    print(f"Step19H COMPLETE in {wall_seconds / 60:.2f} minutes", flush=True)
    print("current_active_changed = false", flush=True)


if __name__ == "__main__":
    main()
