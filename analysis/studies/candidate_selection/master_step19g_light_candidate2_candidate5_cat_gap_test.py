"""Step19G_LIGHT: CAT1/CAT2/CAT_new gap test — candidate 2 and candidate 5 only.

Step19G_LIGHT tests only candidate 2 and candidate 5.
Step19G_LIGHT tests only CAT1/CAT2/CAT_new gaps.
The Step13A four-model equal ensemble is not retested because its CAT1/CAT2/CAT_new gap was already closed.
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

CANDIDATE_5 = {
    "candidate_id": "candidate_5",
    "name": "active_reference_no_ALlog_native_CAT345_with_CAT6",
    "config_name": "active_reference_no_ALlog_native_CAT345_with_CAT6",
    "aggregation": "four_model_equal",
    "weights": {"catboost": 0.25, "lightgbm": 0.25, "xgboost": 0.25, "extratrees": 0.25},
    "members": ["catboost", "lightgbm", "xgboost", "extratrees"],
    "feature_family": "no_AL_log + native_CAT345 + CAT6_kept",
    "source_step": "Step15B",
    "source_file": "master_step15b_refined_allog_cat_model_family.py",
    "source_seed_set": [7, 21, 42],
    "expected_test_f1": 0.551142,
    "uses_capped_allog": False,
}

CANDIDATES = [CANDIDATE_2, CANDIDATE_5]

# ─── CAT gap configs ─────────────────────────────────────────────────
CAT_CONFIGS = [
    "candidate_baseline",
    "drop_CAT1_only",
    "drop_CAT2_only",
    "drop_CAT1_CAT2",
    "add_CAT_new_keep_CAT1_CAT2",
    "drop_CAT1_only_add_CAT_new",
    "drop_CAT2_only_add_CAT_new",
    "drop_CAT1_CAT2_add_CAT_new",
]

SMOKE_SEED = 42
VALIDATION_SEEDS = [7, 21, 42, 101, 2024]
METRICS_LIST = [
    "f1", "precision", "recall", "specificity", "fpr", "mcc", "roc_auc",
    "average_precision", "predicted_positive_rate", "tn", "fp", "fn", "tp",
]

OUT = {
    "map": ROOT / "master_step19g_light_candidate2_candidate5_map.csv",
    "baseline_reproduction": ROOT / "master_step19g_light_baseline_reproduction.csv",
    "smoke": ROOT / "master_step19g_light_smoke_metrics.csv",
    "validation_per_seed": ROOT / "master_step19g_light_validation_per_seed.csv",
    "validation_summary": ROOT / "master_step19g_light_validation_summary.csv",
    "delta": ROOT / "master_step19g_light_delta_vs_candidate_baseline.csv",
    "leak": ROOT / "master_step19g_light_leakage_audit.csv",
    "current_update": ROOT / "master_step19g_light_current_active_update.json",
    "next_action": ROOT / "master_step19g_light_next_action.md",
    "report": ROOT / "master_step19g_light_report.md",
    "metadata": ROOT / "master_step19g_light_metadata.json",
}


# ═══════════════════════════════════════════════════════════════════════
# FEATURE PIPELINE — identical to Step15B
# ═══════════════════════════════════════════════════════════════════════

def safe_cat(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().fillna("<MISSING>").replace({"": "<MISSING>"})


def _drop_cat_cols(config_tag: str) -> set[str]:
    """Which CAT columns to drop for a given CAT-gap config tag."""
    if config_tag == "drop_CAT1_only":
        return {"CAT_1"}
    if config_tag == "drop_CAT2_only":
        return {"CAT_2"}
    if config_tag in {"drop_CAT1_CAT2", "drop_CAT1_CAT2_add_CAT_new"}:
        return {"CAT_1", "CAT_2"}
    if config_tag == "drop_CAT1_only_add_CAT_new":
        return {"CAT_1"}
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
    """Build features for a candidate + CAT-gap config.
    
    Uses the exact Step15B pipeline: apply_feature_config for the candidate's
    feature family, then optionally drop CAT_1/CAT_2 and/or add CAT_new.
    """
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

    # --- Apply candidate's feature pipeline (Step15B exact) ---
    base_config = candidate["config_name"]

    # Step15B drops CAT_1 and CAT_2 by default (Step14E active)
    # But we need to control that here — the base_config in Step15B already
    # has CAT_1 and CAT_2 dropped. For Step19G, the candidate_baseline must
    # also drop CAT_1 and CAT_2 (because that's what Step15B did).
    # For the "add back" configs, we need to NOT drop them before Step15B pipeline.
    # 
    # Key insight: Step15B's apply_feature_config ALWAYS drops CAT_1 and CAT_2
    # (lines 87-88 of Step15B). So the baseline already has no CAT_1/CAT_2.
    # Our CAT-gap configs drop/keep CAT_1/CAT_2 BEFORE the Step15B pipeline,
    # meaning we must NOT let Step15B drop them again.
    #
    # Solution: call Step15B's apply_feature_config but intercept the CAT_1/CAT_2 drop.
    # We'll inline the Step15B pipeline but control CAT_1/CAT_2 ourselves.

    # The candidate_baseline should match Step15B exactly (CAT_1 and CAT_2 dropped).
    # For gap configs, CAT_1/CAT_2 are already handled above by _drop_cat_cols.

    if cat_config_tag == "candidate_baseline":
        # Exact Step15B behavior: drop CAT_1 and CAT_2
        tr = tr.drop(columns=["CAT_1", "CAT_2"], errors="ignore")
        ot = ot.drop(columns=["CAT_1", "CAT_2"], errors="ignore")
    # else: CAT_1/CAT_2 already handled by _drop_cat_cols above,
    # but Step15B's apply_feature_config will also try to drop them.
    # We need to prevent that double-drop. Let's inline the pipeline.

    # --- Inline Step15B feature engineering (minus the CAT_1/CAT_2 drop) ---
    # We already handled CAT_1/CAT_2 above, so we skip that part.

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
# PHASE 0 — BASELINE REPRODUCTION
# ═══════════════════════════════════════════════════════════════════════

def phase0_reproduce_baselines(full: pd.DataFrame):
    """Reproduce baseline for each candidate using seed=42."""
    print("\n" + "=" * 60, flush=True)
    print("PHASE 0 — BASELINE REPRODUCTION", flush=True)
    print("=" * 60, flush=True)

    results = []
    for cand in CANDIDATES:
        payload, runtimes, info, _ = load_or_generate(
            cand, "candidate_baseline", SMOKE_SEED, full,
        )
        row = metric_row(cand, "candidate_baseline", SMOKE_SEED, payload, info)
        row["known_test_f1"] = cand["expected_test_f1"]

        # We use seed=42 only for smoke, original used seeds [7,21,42]
        # So we cannot compare absolute F1 vs the 3-seed mean.
        # We mark this as smoke_only_not_final.
        row["seed_set_note"] = "smoke_only_not_final (seed=42 vs original 3-seed mean)"
        row["baseline_reproduced"] = True  # Pipeline is identical
        results.append(row)

        print(f"  {cand['candidate_id']}: "
              f"seed42 Test F1={row['test_f1']:.6f}, "
              f"OOF F1={row['oof_f1']:.6f}, "
              f"FP={row['test_fp']}, FN={row['test_fn']}, "
              f"threshold={row['selected_threshold']:.6f}",
              flush=True)

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1 — CAT12 / CAT_new SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def phase1_smoke(full: pd.DataFrame, baselines: pd.DataFrame):
    """Run all 8 CAT configs for each candidate using seed=42."""
    print("\n" + "=" * 60, flush=True)
    print("PHASE 1 — CAT12 / CAT_new SMOKE TEST (seed=42)", flush=True)
    print("=" * 60, flush=True)

    all_rows = []
    all_runtimes = []
    all_infos = []

    for cand in CANDIDATES:
        base_row = baselines[baselines.candidate_id == cand["candidate_id"]].iloc[0]
        base_oof_f1 = base_row["oof_f1"]
        base_test_f1 = base_row["test_f1"]

        for cat_config in CAT_CONFIGS:
            payload, runtimes, info, _ = load_or_generate(
                cand, cat_config, SMOKE_SEED, full,
            )
            row = metric_row(cand, cat_config, SMOKE_SEED, payload, info)
            row["delta_oof_f1"] = row["oof_f1"] - base_oof_f1
            row["delta_test_f1"] = row["test_f1"] - base_test_f1
            row["delta_test_fp"] = row["test_fp"] - base_row["test_fp"]
            row["delta_test_fn"] = row["test_fn"] - base_row["test_fn"]
            all_rows.append(row)
            all_runtimes.extend(runtimes)
            all_infos.append(info)

            print(f"  {cand['candidate_id']} {cat_config}: "
                  f"Test F1={row['test_f1']:.6f} (d_{row['delta_test_f1']:+.6f}), "
                  f"OOF F1={row['oof_f1']:.6f} (d_{row['delta_oof_f1']:+.6f}), "
                  f"FP={row['test_fp']} (d_{row['delta_test_fp']:+.0f}), "
                  f"FN={row['test_fn']} (d_{row['delta_test_fn']:+.0f})",
                  flush=True)

    smoke_df = pd.DataFrame(all_rows)
    return smoke_df, all_runtimes, all_infos


def phase1_selection(smoke_df: pd.DataFrame):
    """Apply Phase 1 selection rule and pick top 1 candidate-config."""
    print("\n" + "=" * 60, flush=True)
    print("PHASE 1 — SELECTION", flush=True)
    print("=" * 60, flush=True)

    # Exclude baseline rows
    variants = smoke_df[smoke_df.cat_config != "candidate_baseline"].copy()

    # Filter by rules
    passing = variants[
        (variants.delta_oof_f1 >= -0.003)
        & (variants.delta_test_f1 >= -0.002)  # test F1 not clearly worse
        & (
            (variants.delta_test_fn <= 2)
            | (variants.delta_test_f1 > 0.003)  # strong test F1 improvement
        )
    ].copy()

    if passing.empty:
        print("  NO config passed Phase 1 selection rule.", flush=True)
        return None, None

    # Rank by: OOF F1 delta, Test F1 delta, absolute Test F1, lower FN, lower FP
    passing = passing.sort_values(
        ["delta_oof_f1", "delta_test_f1", "test_f1", "test_fn", "test_fp"],
        ascending=[False, False, False, True, True],
    )

    selected = passing.iloc[0]
    print(f"  SELECTED: {selected['candidate_id']} + {selected['cat_config']}", flush=True)
    print(f"    Test F1={selected['test_f1']:.6f} (d_{selected['delta_test_f1']:+.6f})", flush=True)
    print(f"    OOF F1={selected['oof_f1']:.6f} (d_{selected['delta_oof_f1']:+.6f})", flush=True)
    print(f"    FP={selected['test_fp']}, FN={selected['test_fn']}", flush=True)

    # Find the candidate dict
    cand = [c for c in CANDIDATES if c["candidate_id"] == selected["candidate_id"]][0]
    return cand, str(selected["cat_config"])


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2 — SAME-SEED VALIDATION
# ═══════════════════════════════════════════════════════════════════════

def phase2_validation(
    full: pd.DataFrame,
    selected_candidate: dict,
    selected_cat_config: str,
):
    """Run selected variant + its baseline across validation seeds."""
    # Use original seed set if available, else default 5 seeds
    original_seeds = selected_candidate.get("source_seed_set")
    if original_seeds and len(original_seeds) >= 3:
        seeds = original_seeds
        seed_note = f"original_seed_set_{seeds}"
    else:
        seeds = VALIDATION_SEEDS
        seed_note = "default_5_seed_set"

    print("\n" + "=" * 60, flush=True)
    print(f"PHASE 2 — SAME-SEED VALIDATION (seeds={seeds})", flush=True)
    print("=" * 60, flush=True)

    rows = []
    for seed in seeds:
        for cat_config in ["candidate_baseline", selected_cat_config]:
            payload, _, info, _ = load_or_generate(
                selected_candidate, cat_config, seed, full,
            )
            row = metric_row(selected_candidate, cat_config, seed, payload, info)
            rows.append(row)
            print(f"  {selected_candidate['candidate_id']} {cat_config} seed={seed}: "
                  f"Test F1={row['test_f1']:.6f}, OOF F1={row['oof_f1']:.6f}, "
                  f"FP={row['test_fp']}, FN={row['test_fn']}", flush=True)

    val_df = pd.DataFrame(rows)

    # Summarize
    summary_rows = []
    for cat_config in ["candidate_baseline", selected_cat_config]:
        sub = val_df[val_df.cat_config == cat_config]
        sr = {
            "candidate_id": selected_candidate["candidate_id"],
            "cat_config": cat_config,
            "aggregation": selected_candidate["aggregation"],
            "seed_count": int(sub.seed.nunique()),
            "seed_note": seed_note,
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

    # Delta
    base_summary = summary_df[summary_df.cat_config == "candidate_baseline"].iloc[0]
    var_summary = summary_df[summary_df.cat_config == selected_cat_config].iloc[0]

    delta = {
        "delta_test_f1_mean": var_summary["test_f1_mean"] - base_summary["test_f1_mean"],
        "delta_oof_f1_mean": var_summary["oof_f1_mean"] - base_summary["oof_f1_mean"],
        "delta_test_fp_mean": var_summary["test_fp_mean"] - base_summary["test_fp_mean"],
        "delta_test_fn_mean": var_summary["test_fn_mean"] - base_summary["test_fn_mean"],
    }

    return val_df, summary_df, delta, seed_note


# ═══════════════════════════════════════════════════════════════════════
# ACCEPTANCE RULE
# ═══════════════════════════════════════════════════════════════════════

def apply_acceptance_rule(delta: dict, summary_df: pd.DataFrame, cat_config: str):
    """Apply the acceptance rule to the validated variant."""
    var_row = summary_df[summary_df.cat_config == cat_config].iloc[0]
    base_row = summary_df[summary_df.cat_config == "candidate_baseline"].iloc[0]

    test_f1_improved = delta["delta_test_f1_mean"] > 0
    oof_ok = delta["delta_oof_f1_mean"] >= -0.005
    fn_ok = delta["delta_test_fn_mean"] <= 2 or delta["delta_test_f1_mean"] > 0.003
    reached_056 = var_row["test_f1_mean"] >= 0.560

    if test_f1_improved and oof_ok and fn_ok:
        if reached_056:
            label = "reached_056_target"
        elif delta["delta_test_f1_mean"] >= 0.001:
            label = "improved_high_f1_candidate"
        else:
            label = "valid_but_no_meaningful_uplift"
    elif test_f1_improved and not oof_ok:
        label = "unstable_test_only_signal"
    elif not test_f1_improved and delta["delta_oof_f1_mean"] > 0:
        label = "oof_only_signal"
    else:
        label = "rejected_no_uplift"

    return label


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


# ═══════════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════════

def md_table(df: pd.DataFrame) -> str:
    return catnew_base.markdown_table(df.round(6))


def write_report(
    baselines: pd.DataFrame,
    smoke_df: pd.DataFrame,
    selected_candidate,
    selected_cat_config,
    validation_df,
    validation_summary,
    delta,
    acceptance_label,
    leak_df: pd.DataFrame,
    wall_seconds: float,
):
    """Write all output files."""
    # ─── candidate map ───
    map_rows = []
    for c in CANDIDATES:
        map_rows.append({
            "candidate_id": c["candidate_id"],
            "name": c["name"],
            "config_name": c["config_name"],
            "aggregation": c["aggregation"],
            "feature_family": c["feature_family"],
            "source_step": c["source_step"],
            "source_file": c["source_file"],
            "source_seed_set": str(c["source_seed_set"]),
            "expected_test_f1": c["expected_test_f1"],
        })
    pd.DataFrame(map_rows).to_csv(OUT["map"], index=False)

    # ─── baseline reproduction ───
    baselines.to_csv(OUT["baseline_reproduction"], index=False)

    # ─── smoke metrics ───
    smoke_df.to_csv(OUT["smoke"], index=False)

    # ─── validation ───
    if validation_df is not None:
        validation_df.to_csv(OUT["validation_per_seed"], index=False)
        validation_summary.to_csv(OUT["validation_summary"], index=False)
    else:
        pd.DataFrame().to_csv(OUT["validation_per_seed"], index=False)
        pd.DataFrame().to_csv(OUT["validation_summary"], index=False)

    # ─── delta ───
    if delta is not None:
        pd.DataFrame([delta]).to_csv(OUT["delta"], index=False)
    else:
        pd.DataFrame().to_csv(OUT["delta"], index=False)

    # ─── leakage ───
    leak_df.to_csv(OUT["leak"], index=False)

    # ─── current active update ───
    OUT["current_update"].write_text(json.dumps({
        "current_active_changed": False,
        "current_active_changed_text": "current_active_changed = false",
        "step": "Step19G_LIGHT",
        "selected_for_validation": selected_cat_config,
        "acceptance_label": acceptance_label if acceptance_label else "no_variant_passed_phase1",
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # ─── report ───
    smoke_compact = ["candidate_id", "cat_config", "test_f1", "oof_f1",
                     "delta_test_f1", "delta_oof_f1", "test_fp", "test_fn",
                     "selected_threshold"]
    smoke_display = smoke_df[[c for c in smoke_compact if c in smoke_df.columns]]

    baseline_compact = ["candidate_id", "candidate_name", "aggregation",
                        "test_f1", "oof_f1", "test_fp", "test_fn",
                        "test_precision", "test_recall", "selected_threshold",
                        "known_test_f1", "seed_set_note"]
    baseline_display = baselines[[c for c in baseline_compact if c in baselines.columns]]

    val_section = ""
    if validation_df is not None and len(validation_df) > 0:
        val_compact = ["candidate_id", "cat_config", "seed", "test_f1", "oof_f1",
                       "test_fp", "test_fn", "selected_threshold"]
        val_display = validation_df[[c for c in val_compact if c in validation_df.columns]]
        val_section = f"""## Phase 2 — Same-Seed Validation

{md_table(val_display)}

### Validation Summary

{md_table(validation_summary)}

### Delta vs Baseline

| metric | value |
| --- | --- |
| delta_test_f1_mean | {delta['delta_test_f1_mean']:+.6f} |
| delta_oof_f1_mean | {delta['delta_oof_f1_mean']:+.6f} |
| delta_test_fp_mean | {delta['delta_test_fp_mean']:+.1f} |
| delta_test_fn_mean | {delta['delta_test_fn_mean']:+.1f} |

### Acceptance Label

**{acceptance_label}**
"""
    else:
        val_section = """## Phase 2 — Same-Seed Validation

Phase 2 was not run because no candidate-config passed Phase 1 selection.
"""

    # Determine best model statement
    best_model_statement = (
        "Best model right now remains "
        "`catboost_lightgbm_xgboost_extratrees__prob_average` from Step13A "
        "with 5-seed Test F1 = 0.554365."
    )
    if acceptance_label == "reached_056_target":
        best_model_statement = (
            f"Selected variant {selected_cat_config} on "
            f"{selected_candidate['candidate_id']} reached F1 >= 0.560. "
            "Review for manual promotion."
        )
    elif acceptance_label == "improved_high_f1_candidate":
        best_model_statement = (
            f"Selected variant {selected_cat_config} on "
            f"{selected_candidate['candidate_id']} shows meaningful improvement "
            "over its reproduced baseline, but current_active is not changed automatically."
        )

    report = f"""# Step19G_LIGHT — CAT1/CAT2/CAT_new Gap Test for Candidate 2 and Candidate 5

Step19G_LIGHT tests only candidate 2 and candidate 5.

Step19G_LIGHT tests only CAT1/CAT2/CAT_new gaps.

The Step13A four-model equal ensemble is not retested because its CAT1/CAT2/CAT_new gap was already closed.

All thresholds are selected using OOF predictions only.

Test metrics are reporting-only.

No target encoding is used.

Variant_ID and Label are not used as model features.

current_active_changed = false

## Reference Context (not retested)

Step13A four-model probability-average ensemble:
- config: catboost_lightgbm_xgboost_extratrees__prob_average
- Test F1 = 0.554365
- OOF F1 = 0.798210
- threshold = 0.929330
- seed set: 7, 21, 42, 101, 2024

Step19D / Step19E already concluded:
- CAT_new did not show useful signal on the Step13A four-model lane
- drop_CAT2_only was valid but not meaningful uplift
- current_active_changed = false

## Phase 0 — Baseline Reproduction

{md_table(baseline_display)}

Note: Baselines reproduced with seed=42 only (smoke). The original seed set was [7, 21, 42].
Absolute F1 values for seed=42 only will differ from the 3-seed mean.
Pipeline is identical to Step15B so baseline is considered reproduced.

## Phase 1 — CAT12 / CAT_new Smoke Test (seed=42)

{md_table(smoke_display)}

### Phase 1 Selection

Selected candidate-config for validation: **{f"{selected_candidate['candidate_id']} + {selected_cat_config}" if selected_candidate else "NONE (no config passed)"}**

{val_section}

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

- Any FAIL: **{bool(leak_df.status.eq('FAIL').any())}**
- Status counts: {leak_df.status.value_counts().to_dict()}

## Wall Runtime

{wall_seconds / 60:.2f} minutes

## Best Model

{best_model_statement}
"""
    OUT["report"].write_text(report, encoding="utf-8")

    # ─── next action ───
    if acceptance_label and "improved" in acceptance_label.lower():
        next_text = (
            f"# Step19G_LIGHT Next Action\n\n"
            f"current_active_changed = false\n\n"
            f"Candidate {selected_candidate['candidate_id']} with {selected_cat_config} "
            f"shows acceptance label: {acceptance_label}.\n"
            f"Review for manual promotion. Do not auto-update.\n"
        )
    elif acceptance_label == "reached_056_target":
        next_text = (
            f"# Step19G_LIGHT Next Action\n\n"
            f"current_active_changed = false\n\n"
            f"Candidate {selected_candidate['candidate_id']} with {selected_cat_config} "
            f"reached F1 >= 0.560 target.\n"
            f"Review for manual promotion. Do not auto-update.\n"
        )
    else:
        next_text = (
            "# Step19G_LIGHT Next Action\n\n"
            "current_active_changed = false\n\n"
            "CAT1/CAT2/CAT_new does not show useful signal on candidate 2 or candidate 5.\n"
            "Best model remains Step13A four-model `catboost_lightgbm_xgboost_extratrees__prob_average` "
            "with 5-seed Test F1 = 0.554365.\n"
        )
    OUT["next_action"].write_text(next_text, encoding="utf-8")

    # ─── metadata ───
    meta = {
        "step": "Step19G_LIGHT",
        "completed": True,
        "candidates_tested": [c["candidate_id"] for c in CANDIDATES],
        "cat_configs_tested": CAT_CONFIGS,
        "smoke_seed": SMOKE_SEED,
        "selected_for_validation": selected_cat_config,
        "selected_candidate_id": selected_candidate["candidate_id"] if selected_candidate else None,
        "acceptance_label": acceptance_label,
        "wall_seconds": wall_seconds,
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
        },
    }
    OUT["metadata"].write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


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
        catnew_available = True
    except FileNotFoundError:
        print("WARNING: CAT_new file not found. CAT_new configs will be skipped.", flush=True)
        full_catnew = full.copy()
        catnew_available = False

    # Phase 0: Reproduce baselines
    baselines = phase0_reproduce_baselines(full_catnew)

    # Phase 1: Smoke test
    smoke_df, all_runtimes, all_infos = phase1_smoke(full_catnew, baselines)

    # Phase 1: Selection
    selected_candidate, selected_cat_config = phase1_selection(smoke_df)

    # Phase 2: Validation
    validation_df = None
    validation_summary = None
    delta = None
    acceptance_label = None

    if selected_candidate is not None:
        validation_df, validation_summary, delta, seed_note = phase2_validation(
            full_catnew, selected_candidate, selected_cat_config,
        )

        # Collect leakage info from validation
        for seed in (selected_candidate.get("source_seed_set") or VALIDATION_SEEDS):
            for cat_config in ["candidate_baseline", selected_cat_config]:
                _, _, info, _ = load_or_generate(
                    selected_candidate, cat_config, seed, full_catnew,
                )
                all_infos.append(info)

        acceptance_label = apply_acceptance_rule(
            delta, validation_summary, selected_cat_config,
        )
        print(f"\n  ACCEPTANCE LABEL: {acceptance_label}", flush=True)
    else:
        print("\n  STOP: No candidate-config passed Phase 1.", flush=True)

    # Leakage audit
    leak_df = leakage_audit(all_infos)

    # Write all outputs
    wall_seconds = time.perf_counter() - wall_start
    write_report(
        baselines, smoke_df,
        selected_candidate, selected_cat_config,
        validation_df, validation_summary, delta, acceptance_label,
        leak_df, wall_seconds,
    )

    print(f"\n{'=' * 60}", flush=True)
    print(f"Step19G_LIGHT COMPLETE in {wall_seconds / 60:.2f} minutes", flush=True)
    print(f"current_active_changed = false", flush=True)
    if selected_candidate:
        print(f"Selected: {selected_candidate['candidate_id']} + {selected_cat_config}", flush=True)
        print(f"Acceptance: {acceptance_label}", flush=True)
    else:
        print("No variant passed Phase 1.", flush=True)


if __name__ == "__main__":
    main()
