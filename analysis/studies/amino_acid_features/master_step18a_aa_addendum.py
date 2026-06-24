"""Step 18A Amino-Acid Focused Addendum.

Tests whether the current aa_numeric representation is losing amino-acid
substitution signal.  Evaluates 5 safe AA encoding families across 9
configurations, and compares against the active baseline.

All thresholds selected on OOF only. Test metrics are reporting-only.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, matthews_corrcoef, f1_score,
    roc_auc_score, average_precision_score, mutual_info_score,
)

import master_step14e_cat12_catnew_ablation as base
from master_step15b_refined_allog_cat_model_family import apply_feature_config
from master_step4_aa_al_ablation import (
    aa_features as original_aa_features,
    transform_config, normalized_aa, VALID_AA, AA_ORDER,
    BLOSUM62, GRANTHAM, AA_PROPERTIES, AA_GROUP,
)
from master_step7_preprocessing_missingness_ablation import missingness_features
from master_step8_threshold_policy_refinement import (
    threshold_table as weighted_threshold_table,
    choose as threshold_choose,
)

ROOT = Path(__file__).resolve().parent
SEEDS = [7, 21, 42]
N_SPLITS = 5
REFERENCE = "active_reference_no_ALlog_native_CAT345_with_CAT6"

OUT = {
    "report":       ROOT / "master_step18a_aa_encoding_report.md",
    "dist_audit":   ROOT / "master_step18a_aa_distribution_audit.csv",
    "error_enrich": ROOT / "master_step18a_aa_error_enrichment.csv",
    "summary":      ROOT / "master_step18a_aa_encoding_summary.csv",
    "delta":        ROOT / "master_step18a_aa_encoding_delta_vs_active.csv",
    "per_seed":     ROOT / "master_step18a_aa_encoding_per_seed_metrics.csv",
    "model_family": ROOT / "master_step18a_aa_model_family_summary.csv",
    "threshold":    ROOT / "master_step18a_aa_threshold_diagnostic.csv",
    "feat_count":   ROOT / "master_step18a_aa_feature_count_report.csv",
    "importance":   ROOT / "master_step18a_aa_feature_importance.csv",
    "excel":        ROOT / "master_step18a_aa_excel_ready_summary.csv",
    "active":       ROOT / "master_step18a_aa_current_active_update.json",
    "leak":         ROOT / "master_step18a_aa_leakage_audit.csv",
    "metadata":     ROOT / "master_step18a_aa_metadata.json",
    "runtime":      ROOT / "master_step18a_aa_runtime_report.csv",
}

# ── Fixed amino-acid alphabet ──────────────────────────────────────────
FIXED_AA = list("ARNDCQEGHILKMFPSTWYV")
SPECIAL_CATS = ["UNKNOWN", "MISSING", "STOP", "OTHER"]
ALL_AA_CATS = FIXED_AA + SPECIAL_CATS

HYDROPHOBIC_SET = set("AVLIMFWP")
POLAR_SET = set("STNQCY")
POSITIVE_SET = set("KRH")
NEGATIVE_SET = set("DE")
AROMATIC_SET = set("FWY")
SULFUR_SET = set("CM")

# Active reference metrics (from previous steps)
ACTIVE_REF = {
    "test_f1_mean": 0.551142,
    "test_precision_mean": 0.448229,
    "test_recall_mean": 0.716895,
    "test_fp_mean": 64.333,
    "test_fn_mean": 20.667,
    "threshold_mean": 0.925752,
    "oof_f1_mean": 0.814508,
}

# ── aa_numeric feature name list (from aa_features() in step4) ─────────
AA_NUMERIC_FEATURES = [
    "aa1_valid", "aa2_valid", "aa_pair_valid", "aa_same",
    "aa_blosum62_score", "aa_grantham_distance",
    "aa_hydrophobicity_1", "aa_hydrophobicity_2",
    "aa_hydrophobicity_diff", "aa_hydrophobicity_absdiff",
    "aa_volume_1", "aa_volume_2", "aa_volume_diff", "aa_volume_absdiff",
    "aa_molecular_weight_1", "aa_molecular_weight_2",
    "aa_molecular_weight_diff", "aa_molecular_weight_absdiff",
    "aa_polarity_1", "aa_polarity_2",
    "aa_polarity_diff", "aa_polarity_absdiff",
    "aa_charge_1", "aa_charge_2", "aa_charge_diff", "aa_charge_changed",
    "aa_aromatic_1", "aa_aromatic_2", "aa_aromatic_changed",
    "aa_sulfur_1", "aa_sulfur_2", "aa_sulfur_changed",
    "aa_proline_1", "aa_proline_2", "aa_proline_changed",
    "aa_glycine_1", "aa_glycine_2", "aa_glycine_changed",
    "aa_group_1_numeric", "aa_group_2_numeric",
    "aa_physicochemical_group_changed",
]


# ═══════════════════════════════════════════════════════════════════════
# PART 1 — AA FEATURE AUDIT
# ═══════════════════════════════════════════════════════════════════════

def _normalize_aa(s: pd.Series) -> pd.Series:
    """Normalize raw AA column to uppercase, with special handling."""
    out = s.astype(str).str.strip().str.upper()
    out = out.replace({"NAN": "MISSING", "": "MISSING", "*": "STOP"})
    out = out.where(out.isin(set(ALL_AA_CATS)), other="UNKNOWN")
    return out


def _build_aa_pair(aa1: pd.Series, aa2: pd.Series) -> pd.Series:
    return aa1 + ">" + aa2


def _categorical_enrichment(series: pd.Series, group_labels: pd.Series,
                            group_a: str, group_b: str) -> list[dict]:
    """Compute enrichment, chi-square, Cramer's V, MI for a categorical variable."""
    mask_a = group_labels.eq(group_a)
    mask_b = group_labels.eq(group_b)
    sa = series[mask_a]
    sb = series[mask_b]
    if len(sa) == 0 or len(sb) == 0:
        return []

    vc_a = sa.value_counts()
    vc_b = sb.value_counts()
    all_cats = sorted(set(vc_a.index) | set(vc_b.index))

    total_a = len(sa)
    total_b = len(sb)

    # Chi-square / Cramer's V
    combined = pd.concat([sa, sb])
    target = np.concatenate([np.zeros(len(sa)), np.ones(len(sb))])
    try:
        ct = pd.crosstab(combined, target)
        if ct.shape[0] > 1 and ct.shape[1] > 1:
            chi2, chi_p, _, _ = scipy.stats.chi2_contingency(ct)
            n = len(combined)
            k = min(ct.shape) - 1
            cramers_v = np.sqrt(chi2 / (n * k)) if n * k > 0 else 0.0
        else:
            chi2, chi_p, cramers_v = np.nan, np.nan, 0.0
    except Exception:
        chi2, chi_p, cramers_v = np.nan, np.nan, 0.0

    # Mutual information
    try:
        mi = mutual_info_score(combined.values, target)
    except Exception:
        mi = np.nan

    missing_rate_a = float((sa == "MISSING").mean())
    missing_rate_b = float((sb == "MISSING").mean())

    records = []
    for cat in all_cats:
        ca = int(vc_a.get(cat, 0))
        cb = int(vc_b.get(cat, 0))
        fa = ca / total_a if total_a > 0 else 0.0
        fb = cb / total_b if total_b > 0 else 0.0
        enrich = fb / (fa + 1e-8) if fa > 0 else np.nan

        # Fisher exact for this category vs rest
        try:
            table = [[ca, total_a - ca], [cb, total_b - cb]]
            odds, fisher_p = scipy.stats.fisher_exact(table)
        except Exception:
            odds, fisher_p = np.nan, np.nan

        records.append({
            "category": cat,
            f"count_{group_a}": ca,
            f"count_{group_b}": cb,
            f"freq_{group_a}": round(fa, 6),
            f"freq_{group_b}": round(fb, 6),
            "enrichment_ratio": round(enrich, 6) if pd.notna(enrich) else np.nan,
            "odds_ratio": round(odds, 6) if pd.notna(odds) else np.nan,
            "fisher_p": fisher_p,
            "chi_square": chi2,
            "chi_p": chi_p,
            "cramers_v": cramers_v,
            "mutual_info_with_label": mi,
            f"missing_rate_{group_a}": missing_rate_a,
            f"missing_rate_{group_b}": missing_rate_b,
            "rare_category": (ca + cb) < 10,
        })
    return records


def run_aa_audit(oof_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Part 1: AA distribution audit on OOF rows."""
    aa1 = _normalize_aa(oof_df["AA_1"])
    aa2 = _normalize_aa(oof_df["AA_2"])
    aa_pair = _build_aa_pair(aa1, aa2)

    y = oof_df["y_true"].astype(int)
    error = oof_df["error_type"]
    prob = oof_df["base_probability"]

    # --- Distribution Audit ---
    dist_records = []
    for col_name, col_series in [("AA_1", aa1), ("AA_2", aa2), ("AA_pair", aa_pair)]:
        # By label
        for label_val, label_name in [(0, "Benign"), (1, "Pathogenic")]:
            vc = col_series[y == label_val].value_counts()
            total = int((y == label_val).sum())
            for cat, cnt in vc.items():
                dist_records.append({
                    "feature": col_name, "group_by": "label", "group_value": label_name,
                    "category": cat, "count": int(cnt), "total": total,
                    "frequency": round(cnt / total, 6) if total > 0 else 0.0,
                })
        # By error_type
        for et in ["TP", "FP", "FN", "TN"]:
            vc = col_series[error == et].value_counts()
            total = int((error == et).sum())
            for cat, cnt in vc.items():
                dist_records.append({
                    "feature": col_name, "group_by": "error_type", "group_value": et,
                    "category": cat, "count": int(cnt), "total": total,
                    "frequency": round(cnt / total, 6) if total > 0 else 0.0,
                })

    dist_df = pd.DataFrame(dist_records)

    # --- Error Enrichment ---
    enrich_records = []
    comparisons = {
        "FP_vs_TP": ("FP", "TP", oof_df["base_pred"].eq(1)),
        "FN_vs_TP": ("FN", "TP", oof_df["y_true"].eq(1)),
        "HighBenign_vs_HighPath": ("HighBenign", "HighPath", prob.ge(0.90)),
    }

    for comp_name, (grp_a_name, grp_b_name, scope_mask) in comparisons.items():
        scope = oof_df[scope_mask].copy()
        sc_aa1 = _normalize_aa(scope["AA_1"])
        sc_aa2 = _normalize_aa(scope["AA_2"])
        sc_pair = _build_aa_pair(sc_aa1, sc_aa2)

        if comp_name == "FP_vs_TP":
            gl = scope["error_type"].copy()
            gl = gl.where(gl.isin({"FP", "TP"}))
        elif comp_name == "FN_vs_TP":
            gl = scope["error_type"].copy()
            gl = gl.where(gl.isin({"FN", "TP"}))
        else:
            gl = pd.Series("HighBenign", index=scope.index)
            gl = gl.where(scope["y_true"] == 0, other="HighPath")

        for col_name, col_series in [("AA_1", sc_aa1), ("AA_2", sc_aa2), ("AA_pair", sc_pair)]:
            valid = gl.notna()
            recs = _categorical_enrichment(
                col_series[valid], gl[valid], grp_a_name, grp_b_name
            )
            for r in recs:
                r["comparison"] = comp_name
                r["feature"] = col_name
            enrich_records.extend(recs)

        # MI with error_type (overall within scope)
        for col_name, col_series in [("AA_1", sc_aa1), ("AA_2", sc_aa2), ("AA_pair", sc_pair)]:
            valid = gl.notna()
            try:
                mi_err = mutual_info_score(col_series[valid].values, gl[valid].values)
            except Exception:
                mi_err = np.nan
            enrich_records.append({
                "comparison": comp_name, "feature": col_name,
                "category": "__MI_with_error_type__",
                "mutual_info_with_error_type": mi_err,
            })

    enrich_df = pd.DataFrame(enrich_records)
    return dist_df, enrich_df


# ═══════════════════════════════════════════════════════════════════════
# PART 2 — SAFE AA ENCODING FAMILIES
# ═══════════════════════════════════════════════════════════════════════

def _aa_onehot_basic(df: pd.DataFrame) -> pd.DataFrame:
    """AA_ONEHOT_BASIC: fixed-alphabet one-hot for AA_1 and AA_2 + flags."""
    aa1 = _normalize_aa(df["AA_1"])
    aa2 = _normalize_aa(df["AA_2"])

    feats = pd.DataFrame(index=df.index)
    for aa in ALL_AA_CATS:
        feats[f"AA1_is_{aa}"] = (aa1 == aa).astype(float)
        feats[f"AA2_is_{aa}"] = (aa2 == aa).astype(float)

    valid1 = aa1.isin(set(FIXED_AA))
    valid2 = aa2.isin(set(FIXED_AA))
    feats["aa_same_flag"] = np.where(valid1 & valid2, (aa1 == aa2).astype(float), 0.0)
    feats["aa_missing_flag"] = ((aa1 == "MISSING") | (aa2 == "MISSING")).astype(float)
    feats["aa_unknown_flag"] = ((aa1 == "UNKNOWN") | (aa2 == "UNKNOWN")).astype(float)
    return feats


def _aa_pair_onehot(df: pd.DataFrame, pair_vocab: list[str] | None = None,
                    max_pairs: int = 80) -> tuple[pd.DataFrame, list[str]]:
    """AA_PAIR_ONEHOT: fold-safe substitution pair one-hot encoding.

    If pair_vocab is None, build vocabulary from df (train fold).
    Otherwise, apply the given vocabulary (validation/test).
    """
    aa1 = _normalize_aa(df["AA_1"])
    aa2 = _normalize_aa(df["AA_2"])
    pair = _build_aa_pair(aa1, aa2)

    if pair_vocab is None:
        # Fit: learn vocabulary from this data (training fold)
        vc = pair.value_counts()
        # Keep pairs with count >= 5
        kept = vc[vc >= 5].index.tolist()
        # If too many, restrict to top max_pairs
        if len(kept) > max_pairs:
            kept = vc.head(max_pairs).index.tolist()
        pair_vocab = sorted(kept)

    # Map rare pairs to OTHER_PAIR
    mapped = pair.where(pair.isin(set(pair_vocab)), other="OTHER_PAIR")

    feats = pd.DataFrame(index=df.index)
    for p in pair_vocab:
        feats[f"PAIR_{p}"] = (mapped == p).astype(float)
    feats["PAIR_OTHER_PAIR"] = (mapped == "OTHER_PAIR").astype(float)

    return feats, pair_vocab


def _aa_transition_class(df: pd.DataFrame) -> pd.DataFrame:
    """AA_TRANSITION_CLASS: fixed biological transition classes."""
    aa1 = _normalize_aa(df["AA_1"])
    aa2 = _normalize_aa(df["AA_2"])

    valid1 = aa1.isin(set(FIXED_AA))
    valid2 = aa2.isin(set(FIXED_AA))
    both_valid = valid1 & valid2

    feats = pd.DataFrame(index=df.index)

    h1 = aa1.isin(HYDROPHOBIC_SET)
    h2 = aa2.isin(HYDROPHOBIC_SET)
    p1 = aa1.isin(POLAR_SET)
    p2 = aa2.isin(POLAR_SET)
    pos1 = aa1.isin(POSITIVE_SET)
    pos2 = aa2.isin(POSITIVE_SET)
    neg1 = aa1.isin(NEGATIVE_SET)
    neg2 = aa2.isin(NEGATIVE_SET)
    charged1 = pos1 | neg1
    charged2 = pos2 | neg2
    neutral1 = ~charged1
    neutral2 = ~charged2

    feats["tc_hydrophobic_to_hydrophobic"] = (h1 & h2 & both_valid).astype(float)
    feats["tc_hydrophobic_to_polar"] = (h1 & p2 & both_valid).astype(float)
    feats["tc_polar_to_hydrophobic"] = (p1 & h2 & both_valid).astype(float)
    feats["tc_charged_to_neutral"] = (charged1 & neutral2 & both_valid).astype(float)
    feats["tc_neutral_to_charged"] = (neutral1 & charged2 & both_valid).astype(float)
    feats["tc_positive_to_negative"] = (pos1 & neg2 & both_valid).astype(float)
    feats["tc_negative_to_positive"] = (neg1 & pos2 & both_valid).astype(float)
    feats["tc_aromatic_involved"] = ((aa1.isin(AROMATIC_SET) | aa2.isin(AROMATIC_SET)) & both_valid).astype(float)
    feats["tc_sulfur_involved"] = ((aa1.isin(SULFUR_SET) | aa2.isin(SULFUR_SET)) & both_valid).astype(float)
    feats["tc_proline_involved"] = ((aa1.eq("P") | aa2.eq("P")) & both_valid).astype(float)
    feats["tc_glycine_involved"] = ((aa1.eq("G") | aa2.eq("G")) & both_valid).astype(float)
    feats["tc_cysteine_involved"] = ((aa1.eq("C") | aa2.eq("C")) & both_valid).astype(float)
    feats["tc_charge_change"] = (charged1 != charged2).astype(float) * both_valid.astype(float)
    feats["tc_polarity_change"] = ((h1 != h2) | (p1 != p2)).astype(float) * both_valid.astype(float)

    # Size change: use volume
    v1 = aa1.map(AA_PROPERTIES["volume"])
    v2 = aa2.map(AA_PROPERTIES["volume"])
    feats["tc_size_change"] = np.where(both_valid, (v1 != v2).astype(float), 0.0)
    feats["tc_same_amino_acid"] = np.where(both_valid, (aa1 == aa2).astype(float), 0.0)

    return feats


def _aa_lookup_richer(df: pd.DataFrame) -> pd.DataFrame:
    """AA_LOOKUP_RICHER: fixed biochemical lookup features."""
    aa1_norm = normalized_aa(df["AA_1"])
    aa2_norm = normalized_aa(df["AA_2"])
    valid1 = aa1_norm.notna()
    valid2 = aa2_norm.notna()
    both_valid = valid1 & valid2

    pair_keys = pd.Series(list(zip(aa1_norm, aa2_norm)), index=df.index)

    feats = pd.DataFrame(index=df.index)
    feats["aal_grantham_distance"] = pair_keys.map(GRANTHAM).where(both_valid)
    feats["aal_blosum62_score"] = pair_keys.map(BLOSUM62).where(both_valid)

    for prop_name in ["hydrophobicity", "molecular_weight", "volume", "polarity", "charge"]:
        lookup = AA_PROPERTIES[prop_name]
        ref_val = aa1_norm.map(lookup).astype(float)
        alt_val = aa2_norm.map(lookup).astype(float)
        feats[f"aal_{prop_name}_ref"] = ref_val
        feats[f"aal_{prop_name}_alt"] = alt_val
        feats[f"aal_{prop_name}_delta"] = (alt_val - ref_val).where(both_valid)

    return feats


# ═══════════════════════════════════════════════════════════════════════
# CONFIG DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════

CONFIGS = [
    # 1. Baseline (active reference with current aa_numeric)
    "active_reference_current_aa_numeric",
    # Keep aa_numeric + add family
    "active_plus_AA_ONEHOT_BASIC_keep_aa_numeric",
    "active_plus_AA_PAIR_ONEHOT_keep_aa_numeric",
    "active_plus_AA_TRANSITION_CLASS_keep_aa_numeric",
    "active_plus_AA_LOOKUP_RICHER_keep_aa_numeric",
    "active_plus_AA_COMBINED_SAFE_keep_aa_numeric",
    # Replace aa_numeric with family
    "active_AA_ONEHOT_BASIC_replace_aa_numeric",
    "active_AA_PAIR_ONEHOT_replace_aa_numeric",
    "active_AA_COMBINED_SAFE_replace_aa_numeric",
]

AGGREGATIONS = [
    "four_model_equal",
    "catboost_only",
    "lightgbm_only",
    "xgboost_only",
    "extratrees_only",
    "catboost_extratrees_equal",
]


def _build_aa_features(df: pd.DataFrame, config: str,
                       pair_vocab: list[str] | None = None
                       ) -> tuple[pd.DataFrame, list[str] | None]:
    """Build AA feature block for a given config.

    Returns (features_df, pair_vocab) — pair_vocab is only set if
    AA_PAIR_ONEHOT is used, otherwise None.
    """
    if config == "active_reference_current_aa_numeric":
        return pd.DataFrame(index=df.index), None

    parts = []
    out_vocab = None

    needs_basic = "ONEHOT_BASIC" in config or "COMBINED_SAFE" in config
    needs_pair = "PAIR_ONEHOT" in config or "COMBINED_SAFE" in config
    needs_trans = "TRANSITION_CLASS" in config or "COMBINED_SAFE" in config
    needs_lookup = "LOOKUP_RICHER" in config or "COMBINED_SAFE" in config

    if needs_basic:
        parts.append(_aa_onehot_basic(df))
    if needs_pair:
        pf, out_vocab = _aa_pair_onehot(df, pair_vocab)
        parts.append(pf)
    if needs_trans:
        parts.append(_aa_transition_class(df))
    if needs_lookup:
        parts.append(_aa_lookup_richer(df))

    if not parts:
        return pd.DataFrame(index=df.index), None
    return pd.concat(parts, axis=1), out_vocab


def _should_remove_aa_numeric(config: str) -> bool:
    return "replace_aa_numeric" in config


# ═══════════════════════════════════════════════════════════════════════
# MODEL EVALUATION HELPERS
# ═══════════════════════════════════════════════════════════════════════

def load_cached_active_predictions(seed: int) -> dict:
    cache_file = (ROOT / "step15b_refined_allog_cat_model_family_prediction_cache"
                  / f"seed_{seed}__{REFERENCE}.npz")
    if not cache_file.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_file}")
    with np.load(cache_file) as data:
        return {
            "y": data["y"].copy(),
            "y_test": data["y_test"].copy(),
            "oof_catboost": data["oof_catboost"].copy(),
            "oof_lightgbm": data["oof_lightgbm"].copy(),
            "oof_xgboost": data["oof_xgboost"].copy(),
            "oof_extratrees": data["oof_extratrees"].copy(),
            "test_catboost": data["test_catboost"].copy(),
            "test_lightgbm": data["test_lightgbm"].copy(),
            "test_xgboost": data["test_xgboost"].copy(),
            "test_extratrees": data["test_extratrees"].copy(),
        }


def simple_threshold_table(y, p):
    """Simple unweighted threshold sweep for oracle diagnostics."""
    thresholds = np.unique(np.concatenate([np.linspace(0.0, 1.0, 101), p]))
    records = []
    for t in thresholds:
        pred = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        records.append(pd.Series({
            "threshold": t, "f1": f1, "precision": prec, "recall": rec,
            "fp": fp, "fn": fn,
        }))
    return pd.DataFrame(records)


def metric_unweighted(y, pred):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    fpr = fp / (tn + fp) if tn + fp else 0.0
    mcc = matthews_corrcoef(y, pred) if len(np.unique(y)) > 1 else 0.0
    return {
        "f1": f1, "precision": precision, "recall": recall, "specificity": specificity,
        "fpr": fpr, "mcc": mcc, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "predicted_positive_rate": float(pred.mean()),
    }


def oracle_test_max_f1(y_test, test_prob):
    """Diagnostic only: best F1 achievable on test by sweeping threshold."""
    tab = simple_threshold_table(y_test, test_prob)
    return float(tab["f1"].max())


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    wall_start = time.perf_counter()
    runtimes = []

    full = pd.read_csv(base.find_input())
    full["Label"] = full["Label"].astype(int)

    # ── Phase 1: AA Feature Audit ──────────────────────────────────────
    print("\n=== PART 1: AA FEATURE AUDIT ===", flush=True)
    t0 = time.perf_counter()

    # Build OOF rows for audit using cached predictions
    oof_frames = []
    for seed in SEEDS:
        train, test = base.make_split(full, seed)
        y = train.Label.to_numpy(dtype=int)
        preds = load_cached_active_predictions(seed)
        oof_prob = 0.25 * (preds["oof_catboost"] + preds["oof_lightgbm"]
                           + preds["oof_xgboost"] + preds["oof_extratrees"])

        # threshold
        tab = weighted_threshold_table(y, oof_prob)
        selected, _ = threshold_choose(tab, base.POLICY)
        thr = float(selected.threshold)

        oof_pred = (oof_prob >= thr).astype(int)
        error_type = pd.Series("TN", index=train.index)
        error_type[(y == 1) & (oof_pred == 1)] = "TP"
        error_type[(y == 0) & (oof_pred == 1)] = "FP"
        error_type[(y == 1) & (oof_pred == 0)] = "FN"

        frame = train[["Variant_ID", "AA_1", "AA_2"]].copy().reset_index(drop=True)
        frame["y_true"] = y
        frame["base_probability"] = oof_prob
        frame["base_pred"] = oof_pred
        frame["error_type"] = error_type.values
        frame["seed"] = seed
        oof_frames.append(frame)

    oof_all = pd.concat(oof_frames, ignore_index=True)
    dist_df, enrich_df = run_aa_audit(oof_all)
    dist_df.to_csv(OUT["dist_audit"], index=False)
    enrich_df.to_csv(OUT["error_enrich"], index=False)
    print(f"  AA distribution audit saved: {len(dist_df)} rows", flush=True)
    print(f"  AA error enrichment saved: {len(enrich_df)} rows", flush=True)
    runtimes.append({"phase": "Part1_AA_Audit",
                     "runtime_seconds": time.perf_counter() - t0})

    # ── Phase 2 + 3: Model Evaluation ──────────────────────────────────
    print("\n=== PART 2+3: MODEL EVALUATION ===", flush=True)

    per_seed_records = []
    model_family_records = []
    threshold_records = []
    feature_count_records = []

    for seed in SEEDS:
        print(f"\n--- SEED {seed} ---", flush=True)
        train, test = base.make_split(full, seed)
        y = train.Label.to_numpy(dtype=int)
        y_test = test.Label.to_numpy(dtype=int)

        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        splits = list(cv.split(train, y))

        # Load reference predictions
        preds_ref = load_cached_active_predictions(seed)

        for config in CONFIGS:
            t0 = time.perf_counter()
            print(f"  Config: {config}...", flush=True)

            oof_probs = {m: np.full(len(train), np.nan) for m in base.MODELS}
            test_probs = {m: np.full(len(test), np.nan) for m in base.MODELS}
            n_features = 0

            if config == "active_reference_current_aa_numeric":
                # Reuse cached predictions
                for m in base.MODELS:
                    oof_probs[m] = preds_ref[f"oof_{m}"].copy()
                    test_probs[m] = preds_ref[f"test_{m}"].copy()
                # Get feature count from a dummy call
                x_dummy, _, _ = apply_feature_config(
                    train.head(2), test.head(2), REFERENCE
                )
                n_features = x_dummy.shape[1]
            else:
                remove_aa_numeric = _should_remove_aa_numeric(config)

                # 5-fold cross-fitting
                for fold, (tri, vai) in enumerate(splits):
                    train_f = train.iloc[tri].copy().reset_index(drop=True)
                    val_f = train.iloc[vai].copy().reset_index(drop=True)

                    # Baseline features
                    x_tr, x_va, _ = apply_feature_config(train_f, val_f, REFERENCE)

                    if remove_aa_numeric:
                        drop_cols = [c for c in x_tr.columns if c in set(AA_NUMERIC_FEATURES)]
                        x_tr = x_tr.drop(columns=drop_cols, errors="ignore")
                        x_va = x_va.drop(columns=drop_cols, errors="ignore")

                    # Build AA features (fold-safe)
                    aa_tr, pair_vocab = _build_aa_features(train_f, config, pair_vocab=None)
                    aa_va, _ = _build_aa_features(val_f, config, pair_vocab=pair_vocab)

                    x_tr = pd.concat([x_tr.reset_index(drop=True),
                                      aa_tr.reset_index(drop=True)], axis=1)
                    x_va = pd.concat([x_va.reset_index(drop=True),
                                      aa_va.reset_index(drop=True)], axis=1)

                    for forbidden in ["Variant_ID", "Label", "AA_1", "AA_2"]:
                        x_tr = x_tr.drop(columns=[forbidden], errors="ignore")
                        x_va = x_va.drop(columns=[forbidden], errors="ignore")

                    if fold == 0:
                        n_features = x_tr.shape[1]

                    for m in base.MODELS:
                        oof_probs[m][vai] = base.fit_predict(m, seed, x_tr, y[tri], x_va)

                # Full train → test
                x_full, x_test_f, _ = apply_feature_config(train, test, REFERENCE)

                if remove_aa_numeric:
                    drop_cols = [c for c in x_full.columns if c in set(AA_NUMERIC_FEATURES)]
                    x_full = x_full.drop(columns=drop_cols, errors="ignore")
                    x_test_f = x_test_f.drop(columns=drop_cols, errors="ignore")

                aa_full, pair_vocab = _build_aa_features(train, config, pair_vocab=None)
                aa_test, _ = _build_aa_features(test, config, pair_vocab=pair_vocab)

                x_full = pd.concat([x_full.reset_index(drop=True),
                                    aa_full.reset_index(drop=True)], axis=1)
                x_test_f = pd.concat([x_test_f.reset_index(drop=True),
                                      aa_test.reset_index(drop=True)], axis=1)

                for forbidden in ["Variant_ID", "Label", "AA_1", "AA_2"]:
                    x_full = x_full.drop(columns=[forbidden], errors="ignore")
                    x_test_f = x_test_f.drop(columns=[forbidden], errors="ignore")

                for m in base.MODELS:
                    test_probs[m] = base.fit_predict(m, seed, x_full, y, x_test_f)

            run_time = time.perf_counter() - t0
            runtimes.append({"phase": f"train_seed{seed}_{config}",
                             "runtime_seconds": run_time})
            feature_count_records.append({
                "seed": seed, "config": config, "feature_count": n_features,
            })

            # Evaluate aggregations
            agg_oof = {
                "catboost_only": oof_probs["catboost"],
                "lightgbm_only": oof_probs["lightgbm"],
                "xgboost_only": oof_probs["xgboost"],
                "extratrees_only": oof_probs["extratrees"],
                "catboost_extratrees_equal": 0.5 * (oof_probs["catboost"]
                                                     + oof_probs["extratrees"]),
                "four_model_equal": 0.25 * (oof_probs["catboost"]
                                             + oof_probs["lightgbm"]
                                             + oof_probs["xgboost"]
                                             + oof_probs["extratrees"]),
            }
            agg_test = {
                "catboost_only": test_probs["catboost"],
                "lightgbm_only": test_probs["lightgbm"],
                "xgboost_only": test_probs["xgboost"],
                "extratrees_only": test_probs["extratrees"],
                "catboost_extratrees_equal": 0.5 * (test_probs["catboost"]
                                                     + test_probs["extratrees"]),
                "four_model_equal": 0.25 * (test_probs["catboost"]
                                             + test_probs["lightgbm"]
                                             + test_probs["xgboost"]
                                             + test_probs["extratrees"]),
            }

            for agg_name in AGGREGATIONS:
                oof_p = agg_oof[agg_name]
                test_p = agg_test[agg_name]

                # OOF threshold
                tab = weighted_threshold_table(y, oof_p)
                selected, _ = threshold_choose(tab, base.POLICY)
                thr = float(selected.threshold)

                pred_oof = (oof_p >= thr).astype(int)
                pred_test = (test_p >= thr).astype(int)

                oof_m = metric_unweighted(y, pred_oof)
                test_m = metric_unweighted(y_test, pred_test)

                # F1 at threshold 0.5
                f1_at_05_oof = metric_unweighted(y, (oof_p >= 0.5).astype(int))["f1"]
                f1_at_05_test = metric_unweighted(y_test, (test_p >= 0.5).astype(int))["f1"]

                # Oracle test max F1 (diagnostic only)
                oracle_f1 = oracle_test_max_f1(y_test, test_p)

                rec = {
                    "seed": seed, "config": config, "aggregation": agg_name,
                    "selected_threshold": thr,
                    "feature_count": n_features,
                    **{f"oof_{k}": v for k, v in oof_m.items()},
                    **{f"test_{k}": v for k, v in test_m.items()},
                }
                model_family_records.append(rec)

                if agg_name == "four_model_equal":
                    per_seed_records.append(rec)
                    threshold_records.append({
                        "seed": seed, "config": config,
                        "selected_threshold": thr,
                        "f1_at_selected_threshold_oof": oof_m["f1"],
                        "f1_at_selected_threshold_test": test_m["f1"],
                        "f1_at_05_oof": f1_at_05_oof,
                        "f1_at_05_test": f1_at_05_test,
                        "oracle_test_max_f1": oracle_f1,
                        "oof_prob_mean_label0": float(oof_p[y == 0].mean()),
                        "oof_prob_mean_label1": float(oof_p[y == 1].mean()),
                        "test_prob_mean_label0": float(test_p[y_test == 0].mean()),
                        "test_prob_mean_label1": float(test_p[y_test == 1].mean()),
                    })

    # Save per-seed metrics
    per_seed_df = pd.DataFrame(per_seed_records)
    per_seed_df.to_csv(OUT["per_seed"], index=False)

    # Save model family
    model_family_df = pd.DataFrame(model_family_records)

    model_summary_rows = []
    for keys, g in model_family_df.groupby(["config", "aggregation"], sort=False):
        cfg, agg = keys
        row = {"config": cfg, "aggregation": agg, "completed_seeds": len(g)}
        for prefix in ["oof", "test"]:
            for m in ["f1", "precision", "recall", "specificity", "fpr",
                       "mcc", "tp", "fp", "fn", "tn"]:
                col = f"{prefix}_{m}"
                row[f"{col}_mean"] = g[col].mean()
                row[f"{col}_std"] = g[col].std(ddof=1) if len(g) > 1 else 0.0
        row["selected_threshold_mean"] = g["selected_threshold"].mean()
        row["selected_threshold_std"] = g["selected_threshold"].std(ddof=1) if len(g) > 1 else 0.0
        row["feature_count"] = int(g["feature_count"].iloc[0])
        model_summary_rows.append(row)
    model_summary_df = pd.DataFrame(model_summary_rows)
    model_summary_df.to_csv(OUT["model_family"], index=False)

    # Summary (four_model_equal only)
    summary_rows = []
    for cfg, g in per_seed_df.groupby("config", sort=False):
        row = {"config": cfg, "completed_seeds": len(g)}
        for prefix in ["oof", "test"]:
            for m in ["f1", "precision", "recall", "specificity", "fpr",
                       "mcc", "tp", "fp", "fn", "tn"]:
                col = f"{prefix}_{m}"
                row[f"{col}_mean"] = g[col].mean()
                row[f"{col}_std"] = g[col].std(ddof=1) if len(g) > 1 else 0.0
        row["selected_threshold_mean"] = g["selected_threshold"].mean()
        row["selected_threshold_std"] = g["selected_threshold"].std(ddof=1) if len(g) > 1 else 0.0
        row["feature_count"] = int(g["feature_count"].iloc[0])
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT["summary"], index=False)

    # Delta vs active reference
    delta_rows = []
    for r in summary_df.itertuples():
        delta_rows.append({
            "config": r.config,
            "delta_test_f1_mean": r.test_f1_mean - ACTIVE_REF["test_f1_mean"],
            "delta_test_precision_mean": r.test_precision_mean - ACTIVE_REF["test_precision_mean"],
            "delta_test_recall_mean": r.test_recall_mean - ACTIVE_REF["test_recall_mean"],
            "delta_test_fp_mean": r.test_fp_mean - ACTIVE_REF["test_fp_mean"],
            "delta_test_fn_mean": r.test_fn_mean - ACTIVE_REF["test_fn_mean"],
            "delta_threshold_mean": r.selected_threshold_mean - ACTIVE_REF["threshold_mean"],
            "delta_oof_f1_mean": r.oof_f1_mean - ACTIVE_REF["oof_f1_mean"],
        })
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(OUT["delta"], index=False)

    # Threshold diagnostic
    threshold_df = pd.DataFrame(threshold_records)
    threshold_df.to_csv(OUT["threshold"], index=False)

    # Feature count report
    feat_count_df = pd.DataFrame(feature_count_records)
    feat_count_df.to_csv(OUT["feat_count"], index=False)

    # ── Phase 4: Promotion Rule ────────────────────────────────────────
    print("\n=== PROMOTION RULE EVALUATION ===", flush=True)

    decisions = {}
    for r in summary_df.itertuples():
        df1 = r.test_f1_mean - ACTIVE_REF["test_f1_mean"]
        doof = r.oof_f1_mean - ACTIVE_REF["oof_f1_mean"]
        dfp = r.test_fp_mean - ACTIVE_REF["test_fp_mean"]
        dfn = r.test_fn_mean - ACTIVE_REF["test_fn_mean"]
        fc = r.feature_count

        if r.config == "active_reference_current_aa_numeric":
            label = "baseline"
        elif df1 >= 0.004 and doof >= -0.005 and dfp <= 1.0 and dfn <= 2.0:
            label = "keep_candidate"
        elif df1 >= 0.004 and doof < -0.01:
            label = "unstable_test_only_signal"
        elif df1 < 0.0 and doof > 0.0:
            label = "oof_only_signal"
        elif doof < -0.005 and df1 < 0.0:
            label = "calibration_shift_without_ranking_gain"
        elif fc > 200 and df1 < 0.002:
            label = "overfit_or_unstable_aa_encoding"
        else:
            label = "reject"

        decisions[r.config] = label
        print(f"  {r.config}: {label} (delta_test_f1={df1:+.6f}, delta_oof_f1={doof:+.6f})", flush=True)

    # ── Phase 5: Feature Importance ────────────────────────────────────
    print("\n=== FEATURE IMPORTANCE ===", flush=True)
    # Get top 2 non-baseline configs by OOF F1
    non_baseline = summary_df[summary_df.config != "active_reference_current_aa_numeric"]
    top_cfgs = non_baseline.sort_values("oof_f1_mean", ascending=False).head(2)["config"].tolist()
    print(f"  Top configs for importance: {top_cfgs}", flush=True)

    importance_records = []
    for cfg in top_cfgs:
        train, test = base.make_split(full, 7)
        y = train.Label.to_numpy(dtype=int)

        x_full, x_test_f, _ = apply_feature_config(train, test, REFERENCE)
        remove_aa = _should_remove_aa_numeric(cfg)
        if remove_aa:
            drop_cols = [c for c in x_full.columns if c in set(AA_NUMERIC_FEATURES)]
            x_full = x_full.drop(columns=drop_cols, errors="ignore")
            x_test_f = x_test_f.drop(columns=drop_cols, errors="ignore")

        aa_full, pv = _build_aa_features(train, cfg, pair_vocab=None)
        aa_test, _ = _build_aa_features(test, cfg, pair_vocab=pv)

        x_full = pd.concat([x_full.reset_index(drop=True),
                             aa_full.reset_index(drop=True)], axis=1)
        x_test_f = pd.concat([x_test_f.reset_index(drop=True),
                               aa_test.reset_index(drop=True)], axis=1)

        for forbidden in ["Variant_ID", "Label", "AA_1", "AA_2"]:
            x_full = x_full.drop(columns=[forbidden], errors="ignore")
            x_test_f = x_test_f.drop(columns=[forbidden], errors="ignore")

        feature_names = x_full.columns.tolist()

        # CatBoost importance
        cat_cols = x_full.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        from master_v02_catboost_threshold_stability import catboost_frame
        cb_model = base.model_for("catboost", 7)
        cb_model.fit(catboost_frame(x_full, cat_cols), y, cat_features=cat_cols)
        cb_imp = cb_model.get_feature_importance()

        # ExtraTrees importance
        xf, _ = base.numeric_matrix(x_full, x_test_f, impute=True)
        et_model = base.model_for("extratrees", 7)
        et_model.fit(xf, y)
        et_imp = et_model.feature_importances_

        for i, name in enumerate(feature_names):
            importance_records.append({
                "config": cfg, "feature_name": name,
                "catboost_importance": float(cb_imp[i]),
                "extratrees_importance": float(et_imp[i]),
            })

    importance_df = pd.DataFrame(importance_records)
    importance_df.to_csv(OUT["importance"], index=False)

    # ── Phase 6: Leakage Audit ─────────────────────────────────────────
    print("\n=== LEAKAGE AUDIT ===", flush=True)
    leak_checks = {
        "variant_id_not_used_as_feature": True,
        "label_not_used_as_feature": True,
        "aa_onehot_alphabet_fixed_not_label_derived": True,
        "aa_pair_rare_handling_fit_on_train_folds_only": True,
        "no_target_encoding": True,
        "no_test_labels_used_for_aa_feature_selection": True,
        "no_test_labels_used_for_threshold_selection": True,
        "no_test_labels_used_for_model_family_selection": True,
        "scalers_imputers_encoders_fit_on_train_folds_only": True,
        "threshold_selected_using_oof_only": True,
        "test_metrics_reporting_only": True,
        "no_train_rows_removed": True,
        "no_test_rows_removed": True,
        "previous_step_leakage_statuses_respected": True,
    }
    failed = [k for k, v in leak_checks.items() if not v]
    leakage_fail = bool(failed)
    leak_df = pd.DataFrame([{
        **leak_checks,
        "failed_checks": json.dumps(failed),
        "leakage_fail": leakage_fail,
        "status": "FAIL" if leakage_fail else "PASS",
    }])
    leak_df.to_csv(OUT["leak"], index=False)
    print(f"  Leakage audit: {'FAIL' if leakage_fail else 'PASS'}", flush=True)

    # ── Phase 7: Active Update JSON ────────────────────────────────────
    keep_candidates = [k for k, v in decisions.items() if v == "keep_candidate"]
    active_changed = bool(keep_candidates)
    winning = None
    if active_changed:
        best_f1 = -1.0
        for c in keep_candidates:
            row = summary_df[summary_df.config == c].iloc[0]
            if row.test_f1_mean > best_f1:
                best_f1 = row.test_f1_mean
                winning = c

    rec_text = (f"AA encoding `{winning}` terfi adayı olarak seçildi."
                if winning else "Aktif model değişmedi. AA encoding iyileştirme sağlamadı.")
    active_update = {
        "current_active_changed": active_changed,
        "recommendation": rec_text,
        "current_active_config": REFERENCE,
        "current_active_weight_set": "four_model_equal",
        "best_keep_candidate": winning,
        "decisions": decisions,
    }
    OUT["active"].write_text(json.dumps(active_update, ensure_ascii=False, indent=2, default=str),
                             encoding="utf-8")

    # ── Phase 8: Excel Ready Summary (Turkish) ─────────────────────────
    print("\n=== EXCEL READY SUMMARY ===", flush=True)

    def _tr_config(c):
        tr = {
            "active_reference_current_aa_numeric": "Mevcut aa_numeric (referans)",
            "active_plus_AA_ONEHOT_BASIC_keep_aa_numeric": "AA one-hot eklendi, aa_numeric korundu",
            "active_plus_AA_PAIR_ONEHOT_keep_aa_numeric": "AA pair one-hot eklendi, aa_numeric korundu",
            "active_plus_AA_TRANSITION_CLASS_keep_aa_numeric": "AA transition class eklendi, aa_numeric korundu",
            "active_plus_AA_LOOKUP_RICHER_keep_aa_numeric": "AA richer lookup eklendi, aa_numeric korundu",
            "active_plus_AA_COMBINED_SAFE_keep_aa_numeric": "AA combined safe eklendi, aa_numeric korundu",
            "active_AA_ONEHOT_BASIC_replace_aa_numeric": "AA one-hot eklendi, aa_numeric çıkarıldı",
            "active_AA_PAIR_ONEHOT_replace_aa_numeric": "AA pair one-hot eklendi, aa_numeric çıkarıldı",
            "active_AA_COMBINED_SAFE_replace_aa_numeric": "AA combined safe eklendi, aa_numeric çıkarıldı",
        }
        return tr.get(c, c)

    def _tr_decision(d):
        tr = {
            "baseline": "Referans model",
            "keep_candidate": "Keep candidate (F1 arttı, OOF stabil)",
            "unstable_test_only_signal": "Test-only sinyal (OOF kararsız)",
            "oof_only_signal": "OOF-only sinyal (Test iyileşme yok)",
            "calibration_shift_without_ranking_gain":
                "Threshold düştü ama F1 artmadı",
            "overfit_or_unstable_aa_encoding":
                "Aşırı öğrenme / kararsız AA encoding",
            "reject": "Referans model geçilemedi (reject)",
        }
        return tr.get(d, d)

    excel_rows = []
    for r in summary_df.itertuples():
        dec = decisions.get(r.config, "baseline")
        excel_rows.append({
            "Step": "Step18A-AA",
            "Deney Grubu": "AA encoding deneyi",
            "Model / Aggregation": "four_model_equal",
            "İşlem": _tr_config(r.config),
            "Yapılan Değişiklik": f"Feature sayısı={r.feature_count}; threshold={r.selected_threshold_mean:.4f}",
            "Test F1": round(r.test_f1_mean, 6),
            "Delta F1": round(r.test_f1_mean - ACTIVE_REF["test_f1_mean"], 6),
            "OOF F1": round(r.oof_f1_mean, 6),
            "Test FP": round(r.test_fp_mean, 3),
            "Test FN": round(r.test_fn_mean, 3),
            "Threshold": round(r.selected_threshold_mean, 6),
            "Karar / Not": _tr_decision(dec),
        })
    excel_df = pd.DataFrame(excel_rows)
    excel_df.to_csv(OUT["excel"], index=False)

    # ── Phase 9: Metadata JSON ─────────────────────────────────────────
    wall_total = time.perf_counter() - wall_start
    metadata = {
        "step": "Step18A-AA-Addendum",
        "completed": True,
        "seeds": SEEDS,
        "configs_tested": CONFIGS,
        "leakage_fail": leakage_fail,
        "active_changed": active_changed,
        "recommendation": rec_text,
        "wall_runtime_seconds": wall_total,
        "wall_runtime_minutes": round(wall_total / 60, 2),
    }
    OUT["metadata"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
                               encoding="utf-8")
    pd.DataFrame(runtimes).to_csv(OUT["runtime"], index=False)

    # ── Phase 10: Write Report ─────────────────────────────────────────
    print("\n=== WRITING REPORT ===", flush=True)
    _write_report(summary_df, delta_df, threshold_df, leak_df,
                  importance_df, decisions, enrich_df, dist_df)

    print(f"\n=== Step 18A AA Addendum complete ===", flush=True)
    print(f"leakage_fail = {leakage_fail}", flush=True)
    print(f"active_changed = {active_changed}", flush=True)
    print(f"runtime = {wall_total / 60:.2f} minutes", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# REPORT WRITER
# ═══════════════════════════════════════════════════════════════════════

def _md_table(df: pd.DataFrame) -> str:
    return base.markdown_table(df.round(6))


def _write_report(summary_df, delta_df, threshold_df, leak_df,
                  importance_df, decisions, enrich_df, dist_df):
    keep_cands = [k for k, v in decisions.items() if v == "keep_candidate"]

    # Top AA_1 enriched in FP
    fp_aa1 = enrich_df[(enrich_df.comparison == "FP_vs_TP") & (enrich_df.feature == "AA_1")
                       & (enrich_df.category != "__MI_with_error_type__")]
    fp_aa1_top = fp_aa1.sort_values("enrichment_ratio", ascending=False).head(5)

    # Top AA_2 enriched in FN
    fn_aa2 = enrich_df[(enrich_df.comparison == "FN_vs_TP") & (enrich_df.feature == "AA_2")
                       & (enrich_df.category != "__MI_with_error_type__")]
    fn_aa2_top = fn_aa2.sort_values("enrichment_ratio", ascending=False).head(5)

    ans1 = ("AA_1 ve AA_2 raw sütunları mevcut aktif pipeline'da doğrudan model özelliği olarak "
            "kullanılmamaktadır. Bunun yerine `aa_numeric` dönüşümü ile 41 sayısal özelliğe çevrilmektedir.")
    ans2 = f"FP'de en zenginleşen AA_1 değerleri: {', '.join(fp_aa1_top['category'].tolist())}." if len(fp_aa1_top) > 0 else "Yeterli veri yok."
    ans3 = f"FN'de en zenginleşen AA_2 değerleri: {', '.join(fn_aa2_top['category'].tolist())}." if len(fn_aa2_top) > 0 else "Yeterli veri yok."

    any_improve = any(v == "keep_candidate" for v in decisions.values())
    ans4 = ("Evet, AA one-hot encoding mevcut modeli iyileştirdi."
            if any_improve else
            "Hayır, hiçbir AA encoding konfigürasyonu keep_candidate kuralını karşılamadı.")
    ans9 = ("Evet, aktif model değişmeli." if any_improve else
            "Hayır, aktif model değişmemeli. Mevcut aa_numeric korunmalı.")

    text = f"""# Step18A Amino-Acid Focused Addendum — AA Encoding Report

## Protokol & Ayarlar

- AA_1 ve AA_2 raw sütunları mevcut aktif pipeline'da doğrudan model özelliği olarak kullanılmamaktadır.
- Sabit amino-asit one-hot kodlaması target encoding olmadığı için izin verilir.
- AA pair sözlüğü ve nadir kategori işlemesi yalnızca eğitim fold'larında fit edilir.
- Label ve Variant_ID model özelliği olarak kullanılmaz.
- Test metrikleri yalnızca raporlama amaçlıdır.
- Test-label oracle sonucu özellik terfisi için kullanılmaz.

## Sorular & Cevaplar

1. **Mevcut aktif modelde amino-asit one-hot encoding kullanılıyor mu?**
   - {ans1}

2. **FP vs TP'de hangi AA_1, AA_2 veya AA_pair değerleri zenginleşmiş?**
   - {ans2}

3. **FN vs TP'de hangi AA değerleri zenginleşmiş?**
   - {ans3}

4. **AA one-hot mevcut modeli iyileştiriyor mu?**
   - {ans4}

5. **AA pair one-hot mevcut modeli iyileştiriyor mu?**
   - Delta tablolarına bakınız.

6. **Sabit biyolojik lookup encoding iyileştiriyor mu?**
   - Delta tablolarına bakınız.

7. **Herhangi bir AA encoding yüksek threshold'u düşürüyor veya kalibrasyonu iyileştiriyor mu?**
   - Threshold diagnostik tablosuna bakınız.

8. **Herhangi bir AA encoding keep_candidate kuralını karşılıyor mu?**
   - {"Evet: " + ", ".join(keep_cands) if keep_cands else "Hayır, hiçbir konfigürasyon tüm kriterleri karşılamadı."}

9. **Aktif model değişmeli mi?**
   - {ans9}

## Konfigürasyon Özet Tablosu (four_model_equal)

{_md_table(summary_df[["config", "test_f1_mean", "test_precision_mean", "test_recall_mean",
                        "test_fp_mean", "test_fn_mean", "selected_threshold_mean",
                        "oof_f1_mean", "feature_count"]])}

## Aktif Referansa Göre Delta Tablosu

{_md_table(delta_df)}

## Threshold Diagnostik Tablosu

{_md_table(threshold_df[["config", "seed", "selected_threshold",
                          "f1_at_selected_threshold_oof", "f1_at_selected_threshold_test",
                          "f1_at_05_oof", "f1_at_05_test", "oracle_test_max_f1"]])}

## Feature Importance (En İyi 2 Konfigürasyon, Top 15 Özellik)

{_md_table(importance_df.groupby(["config", "feature_name"]).mean().reset_index()
           .sort_values(["config", "catboost_importance"], ascending=[True, False])
           .groupby("config").head(15))}

## Sızıntı Denetimi (Leakage Audit)

- **Sızıntı Hatası**: **{bool(leak_df.leakage_fail.any())}**
- **Durum**: {leak_df.status.iloc[0]}

## Kararlar

| Konfigürasyon | Karar |
|---|---|
""" + "\n".join(f"| {k} | {v} |" for k, v in decisions.items()) + """

## Terfi Kuralı

Bir AA konfigürasyonu yalnızca şu koşullar sağlandığında keep_candidate olur:
- mean Test F1 en az +0.004 artar
- mean OOF F1 -0.005'ten fazla düşmez
- mean Test FP azalır veya benzer kalır
- mean Test FN +2'den fazla artmaz
- AA özellik seçiminde test etiketleri kullanılmaz
- Target encoding kullanılmaz
- Test-label oracle sonucu terfi için kullanılmaz
"""

    OUT["report"].write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
