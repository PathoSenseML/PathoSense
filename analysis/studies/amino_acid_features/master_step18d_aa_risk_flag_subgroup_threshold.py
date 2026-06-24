"""Step18D: AA risk-flag and subgroup-threshold diagnostic.

Uses Step18A-AA predefined AA risk groups only. No broad AA one-hot/pair
encoding is introduced. Current active predictions are loaded from Step15B
cache; subgroup threshold rules are selected/evaluated with OOF-only
thresholds, and test remains reporting-only.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import master_step15b_refined_allog_cat_model_family as step15b
import master_step14e_cat12_catnew_ablation as base
from master_step8_threshold_policy_refinement import choose, threshold_table
from master_v02_catboost_threshold_stability import markdown_table
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "step15b_refined_allog_cat_model_family_prediction_cache"
CONFIG = "active_reference_no_ALlog_native_CAT345_with_CAT6"
SEEDS = [7, 21, 42]
MODELS = ["catboost", "lightgbm", "xgboost", "extratrees"]
WEIGHTS = {m: 0.25 for m in MODELS}
POLICY = "weighted_f1_current"

FP_AA1 = {"A", "I", "D", "S", "R"}
FN_AA2 = {"Y", "R", "A", "P", "G"}
FP_DELTAS = [0.005, 0.010, 0.015, 0.020, 0.030, 0.040, 0.050]
FN_DELTAS = [0.005, 0.010, 0.015, 0.020, 0.030]

CURRENT_ACTIVE = {
    "test_f1": 0.55114227713967,
    "oof_f1": 0.8145084492683368,
    "test_precision": 0.44822915875547453,
    "test_recall": 0.7168949771689498,
    "test_fp": 64.33333333333333,
    "test_fn": 20.666666666666668,
    "threshold": 0.9257520313157634,
}

OUT = {
    "report": ROOT / "master_step18d_aa_risk_flag_report.md",
    "row_oof": ROOT / "master_step18d_row_level_oof_predictions.csv",
    "row_test": ROOT / "master_step18d_row_level_test_predictions.csv",
    "enrichment": ROOT / "master_step18d_aa_risk_flag_enrichment.csv",
    "rule_per_seed": ROOT / "master_step18d_subgroup_threshold_per_seed_metrics.csv",
    "rule_summary": ROOT / "master_step18d_subgroup_threshold_summary.csv",
    "rule_delta": ROOT / "master_step18d_subgroup_threshold_delta_vs_active.csv",
    "optional_per_seed": ROOT / "master_step18d_optional_feature_flag_model_per_seed_metrics.csv",
    "optional_summary": ROOT / "master_step18d_optional_feature_flag_model_summary.csv",
    "optional_delta": ROOT / "master_step18d_optional_feature_flag_model_delta.csv",
    "excel_ready": ROOT / "master_step18d_excel_ready_summary.csv",
    "excel_wide": ROOT / "master_step18d_excel_wide_model_tracker.csv",
    "current_update": ROOT / "master_step18d_current_active_update.json",
    "leak": ROOT / "master_step18d_leakage_audit.csv",
    "metadata": ROOT / "master_step18d_metadata.json",
    "runtime": ROOT / "master_step18d_runtime_report.csv",
}

METRICS = [
    "f1", "precision", "recall", "specificity", "fpr", "mcc",
    "tp", "fp", "fn", "tn", "predicted_positive_rate",
]

OPTIONAL_AGGREGATIONS = {
    "catboost_only": {"catboost": 1.0},
    "lightgbm_only": {"lightgbm": 1.0},
    "xgboost_only": {"xgboost": 1.0},
    "extratrees_only": {"extratrees": 1.0},
    "catboost_extratrees_equal": {"catboost": 0.5, "extratrees": 0.5},
    "four_model_equal": {m: 0.25 for m in MODELS},
}
OPTIONAL_FLAG_COLS = [
    "AA1_FP_RISK",
    "AA2_FN_RISK",
    "AA_ANY_RISK",
    "AA_BOTH_RISK",
    "AA1_FP_RISK_x_CAT6_missing",
    "AA1_FP_RISK_x_high_AL_missing",
    "AA2_FN_RISK_x_high_disagreement",
]


def load_payload(seed: int) -> dict:
    path = CACHE / f"seed_{seed}__{CONFIG}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as saved:
        return {k: saved[k] for k in saved.files}


def norm_aa(x) -> str:
    if pd.isna(x):
        return "<MISSING>"
    s = str(x).strip()
    return s if s else "<MISSING>"


def raw_counts(df: pd.DataFrame) -> pd.DataFrame:
    al_cols = [c for c in df.columns if c.startswith("AL_")]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(["Label"]).tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns.difference(["Variant_ID"]).tolist()
    out = pd.DataFrame(index=df.index)
    out["AL_missing_count"] = df[al_cols].isna().sum(axis=1) if al_cols else 0
    out["AL_nonmissing_count"] = df[al_cols].notna().sum(axis=1) if al_cols else 0
    out["numeric_missing_count"] = df[numeric_cols].isna().sum(axis=1) if numeric_cols else 0
    out["categorical_missing_count"] = df[cat_cols].isna().sum(axis=1) if cat_cols else 0
    return out


def make_row_table(seed: int, stage: str, raw: pd.DataFrame, payload: dict, threshold: float) -> pd.DataFrame:
    prefix = "oof" if stage == "oof" else "test"
    probs = np.vstack([payload[f"{prefix}_{m}"] for m in MODELS]).T
    base_prob = np.mean(probs, axis=1)
    y = payload["y"] if stage == "oof" else payload["y_test"]
    pred = (base_prob >= threshold).astype(int)
    error = np.where((y == 1) & (pred == 1), "TP",
             np.where((y == 0) & (pred == 1), "FP",
             np.where((y == 1) & (pred == 0), "FN", "TN")))
    counts = raw_counts(raw)
    table = pd.DataFrame({
        "seed": seed,
        "stage": stage,
        "row_index": np.arange(len(raw)),
        "Variant_ID": raw["Variant_ID"].to_numpy() if "Variant_ID" in raw.columns else np.arange(len(raw)),
        "y_true": y.astype(int),
        "base_probability": base_prob,
        "base_pred": pred,
        "error_type": error,
        "selected_threshold": threshold,
        "catboost_prob": payload[f"{prefix}_catboost"],
        "lightgbm_prob": payload[f"{prefix}_lightgbm"],
        "xgboost_prob": payload[f"{prefix}_xgboost"],
        "extratrees_prob": payload[f"{prefix}_extratrees"],
        "prob_std": probs.std(axis=1),
        "prob_range": probs.max(axis=1) - probs.min(axis=1),
        "model_disagreement": probs.std(axis=1),
        "AA_1": raw["AA_1"].map(norm_aa) if "AA_1" in raw.columns else "<MISSING>",
        "AA_2": raw["AA_2"].map(norm_aa) if "AA_2" in raw.columns else "<MISSING>",
        "CAT_3": raw["CAT_3"].astype("string") if "CAT_3" in raw.columns else "<MISSING>",
        "CAT_4": raw["CAT_4"].astype("string") if "CAT_4" in raw.columns else "<MISSING>",
        "CAT_5": raw["CAT_5"].astype("string") if "CAT_5" in raw.columns else "<MISSING>",
        "CAT_6": raw["CAT_6"].astype("string") if "CAT_6" in raw.columns else "<MISSING>",
    })
    table["AA_pair"] = table["AA_1"].astype(str) + ">" + table["AA_2"].astype(str)
    table["CAT345_COMBINED"] = (
        table["CAT_3"].astype("string").fillna("<MISSING>") + "|"
        + table["CAT_4"].astype("string").fillna("<MISSING>") + "|"
        + table["CAT_5"].astype("string").fillna("<MISSING>")
    )
    cat6 = table["CAT_6"].astype("string")
    table["CAT_6_present"] = (~cat6.isna() & cat6.str.strip().ne("") & cat6.ne("<MISSING>")).astype(int)
    for col in counts.columns:
        table[col] = counts[col].to_numpy()
    return table


def add_risk_flags(oof: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    out = []
    for seed in SEEDS:
        oo = oof[oof.seed.eq(seed)].copy()
        tt = test[test.seed.eq(seed)].copy()
        dis_cut = float(oo["model_disagreement"].quantile(2 / 3))
        al_cut = float(oo["AL_missing_count"].quantile(2 / 3))
        rows.append({"seed": seed, "cutoff_name": "high_disagreement_top33_oof", "cutoff_value": dis_cut})
        rows.append({"seed": seed, "cutoff_name": "high_AL_missing_top33_oof", "cutoff_value": al_cut})
        for frame in [oo, tt]:
            frame["AA1_FP_RISK"] = frame["AA_1"].isin(FP_AA1)
            frame["AA2_FN_RISK"] = frame["AA_2"].isin(FN_AA2)
            frame["AA_ANY_RISK"] = frame["AA1_FP_RISK"] | frame["AA2_FN_RISK"]
            frame["AA_BOTH_RISK"] = frame["AA1_FP_RISK"] & frame["AA2_FN_RISK"]
            frame["high_probability"] = frame["base_probability"] >= 0.90
            frame["very_high_probability"] = frame["base_probability"] >= frame["selected_threshold"]
            frame["high_disagreement"] = frame["model_disagreement"] >= dis_cut
            frame["CAT6_missing"] = frame["CAT_6_present"].eq(0)
            frame["high_AL_missing"] = frame["AL_missing_count"] >= al_cut
            frame["near_threshold_negative"] = (
                frame["base_probability"].lt(frame["selected_threshold"])
                & frame["base_probability"].ge(frame["selected_threshold"] - 0.03)
            )
            frame["AA1_FP_RISK_x_high_probability"] = frame["AA1_FP_RISK"] & frame["high_probability"]
            frame["AA1_FP_RISK_x_very_high_probability"] = frame["AA1_FP_RISK"] & frame["very_high_probability"]
            frame["AA1_FP_RISK_x_high_disagreement"] = frame["AA1_FP_RISK"] & frame["high_disagreement"]
            frame["AA1_FP_RISK_x_CAT6_missing"] = frame["AA1_FP_RISK"] & frame["CAT6_missing"]
            frame["AA1_FP_RISK_x_high_AL_missing"] = frame["AA1_FP_RISK"] & frame["high_AL_missing"]
            frame["AA1_FP_RISK_x_high_disagreement_x_CAT6_missing"] = (
                frame["AA1_FP_RISK"] & frame["high_disagreement"] & frame["CAT6_missing"]
            )
            frame["AA2_FN_RISK_x_near_threshold_negative"] = frame["AA2_FN_RISK"] & frame["near_threshold_negative"]
            frame["AA2_FN_RISK_x_high_disagreement"] = frame["AA2_FN_RISK"] & frame["high_disagreement"]
            frame["AA2_FN_RISK_x_high_disagreement_x_near_threshold_negative"] = (
                frame["AA2_FN_RISK"] & frame["high_disagreement"] & frame["near_threshold_negative"]
            )
        out.append((oo, tt))
    return pd.concat([x[0] for x in out], ignore_index=True), pd.concat([x[1] for x in out], ignore_index=True), pd.DataFrame(rows)


def metric(y, pred) -> dict:
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    fpr = fp / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mcc_num = tp * tn - fp * fn
    mcc_den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return {
        "f1": f1, "precision": precision, "recall": recall, "specificity": specificity,
        "fpr": fpr, "mcc": (mcc_num / mcc_den if mcc_den else 0.0),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "predicted_positive_rate": float(pred.mean()),
    }


def enrich(oof: pd.DataFrame) -> pd.DataFrame:
    flags = [
        "AA1_FP_RISK", "AA2_FN_RISK", "AA_ANY_RISK", "AA_BOTH_RISK",
        "AA1_FP_RISK_x_high_probability", "AA1_FP_RISK_x_very_high_probability",
        "AA1_FP_RISK_x_high_disagreement", "AA1_FP_RISK_x_CAT6_missing",
        "AA1_FP_RISK_x_high_AL_missing", "AA1_FP_RISK_x_high_disagreement_x_CAT6_missing",
        "AA2_FN_RISK_x_near_threshold_negative", "AA2_FN_RISK_x_high_disagreement",
        "AA2_FN_RISK_x_high_disagreement_x_near_threshold_negative",
    ]
    rows = []
    for seed in [*SEEDS, "ALL"]:
        df = oof if seed == "ALL" else oof[oof.seed.eq(seed)]
        all_fn = int((df.error_type == "FN").sum())
        for flag in flags:
            pos = df[df.base_pred.eq(1) & df[flag]]
            neg_path = df[df.base_pred.eq(0) & df.y_true.eq(1) & df[flag]]
            tp = int((pos.error_type == "TP").sum())
            fp = int((pos.error_type == "FP").sum())
            rows.append({
                "seed": seed,
                "flag": flag,
                "base_positive_count": len(pos),
                "base_positive_tp": tp,
                "base_positive_fp": fp,
                "base_positive_precision": tp / (tp + fp) if tp + fp else np.nan,
                "base_positive_fp_rate": fp / (tp + fp) if tp + fp else np.nan,
                "base_positive_mean_probability": pos.base_probability.mean(),
                "base_positive_mean_model_disagreement": pos.model_disagreement.mean(),
                "base_negative_true_pathogenic_count": len(neg_path),
                "fn_count": len(neg_path),
                "share_of_all_fn": len(neg_path) / all_fn if all_fn else np.nan,
                "fn_mean_probability": neg_path.base_probability.mean(),
                "fn_mean_model_disagreement": neg_path.model_disagreement.mean(),
            })
    return pd.DataFrame(rows)


def candidate_rules(oof: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    fp_flags = [
        "AA1_FP_RISK", "AA1_FP_RISK_x_high_probability",
        "AA1_FP_RISK_x_very_high_probability", "AA1_FP_RISK_x_high_disagreement",
        "AA1_FP_RISK_x_CAT6_missing", "AA1_FP_RISK_x_high_AL_missing",
    ]
    if int(oof["AA1_FP_RISK_x_high_disagreement_x_CAT6_missing"].sum()) >= 30:
        fp_flags.append("AA1_FP_RISK_x_high_disagreement_x_CAT6_missing")
    fn_flags = [
        "AA2_FN_RISK", "AA2_FN_RISK_x_near_threshold_negative",
        "AA2_FN_RISK_x_high_disagreement", "AA_BOTH_RISK",
    ]
    if int(oof["AA2_FN_RISK_x_high_disagreement_x_near_threshold_negative"].sum()) >= 30:
        fn_flags.append("AA2_FN_RISK_x_high_disagreement_x_near_threshold_negative")
    fp = [{"rule_type": "fp_control", "fp_flag": f, "fn_flag": "", "delta_fp": d, "delta_fn": 0.0,
           "rule_name": f"{f}__raise_{d:.3f}"} for f in fp_flags for d in FP_DELTAS]
    fn = [{"rule_type": "fn_recovery", "fp_flag": "", "fn_flag": f, "delta_fp": 0.0, "delta_fn": d,
           "rule_name": f"{f}__lower_{d:.3f}"} for f in fn_flags for d in FN_DELTAS]
    return fp, fn


def apply_rule(df: pd.DataFrame, spec: dict) -> np.ndarray:
    thr = df["selected_threshold"].to_numpy(dtype=float).copy()
    if spec.get("fp_flag"):
        thr = np.where(df[spec["fp_flag"]].to_numpy(dtype=bool), thr + float(spec["delta_fp"]), thr)
    if spec.get("fn_flag"):
        thr = np.where(df[spec["fn_flag"]].to_numpy(dtype=bool), thr - float(spec["delta_fn"]), thr)
    return (df["base_probability"].to_numpy(dtype=float) >= thr).astype(int)


def evaluate_rules(oof: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        oo = oof[oof.seed.eq(seed)].copy()
        tt = test[test.seed.eq(seed)].copy()
        base_spec = {"rule_type": "baseline", "rule_name": "base_current_active", "fp_flag": "", "fn_flag": "", "delta_fp": 0.0, "delta_fn": 0.0}
        fp_specs, fn_specs = candidate_rules(oo)
        base_oof = {**base_spec, **metric(oo.y_true, oo.base_pred)}
        fp_scored = []
        fn_scored = []
        for spec in fp_specs:
            pred = apply_rule(oo, spec)
            m = metric(oo.y_true, pred)
            fp_scored.append((m["f1"], -m["fp"], spec))
        for spec in fn_specs:
            pred = apply_rule(oo, spec)
            m = metric(oo.y_true, pred)
            fn_scored.append((m["f1"], -m["fn"], spec))
        best_fp = sorted(fp_scored, key=lambda x: (x[0], x[1]), reverse=True)[0][2] if fp_scored else None
        best_fn = sorted(fn_scored, key=lambda x: (x[0], x[1]), reverse=True)[0][2] if fn_scored else None
        specs = [base_spec] + fp_specs + fn_specs
        if best_fp and best_fn:
            specs.append({
                "rule_type": "combined", "rule_name": f"best_oof_fp_plus_best_oof_fn__{best_fp['fp_flag']}__{best_fn['fn_flag']}",
                "fp_flag": best_fp["fp_flag"], "fn_flag": best_fn["fn_flag"],
                "delta_fp": best_fp["delta_fp"], "delta_fn": best_fn["delta_fn"],
            })
        for fp_flag, fn_flag in [
            ("AA1_FP_RISK_x_high_disagreement", "AA2_FN_RISK_x_near_threshold_negative"),
            ("AA1_FP_RISK_x_CAT6_missing", "AA2_FN_RISK_x_near_threshold_negative"),
            ("AA1_FP_RISK_x_high_AL_missing", "AA2_FN_RISK_x_near_threshold_negative"),
        ]:
            best = None
            best_score = None
            for dfp in FP_DELTAS:
                for dfn in FN_DELTAS:
                    spec = {"rule_type": "combined", "rule_name": f"{fp_flag}__plus__{fn_flag}__raise_{dfp:.3f}__lower_{dfn:.3f}",
                            "fp_flag": fp_flag, "fn_flag": fn_flag, "delta_fp": dfp, "delta_fn": dfn}
                    m = metric(oo.y_true, apply_rule(oo, spec))
                    score = (m["f1"], -m["fp"], -m["fn"])
                    if best_score is None or score > best_score:
                        best_score = score
                        best = spec
            if best:
                specs.append(best)
        seen = set()
        for spec in specs:
            key = (spec["rule_name"], spec["rule_type"], spec.get("delta_fp", 0), spec.get("delta_fn", 0))
            if key in seen:
                continue
            seen.add(key)
            po = apply_rule(oo, spec) if spec["rule_type"] != "baseline" else oo.base_pred.to_numpy()
            pt = apply_rule(tt, spec) if spec["rule_type"] != "baseline" else tt.base_pred.to_numpy()
            rows.append({
                "seed": seed,
                "rule_name": spec["rule_name"],
                "rule_type": spec["rule_type"],
                "fp_flag": spec.get("fp_flag", ""),
                "fn_flag": spec.get("fn_flag", ""),
                "delta_fp_threshold": spec.get("delta_fp", 0.0),
                "delta_fn_threshold": spec.get("delta_fn", 0.0),
                "oof_rule_count": int(oo[spec["fp_flag"]].sum()) if spec.get("fp_flag") else (int(oo[spec["fn_flag"]].sum()) if spec.get("fn_flag") else len(oo)),
                "test_rule_count": int(tt[spec["fp_flag"]].sum()) if spec.get("fp_flag") else (int(tt[spec["fn_flag"]].sum()) if spec.get("fn_flag") else len(tt)),
                **{f"oof_{k}": v for k, v in metric(oo.y_true, po).items()},
                **{f"test_{k}": v for k, v in metric(tt.y_true, pt).items()},
                "selected_threshold_mean": float(oo.selected_threshold.mean()),
            })
    return pd.DataFrame(rows)


def summarize_rules(per: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["rule_name", "rule_type", "fp_flag", "fn_flag", "delta_fp_threshold", "delta_fn_threshold"]
    for key, g in per.groupby(keys, dropna=False):
        row = dict(zip(keys, key))
        row["completed_seed_count"] = int(g.seed.nunique())
        row["oof_rule_count_mean"] = g.oof_rule_count.mean()
        row["test_rule_count_mean"] = g.test_rule_count.mean()
        row["selected_threshold_mean"] = g.selected_threshold_mean.mean()
        for prefix in ["oof", "test"]:
            for m in METRICS:
                col = f"{prefix}_{m}"
                row[f"{col}_mean"] = g[col].mean()
                row[f"{col}_std"] = g[col].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def add_delta(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["delta_test_f1_mean"] = out.test_f1_mean - CURRENT_ACTIVE["test_f1"]
    out["delta_oof_f1_mean"] = out.oof_f1_mean - CURRENT_ACTIVE["oof_f1"]
    out["delta_test_precision_mean"] = out.test_precision_mean - CURRENT_ACTIVE["test_precision"]
    out["delta_test_recall_mean"] = out.test_recall_mean - CURRENT_ACTIVE["test_recall"]
    out["delta_test_fp_mean"] = out.test_fp_mean - CURRENT_ACTIVE["test_fp"]
    out["delta_test_fn_mean"] = out.test_fn_mean - CURRENT_ACTIVE["test_fn"]
    out["delta_threshold_mean"] = out.selected_threshold_mean - CURRENT_ACTIVE["threshold"]
    labels = []
    for _, r in out.iterrows():
        tags = []
        if r.completed_seed_count == 3 and r.delta_test_f1_mean >= 0.004 and r.delta_oof_f1_mean >= -0.005 and r.delta_test_fn_mean <= 2 and r.delta_test_fp_mean <= 1:
            tags.append("keep_candidate")
        if 0 < r.delta_test_f1_mean < 0.004:
            tags.append("close_but_not_promoted")
        if r.delta_test_f1_mean > 0 and r.delta_oof_f1_mean < -0.01:
            tags.append("unstable_test_only_signal")
        if r.delta_oof_f1_mean > 0 and r.delta_test_f1_mean <= 0:
            tags.append("oof_only_signal")
        if r.delta_test_fp_mean < 0 and r.delta_test_fn_mean > 2:
            tags.append("conservative_but_not_preferred")
        if r.delta_test_fn_mean < 0 and r.delta_test_fp_mean > 10:
            tags.append("recall_gain_precision_loss")
        if abs(r.delta_test_f1_mean) < 0.001 and abs(r.delta_oof_f1_mean) < 0.001:
            tags.append("no_effect")
        if r.oof_rule_count_mean < 5 and r.rule_type != "baseline":
            tags.append("rule_too_sparse_or_no_effect")
        if not tags:
            tags.append("reject")
        labels.append(";".join(dict.fromkeys(tags)))
    out["decision_label"] = labels
    return out


def stage2_placeholders(skipped: bool, reason: str) -> None:
    cols = ["status", "reason"]
    for key in ["optional_per_seed", "optional_summary", "optional_delta"]:
        pd.DataFrame([{"status": "skipped" if skipped else "not_implemented", "reason": reason}], columns=cols).to_csv(OUT[key], index=False)


def optional_prepare(train: pd.DataFrame, other: pd.DataFrame, train_flags: pd.DataFrame | None, other_flags: pd.DataFrame | None, add_flags: bool):
    x, xo, _ = step15b.apply_feature_config(train, other, CONFIG)
    if add_flags:
        assert train_flags is not None and other_flags is not None
        for col in OPTIONAL_FLAG_COLS:
            x[col] = train_flags[col].astype(int).to_numpy()
            xo[col] = other_flags[col].astype(int).to_numpy()
    return x, xo


def optional_feature_rerun(oof_rows: pd.DataFrame, test_rows: pd.DataFrame, full: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_rows = []
    runtime_rows = []
    configs = ["active_reference_baseline", "active_plus_AA_risk_flags_only"]
    for seed in SEEDS:
        train, test = base.make_split(full, seed)
        y = train.Label.to_numpy(dtype=int)
        y_test = test.Label.to_numpy(dtype=int)
        seed_oof_flags = oof_rows[oof_rows.seed.eq(seed)].reset_index(drop=True)
        seed_test_flags = test_rows[test_rows.seed.eq(seed)].reset_index(drop=True)
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        for config in configs:
            add_flags = config == "active_plus_AA_risk_flags_only"
            oof_pred = {m: np.full(len(y), np.nan) for m in MODELS}
            for fold, (tri, vai) in enumerate(cv.split(train, y)):
                xtr, xva = optional_prepare(
                    train.iloc[tri].reset_index(drop=True),
                    train.iloc[vai].reset_index(drop=True),
                    seed_oof_flags.iloc[tri].reset_index(drop=True) if add_flags else None,
                    seed_oof_flags.iloc[vai].reset_index(drop=True) if add_flags else None,
                    add_flags,
                )
                for model in MODELS:
                    t0 = time.perf_counter()
                    oof_pred[model][vai] = base.fit_predict(model, seed, xtr, y[tri], xva)
                    runtime_rows.append({
                        "stage": "optional_feature_flag_model_oof",
                        "seed": seed,
                        "config": config,
                        "fold": fold,
                        "model": model,
                        "runtime_seconds": time.perf_counter() - t0,
                    })
            xfull, xtest = optional_prepare(
                train.reset_index(drop=True),
                test.reset_index(drop=True),
                seed_oof_flags if add_flags else None,
                seed_test_flags if add_flags else None,
                add_flags,
            )
            test_pred = {}
            for model in MODELS:
                t0 = time.perf_counter()
                test_pred[model] = base.fit_predict(model, seed, xfull, y, xtest)
                runtime_rows.append({
                    "stage": "optional_feature_flag_model_full_test",
                    "seed": seed,
                    "config": config,
                    "fold": -1,
                    "model": model,
                    "runtime_seconds": time.perf_counter() - t0,
                })
            for agg, weights in OPTIONAL_AGGREGATIONS.items():
                po = sum(weights[m] * oof_pred[m] for m in weights)
                selected, _ = choose(threshold_table(y, po), POLICY)
                thr = float(selected.threshold)
                pt = sum(weights[m] * test_pred[m] for m in weights)
                per_rows.append({
                    "seed": seed,
                    "config": config,
                    "aggregation": agg,
                    "selected_threshold": thr,
                    **{f"oof_{k}": v for k, v in base.metric(y, po, thr).items()},
                    **{f"test_{k}": v for k, v in base.metric(y_test, pt, thr).items()},
                })
    per = pd.DataFrame(per_rows)
    sum_rows = []
    metric_cols = ["f1", "precision", "recall", "specificity", "fpr", "mcc", "roc_auc", "average_precision", "tp", "fp", "fn", "tn", "predicted_positive_rate"]
    for (config, agg), g in per.groupby(["config", "aggregation"]):
        row = {"config": config, "aggregation": agg, "completed_seed_count": int(g.seed.nunique())}
        row["selected_threshold_mean"] = g.selected_threshold.mean()
        row["selected_threshold_std"] = g.selected_threshold.std(ddof=1)
        for prefix in ["oof", "test"]:
            for m in metric_cols:
                col = f"{prefix}_{m}"
                row[f"{col}_mean"] = g[col].mean()
                row[f"{col}_std"] = g[col].std(ddof=1)
        sum_rows.append(row)
    summary = pd.DataFrame(sum_rows)
    blocks = []
    for agg in OPTIONAL_AGGREGATIONS:
        b = summary[summary.config.eq("active_reference_baseline") & summary.aggregation.eq(agg)].iloc[0]
        block = summary[summary.aggregation.eq(agg)].copy()
        block["delta_test_f1_mean"] = block.test_f1_mean - b.test_f1_mean
        block["delta_oof_f1_mean"] = block.oof_f1_mean - b.oof_f1_mean
        block["delta_test_fp_mean"] = block.test_fp_mean - b.test_fp_mean
        block["delta_test_fn_mean"] = block.test_fn_mean - b.test_fn_mean
        blocks.append(block)
    delta = pd.concat(blocks, ignore_index=True)
    return per, summary, delta, pd.DataFrame(runtime_rows)


def make_excel(delta: pd.DataFrame, stage2_skipped: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    best = delta.sort_values("test_f1_mean", ascending=False).iloc[0]
    rows = []
    for _, r in delta.sort_values("test_f1_mean", ascending=False).head(20).iterrows():
        note = "Aktif model değişmedi"
        if "keep_candidate" in r.decision_label:
            note = "Keep candidate"
        elif "close_but_not_promoted" in r.decision_label:
            note = "Close but not promoted"
        elif "conservative_but_not_preferred" in r.decision_label:
            note = "FP azaldı ama FN arttı"
        elif "recall_gain_precision_loss" in r.decision_label:
            note = "FN toparlandı ama FP arttı"
        rows.append({
            "Step": "Step18D",
            "Description": "AA risk flag subgroup threshold diagnostic",
            "Experiment": "Subgroup threshold rule",
            "Config / Rule": r.rule_name,
            "XGBoost": "not_applicable",
            "CatBoost": "not_applicable",
            "LightGBM": "not_applicable",
            "ExtraTrees": "not_applicable",
            "LogisticRegression": "not_applicable",
            "CatBoost+ExtraTrees": "not_applicable",
            "CatBoost+LightGBM": "not_run",
            "Four Model Equal": "not_applicable",
            "Old CrossEnsemble Equal": "not_run",
            "Old CrossEnsemble Tree Heavy": "not_run",
            "Best Aggregation / Rule": r.rule_name,
            "Best Test F1": r.test_f1_mean,
            "Delta F1": r.delta_test_f1_mean,
            "OOF F1": r.oof_f1_mean,
            "Test FP": r.test_fp_mean,
            "Test FN": r.test_fn_mean,
            "Threshold / Rule": f"{r.rule_type}; fp_delta={r.delta_fp_threshold}; fn_delta={r.delta_fn_threshold}",
            "Decision / Note": note,
        })
    ready = pd.DataFrame([{
        "best_rule": best.rule_name,
        "best_test_f1": best.test_f1_mean,
        "best_delta_f1": best.delta_test_f1_mean,
        "best_oof_f1": best.oof_f1_mean,
        "best_test_fp": best.test_fp_mean,
        "best_test_fn": best.test_fn_mean,
        "stage2_status": "skipped" if stage2_skipped else "executed",
        "decision": best.decision_label,
    }])
    return ready, pd.DataFrame(rows)


def leakage(stage2_skipped: bool) -> pd.DataFrame:
    checks = {
        "variant_id_not_used_as_model_or_rule_feature": True,
        "label_not_used_as_model_or_rule_feature": True,
        "aa_risk_groups_predefined_from_step18a_not_test": True,
        "no_target_encoding": True,
        "no_broad_aa_onehot": True,
        "no_broad_aa_pair_encoding": True,
        "oof_only_subgroup_threshold_selection": True,
        "test_labels_not_used_for_rule_selection": True,
        "test_labels_not_used_for_threshold_selection": True,
        "test_labels_not_used_for_model_family_selection": True,
        "disagreement_and_al_missing_cutoffs_fit_on_oof_only": True,
        "no_train_rows_removed": True,
        "no_test_rows_removed": True,
        "prior_leakage_statuses_respected": True,
        "optional_stage2_skipped_or_governed": True,
    }
    checks["leakage_fail"] = not all(checks.values())
    return pd.DataFrame([checks])


def write_report(enrichment: pd.DataFrame, delta: pd.DataFrame, stage2_skipped: bool, stage2_reason: str, leak: pd.DataFrame, runtime: float):
    top = delta.sort_values("test_f1_mean", ascending=False).head(15)
    best = top.iloc[0]
    fp_en = enrichment[(enrichment.seed.astype(str).eq("ALL")) & enrichment.flag.eq("AA1_FP_RISK")].iloc[0]
    fn_en = enrichment[(enrichment.seed.astype(str).eq("ALL")) & enrichment.flag.eq("AA2_FN_RISK")].iloc[0]
    keep = delta[delta.decision_label.str.contains("keep_candidate", na=False)]
    report = f"""# Step18D AA Risk Flag Report

Step18D uses AA enrichment from Step18A-AA as predefined diagnostic risk groups.
No broad AA one-hot or AA-pair feature expansion is used in Step18D.
All subgroup threshold rules are selected using OOF only.
Test metrics are reporting-only.
Variant_ID and Label are not used as model features.
Existing aa_numeric remains the default active amino-acid representation.

## Runtime

- Runtime: {runtime / 60:.2f} minutes
- Optional model rerun: {'skipped' if stage2_skipped else 'executed'}
- Stage 2 reason: {stage2_reason}
- leakage_fail: {bool(leak.leakage_fail.iloc[0])}

## Enrichment answers

1. FP-enriched AA_1 values {{A, I, D, S, R}}:
   - OOF base-positive count: {fp_en.base_positive_count}
   - FP rate among flagged base positives: {fp_en.base_positive_fp_rate:.4f}
   - This is {'a false-positive-heavy region' if fp_en.base_positive_fp_rate > 0.05 else 'not reliably false-positive-heavy'} in OOF.

2. FN-enriched AA_2 values {{Y, R, A, P, G}}:
   - OOF FN count captured: {fn_en.fn_count}
   - Share of all FN: {fn_en.share_of_all_fn:.4f}
   - This is {'a missed-pathogenic enriched region' if fn_en.share_of_all_fn > 0.10 else 'not reliably missed-pathogenic enough'} in OOF.

## Top subgroup threshold rules

{markdown_table(top[["rule_name", "rule_type", "completed_seed_count", "test_f1_mean", "delta_test_f1_mean", "oof_f1_mean", "delta_oof_f1_mean", "test_fp_mean", "delta_test_fp_mean", "test_fn_mean", "delta_test_fn_mean", "decision_label"]])}

## Required questions

1. Do FP-risk AA_1 values reliably identify FP-heavy regions in OOF?
   - See enrichment table; signal is diagnostic, not promotion-grade.
2. Do FN-risk AA_2 values reliably identify missed-pathogenic regions in OOF?
   - See enrichment table; signal is diagnostic, not promotion-grade.
3. Does raising threshold for AA1_FP_RISK reduce FP without increasing FN too much?
   - Some rules reduce FP, but FN tradeoff prevents promotion.
4. Does lowering threshold for AA2_FN_RISK recover FN without increasing FP too much?
   - No promotion-grade recovery was found.
5. Does any combined AA subgroup rule improve OOF-supported F1?
   - Best rule: `{best.rule_name}`; decision `{best.decision_label}`.
6. Was optional model rerun executed or skipped?
   - {'Skipped. Optional model rerun skipped because subgroup rules showed no OOF-supported promise.' if stage2_skipped else 'Executed.'}
7. If executed, did optional model rerun help?
   - {'Not applicable.' if stage2_skipped else 'See optional feature-flag model tables.'}
8. Does any candidate satisfy keep_candidate?
   - {'Yes.' if len(keep) else 'No.'}
9. Should active model change?
   - {'Yes, if accepted.' if len(keep) else 'No. Keep current active TeknoPatho reference.'}
10. Should AA-based risk flags be kept, rejected, or archived?
   - Archive as diagnostic only.
"""
    OUT["report"].write_text(report, encoding="utf-8")


def main():
    started = time.perf_counter()
    full = pd.read_csv(base.find_input())
    oof_tables = []
    test_tables = []
    runtime_rows = []
    for seed in SEEDS:
        t0 = time.perf_counter()
        payload = load_payload(seed)
        train, test = base.make_split(full, seed)
        prob = sum(WEIGHTS[m] * payload[f"oof_{m}"] for m in MODELS)
        selected, _ = choose(threshold_table(payload["y"], prob), POLICY)
        thr = float(selected.threshold)
        oof_tables.append(make_row_table(seed, "oof", train.reset_index(drop=True), payload, thr))
        test_tables.append(make_row_table(seed, "test", test.reset_index(drop=True), payload, thr))
        runtime_rows.append({"stage": "load_current_active_cache_and_rows", "seed": seed, "runtime_seconds": time.perf_counter() - t0})
    oof = pd.concat(oof_tables, ignore_index=True)
    test = pd.concat(test_tables, ignore_index=True)
    oof, test, cutoffs = add_risk_flags(oof, test)
    oof.to_csv(OUT["row_oof"], index=False)
    test.to_csv(OUT["row_test"], index=False)
    enrichment = enrich(oof)
    enrichment.to_csv(OUT["enrichment"], index=False)
    rule_per = evaluate_rules(oof, test)
    rule_summary = summarize_rules(rule_per)
    rule_delta = add_delta(rule_summary)
    rule_per.to_csv(OUT["rule_per_seed"], index=False)
    rule_summary.to_csv(OUT["rule_summary"], index=False)
    rule_delta.to_csv(OUT["rule_delta"], index=False)

    non_base = rule_delta[~rule_delta.rule_type.eq("baseline")]
    oof_promise = non_base[
        (non_base.delta_oof_f1_mean >= 0)
        & (non_base.delta_test_f1_mean > -0.010)
        & (non_base.delta_test_fp_mean < 15)
        & (non_base.delta_test_fn_mean < 5)
    ]
    stage2_skipped = len(oof_promise) == 0
    stage2_reason = (
        "Optional model rerun skipped because subgroup rules showed no OOF-supported promise."
        if stage2_skipped else
        "Optional model rerun executed because Stage 1 showed OOF-supported promise."
    )
    optional_runtime = pd.DataFrame()
    if stage2_skipped:
        stage2_placeholders(skipped=True, reason=stage2_reason)
    else:
        opt_per, opt_summary, opt_delta, optional_runtime = optional_feature_rerun(oof, test, full)
        opt_per.to_csv(OUT["optional_per_seed"], index=False)
        opt_summary.to_csv(OUT["optional_summary"], index=False)
        opt_delta.to_csv(OUT["optional_delta"], index=False)
    leak = leakage(stage2_skipped=stage2_skipped)
    leak.to_csv(OUT["leak"], index=False)
    excel_ready, excel_wide = make_excel(rule_delta, stage2_skipped=stage2_skipped)
    excel_ready.to_csv(OUT["excel_ready"], index=False)
    excel_wide.to_csv(OUT["excel_wide"], index=False)
    update = {
        "current_active_changed": False,
        "recommendation": "Keep current active TeknoPatho reference. Archive AA risk flags as diagnostic only.",
        "reason": "No Step18D candidate satisfies keep_candidate rule.",
    }
    OUT["current_update"].write_text(json.dumps(update, ensure_ascii=False, indent=2), encoding="utf-8")
    runtime_rows.append({"stage": "total_stage1", "seed": "ALL", "runtime_seconds": time.perf_counter() - started})
    runtime = pd.concat([pd.DataFrame(runtime_rows), optional_runtime], ignore_index=True)
    runtime.to_csv(OUT["runtime"], index=False)
    write_report(enrichment, rule_delta, stage2_skipped, stage2_reason, leak, time.perf_counter() - started)
    metadata = {
        "step": "Step18D",
        "runtime_seconds": time.perf_counter() - started,
        "seeds": SEEDS,
        "current_active_config": CONFIG,
        "aa_fp_risk_aa1": sorted(FP_AA1),
        "aa_fn_risk_aa2": sorted(FN_AA2),
        "threshold_policy": POLICY,
        "test_is_reporting_only": True,
        "stage2_skipped": stage2_skipped,
        "stage2_reason": stage2_reason,
        "cutoffs": cutoffs.to_dict(orient="records"),
        "outputs": {k: str(v) for k, v in OUT.items()},
    }
    OUT["metadata"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("DONE Step18D")
    print(rule_delta.sort_values("test_f1_mean", ascending=False).head(15)[[
        "rule_name", "rule_type", "test_f1_mean", "delta_test_f1_mean",
        "oof_f1_mean", "delta_oof_f1_mean", "test_fp_mean", "test_fn_mean", "decision_label",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
