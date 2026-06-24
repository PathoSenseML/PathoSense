"""MASTER Step 11B: train-only column filtering / feature cleanup ablation.

This is an ablation evidence script, not a final model-selection step.
All filtering rules are computed on the train feature matrix only; test is
reporting-only.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from master_step3b_repeated_split_catboost import find_input, leakage_audit, make_split
from master_step4_aa_al_ablation import transform_config
from master_step7_preprocessing_missingness_ablation import missingness_features
from master_step8_threshold_policy_refinement import choose, threshold_table
from master_step9_error_analysis import confidence, err
from master_v02_catboost_threshold_stability import catboost_frame, make_model, markdown_table

ROOT = Path(__file__).resolve().parent
SEEDS = [7, 21, 42, 101, 2024]
N_SPLITS = 5
POLICY = "weighted_f1_current"
CONFIGS = [
    "baseline_current",
    "drop_train_constant_features",
    "drop_train_near_constant_features",
    "drop_exact_duplicate_features",
    "drop_high_missing_features_98",
    "drop_low_info_combined",
    "drop_oof_consistent_shift_top_features_conservative",
    "drop_high_missing_features_95",
    "drop_EK_consistent_shift_candidates",
    "drop_AL_consistent_shift_candidates",
]
PARTIAL_ROWS = ROOT / ".master_step11b_partial_row_predictions.csv"
PARTIAL_RUNTIME = ROOT / ".master_step11b_partial_runtime.csv"
OUT = {
    "script": ROOT / "master_step11b_column_filtering_ablation.py",
    "test": ROOT / "master_step11b_seed_config_metrics.csv",
    "oof": ROOT / "master_step11b_oof_metrics_by_seed_config.csv",
    "summary": ROOT / "master_step11b_repeated_test_summary_by_config.csv",
    "delta": ROOT / "master_step11b_delta_vs_baseline.csv",
    "filter": ROOT / "master_step11b_feature_filtering_report.csv",
    "dropped": ROOT / "master_step11b_dropped_features_by_seed_config.csv",
    "subgroup": ROOT / "master_step11b_subgroup_metrics.csv",
    "rows": ROOT / "master_step11b_row_level_predictions.csv",
    "transition": ROOT / "master_step11b_error_transition_vs_baseline.csv",
    "transition_profiles": ROOT / "master_step11b_transition_group_profiles.csv",
    "leak": ROOT / "master_step11b_leakage_audit.csv",
    "runtime": ROOT / "master_step11b_runtime_report.csv",
    "report": ROOT / "master_step11b_report.md",
    "metadata": ROOT / "master_step11b_metadata.json",
}


def metric(y, prob, threshold):
    y = np.asarray(y, int)
    prob = np.asarray(prob, float)
    if len(y) == 0:
        return {k: np.nan for k in ["f1", "precision", "recall", "specificity", "fpr", "balanced_accuracy",
                                    "predicted_positive_rate", "tn", "fp", "fn", "tp"]}
    pred = (prob >= threshold).astype(int)
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    fpr = fp / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"f1": f1, "precision": precision, "recall": recall, "specificity": specificity, "fpr": fpr,
            "balanced_accuracy": (recall + specificity) / 2, "predicted_positive_rate": float(pred.mean()),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def feature_group(name: str) -> str:
    low = str(name).lower()
    if any(t.lower() in low for t in ["row_missing", "missing_rate", "missing_count", "all_nan", "high_missing",
                                      "aa_missing", "al_missing", "cat_missing", "ek_missing"]):
        return "missingness"
    if str(name).startswith("AL_"):
        return "AL"
    if str(name).startswith("CAT_"):
        return "CAT"
    if str(name).startswith("EK_"):
        return "EK"
    if any(t in low for t in ["aa_", "blosum", "grantham", "hydrophobicity", "molecular_weight", "charge", "polarity"]):
        return "AA_derived"
    return "other"


def context(frame, raw):
    m = missingness_features(frame, raw)
    c = pd.DataFrame({"Variant_ID": frame.Variant_ID.astype(str).to_numpy(), "true_label": frame.Label.to_numpy(int)})
    c["raw_missing_rate"] = m.row_missing_rate.to_numpy()
    c["high_missing_80_flag"] = m.high_missing_80_flag.to_numpy()
    c["high_missing_90_flag"] = m.high_missing_90_flag.to_numpy()
    c["high_missing_95_flag"] = m.high_missing_95_flag.to_numpy()
    c["all_nan_flag"] = m.all_nan_flag.to_numpy()
    c["AA_missing_flag"] = m.AA_missing_flag.to_numpy()
    c["AL_missing_rate"] = m.AL_missing_rate.to_numpy()
    c["CAT_missing_rate"] = m.CAT_missing_rate.to_numpy()
    c["EK_missing_rate"] = m.EK_missing_rate.to_numpy()
    c["CAT_6_value"] = frame["CAT_6"].to_numpy() if "CAT_6" in frame else np.nan
    c["CAT6_nonmissing_flag"] = frame["CAT_6"].notna().astype(int).to_numpy() if "CAT_6" in frame else 0
    c["AA_pair"] = frame.AA_1.astype("string").fillna("<NA>").to_numpy() + ">" + frame.AA_2.astype("string").fillna("<NA>").to_numpy()
    return c


def row_block(seed, config, role, fold_id, frame, raw, prob, threshold):
    c = context(frame, raw)
    prob = np.asarray(prob, float)
    pred = (prob >= threshold).astype(int)
    margin = prob - threshold
    c.insert(0, "seed", seed)
    c.insert(1, "config_name", config)
    c.insert(2, "split_role", role)
    c.insert(3, "fold_id", np.asarray(fold_id, int))
    c["pred_proba_pathogenic"] = prob
    c["selected_threshold"] = threshold
    c["pred_label"] = pred
    c["error_type"] = err(c.true_label.to_numpy(), pred)
    c["margin"] = margin
    c["abs_margin"] = np.abs(margin)
    c["confidence_bin"] = confidence(np.abs(margin))
    return c


def prepare_xy(train, test, raw, al):
    x, xt, _, status = transform_config(train, test, raw, al, "aa_numeric")
    if status != "OK":
        raise RuntimeError(status)
    x = pd.concat([x.reset_index(drop=True), missingness_features(train, raw)], axis=1)
    xt = pd.concat([xt.reset_index(drop=True), missingness_features(test, raw)], axis=1)
    forbidden = [c for c in ["Variant_ID", "Label", "AA_1", "AA_2"] if c in x.columns]
    if forbidden:
        x = x.drop(columns=forbidden)
        xt = xt.drop(columns=[c for c in forbidden if c in xt.columns])
    return x, xt


def stats_and_duplicates(x):
    rows = {}
    for col in x.columns:
        s = x[col]
        vc = s.value_counts(dropna=False)
        top = float(vc.iloc[0] / len(s)) if len(vc) else np.nan
        var = pd.to_numeric(s, errors="coerce").var(ddof=0) if pd.api.types.is_numeric_dtype(s) else np.nan
        rows[col] = {
            "missing_rate": float(s.isna().mean()),
            "nunique": int(s.nunique(dropna=False)),
            "top_freq": top,
            "variance": float(var) if pd.notna(var) else np.nan,
        }
    duplicate_drop = {}
    duplicate_keep = {}
    group_id = 0
    seen = {}
    for col in x.columns:
        key = tuple(pd.util.hash_pandas_object(x[col], index=False).tolist())
        seen.setdefault(key, []).append(col)
    for cols in seen.values():
        if len(cols) <= 1:
            continue
        group_id += 1
        keep = sorted(cols, key=lambda c: (rows[c]["missing_rate"], len(c), c))[0]
        for c in cols:
            if c != keep:
                duplicate_drop[c] = group_id
                duplicate_keep[c] = keep
    return rows, duplicate_drop, duplicate_keep


def step10_candidates():
    path = ROOT / "master_step10_feature_shift_consistency.csv"
    if not path.exists():
        return {"top": [], "EK": [], "AL": []}, False
    d = pd.read_csv(path)
    if "test_only_observation" not in d:
        d["test_only_observation"] = False
    cand = d[(d.stable_shift_candidate == True) & (d.test_only_observation == False)].copy()
    if "stable_candidate_rank" in cand:
        cand = cand.sort_values(["stable_candidate_rank", "feature"])
    else:
        cand = cand.sort_values("feature")
    groups = ["EK", "AL", "CAT", "AA_missing", "AA_derived", "other"]
    top = cand[cand.feature_group.isin(groups)]["feature"].drop_duplicates().head(10).tolist()
    ek = cand[cand.feature_group.eq("EK")]["feature"].drop_duplicates().head(10).tolist()
    al = cand[cand.feature_group.eq("AL")]["feature"].drop_duplicates().head(10).tolist()
    return {"top": top, "EK": ek, "AL": al}, True


def filter_plan(config, x, stat_rows, dup_drop, dup_keep, candidates):
    reasons = defaultdict(list)
    kept = {}
    dupids = {}

    def add(cols, reason):
        for c in cols:
            if c in x.columns:
                reasons[c].append(reason)

    constant = [c for c, s in stat_rows.items() if s["nunique"] <= 1]
    near = [c for c, s in stat_rows.items() if s["top_freq"] >= 0.995 or (pd.notna(s["variance"]) and s["variance"] <= 1e-12)]
    miss95 = [c for c, s in stat_rows.items() if s["missing_rate"] >= 0.95]
    miss98 = [c for c, s in stat_rows.items() if s["missing_rate"] >= 0.98]
    dupcols = list(dup_drop)
    for c in dupcols:
        kept[c] = dup_keep[c]
        dupids[c] = dup_drop[c]

    if config == "drop_train_constant_features":
        add(constant, "constant")
    elif config == "drop_train_near_constant_features":
        add(near, "near_constant")
    elif config == "drop_exact_duplicate_features":
        add(dupcols, "duplicate")
    elif config == "drop_high_missing_features_95":
        add(miss95, "high_missing_95")
    elif config == "drop_high_missing_features_98":
        add(miss98, "high_missing_98")
    elif config == "drop_low_info_combined":
        add(constant, "constant")
        add(near, "near_constant")
        add(dupcols, "duplicate")
        add(miss98, "high_missing_98")
    elif config == "drop_oof_consistent_shift_top_features_conservative":
        add(candidates["top"], "step10_consistent_shift_candidate")
    elif config == "drop_EK_consistent_shift_candidates":
        add(candidates["EK"], "EK_consistent_shift_candidate")
    elif config == "drop_AL_consistent_shift_candidates":
        add(candidates["AL"], "AL_consistent_shift_candidate")

    dropped = sorted(reasons)
    report = []
    for c in dropped:
        s = stat_rows[c]
        report.append({
            "feature": c,
            "reason": "|".join(sorted(set(reasons[c]))),
            "kept_representative_feature": kept.get(c, ""),
            "duplicate_group_id": dupids.get(c, np.nan),
            "train_missing_rate": s["missing_rate"],
            "train_nunique": s["nunique"],
            "train_top_value_frequency": s["top_freq"],
            "train_variance": s["variance"],
            "feature_group": feature_group(c),
        })
    return dropped, report


def train_config(seed, config, train, test, raw, al, dropped):
    x, xt = prepare_xy(train, test, raw, al)
    original_count = x.shape[1]
    x = x.drop(columns=[c for c in dropped if c in x.columns])
    xt = xt.drop(columns=[c for c in dropped if c in xt.columns])
    features = x.columns.tolist()
    cat = x.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    y = train.Label.to_numpy(int)
    oof = np.full(len(y), np.nan)
    fold_id = np.full(len(y), -99, int)
    cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)
    t0 = time.perf_counter()
    for fold, (fi, vi) in enumerate(cv.split(x, y)):
        model = make_model(seed)
        model.fit(catboost_frame(x.iloc[fi], cat), y[fi], cat_features=cat)
        oof[vi] = model.predict_proba(catboost_frame(x.iloc[vi], cat))[:, 1]
        fold_id[vi] = fold
    oof_seconds = time.perf_counter() - t0
    selected, _ = choose(threshold_table(y, oof), POLICY)
    threshold = float(selected.threshold)
    t1 = time.perf_counter()
    model = make_model(seed)
    model.fit(catboost_frame(x, cat), y, cat_features=cat)
    ptest = model.predict_proba(catboost_frame(xt, cat))[:, 1]
    final_seconds = time.perf_counter() - t1
    rows = pd.concat([
        row_block(seed, config, "train_oof", fold_id, train, raw, oof, threshold),
        row_block(seed, config, "test_reporting", np.full(len(test), -1), test, raw, ptest, threshold),
    ], ignore_index=True)
    return rows, features, original_count, len(features), oof_seconds, final_seconds, threshold


def reuse_step10(seed, train, test, raw):
    path = ROOT / "master_step10_row_level_predictions_with_folds.csv"
    if not path.exists():
        return None
    step10 = pd.read_csv(path)
    out = []
    for role, frame in [("train_oof", train), ("test_reporting", test)]:
        s = step10[(step10.seed.eq(seed)) & step10.split_role.eq(role)].reset_index(drop=True)
        if len(s) != len(frame):
            return None
        if not np.array_equal(s.Variant_ID.astype(str), frame.Variant_ID.astype(str)):
            return None
        th = float(s.selected_threshold_from_oof.iloc[0])
        out.append(row_block(seed, "baseline_current", role, s.fold_id.to_numpy(), frame, raw, s.pred_proba_pathogenic.to_numpy(), th))
    return pd.concat(out, ignore_index=True)


def rows_from_cache(seed, config, source_config, cached):
    r = cached.copy()
    r["config_name"] = config
    return r


def metrics_tables(rows):
    test, oof = [], []
    for (seed, config, role), d in rows.groupby(["seed", "config_name", "split_role"]):
        th = float(d.selected_threshold.iloc[0])
        m = metric(d.true_label, d.pred_proba_pathogenic, th)
        rec = {"seed": seed, "config_name": config, "selected_threshold": th, **m}
        if role == "test_reporting":
            rec["test_usage"] = "reporting_only"
            rec["auc"] = roc_auc_score(d.true_label, d.pred_proba_pathogenic)
            rec["average_precision"] = average_precision_score(d.true_label, d.pred_proba_pathogenic)
            test.append(rec)
        else:
            oof.append(rec)
    return pd.DataFrame(test), pd.DataFrame(oof)


def aggregate(test, oof, drop_summary):
    rows = []
    for config, t in test.groupby("config_name", sort=False):
        oo = oof[oof.config_name.eq(config)]
        r = {"config_name": config, "n_seeds": int(t.seed.nunique())}
        for c in ["f1", "precision", "recall", "specificity", "fpr", "balanced_accuracy", "fp", "fn", "selected_threshold"]:
            r[f"test_{c}_mean"] = t[c].mean()
            r[f"test_{c}_std"] = t[c].std(ddof=1)
        for c in ["f1", "recall", "fpr"]:
            r[f"oof_{c}_mean"] = oo[c].mean()
            r[f"oof_{c}_std"] = oo[c].std(ddof=1)
        ds = drop_summary[drop_summary.config_name.eq(config)]
        r["dropped_feature_count_mean"] = ds.dropped_feature_count.mean()
        r["dropped_feature_count_std"] = ds.dropped_feature_count.std(ddof=1)
        rows.append(r)
    return pd.DataFrame(rows)


def delta_table(summary):
    b = summary[summary.config_name.eq("baseline_current")].iloc[0]
    out = summary.copy()
    for c in ["test_f1_mean", "test_precision_mean", "test_recall_mean", "test_specificity_mean", "test_fpr_mean",
              "test_fp_mean", "test_fn_mean", "oof_f1_mean", "oof_recall_mean", "oof_fpr_mean",
              "dropped_feature_count_mean"]:
        out["delta_" + c] = out[c] - b[c]
    out["oof_test_delta_same_direction"] = np.sign(out.delta_test_f1_mean) == np.sign(out.delta_oof_f1_mean)
    out.loc[out.config_name.eq("baseline_current"), "oof_test_delta_same_direction"] = True
    labels = []
    for _, r in out.iterrows():
        if r.config_name == "baseline_current":
            labels.append("reference")
        elif (r.delta_test_f1_mean >= .010 and r.delta_oof_f1_mean >= .005 and r.delta_test_fn_mean <= 2
              and r.delta_test_recall_mean >= -.03 and r.oof_test_delta_same_direction):
            labels.append("strong_positive_evidence")
        elif (.005 <= r.delta_test_f1_mean < .010 and r.delta_test_fp_mean < 0 and r.delta_test_fn_mean <= 2
              and r.oof_test_delta_same_direction):
            labels.append("weak_positive_evidence")
        elif (r.delta_test_f1_mean < 0 or r.delta_test_fn_mean > 3 or r.delta_test_recall_mean < -.04
              or not r.oof_test_delta_same_direction):
            labels.append("negative_evidence")
        elif (abs(r.delta_test_f1_mean) < .005 and (r.delta_test_fp_mean < 0 or r.delta_test_fn_mean > 0)):
            labels.append("trade_off")
        else:
            labels.append("inconclusive")
    out["ablation_evidence_label"] = labels
    return out


def subgroup_masks(d):
    return {
        "all_rows": np.ones(len(d), bool),
        "high_missing_gt_80": d.raw_missing_rate.gt(.80),
        "high_missing_gt_90": d.raw_missing_rate.gt(.90),
        "high_missing_gt_95": d.raw_missing_rate.gt(.95),
        "all_nan": d.all_nan_flag.eq(1),
        "aa_missing": d.AA_missing_flag.eq(1),
        "non_aa_missing": d.AA_missing_flag.eq(0),
        "AL_missing_rate_gt_80": d.AL_missing_rate.gt(.80),
        "CAT_missing_rate_gt_80": d.CAT_missing_rate.gt(.80),
        "EK_missing_rate_gt_80": d.EK_missing_rate.gt(.80),
        "CAT_6_missing": d.CAT6_nonmissing_flag.eq(0),
        "CAT_6_nonmissing": d.CAT6_nonmissing_flag.eq(1),
    }


def subgroup_rows(rows):
    out = []
    for (seed, config, role), d in rows.groupby(["seed", "config_name", "split_role"]):
        th = float(d.selected_threshold.iloc[0])
        for name, mask in subgroup_masks(d).items():
            g = d[np.asarray(mask, bool)]
            m = metric(g.true_label, g.pred_proba_pathogenic, th)
            out.append({"seed": seed, "config_name": config, "split_role": role, "subgroup": name, "n": len(g),
                        "benign_count": int(g.true_label.eq(0).sum()), "pathogenic_count": int(g.true_label.eq(1).sum()),
                        **m, "mean_probability": g.pred_proba_pathogenic.mean(), "mean_margin": g.margin.mean(),
                        "small_n_warning": bool(len(g) < 20)})
    return pd.DataFrame(out)


def transition_tables(rows):
    parts = []
    profiles = []
    base = rows[rows.config_name.eq("baseline_current")]
    for config in [c for c in rows.config_name.unique() if c != "baseline_current"]:
        cand = rows[rows.config_name.eq(config)]
        z = base.merge(cand, on=["seed", "split_role", "Variant_ID"], suffixes=("_baseline", "_candidate"), validate="one_to_one")
        choices = [
            z.error_type_baseline.eq("FP") & z.error_type_candidate.eq("TN"),
            z.error_type_baseline.eq("TN") & z.error_type_candidate.eq("FP"),
            z.error_type_baseline.eq("FN") & z.error_type_candidate.eq("TP"),
            z.error_type_baseline.eq("TP") & z.error_type_candidate.eq("FN"),
        ]
        names = ["fixed_fp_vs_baseline", "introduced_fp_vs_baseline", "fixed_fn_vs_baseline", "introduced_fn_vs_baseline"]
        z["config_name"] = config
        z["transition_group"] = np.select(choices, names, default="unchanged_or_other")
        z["proba_delta"] = z.pred_proba_pathogenic_candidate - z.pred_proba_pathogenic_baseline
        parts.append(z[z.transition_group.ne("unchanged_or_other")])
    trans = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not trans.empty:
        for (config, role, group), g in trans.groupby(["config_name", "split_role", "transition_group"]):
            profiles.append({"config_name": config, "split_role": role, "transition_group": group, "n": len(g),
                             "label_distribution": json.dumps(g.true_label_baseline.value_counts().sort_index().to_dict()),
                             "raw_missing_rate_mean": g.raw_missing_rate_baseline.mean(),
                             "AA_missing_rate": g.AA_missing_flag_baseline.mean(),
                             "AL_missing_rate": g.AL_missing_rate_baseline.mean(),
                             "CAT_missing_rate": g.CAT_missing_rate_baseline.mean(),
                             "EK_missing_rate": g.EK_missing_rate_baseline.mean(),
                             "high_missing_90_rate": g.high_missing_90_flag_baseline.mean(),
                             "all_nan_rate": g.all_nan_flag_baseline.mean(),
                             "CAT6_nonmissing_rate": g.CAT6_nonmissing_flag_baseline.mean(),
                             "top_AA_pair": g.AA_pair_baseline.mode().iloc[0] if len(g.AA_pair_baseline.mode()) else "",
                             "mean_proba_delta": g.proba_delta.mean()})
    return trans, pd.DataFrame(profiles)


def append_csv(path, df):
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def main():
    wall = time.perf_counter()
    full = pd.read_csv(find_input())
    full["Label"] = full.Label.astype(int)
    raw = [c for c in full.columns if c not in ["Variant_ID", "Label"]]
    al = [c for c in raw if c.startswith("AL_") and pd.api.types.is_numeric_dtype(full[c])]
    candidates, uses_step10 = step10_candidates()

    done = set()
    if PARTIAL_RUNTIME.exists():
        rt0 = pd.read_csv(PARTIAL_RUNTIME)
        if {"seed", "config_name", "status"}.issubset(rt0.columns):
            done = set(map(tuple, rt0[rt0.status.eq("completed")][["seed", "config_name"]].values.tolist()))

    all_rows = [pd.read_csv(PARTIAL_ROWS)] if PARTIAL_ROWS.exists() else []
    runtime_rows, filter_rows, dropped_rows, leak_rows = [], [], [], []
    row_cache = {}
    feature_cache = {}

    for seed in SEEDS:
        train, test = make_split(full, seed)
        base_x, _ = prepare_xy(train, test, raw, al)
        stat_rows, dup_drop, dup_keep = stats_and_duplicates(base_x)
        plans = {}
        reports = {}
        for config in CONFIGS:
            dropped, report = filter_plan(config, base_x, stat_rows, dup_drop, dup_keep, candidates)
            plans[config] = dropped
            reports[config] = report
            groups = pd.Series([feature_group(c) for c in dropped], dtype=object)
            dropped_rows.append({"seed": seed, "config_name": config, "original_feature_count": base_x.shape[1],
                                 "final_feature_count": base_x.shape[1] - len(dropped),
                                 "dropped_feature_count": len(dropped),
                                 "dropped_features": json.dumps(dropped, ensure_ascii=False),
                                 "dropped_feature_groups": json.dumps(groups.value_counts().to_dict(), ensure_ascii=False),
                                 "dropped_AL_count": int((groups == "AL").sum()), "dropped_CAT_count": int((groups == "CAT").sum()),
                                 "dropped_EK_count": int((groups == "EK").sum()), "dropped_AA_derived_count": int((groups == "AA_derived").sum()),
                                 "dropped_missingness_count": int((groups == "missingness").sum()), "train_n": len(train), "test_n": len(test)})
            for r in report:
                rr = dict(r)
                rr.update({"seed": seed, "config_name": config})
                filter_rows.append(rr)

        for config in CONFIGS:
            if (seed, config) in done:
                print(f"STEP11B seed={seed} config={config} already completed; skipping", flush=True)
                continue
            dropped = plans[config]
            fingerprint = tuple(sorted([c for c in base_x.columns if c not in set(dropped)]))
            t0 = time.perf_counter()
            source = "RETRAINED_FROZEN_FACTORY"
            print(f"STEP11B seed={seed} config={config} drop={len(dropped)}", flush=True)
            if config == "baseline_current":
                rows = reuse_step10(seed, train, test, raw)
                if rows is not None:
                    x_now = base_x
                    features = base_x.columns.tolist()
                    original = final = base_x.shape[1]
                    oof_seconds = final_seconds = 0.0
                    th = float(rows.selected_threshold.iloc[0])
                    source = "REUSED_STEP10_EXACT"
                else:
                    rows, features, original, final, oof_seconds, final_seconds, th = train_config(seed, config, train, test, raw, al, dropped)
            elif fingerprint in row_cache:
                rows = rows_from_cache(seed, config, row_cache[fingerprint][0], row_cache[fingerprint][1])
                features, original, final, oof_seconds, final_seconds, th = feature_cache[fingerprint]
                source = f"REUSED_IDENTICAL_FEATURE_SET:{row_cache[fingerprint][0]}"
            else:
                rows, features, original, final, oof_seconds, final_seconds, th = train_config(seed, config, train, test, raw, al, dropped)

            row_cache[fingerprint] = (config, rows.copy())
            feature_cache[fingerprint] = (features, original, final, oof_seconds, final_seconds, th)
            elapsed = time.perf_counter() - t0
            all_rows.append(rows)
            append_csv(PARTIAL_ROWS, rows)
            runtime = {"seed": seed, "config_name": config, "status": "completed", "prediction_source": source,
                       "runtime_seconds": elapsed, "oof_seconds": oof_seconds, "final_fit_seconds": final_seconds,
                       "dropped_feature_count": len(dropped)}
            runtime_rows.append(runtime)
            append_csv(PARTIAL_RUNTIME, pd.DataFrame([runtime]))
            try:
                audit, _ = leakage_audit(full, train, test, features, seed, int(full.duplicated().sum()))
            except Exception as exc:
                audit = {"seed": seed, "status": "FAIL", "details": json.dumps({"error": str(exc)})}
            audit.update({"config_name": config, "threshold_selected_on_test": False, "config_selected_on_test": False,
                          "feature_selected_on_test": False, "test_rows_dropped": False, "train_rows_dropped": False,
                          "preprocessing_fitted_on_test": False, "label_used_in_transform": False,
                          "saved_probability_label_mismatch": False, "dropped_feature_list_uses_test_statistics": False})
            hard_fail = (audit.get("status") == "FAIL" or "Variant_ID" in features or "Label" in features
                         or "AA_1" in features or "AA_2" in features)
            audit["leakage_fail"] = bool(hard_fail)
            audit["leakage_warning"] = bool(audit.get("status") == "WARNING")
            leak_rows.append(audit)
            print(f"  completed source={source} minutes={elapsed/60:.2f}", flush=True)

    rows = pd.concat(all_rows, ignore_index=True).drop_duplicates(["seed", "config_name", "split_role", "Variant_ID"], keep="last")
    drop_df = pd.DataFrame(dropped_rows).drop_duplicates(["seed", "config_name"])
    filter_df = pd.DataFrame(filter_rows)
    leak = pd.DataFrame(leak_rows)
    runtime = pd.concat([pd.read_csv(PARTIAL_RUNTIME), pd.DataFrame(runtime_rows)], ignore_index=True).drop_duplicates(["seed", "config_name"], keep="last")

    test, oof = metrics_tables(rows)
    test = test.merge(drop_df[["seed", "config_name", "original_feature_count", "final_feature_count", "dropped_feature_count",
                              "dropped_features", "dropped_feature_groups", "dropped_AL_count", "dropped_CAT_count",
                              "dropped_EK_count", "dropped_AA_derived_count", "dropped_missingness_count", "train_n", "test_n"]],
                      on=["seed", "config_name"], how="left")
    oof = oof.merge(drop_df[["seed", "config_name", "dropped_feature_count"]], on=["seed", "config_name"], how="left")
    summary = aggregate(test, oof, drop_df)
    delta = delta_table(summary)
    sub = subgroup_rows(rows)
    trans, trans_profiles = transition_tables(rows)

    rows.to_csv(OUT["rows"], index=False)
    test.to_csv(OUT["test"], index=False)
    oof.to_csv(OUT["oof"], index=False)
    summary.to_csv(OUT["summary"], index=False)
    delta.to_csv(OUT["delta"], index=False)
    filter_df.to_csv(OUT["filter"], index=False)
    drop_df.to_csv(OUT["dropped"], index=False)
    sub.to_csv(OUT["subgroup"], index=False)
    trans.to_csv(OUT["transition"], index=False)
    trans_profiles.to_csv(OUT["transition_profiles"], index=False)
    leak.to_csv(OUT["leak"], index=False)
    runtime.to_csv(OUT["runtime"], index=False)

    b = summary[summary.config_name.eq("baseline_current")].iloc[0]
    nonbase = summary[~summary.config_name.eq("baseline_current")].sort_values("test_f1_mean", ascending=False)
    best = nonbase.iloc[0]
    d_best = delta[delta.config_name.eq(best.config_name)].iloc[0]
    rejected = delta[delta.ablation_evidence_label.eq("negative_evidence")]["config_name"].tolist()
    hypotheses = delta[(~delta.config_name.eq("baseline_current")) & (~delta.ablation_evidence_label.eq("negative_evidence"))]["config_name"].tolist()
    step14 = delta[delta.ablation_evidence_label.isin(["strong_positive_evidence", "weak_positive_evidence"])]["config_name"].tolist()

    report = f"""# MASTER Step 11B — column filtering / feature cleanup ablation

Bu çalışma final model seçimi değildir; kontrollü ablation evidence üretir. Test seti reporting-only kaldı, train/test satır silme yapılmadı ve feature filtering listeleri sadece train feature matrix üzerinden hesaplandı.

## Özet

- Baseline F1 / FP / FN: {b.test_f1_mean:.6f} / {b.test_fp_mean:.2f} / {b.test_fn_mean:.2f}
- En iyi reporting-only F1 config: `{best.config_name}` = {best.test_f1_mean:.6f}
- En iyi config delta F1 / FP / FN: {d_best.delta_test_f1_mean:+.6f} / {d_best.delta_test_fp_mean:+.2f} / {d_best.delta_test_fn_mean:+.2f}
- OOF/test delta yönü aynı mı: {bool(d_best.oof_test_delta_same_direction)}
- Leakage FAIL/WARNING: {bool(leak.leakage_fail.any())}/{bool(leak.leakage_warning.any())}

## Config karşılaştırması

{markdown_table(summary[["config_name","test_f1_mean","test_f1_std","test_fp_mean","test_fn_mean","oof_f1_mean","dropped_feature_count_mean"]].round(6))}

## Baseline'a göre delta ve karar dili

{markdown_table(delta[["config_name","delta_test_f1_mean","delta_test_fp_mean","delta_test_fn_mean","delta_oof_f1_mean","oof_test_delta_same_direction","ablation_evidence_label"]].round(6))}

## Governance

- Threshold seçimi sadece ilgili config train OOF prediction'larından `{POLICY}` ile yapıldı.
- Testten threshold/config/feature seçimi yapılmadı.
- `Variant_ID`, `Label`, ham `AA_1`, ham `AA_2` model feature listesine sokulmadı.
- Step10 consistency kaynaklı configler hypothesis-only ve doğrudan promote edilemez.

## Yorum

- Rejected configler: {rejected}
- Hypothesis olarak kalabilecek configler: {hypotheses}
- Step 14 revalidation adayı: {step14}
- Step 12/13 için yön: preregistered AA/AL hipotezleri korunmalı; OOF/test ve FN/recall guardrail şartı devam etmeli.
"""
    OUT["report"].write_text(report, encoding="utf-8")
    metadata = {
        "project_folder": "TeknoPatho",
        "step": "Step 11B column filtering / feature cleanup ablation",
        "seed_list": SEEDS,
        "feature_set": "aa_numeric",
        "model": "CatBoostClassifier",
        "threshold_policy": POLICY,
        "configs": CONFIGS,
        "test_usage": "reporting_only",
        "test_rows_dropped": False,
        "train_rows_dropped": False,
        "feature_filtering_train_only": True,
        "model_selection_from_test": False,
        "threshold_selection_from_test": False,
        "feature_selection_from_test": False,
        "uses_step10_consistency_file": uses_step10,
        "hypothesis_only_configs": ["drop_oof_consistent_shift_top_features_conservative", "drop_EK_consistent_shift_candidates", "drop_AL_consistent_shift_candidates"],
        "near_constant_top_frequency_threshold": 0.995,
        "near_constant_numeric_variance_threshold": 1e-12,
        "baseline_prediction_source": "master_step10_row_level_predictions_with_folds.csv",
        "runtime_seconds": time.perf_counter() - wall,
    }
    OUT["metadata"].write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Step 11B final ===", flush=True)
    print(f"Leakage FAIL/WARNING: {bool(leak.leakage_fail.any())}/{bool(leak.leakage_warning.any())}", flush=True)
    print(f"Baseline F1/FP/FN: {b.test_f1_mean:.6f}/{b.test_fp_mean:.2f}/{b.test_fn_mean:.2f}", flush=True)
    print(f"Best config: {best.config_name} F1/FP/FN={best.test_f1_mean:.6f}/{best.test_fp_mean:.2f}/{best.test_fn_mean:.2f}", flush=True)
    print(f"Best delta F1/FP/FN: {d_best.delta_test_f1_mean:+.6f}/{d_best.delta_test_fp_mean:+.2f}/{d_best.delta_test_fn_mean:+.2f}; same direction={bool(d_best.oof_test_delta_same_direction)}", flush=True)
    print(f"Least FN increase: {delta[~delta.config_name.eq('baseline_current')].sort_values('delta_test_fn_mean').iloc[0].config_name} {delta[~delta.config_name.eq('baseline_current')].sort_values('delta_test_fn_mean').iloc[0].delta_test_fn_mean:+.2f}; most FP reduction: {delta[~delta.config_name.eq('baseline_current')].sort_values('delta_test_fp_mean').iloc[0].config_name} {delta[~delta.config_name.eq('baseline_current')].sort_values('delta_test_fp_mean').iloc[0].delta_test_fp_mean:+.2f}", flush=True)
    print(f"Rejected: {rejected}", flush=True)
    print(f"Remaining hypotheses: {hypotheses}", flush=True)
    print(f"Step14 candidates: {step14}", flush=True)
    print("Step12/13: preregistered AA/AL hypotheses only; retain OOF/test and FN/recall guardrails", flush=True)


if __name__ == "__main__":
    main()
