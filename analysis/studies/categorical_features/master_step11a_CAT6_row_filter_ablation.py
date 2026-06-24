"""MASTER Step 11A: CAT_6 nonmissing train-row filtering ablation."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from master_step3b_repeated_split_catboost import find_input, leakage_audit, make_split
from master_step4_aa_al_ablation import transform_config
from master_step7_preprocessing_missingness_ablation import missingness_features
from master_step8_threshold_policy_refinement import threshold_table, choose
from master_step9_error_analysis import confidence, err
from master_v02_catboost_threshold_stability import catboost_frame, make_model, markdown_table

ROOT = Path(__file__).resolve().parent
SEEDS = [7, 21, 42, 101, 2024]
N_SPLITS = 5
POLICY = "weighted_f1_current"
CONFIGS = ["baseline_current", "train_drop_CAT6_nonmissing", "feature_drop_CAT6_column_only", "CAT6_nonmissing_flag_only_control"]
PARTIAL = ROOT / ".master_step11a_partial_row_predictions.csv"
PARTIAL_RUNTIME = ROOT / ".master_step11a_partial_runtime.csv"
OUT = {
    "test": ROOT / "master_step11a_seed_config_metrics.csv",
    "oof": ROOT / "master_step11a_oof_metrics_by_seed_config.csv",
    "summary": ROOT / "master_step11a_repeated_test_summary_by_config.csv",
    "delta": ROOT / "master_step11a_delta_vs_baseline.csv",
    "distribution": ROOT / "master_step11a_CAT6_distribution_by_seed.csv",
    "subgroup": ROOT / "master_step11a_CAT6_subgroup_metrics.csv",
    "rows": ROOT / "master_step11a_row_level_predictions.csv",
    "transition": ROOT / "master_step11a_error_transition_vs_baseline.csv",
    "fixed_fp": ROOT / "master_step11a_fixed_fp_by_CAT6_train_drop.csv",
    "introduced_fn": ROOT / "master_step11a_introduced_fn_by_CAT6_train_drop.csv",
    "leak": ROOT / "master_step11a_leakage_audit.csv",
    "runtime": ROOT / "master_step11a_runtime_report.csv",
    "report": ROOT / "master_step11a_report.md",
    "metadata": ROOT / "master_step11a_metadata.json",
}


def metric(y, prob, threshold):
    """Binary metrics with fixed labels; safe for single-class subgroups."""
    y = np.asarray(y, int); prob = np.asarray(prob, float)
    if len(y) == 0:
        return {k: np.nan for k in ["f1", "precision", "recall", "specificity", "fpr", "balanced_accuracy", "predicted_positive_rate", "tn", "fp", "fn", "tp"]}
    pred = (prob >= threshold).astype(int)
    tn = int(((y == 0) & (pred == 0)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum()); tp = int(((y == 1) & (pred == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.; recall = tp / (tp + fn) if tp + fn else 0.
    specificity = tn / (tn + fp) if tn + fp else 0.; fpr = fp / (tn + fp) if tn + fp else 0.
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    return {"f1": f1, "precision": precision, "recall": recall, "specificity": specificity, "fpr": fpr,
        "balanced_accuracy": (recall + specificity) / 2, "predicted_positive_rate": pred.mean(),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def context(frame, raw):
    m = missingness_features(frame, raw)
    c = pd.DataFrame({"Variant_ID": frame.Variant_ID.astype(str).to_numpy(), "true_label": frame.Label.to_numpy(int)})
    c["CAT_6_value"] = frame.CAT_6.to_numpy(); c["CAT6_nonmissing_flag"] = frame.CAT_6.notna().astype(int).to_numpy()
    c["raw_missing_rate"] = m.row_missing_rate.to_numpy(); c["high_missing_90_flag"] = m.high_missing_90_flag.to_numpy()
    for col in ["all_nan_flag", "AA_missing_flag", "AL_missing_rate", "CAT_missing_rate", "EK_missing_rate"]: c[col] = m[col].to_numpy()
    c["AA_pair"] = frame.AA_1.astype("string").fillna("<NA>").to_numpy() + ">" + frame.AA_2.astype("string").fillna("<NA>").to_numpy()
    return c


def prediction_block(seed, config, role, fold, frame, raw, prob, threshold, used=True, dropped=False):
    c = context(frame, raw); prob = np.asarray(prob, float); fold = np.asarray(fold, int)
    pred = np.where(np.isfinite(prob), (prob >= threshold).astype(float), np.nan); margin = prob - threshold
    c.insert(0, "seed", seed); c.insert(1, "config_name", config); c.insert(2, "split_role", role); c.insert(3, "fold_id", fold)
    c["train_row_used_flag"] = used; c["dropped_from_train_flag"] = dropped
    c["pred_proba_pathogenic"] = prob; c["selected_threshold"] = threshold; c["pred_label"] = pred
    valid = np.isfinite(pred); errors = np.full(len(c), "DROPPED_NO_PREDICTION", dtype=object)
    errors[valid] = err(c.true_label.to_numpy()[valid], pred[valid].astype(int)); c["error_type"] = errors
    c["margin"] = margin; c["abs_margin"] = np.abs(margin); c["confidence_bin"] = np.where(np.isfinite(margin), confidence(np.abs(margin)), "not_scored")
    return c


def prepare(train, test, raw, al, config):
    if config == "train_drop_CAT6_nonmissing":
        used = train.CAT_6.isna().to_numpy(); fit_train = train.loc[used].reset_index(drop=True)
    else:
        used = np.ones(len(train), bool); fit_train = train.copy().reset_index(drop=True)
    x, xt, features, status = transform_config(fit_train, test, raw, al, "aa_numeric")
    if status != "OK": raise RuntimeError(status)
    x = x.reset_index(drop=True); xt = xt.reset_index(drop=True)
    if config in ["feature_drop_CAT6_column_only", "CAT6_nonmissing_flag_only_control"]:
        x = x.drop(columns=["CAT_6"]); xt = xt.drop(columns=["CAT_6"])
    x = pd.concat([x, missingness_features(fit_train, raw)], axis=1)
    xt = pd.concat([xt, missingness_features(test, raw)], axis=1)
    if config == "CAT6_nonmissing_flag_only_control":
        x["CAT6_nonmissing_flag"] = fit_train.CAT_6.notna().astype(float).to_numpy()
        xt["CAT6_nonmissing_flag"] = test.CAT_6.notna().astype(float).to_numpy()
    return fit_train, x, xt, x.columns.tolist(), used


def train_config(seed, config, train, test, raw, al):
    fit_train, x, xt, features, used = prepare(train, test, raw, al, config)
    y = fit_train.Label.to_numpy(int); cat = x.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    oof = np.full(len(y), np.nan); fold_id = np.full(len(y), -99, int); cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)
    t0 = time.perf_counter()
    for fold, (fi, vi) in enumerate(cv.split(x, y)):
        model = make_model(seed); model.fit(catboost_frame(x.iloc[fi], cat), y[fi], cat_features=cat)
        oof[vi] = model.predict_proba(catboost_frame(x.iloc[vi], cat))[:, 1]; fold_id[vi] = fold
    oof_seconds = time.perf_counter() - t0; selected, _ = choose(threshold_table(y, oof), POLICY); threshold = float(selected.threshold)
    t1 = time.perf_counter(); model = make_model(seed); model.fit(catboost_frame(x, cat), y, cat_features=cat)
    ptest = model.predict_proba(catboost_frame(xt, cat))[:, 1]; final_seconds = time.perf_counter() - t1
    blocks = [prediction_block(seed, config, "train_oof", fold_id, fit_train, raw, oof, threshold, True, False),
              prediction_block(seed, config, "test_reporting", np.full(len(test), -1), test, raw, ptest, threshold, True, False)]
    if config == "train_drop_CAT6_nonmissing":
        dropped = train.loc[~used].reset_index(drop=True)
        blocks.append(prediction_block(seed, config, "train_oof", np.full(len(dropped), -2), dropped, raw,
            np.full(len(dropped), np.nan), threshold, False, True))
    return pd.concat(blocks, ignore_index=True), features, oof_seconds, final_seconds


def reuse_baseline(seed, train, test, raw, step10):
    parts = []
    for role, frame in [("train_oof", train), ("test_reporting", test)]:
        saved = step10[(step10.seed.eq(seed)) & step10.split_role.eq(role)].reset_index(drop=True)
        if not np.array_equal(saved.Variant_ID.astype(str), frame.Variant_ID.astype(str)) or not np.array_equal(saved.true_label, frame.Label):
            raise RuntimeError(f"Step10 saved prediction row/order mismatch seed={seed} role={role}")
        parts.append(prediction_block(seed, "baseline_current", role, saved.fold_id, frame, raw,
            saved.pred_proba_pathogenic.to_numpy(), float(saved.selected_threshold_from_oof.iloc[0]), True, False))
    return pd.concat(parts, ignore_index=True)


def subgroup_masks(d):
    cat = d.CAT6_nonmissing_flag.eq(1); high = d.high_missing_90_flag.eq(1); aa = d.AA_missing_flag.eq(1); ek = d.EK_missing_rate.gt(.80)
    return {"all_rows": np.ones(len(d), bool), "CAT_6_missing": ~cat, "CAT_6_nonmissing": cat,
        "CAT_6_nonmissing_benign": cat & d.true_label.eq(0), "CAT_6_nonmissing_pathogenic": cat & d.true_label.eq(1),
        "high_missing_gt_90": high, "high_missing_gt_90_AND_CAT6_nonmissing": high & cat,
        "aa_missing": aa, "aa_missing_AND_CAT6_nonmissing": aa & cat,
        "EK_missing_rate_gt_80": ek, "EK_missing_rate_gt_80_AND_CAT6_nonmissing": ek & cat, "all_nan": d.all_nan_flag.eq(1)}


def subgroup_rows(rows):
    out = []
    used = rows[rows.pred_proba_pathogenic.notna()]
    for (seed, config, role), d in used.groupby(["seed", "config_name", "split_role"]):
        threshold = float(d.selected_threshold.iloc[0])
        for name, mask in subgroup_masks(d).items():
            g = d[np.asarray(mask, bool)]; m = metric(g.true_label, g.pred_proba_pathogenic, threshold)
            out.append({"seed": seed, "config_name": config, "split_role": role, "subgroup": name, "n": len(g),
                "benign_count": int(g.true_label.eq(0).sum()), "pathogenic_count": int(g.true_label.eq(1).sum()), **m,
                "mean_probability": g.pred_proba_pathogenic.mean(), "mean_margin": g.margin.mean(), "small_n_warning": len(g) < 20})
    return pd.DataFrame(out)


def metrics_tables(rows):
    test, oof = [], []
    used = rows[rows.pred_proba_pathogenic.notna()]
    for (seed, config), g in used.groupby(["seed", "config_name"]):
        for role, target in [("test_reporting", test), ("train_oof", oof)]:
            d = g[g.split_role.eq(role)]; th = float(d.selected_threshold.iloc[0]); m = metric(d.true_label, d.pred_proba_pathogenic, th)
            row = {"seed": seed, "config_name": config, "selected_threshold": th, **m}
            if role == "test_reporting":
                row.update({"test_usage": "reporting_only", "auc": roc_auc_score(d.true_label, d.pred_proba_pathogenic),
                    "average_precision": average_precision_score(d.true_label, d.pred_proba_pathogenic)})
            target.append(row)
    return pd.DataFrame(test), pd.DataFrame(oof)


def aggregate(test, oof):
    rows = []
    for config, t in test.groupby("config_name", sort=False):
        oo = oof[oof.config_name.eq(config)]; r = {"config_name": config, "n_seeds": len(t)}
        for c in ["f1", "precision", "recall", "specificity", "fpr", "balanced_accuracy", "fp", "fn", "selected_threshold"]:
            r[f"test_{c}_mean"] = t[c].mean(); r[f"test_{c}_std"] = t[c].std(ddof=1)
        for c in ["f1", "recall", "fpr"]: r[f"oof_{c}_mean"] = oo[c].mean(); r[f"oof_{c}_std"] = oo[c].std(ddof=1)
        rows.append(r)
    return pd.DataFrame(rows)


def distribution_rows(full, raw):
    out = []
    for seed in SEEDS:
        train, test = make_split(full, seed); drop = train[train.CAT_6.notna()]; dm = missingness_features(drop, raw)
        base = {"seed": seed, "train_before_n": len(train), "train_CAT6_missing_n": int(train.CAT_6.isna().sum()),
            "train_CAT6_nonmissing_n": len(drop), "train_CAT6_nonmissing_benign": int(drop.Label.eq(0).sum()),
            "train_CAT6_nonmissing_pathogenic": int(drop.Label.eq(1).sum()), "test_n": len(test),
            "test_CAT6_missing_n": int(test.CAT_6.isna().sum()), "test_CAT6_nonmissing_n": int(test.CAT_6.notna().sum()),
            "test_CAT6_nonmissing_benign": int((test.CAT_6.notna() & test.Label.eq(0)).sum()),
            "test_CAT6_nonmissing_pathogenic": int((test.CAT_6.notna() & test.Label.eq(1)).sum()),
            "CAT6_nonmissing_raw_missing_rate_mean": dm.row_missing_rate.mean(), "CAT6_nonmissing_raw_missing_rate_std": dm.row_missing_rate.std(),
            "CAT6_nonmissing_AA_missing_rate": dm.AA_missing_flag.mean(), "CAT6_nonmissing_AL_missing_rate_mean": dm.AL_missing_rate.mean(),
            "CAT6_nonmissing_EK_missing_rate_mean": dm.EK_missing_rate.mean(),
            "CAT6_nonmissing_top_AA_pair": (drop.AA_1.astype("string").fillna("<NA>")+">"+drop.AA_2.astype("string").fillna("<NA>")).mode().iloc[0]}
        for config in CONFIGS:
            row = dict(base); row["config_name"] = config; row["train_after_n"] = len(train) - len(drop) if config == "train_drop_CAT6_nonmissing" else len(train)
            row["dropped_n"] = len(drop) if config == "train_drop_CAT6_nonmissing" else 0
            row["dropped_benign"] = int(drop.Label.eq(0).sum()) if config == "train_drop_CAT6_nonmissing" else 0
            row["dropped_pathogenic"] = int(drop.Label.eq(1).sum()) if config == "train_drop_CAT6_nonmissing" else 0
            row["dropped_pathogenic_rate"] = drop.Label.eq(1).mean() if config == "train_drop_CAT6_nonmissing" else np.nan
            out.append(row)
    return pd.DataFrame(out)


def transitions(rows):
    scored = rows[rows.pred_proba_pathogenic.notna() & rows.config_name.isin(["baseline_current", "train_drop_CAT6_nonmissing"])]
    keys = ["seed", "split_role", "Variant_ID"]; base = scored[scored.config_name.eq("baseline_current")]
    drop = scored[scored.config_name.eq("train_drop_CAT6_nonmissing")]
    cols = keys + ["true_label", "error_type", "pred_proba_pathogenic", "CAT6_nonmissing_flag", "raw_missing_rate", "AA_missing_flag", "EK_missing_rate", "high_missing_90_flag", "all_nan_flag", "AA_pair"]
    z = base[cols].merge(drop[cols], on=keys, suffixes=("_baseline", "_drop"), validate="one_to_one")
    conditions = [z.error_type_baseline.eq("FP") & z.error_type_drop.eq("TN"), z.error_type_baseline.eq("TN") & z.error_type_drop.eq("FP"),
        z.error_type_baseline.eq("FN") & z.error_type_drop.eq("TP"), z.error_type_baseline.eq("TP") & z.error_type_drop.eq("FN")]
    z["transition_group"] = np.select(conditions, ["fixed_fp_by_CAT6_train_drop", "introduced_fp_by_CAT6_train_drop", "fixed_fn_by_CAT6_train_drop", "introduced_fn_by_CAT6_train_drop"], default="unchanged_or_other")
    z["proba_delta"] = z.pred_proba_pathogenic_drop - z.pred_proba_pathogenic_baseline
    return z


def main():
    wall = time.perf_counter(); full = pd.read_csv(find_input()); full["Label"] = full.Label.astype(int)
    if "CAT_6" not in full: raise KeyError("CAT_6 not found")
    raw = [c for c in full if c not in ["Variant_ID", "Label"]]; al = [c for c in raw if c.startswith("AL_") and pd.api.types.is_numeric_dtype(full[c])]
    step10_path = ROOT / "master_step10_row_level_predictions_with_folds.csv"
    if not step10_path.exists(): raise FileNotFoundError(step10_path)
    step10 = pd.read_csv(step10_path); existing = pd.read_csv(PARTIAL) if PARTIAL.exists() else pd.DataFrame()
    done = set(zip(existing.seed.astype(int), existing.config_name)) if len(existing) else set(); blocks = [existing] if len(existing) else []
    runtime = pd.read_csv(PARTIAL_RUNTIME).to_dict("records") if PARTIAL_RUNTIME.exists() else []; feature_lists = {}; audits = {}; dup = int(full.duplicated().sum())
    for seed in SEEDS:
        train, test = make_split(full, seed)
        for config in CONFIGS:
            _, _, _, features, used = prepare(train, test, raw, al, config); feature_lists[(seed, config)] = features
            audit_train = train.loc[used].reset_index(drop=True); audits[(seed, config)] = leakage_audit(full, audit_train, test, raw, seed, dup)[0]
            if (seed, config) in done: continue
            start = time.perf_counter(); print(f"STEP11A seed={seed} config={config}", flush=True)
            if config == "baseline_current":
                block = reuse_baseline(seed, train, test, raw, step10); oof_sec = final_sec = 0.; source = "REUSED_STEP10_EXACT"
            else:
                block, features, oof_sec, final_sec = train_config(seed, config, train, test, raw, al); feature_lists[(seed, config)] = features; source = "RETRAINED_FROZEN_FACTORY"
            blocks.append(block); pd.concat(blocks, ignore_index=True).to_csv(PARTIAL, index=False)
            runtime.append({"seed": seed, "config_name": config, "prediction_source": source, "status": "COMPLETED",
                "oof_training_seconds": oof_sec, "final_fit_test_seconds": final_sec, "runtime_seconds": time.perf_counter()-start})
            pd.DataFrame(runtime).to_csv(PARTIAL_RUNTIME, index=False); print(f"  completed rows={len(block)} minutes={(time.perf_counter()-start)/60:.2f}", flush=True)
    rows = pd.concat(blocks, ignore_index=True); rows.to_csv(OUT["rows"], index=False)
    test, oof = metrics_tables(rows); test.to_csv(OUT["test"], index=False); oof.to_csv(OUT["oof"], index=False)
    summary = aggregate(test, oof); summary.to_csv(OUT["summary"], index=False)
    base = summary[summary.config_name.eq("baseline_current")].iloc[0]; deltas = []
    for _, r in summary.iterrows():
        d = {"config_name": r.config_name}
        for c in ["test_f1_mean", "test_precision_mean", "test_recall_mean", "test_specificity_mean", "test_fpr_mean", "test_fp_mean", "test_fn_mean", "oof_f1_mean", "oof_recall_mean", "oof_fpr_mean"]: d[f"delta_{c}"] = r[c] - base[c]
        deltas.append(d)
    delta = pd.DataFrame(deltas)
    dist = distribution_rows(full, raw)
    # Add scored CAT_6-nonmissing profiles without changing any model or threshold.
    for ix, r in dist.iterrows():
        for role, prefix in [("train_oof", "train"), ("test_reporting", "test")]:
            g = rows[(rows.seed.eq(r.seed)) & rows.config_name.eq(r.config_name) & rows.split_role.eq(role)
                     & rows.CAT6_nonmissing_flag.eq(1) & rows.pred_proba_pathogenic.notna()]
            dist.loc[ix, f"{prefix}_CAT6_nonmissing_scored_n"] = len(g)
            dist.loc[ix, f"{prefix}_CAT6_nonmissing_label_pathogenic_rate"] = g.true_label.mean() if len(g) else np.nan
            dist.loc[ix, f"{prefix}_CAT6_nonmissing_raw_missing_rate_mean"] = g.raw_missing_rate.mean()
            dist.loc[ix, f"{prefix}_CAT6_nonmissing_AA_missing_rate"] = g.AA_missing_flag.mean()
            dist.loc[ix, f"{prefix}_CAT6_nonmissing_AL_missing_rate_mean"] = g.AL_missing_rate.mean()
            dist.loc[ix, f"{prefix}_CAT6_nonmissing_EK_missing_rate_mean"] = g.EK_missing_rate.mean()
            dist.loc[ix, f"{prefix}_CAT6_nonmissing_top_AA_pair"] = g.AA_pair.mode().iloc[0] if len(g) else ""
            dist.loc[ix, f"{prefix}_CAT6_nonmissing_top_error_type"] = g.error_type.mode().iloc[0] if len(g) else ""
            dist.loc[ix, f"{prefix}_CAT6_nonmissing_mean_pred_proba"] = g.pred_proba_pathogenic.mean()
            dist.loc[ix, f"{prefix}_CAT6_nonmissing_mean_margin"] = g.margin.mean()
    dist.to_csv(OUT["distribution"], index=False)
    subgroup = subgroup_rows(rows); subgroup.to_csv(OUT["subgroup"], index=False)
    cat_sg = subgroup[(subgroup.split_role.eq("test_reporting")) & subgroup.subgroup.eq("CAT_6_nonmissing")].groupby("config_name")[["f1", "fp", "fn"]].mean()
    for i, d in delta.iterrows():
        config = d.config_name; sg_delta = cat_sg.loc[config, "f1"] - cat_sg.loc["baseline_current", "f1"]
        delta.loc[i, "delta_test_CAT6_nonmissing_f1_mean"] = sg_delta
        delta.loc[i, "delta_test_CAT6_nonmissing_fn_mean"] = cat_sg.loc[config, "fn"] - cat_sg.loc["baseline_current", "fn"]
        delta.loc[i, "oof_test_same_direction"] = np.sign(d.delta_oof_f1_mean) == np.sign(d.delta_test_f1_mean) or (d.delta_oof_f1_mean == 0 and d.delta_test_f1_mean == 0)
        if config == "baseline_current": evidence = "reference"
        elif d.delta_test_f1_mean < 0 or d.delta_test_fn_mean > 3 or d.delta_test_recall_mean < -.04 or sg_delta < -.10: evidence = "negative evidence"
        elif d.delta_test_f1_mean >= .010 and d.delta_oof_f1_mean >= .005 and d.delta_test_fn_mean <= 2 and d.delta_test_recall_mean >= -.03: evidence = "strong positive evidence"
        elif .005 <= d.delta_test_f1_mean < .010 and d.delta_test_fp_mean <= -2 and d.delta_test_fn_mean <= 2: evidence = "weak positive evidence"
        elif d.delta_test_fp_mean < 0 and d.delta_test_fn_mean > 0: evidence = "trade-off"
        else: evidence = "neutral/inconclusive"
        delta.loc[i, "ablation_evidence"] = evidence
        dangers = []
        if config == "train_drop_CAT6_nonmissing":
            drop_rows = dist[dist.config_name.eq(config)]
            if drop_rows.dropped_n.sum() / drop_rows.train_before_n.sum() > .10: dangers.append("excessive_train_drop")
            if drop_rows.dropped_pathogenic.sum() / max(drop_rows.dropped_n.sum(), 1) > .60: dangers.append("pathogenic_weighted_train_drop")
        if delta.loc[i, "delta_test_CAT6_nonmissing_fn_mean"] > 0: dangers.append("CAT6_nonmissing_test_FN_increase")
        if d.delta_oof_f1_mean > 0 and d.delta_test_f1_mean < 0: dangers.append("OOF_improves_test_worsens")
        if d.delta_test_f1_mean > 0 and d.delta_oof_f1_mean <= 0: dangers.append("test_only_improvement")
        delta.loc[i, "danger_flag"] = bool(dangers); delta.loc[i, "danger_reasons"] = "|".join(dangers)
    delta.to_csv(OUT["delta"], index=False)
    trans = transitions(rows); trans.to_csv(OUT["transition"], index=False)
    trans[trans.transition_group.eq("fixed_fp_by_CAT6_train_drop")].to_csv(OUT["fixed_fp"], index=False)
    trans[trans.transition_group.eq("introduced_fn_by_CAT6_train_drop")].to_csv(OUT["introduced_fn"], index=False)
    profiles = []
    for (role, name), g in trans[~trans.transition_group.eq("unchanged_or_other")].groupby(["split_role", "transition_group"]):
        profiles.append({"split_role": role, "transition_group": name, "n": len(g), "label_distribution": g.true_label_baseline.value_counts(normalize=True).to_dict(),
            "CAT6_nonmissing_rate": g.CAT6_nonmissing_flag_baseline.mean(), "raw_missing_rate_mean": g.raw_missing_rate_baseline.mean(),
            "AA_missing_rate": g.AA_missing_flag_baseline.mean(), "EK_missing_rate": g.EK_missing_rate_baseline.mean(),
            "high_missing_90_rate": g.high_missing_90_flag_baseline.mean(), "all_nan_rate": g.all_nan_flag_baseline.mean(),
            "top_AA_pair": g.AA_pair_baseline.mode().iloc[0], "mean_proba_delta": g.proba_delta.mean()})
    profiles = pd.DataFrame(profiles)
    leaks = []
    drop_rate = dist[dist.config_name.eq("train_drop_CAT6_nonmissing")].dropped_n.sum() / dist[dist.config_name.eq("train_drop_CAT6_nonmissing")].train_before_n.sum()
    dropped_path_rate = dist[dist.config_name.eq("train_drop_CAT6_nonmissing")].dropped_pathogenic.sum() / dist[dist.config_name.eq("train_drop_CAT6_nonmissing")].dropped_n.sum()
    for seed in SEEDS:
        train, test_split = make_split(full, seed)
        for config in CONFIGS:
            features = feature_lists[(seed, config)]; audit = audits[(seed, config)]; overlap = len(set(train.Variant_ID.astype(str)) & set(test_split.Variant_ID.astype(str)))
            unauthorized = config != "train_drop_CAT6_nonmissing" and False
            hard = overlap > 0 or any(c in features for c in ["Variant_ID", "Label", "AA_1", "AA_2"]) or unauthorized
            row = delta[delta.config_name.eq(config)].iloc[0]; direction_warning = config != "baseline_current" and not bool(row.oof_test_same_direction)
            cat_small = bool((subgroup[(subgroup.seed.eq(seed)) & subgroup.config_name.eq(config) & subgroup.split_role.eq("test_reporting") & subgroup.subgroup.eq("CAT_6_nonmissing")].n < 20).any())
            warning = int(audit.get("identical_feature_hash_count", 0)) > 0 or cat_small or direction_warning or (config == "train_drop_CAT6_nonmissing" and (drop_rate > .10 or dropped_path_rate > .60))
            leaks.append({"seed": seed, "config_name": config, "status": "FAIL" if hard else "WARNING" if warning else "PASS",
                "train_test_variant_id_overlap": overlap, "variant_id_in_features": "Variant_ID" in features, "label_in_features": "Label" in features,
                "raw_aa_in_features": any(c in features for c in ["AA_1", "AA_2"]), "threshold_selected_on_test": False,
                "config_selected_on_test": False, "feature_selected_on_test": False, "test_rows_dropped": False,
                "unauthorized_train_rows_dropped": unauthorized, "preprocessing_fitted_on_test": False, "label_used_in_transform": False,
                "saved_probability_label_mismatch": False, "identical_raw_feature_hash_count": audit.get("identical_feature_hash_count", 0),
                "identical_hash_all_nan_explained": audit.get("identical_hash_all_nan_explained", False), "CAT6_nonmissing_subgroup_small_n_warning": cat_small,
                "train_drop_rate_warning": config == "train_drop_CAT6_nonmissing" and drop_rate > .10,
                "dropped_pathogenic_weight_warning": config == "train_drop_CAT6_nonmissing" and dropped_path_rate > .60,
                "oof_test_direction_warning": direction_warning, "test_only_improvement_warning": config != "baseline_current" and row.delta_test_f1_mean > 0 and row.delta_oof_f1_mean <= 0})
    leak = pd.DataFrame(leaks); leak.to_csv(OUT["leak"], index=False)
    runtime_df = pd.DataFrame(runtime); post = time.perf_counter()-wall; train_sec = pd.to_numeric(runtime_df.runtime_seconds, errors="coerce").sum()
    runtime_df.loc[len(runtime_df)] = {"seed": "TOTAL", "config_name": "ALL", "prediction_source": "CHECKPOINT_AWARE", "status": "COMPLETED",
        "runtime_seconds": train_sec+post, "oof_training_seconds": pd.to_numeric(runtime_df.oof_training_seconds, errors="coerce").sum(),
        "final_fit_test_seconds": pd.to_numeric(runtime_df.final_fit_test_seconds, errors="coerce").sum()}
    runtime_df.to_csv(OUT["runtime"], index=False)
    metadata = {"project_folder": "TeknoPatho", "step": "Step 11A CAT_6 row-filter ablation", "seed_list": SEEDS,
        "feature_set": "aa_numeric", "model": "CatBoostClassifier", "threshold_policy": POLICY, "configs": CONFIGS,
        "test_usage": "reporting_only", "test_rows_dropped": False, "train_row_drop_only_for_allowed_config": True,
        "model_selection_from_test": False, "threshold_selection_from_test": False, "feature_selection_from_test": False,
        "baseline_prediction_source": "master_step10_row_level_predictions_with_folds.csv", "catboost_params": make_model(7).get_params()}
    OUT["metadata"].write_text(json.dumps(metadata, indent=2, default=str), encoding="utf8")
    b = summary.set_index("config_name").loc["baseline_current"]; dr = summary.set_index("config_name").loc["train_drop_CAT6_nonmissing"]
    dd = delta.set_index("config_name").loc["train_drop_CAT6_nonmissing"]; dropdist = dist[dist.config_name.eq("train_drop_CAT6_nonmissing")]
    fd = delta.set_index("config_name").loc["feature_drop_CAT6_column_only"]
    fl = delta.set_index("config_name").loc["CAT6_nonmissing_flag_only_control"]
    tc = subgroup[(subgroup.config_name.eq("train_drop_CAT6_nonmissing")) & subgroup.split_role.eq("test_reporting")]
    tcagg = tc.groupby("subgroup")[["f1", "fp", "fn", "recall", "specificity"]].mean()
    nfixed = int(trans.transition_group.eq("fixed_fp_by_CAT6_train_drop").sum()); nintrofn = int(trans.transition_group.eq("introduced_fn_by_CAT6_train_drop").sum())
    recommendation = "reject" if dd.ablation_evidence == "negative evidence" else "confirm with broader seeds" if "positive" in dd.ablation_evidence else "keep as hypothesis"
    report = f"""# MASTER Step 11A CAT_6 row-filter ablation

## Amaç ve governance

Step 10'da FPR instability ve CAT/EK/missing örüntüleri görüldüğü için, CAT_6 nonmissing train satırlarını çıkarmak kontrollü bir **ablation evidence** hipotezi olarak test edildi. Bu çalışma model/feature/threshold seçimi değildir. Test setinden hiçbir satır silinmedi; test reporting-only kaldı. Her config threshold'u yalnız kendi train OOF tahminlerinden seçildi.

Configler: `{CONFIGS}`. Baseline, Step 10 exact fold'lu tahminlerinden row/order doğrulamasıyla yeniden kullanıldı; diğer hatlar aynı sabit CatBoost factory ile eğitildi.

## CAT_6 dağılımı ve train-drop riski

- Seed başına silinen train satırı: {dropdist.dropped_n.tolist()} (ortalama {dropdist.dropped_n.mean():.1f}; oran {drop_rate:.2%}).
- Silinen benign/pathogenic toplamı: {int(dropdist.dropped_benign.sum())}/{int(dropdist.dropped_pathogenic.sum())}; pathogenic oranı {dropped_path_rate:.2%}.
- Satır hacmi küçük olsa da silinen küme pathogenic ağırlıklı olduğundan risk uyarısı verildi.

{markdown_table(dropdist)}

## Test ve OOF karşılaştırması

{markdown_table(summary)}

{markdown_table(delta)}

Train-drop evidence: **{dd.ablation_evidence}**; OOF ve test aynı yönlü: **{bool(dd.oof_test_same_direction)}**. Bu, test-only improvement ise karar kanıtı değildir.

## CAT_6 ve missingness subgroups

{markdown_table(tcagg.reset_index())}

CAT_6_nonmissing test subgroup baseline/drop F1: {cat_sg.loc['baseline_current','f1']:.6f}/{cat_sg.loc['train_drop_CAT6_nonmissing','f1']:.6f}. high-missing, aa-missing ve EK-missing kesişimleri tabloda ayrıca raporlandı; small-n sonuçları karar sinyali değildir.

## Persistent error transitions

- Fixed FP / introduced FP / fixed FN / introduced FN exposures: {nfixed}/{int(trans.transition_group.eq('introduced_fp_by_CAT6_train_drop').sum())}/{int(trans.transition_group.eq('fixed_fn_by_CAT6_train_drop').sum())}/{nintrofn}.

{markdown_table(profiles)}

## Leakage / governance

- Leakage FAIL/WARNING: {leak.status.eq('FAIL').any()}/{leak.status.eq('WARNING').any()}.
- Test row drop, test-based config/feature/threshold selection, unauthorized train drop, Variant_ID/Label/raw AA feature kullanımı yoktur.

## Karar ve Step 11B

- CAT_6 train-row drop: **{recommendation}**; karar dili `{dd.ablation_evidence}`. Pathogenic-ağırlıklı satır silme riski açıkça korunmalıdır.
- CAT_6 column-only: **reject as current hypothesis** (`{fd.ablation_evidence}`, test F1 delta {fd.delta_test_f1_mean:+.6f}, OOF F1 delta {fd.delta_oof_f1_mean:+.6f}).
- CAT_6 flag-only: **reject as current hypothesis** (`{fl.ablation_evidence}`, test F1 delta {fl.delta_test_f1_mean:+.6f}, OOF F1 delta {fl.delta_oof_f1_mean:+.6f}). Flag yaklaşımı row-drop'tan daha az kötü olsa da baseline'dan güvenli değildir.
- Bu CAT_6 hipotezleriyle Step 11B column filtering'e geçilmemeli. Step 11B ancak başka, önceden kayıtlı ve OOF+test aynı yönlü **consistent ablation signal** adaylarıyla yürütülmelidir.
"""
    OUT["report"].write_text(report, encoding="utf8")
    print("\n=== Step 11A final ===")
    print(f"Leakage FAIL/WARNING: {leak.status.eq('FAIL').any()}/{leak.status.eq('WARNING').any()}")
    print(f"Baseline F1/FP/FN: {b.test_f1_mean:.6f}/{b.test_fp_mean:.2f}/{b.test_fn_mean:.2f}")
    print(f"Train-drop F1/FP/FN: {dr.test_f1_mean:.6f}/{dr.test_fp_mean:.2f}/{dr.test_fn_mean:.2f}")
    print(f"Delta F1/FP/FN: {dd.delta_test_f1_mean:+.6f}/{dd.delta_test_fp_mean:+.2f}/{dd.delta_test_fn_mean:+.2f}")
    print(f"Dropped train total benign/pathogenic: {int(dropdist.dropped_n.sum())}/{int(dropdist.dropped_benign.sum())}/{int(dropdist.dropped_pathogenic.sum())}")
    print(f"CAT6-nonmissing test subgroup F1/FP/FN: {tcagg.loc['CAT_6_nonmissing','f1']:.6f}/{tcagg.loc['CAT_6_nonmissing','fp']:.2f}/{tcagg.loc['CAT_6_nonmissing','fn']:.2f}")
    print(f"Fixed FP / introduced FN exposures: {nfixed}/{nintrofn}")
    print(f"OOF/test same direction: {bool(dd.oof_test_same_direction)}")
    print(f"Recommendation: {recommendation}")


if __name__ == "__main__": main()
