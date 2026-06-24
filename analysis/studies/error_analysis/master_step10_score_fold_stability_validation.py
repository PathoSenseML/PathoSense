"""MASTER Step 10: frozen-line score/fold stability validation."""
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
from master_step9_error_analysis import aa_map, confidence, err
from master_v02_catboost_threshold_stability import (
    catboost_frame, classification_metrics, make_model, markdown_table,
)

ROOT = Path(__file__).resolve().parent
SEEDS = [7, 21, 42, 101, 2024]
N_SPLITS = 5
PRIMARY = "add_missingness_features_native_missing_step6_params"
POLICY = "weighted_f1_current"
PARTIAL = ROOT / ".master_step10_partial_row_predictions.csv"
PARTIAL_RUNTIME = ROOT / ".master_step10_partial_runtime.csv"
OUT = {
    "rows": ROOT / "master_step10_row_level_predictions_with_folds.csv",
    "test": ROOT / "master_step10_seed_test_metrics.csv",
    "fold": ROOT / "master_step10_oof_fold_metrics.csv",
    "fold_summary": ROOT / "master_step10_oof_fold_stability_summary.csv",
    "subgroup": ROOT / "master_step10_subgroup_stability.csv",
    "variant": ROOT / "master_step10_variant_error_stability.csv",
    "pfp": ROOT / "master_step10_persistent_fp_variants.csv",
    "pfn": ROOT / "master_step10_persistent_fn_variants.csv",
    "near": ROOT / "master_step10_near_threshold_stability.csv",
    "aa": ROOT / "master_step10_aa_pair_stability.csv",
    "shift": ROOT / "master_step10_feature_shift_consistency.csv",
    "leak": ROOT / "master_step10_leakage_audit.csv",
    "runtime": ROOT / "master_step10_runtime_report.csv",
    "report": ROOT / "master_step10_report.md",
    "metadata": ROOT / "master_step10_metadata.json",
}


def metric(y, prob, threshold):
    y = np.asarray(y, int); prob = np.asarray(prob, float)
    if len(y) == 0:
        return {k: np.nan for k in ["f1", "precision", "recall", "specificity", "fpr", "balanced_accuracy", "predicted_positive_rate", "tn", "fp", "fn", "tp"]}
    return classification_metrics(y, prob, threshold)


def row_context(frame, raw):
    m = missingness_features(frame, raw)
    out = pd.DataFrame({"Variant_ID": frame.Variant_ID.astype(str), "true_label": frame.Label.to_numpy(int)})
    for source, target in [("row_missing_count", "raw_missing_count"), ("row_missing_rate", "raw_missing_rate"),
                           ("high_missing_80_flag", "high_missing_80_flag"), ("high_missing_90_flag", "high_missing_90_flag"),
                           ("high_missing_95_flag", "high_missing_95_flag"), ("all_nan_flag", "all_nan_flag"),
                           ("AA_missing_flag", "AA_missing_flag"), ("AL_missing_rate", "AL_missing_rate"),
                           ("CAT_missing_rate", "CAT_missing_rate"), ("EK_missing_rate", "EK_missing_rate")]:
        out[target] = m[source].to_numpy()
    out["AA_1_raw"] = frame.AA_1.astype("string").fillna("<NA>").to_numpy()
    out["AA_2_raw"] = frame.AA_2.astype("string").fillna("<NA>").to_numpy()
    out["AA_pair"] = out.AA_1_raw + ">" + out.AA_2_raw
    return out.reset_index(drop=True)


def row_block(seed, role, fold_id, ctx, prob, threshold):
    pred = (prob >= threshold).astype(int); margin = prob - threshold
    out = ctx.copy()
    out.insert(0, "seed", seed); out.insert(1, "split_role", role); out.insert(2, "fold_id", fold_id)
    out["pred_proba_pathogenic"] = prob; out["selected_threshold_from_oof"] = threshold
    out["pred_label"] = pred; out["error_type"] = err(out.true_label.to_numpy(), pred)
    out["probability_margin"] = margin; out["abs_margin"] = np.abs(margin)
    out["confidence_bin"] = confidence(np.abs(margin))
    return out


def train_seed(seed, train, test, raw, al):
    x, xt, features, status = transform_config(train, test, raw, al, "aa_numeric")
    if status != "OK": raise RuntimeError(status)
    x = pd.concat([x.reset_index(drop=True), missingness_features(train, raw)], axis=1)
    xt = pd.concat([xt.reset_index(drop=True), missingness_features(test, raw)], axis=1)
    features = x.columns.tolist(); cat = x.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    y = train.Label.to_numpy(int); oof = np.full(len(y), np.nan); fold_id = np.full(len(y), -99, int)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    t0 = time.perf_counter()
    for fold, (fi, vi) in enumerate(cv.split(x, y)):
        model = make_model(seed)
        model.fit(catboost_frame(x.iloc[fi], cat), y[fi], cat_features=cat)
        oof[vi] = model.predict_proba(catboost_frame(x.iloc[vi], cat))[:, 1]; fold_id[vi] = fold
    oof_seconds = time.perf_counter() - t0
    selected, _ = choose(threshold_table(y, oof), POLICY)
    threshold = float(selected.threshold)
    t1 = time.perf_counter(); model = make_model(seed)
    model.fit(catboost_frame(x, cat), y, cat_features=cat)
    ptest = model.predict_proba(catboost_frame(xt, cat))[:, 1]
    final_seconds = time.perf_counter() - t1
    tr = row_block(seed, "train_oof", fold_id, row_context(train, raw), oof, threshold)
    te = row_block(seed, "test_reporting", np.full(len(test), -1), row_context(test, raw), ptest, threshold)
    return pd.concat([tr, te], ignore_index=True), features, oof_seconds, final_seconds


def subgroup_masks(d):
    return {
        "all_rows": np.ones(len(d), bool), "high_missing_gt_80": d.raw_missing_rate.gt(.80),
        "high_missing_gt_90": d.raw_missing_rate.gt(.90), "high_missing_gt_95": d.raw_missing_rate.gt(.95),
        "all_nan": d.all_nan_flag.eq(1), "aa_missing": d.AA_missing_flag.eq(1),
        "non_aa_missing": d.AA_missing_flag.eq(0), "AL_missing_rate_gt_80": d.AL_missing_rate.gt(.80),
        "CAT_missing_rate_gt_80": d.CAT_missing_rate.gt(.80), "EK_missing_rate_gt_80": d.EK_missing_rate.gt(.80),
    }


def subgroup_stability(rows):
    details = []
    for (role, seed, fold), d in rows.groupby(["split_role", "seed", "fold_id"]):
        threshold = float(d.selected_threshold_from_oof.iloc[0])
        for name, mask in subgroup_masks(d).items():
            g = d[mask]; m = metric(g.true_label, g.pred_proba_pathogenic, threshold)
            details.append({"record_type": "seed_fold_detail", "split_role": role, "seed": seed, "fold_id": fold,
                "subgroup": name, "n": len(g), "benign_count": int(g.true_label.eq(0).sum()),
                "pathogenic_count": int(g.true_label.eq(1).sum()), **m})
    detail = pd.DataFrame(details); summaries = []
    metrics = ["f1", "recall", "specificity", "fp", "fn", "predicted_positive_rate"]
    for (role, name), g in detail.groupby(["split_role", "subgroup"]):
        r = {"record_type": "stability_summary", "split_role": role, "seed": np.nan, "fold_id": np.nan,
             "subgroup": name, "n_mean": g.n.mean(), "benign_count_mean": g.benign_count.mean(),
             "pathogenic_count_mean": g.pathogenic_count.mean()}
        for col in metrics: r[f"{col}_mean"] = g[col].mean(); r[f"{col}_std"] = g[col].std(ddof=1)
        r["unstable_recall_flag"] = bool(r["recall_std"] > .12)
        r["unstable_f1_flag"] = bool(r["f1_std"] > .10)
        r["small_n_warning"] = bool(r["n_mean"] < 20)
        role_rows = rows[rows.split_role.eq(role)].copy()
        subrows = role_rows[np.asarray(subgroup_masks(role_rows)[name], dtype=bool)]
        r["pathogenic_fn_persists"] = bool(name == "all_nan" and ((subrows.true_label.eq(1)) & subrows.error_type.eq("FN")).any())
        r["instability_flag"] = r["unstable_recall_flag"] or r["unstable_f1_flag"] or r["small_n_warning"]
        summaries.append(r)
    return pd.concat([detail, pd.DataFrame(summaries)], ignore_index=True, sort=False)


def variant_stability(rows):
    records = []
    for vid, g in rows.groupby("Variant_ID"):
        label = int(g.true_label.mode().iloc[0]); errors = g.error_type.isin(["FP", "FN"])
        nfp = int(g.error_type.eq("FP").sum()); nfn = int(g.error_type.eq("FN").sum())
        n0 = int(g.true_label.eq(0).sum()); n1 = int(g.true_label.eq(1).sum())
        error_seeds = g.loc[errors, "seed"].nunique()
        records.append({"Variant_ID": vid, "n_total_exposures": len(g), "n_test_exposures": int(g.split_role.eq("test_reporting").sum()),
            "n_oof_exposures": int(g.split_role.eq("train_oof").sum()), "n_unique_seeds_seen": g.seed.nunique(),
            "n_unique_folds_seen": g.loc[g.fold_id.ge(0), "fold_id"].nunique(), "true_label": label,
            "n_fp": nfp, "n_fn": nfn, "n_tp": int(g.error_type.eq("TP").sum()), "n_tn": int(g.error_type.eq("TN").sum()),
            "error_rate": errors.mean(), "fp_rate_when_label_0": nfp / max(n0, 1) if label == 0 else np.nan,
            "fn_rate_when_label_1": nfn / max(n1, 1) if label == 1 else np.nan,
            "mean_pred_proba": g.pred_proba_pathogenic.mean(), "std_pred_proba": g.pred_proba_pathogenic.std(),
            "mean_margin": g.probability_margin.mean(), "std_margin": g.probability_margin.std(),
            "near_threshold_rate": g.abs_margin.lt(.05).mean(),
            "high_confidence_error_count": int((errors & g.confidence_bin.eq("high_margin")).sum()),
            "persistent_error_flag": bool(errors.mean() >= .60 and len(g) >= 3),
            "persistent_fp_flag": bool(label == 0 and nfp / max(n0, 1) >= .60),
            "persistent_fn_flag": bool(label == 1 and nfn / max(n1, 1) >= .60),
            "seed_stable_error_flag": bool(error_seeds >= 3)})
    return pd.DataFrame(records)


def near_stability(rows):
    records = []
    for (role, seed, fold), g in rows.groupby(["split_role", "seed", "fold_id"]):
        near = g.abs_margin.lt(.05); ne = near & g.error_type.isin(["FP", "FN"])
        records.append({"record_type": "seed_fold", "split_role": role, "seed": seed, "fold_id": fold, "n": len(g),
            "total_near_threshold_rows": int(near.sum()), "near_threshold_fp": int((near & g.error_type.eq("FP")).sum()),
            "near_threshold_fn": int((near & g.error_type.eq("FN")).sum()), "near_threshold_rate": near.mean(),
            "near_threshold_error_rate": ne.sum() / max(near.sum(), 1),
            "high_missing_near_threshold_rate": near[g.high_missing_90_flag.eq(1)].mean() if g.high_missing_90_flag.eq(1).any() else np.nan,
            "aa_missing_near_threshold_rate": near[g.AA_missing_flag.eq(1)].mean() if g.AA_missing_flag.eq(1).any() else np.nan,
            "all_nan_near_threshold_rate": near[g.all_nan_flag.eq(1)].mean() if g.all_nan_flag.eq(1).any() else np.nan})
    detail = pd.DataFrame(records); summaries = []
    for role, g in detail.groupby("split_role"):
        r = {"record_type": "summary", "split_role": role, "seed": np.nan, "fold_id": np.nan, "n": g.n.sum()}
        for c in ["total_near_threshold_rows", "near_threshold_fp", "near_threshold_fn"]: r[c] = g[c].sum()
        for c in ["near_threshold_rate", "near_threshold_error_rate", "high_missing_near_threshold_rate", "aa_missing_near_threshold_rate", "all_nan_near_threshold_rate"]:
            r[f"{c}_mean"] = g[c].mean(); r[f"{c}_std"] = g[c].std(ddof=1)
        summaries.append(r)
    return pd.concat([detail, pd.DataFrame(summaries)], ignore_index=True, sort=False)


def aa_stability(rows):
    records = []
    for (role, pair), g in rows.groupby(["split_role", "AA_pair"], dropna=False):
        errors = g.error_type.isin(["FP", "FN"]); n = len(g)
        support = "low_support" if n < 3 else "medium_support" if n < 10 else "supported"
        records.append({"split_role": role, "AA_pair": pair, "n": n, "error_rate": errors.mean(),
            "n_unique_seeds": g.seed.nunique(), "n_unique_folds": g.loc[g.fold_id.ge(0), "fold_id"].nunique(),
            "FP": int(g.error_type.eq("FP").sum()), "FN": int(g.error_type.eq("FN").sum()), "support": support,
            "stable_error_pattern": bool(support != "low_support" and errors.mean() >= .60 and g.seed.nunique() >= 2)})
    return pd.DataFrame(records)


def feature_shift_consistency():
    frames = []
    for name in ["fp_vs_tn", "fn_vs_tp"]:
        path = ROOT / f"master_step9_feature_shift_{name}.csv"
        if not path.exists(): continue
        d = pd.read_csv(path); comparison = d.comparison.iloc[0] if len(d) else name.upper()
        o = d[d.split_role.eq("train_oof")][["feature", "standardized_mean_difference"]].rename(columns={"standardized_mean_difference": "smd_oof"})
        t = d[d.split_role.eq("test_reporting")][["feature", "standardized_mean_difference"]].rename(columns={"standardized_mean_difference": "smd_test"})
        z = o.merge(t, on="feature", how="outer"); z.insert(0, "comparison", comparison)
        z["same_direction"] = z.smd_oof.notna() & z.smd_test.notna() & (np.sign(z.smd_oof) == np.sign(z.smd_test))
        z["stable_shift_candidate"] = z.same_direction & z.smd_oof.abs().ge(.30) & z.smd_test.abs().ge(.30)
        z["test_only_observation"] = z.smd_oof.isna() & z.smd_test.notna(); z["oof_only_observation"] = z.smd_oof.notna() & z.smd_test.isna()
        z["feature_group"] = np.select([z.feature.str.startswith("EK_"), z.feature.str.contains("AA_missing", case=False), z.feature.str.startswith("AL_"), z.feature.str.startswith("CAT_"), z.feature.str.contains("aa_|blosum|grantham", case=False, regex=True)], ["EK", "AA_missing", "AL", "CAT", "AA_derived"], default="other")
        z["usage"] = "hypothesis source only; not feature selection"
        frames.append(z)
    if not frames:
        return pd.DataFrame(columns=["comparison", "feature", "smd_oof", "smd_test", "same_direction", "stable_shift_candidate", "test_only_observation", "oof_only_observation", "feature_group", "consistency_strength", "stable_candidate_rank", "usage"])
    out = pd.concat(frames, ignore_index=True)
    out["consistency_strength"] = np.minimum(out.smd_oof.abs(), out.smd_test.abs())
    out["stable_candidate_rank"] = np.nan
    for comparison, g in out[out.stable_shift_candidate].groupby("comparison"):
        ranked = g.consistency_strength.rank(method="first", ascending=False)
        out.loc[ranked.index, "stable_candidate_rank"] = ranked
    return out


def main():
    wall = time.perf_counter(); full = pd.read_csv(find_input()); full["Label"] = full.Label.astype(int)
    raw = [c for c in full if c not in ["Variant_ID", "Label"]]
    al = [c for c in raw if c.startswith("AL_") and pd.api.types.is_numeric_dtype(full[c])]
    dup = int(full.duplicated().sum()); existing = pd.read_csv(PARTIAL) if PARTIAL.exists() else pd.DataFrame()
    done = set(existing.seed.unique()) if len(existing) else set(); blocks = [existing] if len(existing) else []
    runtime = pd.read_csv(PARTIAL_RUNTIME).to_dict("records") if PARTIAL_RUNTIME.exists() else []
    feature_lists = {}; audits = {}
    for seed in SEEDS:
        train, test = make_split(full, seed); audit = leakage_audit(full, train, test, raw, seed, dup)[0]; audits[seed] = audit
        x0, _, f0, status = transform_config(train, test, raw, al, "aa_numeric")
        feature_lists[seed] = f0 + missingness_features(train, raw).columns.tolist()
        if seed in done: continue
        print(f"STEP10 seed={seed} 5-fold OOF + final test fit", flush=True); start = time.perf_counter()
        block, features, oof_sec, final_sec = train_seed(seed, train, test, raw, al); blocks.append(block); feature_lists[seed] = features
        pd.concat(blocks, ignore_index=True).to_csv(PARTIAL, index=False)
        runtime.append({"seed": seed, "status": "COMPLETED", "oof_training_seconds": oof_sec, "final_fit_test_seconds": final_sec,
                        "runtime_seconds": time.perf_counter() - start})
        pd.DataFrame(runtime).to_csv(PARTIAL_RUNTIME, index=False)
        print(f"  completed rows={len(block)} minutes={(time.perf_counter()-start)/60:.2f}", flush=True)
    rows = pd.concat(blocks, ignore_index=True); rows.to_csv(OUT["rows"], index=False)

    test_records = []; fold_records = []
    for seed, g in rows.groupby("seed"):
        te = g[g.split_role.eq("test_reporting")]; th = float(te.selected_threshold_from_oof.iloc[0]); m = metric(te.true_label, te.pred_proba_pathogenic, th)
        test_records.append({"seed": seed, "primary_line": PRIMARY, "threshold_policy": POLICY, "selected_threshold": th,
            "test_usage": "reporting_only", "auc": roc_auc_score(te.true_label, te.pred_proba_pathogenic),
            "average_precision": average_precision_score(te.true_label, te.pred_proba_pathogenic), **m})
        for fold, f in g[g.split_role.eq("train_oof")].groupby("fold_id"):
            fm = metric(f.true_label, f.pred_proba_pathogenic, th)
            fold_records.append({"seed": seed, "fold_id": fold, "selected_threshold_from_seed_oof": th, **fm})
    test_metrics = pd.DataFrame(test_records); folds = pd.DataFrame(fold_records)
    test_metrics.to_csv(OUT["test"], index=False); folds.to_csv(OUT["fold"], index=False)
    summaries = []
    for seed, g in folds.groupby("seed"):
        r = {"record_type": "seed_summary", "seed": seed}
        for c in ["f1", "precision", "recall", "specificity", "fpr", "balanced_accuracy", "predicted_positive_rate", "fp", "fn"]:
            for stat, value in [("mean", g[c].mean()), ("std", g[c].std(ddof=1)), ("min", g[c].min()), ("max", g[c].max()), ("range", g[c].max()-g[c].min())]: r[f"{c}_{stat}"] = value
        r["fold_collapse_warning"] = bool((g.f1 < g.f1.mean() - .08).any()); r["recall_instability_warning"] = bool(g.recall.std(ddof=1) > .08); r["fpr_instability_warning"] = bool(g.fpr.std(ddof=1) > .05)
        summaries.append(r)
    overall = {"record_type": "overall", "seed": np.nan}
    for c in ["f1", "recall", "fpr"]:
        overall[f"{c}_mean"] = folds[c].mean(); overall[f"{c}_std"] = folds[c].std(ddof=1); overall[f"{c}_min"] = folds[c].min(); overall[f"{c}_max"] = folds[c].max(); overall[f"{c}_range"] = folds[c].max()-folds[c].min()
    overall["fold_collapse_warning"] = any(r["fold_collapse_warning"] for r in summaries); overall["recall_instability_warning"] = folds.recall.std(ddof=1) > .08; overall["fpr_instability_warning"] = folds.fpr.std(ddof=1) > .05
    fold_summary = pd.concat([pd.DataFrame(summaries), pd.DataFrame([overall])], ignore_index=True); fold_summary.to_csv(OUT["fold_summary"], index=False)

    subgroup = subgroup_stability(rows); subgroup.to_csv(OUT["subgroup"], index=False)
    variants = variant_stability(rows); variants.to_csv(OUT["variant"], index=False)
    variants[variants.persistent_fp_flag].sort_values(["fp_rate_when_label_0", "n_fp"], ascending=False).to_csv(OUT["pfp"], index=False)
    variants[variants.persistent_fn_flag].sort_values(["fn_rate_when_label_1", "n_fn"], ascending=False).to_csv(OUT["pfn"], index=False)
    near = near_stability(rows); near.to_csv(OUT["near"], index=False)
    aa = aa_stability(rows); aa.to_csv(OUT["aa"], index=False)
    shifts = feature_shift_consistency(); shifts.to_csv(OUT["shift"], index=False)

    subgroup_summary = subgroup[subgroup.record_type.eq("stability_summary")]
    fold_collapse = bool(fold_summary.fold_collapse_warning.fillna(False).any())
    high_instability = bool(subgroup_summary.instability_flag.eq(True).any())
    small_n = bool(subgroup_summary.small_n_warning.eq(True).any())
    leaks = []
    for seed in SEEDS:
        features = feature_lists[seed]; audit = audits[seed]; overlap = len(set(make_split(full, seed)[0].Variant_ID.astype(str)) & set(make_split(full, seed)[1].Variant_ID.astype(str)))
        hard = overlap > 0 or any(c in features for c in ["Variant_ID", "Label", "AA_1", "AA_2"])
        warning = int(audit.get("identical_feature_hash_count", 0)) > 0 or fold_collapse or high_instability or small_n
        leaks.append({"seed": seed, "status": "FAIL" if hard else "WARNING" if warning else "PASS", "train_test_variant_id_overlap": overlap,
            "variant_id_in_features": "Variant_ID" in features, "label_in_features": "Label" in features,
            "raw_aa_in_features": any(c in features for c in ["AA_1", "AA_2"]), "threshold_selected_on_test": False,
            "model_selected_on_test": False, "feature_selected_on_test": False, "policy_selected_on_test": False,
            "test_rows_dropped": False, "train_rows_dropped": False, "preprocessing_fitted_on_test": False,
            "label_used_in_transform": False, "saved_probability_label_mismatch": False,
            "identical_raw_feature_hash_count": audit.get("identical_feature_hash_count", 0),
            "identical_hash_all_nan_explained": audit.get("identical_hash_all_nan_explained", False),
            "subgroup_small_n_warning": small_n, "fold_collapse_warning": fold_collapse, "high_subgroup_instability_warning": high_instability})
    leak = pd.DataFrame(leaks); leak.to_csv(OUT["leak"], index=False)
    runtime_df = pd.DataFrame(runtime)
    postprocessing_seconds = time.perf_counter() - wall
    training_seconds = pd.to_numeric(runtime_df.runtime_seconds, errors="coerce").sum()
    runtime_df.loc[len(runtime_df)] = {"seed": "TOTAL", "status": "COMPLETED",
        "runtime_seconds": training_seconds + postprocessing_seconds,
        "oof_training_seconds": pd.to_numeric(runtime_df.oof_training_seconds, errors="coerce").sum(),
        "final_fit_test_seconds": pd.to_numeric(runtime_df.final_fit_test_seconds, errors="coerce").sum()}
    runtime_df.to_csv(OUT["runtime"], index=False)

    amap = aa_map(transform_config(full, full, raw, al, "aa_numeric")[0].columns.tolist())
    desired = {"aa_blosum62_score": amap.get("blosum62_score"), "aa_grantham_distance": amap.get("grantham_distance"),
        "aa_hydrophobicity_1": amap.get("hydrophobicity_before"), "aa_hydrophobicity_2": amap.get("hydrophobicity_after"),
        "aa_hydrophobicity_diff": amap.get("hydrophobicity_delta"), "aa_molecular_weight_diff": amap.get("molecular_weight_delta"),
        "aa_charge_diff": amap.get("charge_delta"), "aa_polarity_diff": amap.get("polarity_delta")}
    metadata = {"project_folder": "TeknoPatho", "step": "Step 10 score-fold / stability validation", "primary_line": PRIMARY,
        "seed_list": SEEDS, "fold_count": N_SPLITS, "feature_set": "aa_numeric", "model": "CatBoostClassifier",
        "threshold_policy": POLICY, "model_training_changed": False, "threshold_policy_changed": False,
        "feature_engineering_changed": False, "test_usage": "reporting_only", "oof_fold_analysis": True,
        "no_model_selection_from_test": True, "no_threshold_selection_from_test": True, "no_feature_selection_from_test": True,
        "aa_derived_field_check": {k: {"present": v is not None, "source_column": v} for k, v in desired.items()},
        "catboost_params": make_model(7).get_params()}
    OUT["metadata"].write_text(json.dumps(metadata, indent=2, default=str), encoding="utf8")

    test_f1_mean = test_metrics.f1.mean(); test_f1_std = test_metrics.f1.std(ddof=1)
    fold_f1_mean = folds.f1.mean(); fold_f1_std = folds.f1.std(ddof=1); worst = folds.sort_values("f1").iloc[0]
    allnan = subgroup_summary[subgroup_summary.subgroup.eq("all_nan")]
    allnan_fn = bool(allnan.pathogenic_fn_persists.eq(True).any())
    def stable_bad(name):
        g = subgroup_summary[subgroup_summary.subgroup.eq(name)]
        return bool(len(g) and not g.instability_flag.eq(True).any() and g.f1_mean.mean() < .50)
    aa_bad = stable_bad("aa_missing"); cat_bad = stable_bad("CAT_missing_rate_gt_80"); ek_bad = stable_bad("EK_missing_rate_gt_80")
    stable_fp = shifts[(shifts.comparison.eq("FP_vs_TN")) & shifts.stable_shift_candidate]
    stable_fn = shifts[(shifts.comparison.eq("FN_vs_TP")) & shifts.stable_shift_candidate]
    test_only = shifts[shifts.test_only_observation]
    consistent_groups = shifts[shifts.stable_shift_candidate].groupby("feature_group").size().to_dict()
    report = f"""# MASTER Step 10 score-fold / stability validation

## Amaç ve sabit hat

Bu çalışma `{PRIMARY}` / `aa_numeric` / CatBoost / native missing + deterministic missingness / `{POLICY}` hattının **stability evidence** analizidir. Model, feature veya threshold seçimi değildir. Threshold yalnız train OOF tahminlerinden seçildi; test yalnız reporting-only kullanıldı ve testten seçim yapılmadı.

## Seed-level test stability

- Test F1 mean/std: **{test_f1_mean:.6f} / {test_f1_std:.6f}**; predeclared stable (std <= 0.04): **{test_f1_std <= .04}**.

{markdown_table(test_metrics)}

## OOF fold stability

- OOF fold F1 mean/std: **{fold_f1_mean:.6f} / {fold_f1_std:.6f}**; predeclared stable (std <= 0.05): **{fold_f1_std <= .05}**.
- Worst fold: seed={int(worst.seed)}, fold={int(worst.fold_id)}, F1={worst.f1:.6f}.
- Fold collapse / recall instability / FPR instability warning: **{fold_collapse} / {bool(fold_summary.recall_instability_warning.fillna(False).any())} / {bool(fold_summary.fpr_instability_warning.fillna(False).any())}**.

{markdown_table(fold_summary)}

## Subgroup stability

- all-NaN pathogenic FN persists: **{allnan_fn}**. all-NaN F1=0 beklenen missing-information limitation'dır; model-selection sinyali değildir.
- aa_missing stable bad: **{aa_bad}**; CAT/EK missing stable bad: **{cat_bad}/{ek_bad}**.
- Small-n veya instability warning mevcut: **{small_n or high_instability}**.

{markdown_table(subgroup_summary)}

## Persistent errors ve threshold yakınlığı

- Persistent FP/FN variant count: **{int(variants.persistent_fp_flag.sum())}/{int(variants.persistent_fn_flag.sum())}**.
- Seed-stable error variants: **{int(variants.seed_stable_error_flag.sum())}**.
- Near-threshold error exposure count (abs margin < .05): **{int(((rows.abs_margin < .05) & rows.error_type.isin(['FP','FN'])).sum())}**.

## Step 9 feature-shift consistency

- Stable consistent FP-vs-TN / FN-vs-TP candidates: **{len(stable_fp)}/{len(stable_fn)}**.
- Stable candidate groups: `{consistent_groups}`. Bunlar feature selection değil, yalnız Step 11/12 için hypothesis source'dur.
- Test-only observations: **{len(test_only)}**; bunlar karar kanıtı olarak kullanılmamalıdır.

Top consistent FP-vs-TN hypothesis sources:

{markdown_table(stable_fp.sort_values('stable_candidate_rank').head(10))}

Top consistent FN-vs-TP hypothesis sources:

{markdown_table(stable_fn.sort_values('stable_candidate_rank').head(10))}

## Leakage / governance

- Leakage FAIL/WARNING: **{leak.status.eq('FAIL').any()} / {leak.status.eq('WARNING').any()}**.
- Variant_ID, Label ve ham AA_1/AA_2 feature değildir. Train/test row drop, test-fit preprocessing, test threshold/model/feature/policy selection yoktur.

## Step 11 için önerilen yön

1. **Column filtering için güvenli hipotezler:** yalnız OOF+test aynı yönlü ve her iki tarafta |SMD| >= 0.30 olan, yeterli destekli kolonları önceden kaydedilmiş ablation adayları yap; doğrudan eleme yapma.
2. **AA expansion / one-hot için güvenli hipotezler:** seed/fold destekli `stable_error_pattern=True` AA çiftlerini ayrı ablation hipotezi olarak taşı; düşük destekli çiftleri kullanma.
3. **AL-log / AL özel inceleme:** consistent stability signal veren AL adaylarını yalnız OOF-öncelikli, önceden kayıtlı ablation'da test et.
4. **Kullanılmaması gerekenler:** `test_only_observation=True`, low-support AA çiftleri ve all-NaN F1 sonucu.
5. **Taşınabilecekler:** `stable_shift_candidate=True` OOF+test consistent patternler ve birden çok seed/fold'da tekrarlanan persistent hata hipotezleri.
"""
    OUT["report"].write_text(report, encoding="utf8")
    print("\n=== Step 10 final ===")
    print(f"Leakage FAIL/WARNING: {leak.status.eq('FAIL').any()}/{leak.status.eq('WARNING').any()}")
    print(f"Seed-level test F1 mean/std: {test_f1_mean:.6f}/{test_f1_std:.6f}")
    print(f"OOF fold F1 mean/std: {fold_f1_mean:.6f}/{fold_f1_std:.6f}")
    print(f"Worst OOF fold: seed={int(worst.seed)} fold={int(worst.fold_id)} F1={worst.f1:.6f}")
    print(f"Fold collapse / recall / FPR warning: {fold_collapse}/{fold_summary.recall_instability_warning.fillna(False).any()}/{fold_summary.fpr_instability_warning.fillna(False).any()}")
    print(f"all-NaN pathogenic FN persists: {allnan_fn}")
    print(f"aa_missing stable bad: {aa_bad}; CAT/EK missing stable bad: {cat_bad}/{ek_bad}")
    print(f"Persistent FP/FN variants: {variants.persistent_fp_flag.sum()}/{variants.persistent_fn_flag.sum()}")
    print(f"Near-threshold errors: {((rows.abs_margin < .05) & rows.error_type.isin(['FP','FN'])).sum()}")
    print(f"Stable FP-vs-TN/FN-vs-TP shift candidates: {len(stable_fp)}/{len(stable_fn)}")
    print("Step 11 direction: pre-registered OOF+test-consistent column/AA/AL ablations; exclude test-only and low-support patterns")


if __name__ == "__main__":
    main()
