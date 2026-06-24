"""MASTER Step 12: preregistered amino-acid biochemical feature ablation.

Test is reporting-only and is not used for threshold, feature, or model selection.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from master_step3b_repeated_split_catboost import find_input, make_split
from master_step4_aa_al_ablation import AA_ORDER, BLOSUM62, GRANTHAM, transform_config
from master_step7_preprocessing_missingness_ablation import missingness_features
from master_step8_threshold_policy_refinement import choose, threshold_table
from master_step9_error_analysis import confidence, err
from master_v02_catboost_threshold_stability import catboost_frame, make_model, markdown_table

ROOT = Path(__file__).resolve().parent
SEEDS = [7, 21, 42, 101, 2024]
SMOKE_SEEDS = [7]
CONFIGS = [
    "baseline_current",
    "aa_biochem_conservative",
    "aa_biochem_full_numeric",
    "aa_biochem_full_numeric_plus_pair_cat",
]
SMOKE_CONFIGS = ["baseline_current", "aa_biochem_conservative"]
N_SPLITS = 5
POLICY = "weighted_f1_current"
PARTIAL_ROWS = ROOT / ".master_step12_partial_row_predictions.csv"
PARTIAL_RUNTIME = ROOT / ".master_step12_partial_runtime.csv"
OUT = {
    "report": ROOT / "master_step12_report.md",
    "metadata": ROOT / "master_step12_metadata.json",
    "leak": ROOT / "master_step12_leakage_audit.csv",
    "oof": ROOT / "master_step12_oof_metrics_by_seed_config.csv",
    "test": ROOT / "master_step12_test_metrics_by_seed_config.csv",
    "summary": ROOT / "master_step12_repeated_test_summary_by_config.csv",
    "delta": ROOT / "master_step12_delta_vs_baseline.csv",
    "importance": ROOT / "master_step12_aa_feature_importance_diagnostic.csv",
    "pair_subgroup": ROOT / "master_step12_aa_pair_subgroup_metrics.csv",
    "rows": ROOT / "master_step12_row_level_predictions.csv",
    "runtime": ROOT / "master_step12_runtime_report.csv",
}
RUNTIME_ESTIMATE = {
    "expected_catboost_fit_count_theoretical": 120,
    "expected_catboost_fit_count_with_baseline_reuse": 90,
    "expected_fold_count_theoretical": 100,
    "expected_new_fold_count_with_baseline_reuse": 75,
    "smoke_minutes": {"optimistic": 3, "expected": 6, "pessimistic": 12},
    "seed7_full_minutes": {"optimistic": 10, "expected": 20, "pessimistic": 35},
    "full_grid_minutes": {"optimistic": 55, "expected": 105, "pessimistic": 190},
    "staged_execution": True,
}

AA = list(AA_ORDER)
VALID_AA = set(AA)
HYDRO = dict(zip(AA, [1.8, -4.5, -3.5, -3.5, 2.5, -3.5, -3.5, -0.4, -3.2, 4.5, 3.8, -3.9, 1.9, 2.8, -1.6, -0.8, -0.7, -0.9, -1.3, 4.2]))
MW = dict(zip(AA, [89.09, 174.20, 132.12, 133.10, 121.16, 146.15, 147.13, 75.07, 155.16, 131.17, 131.17, 146.19, 149.21, 165.19, 115.13, 105.09, 119.12, 204.23, 181.19, 117.15]))
POLAR = dict(zip(AA, [8.1, 10.5, 11.6, 13.0, 5.5, 10.5, 12.3, 9.0, 10.4, 5.2, 4.9, 11.3, 5.7, 5.2, 8.0, 9.2, 8.6, 5.4, 6.2, 5.9]))
CHARGE = {"A": 0, "R": 1, "N": 0, "D": -1, "C": 0, "Q": 0, "E": -1, "G": 0, "H": 1, "I": 0, "L": 0, "K": 1, "M": 0, "F": 0, "P": 0, "S": 0, "T": 0, "W": 0, "Y": 0, "V": 0}
AROMATIC = set("FWY")
POLAR_SET = set("RNDCEQHKSTY")
NONPOLAR_SET = set("AGILMFPWV")


def norm_aa(s):
    x = s.astype("string").str.strip().str.upper()
    return x.where(x.isin(VALID_AA))


def percentile_map(scores):
    vals = np.array(list(scores.values()), dtype=float)
    ranks = {v: float((vals <= v).mean()) for v in sorted(set(vals))}
    return {k: ranks[v] for k, v in scores.items()}


BLOSUM_PERCENTILE = percentile_map(BLOSUM62)


def aa_biochem_features(frame: pd.DataFrame, include_full: bool, include_pair_cat: bool) -> pd.DataFrame:
    aa1_raw = frame["AA_1"].astype("string")
    aa2_raw = frame["AA_2"].astype("string")
    aa1 = norm_aa(frame["AA_1"])
    aa2 = norm_aa(frame["AA_2"])
    missing_pair = aa1_raw.isna() | aa2_raw.isna() | aa1_raw.str.strip().eq("") | aa2_raw.str.strip().eq("")
    valid = aa1.notna() & aa2.notna()
    invalid = (~missing_pair) & (~valid)
    keys = pd.Series(list(zip(aa1, aa2)), index=frame.index)
    gd = keys.map(GRANTHAM).where(valid).astype(float)
    bl = keys.map(BLOSUM62).where(valid).astype(float)
    out = pd.DataFrame(index=frame.index)
    out["step12_aa_valid_pair_flag"] = valid.astype(float)
    out["step12_aa_missing_pair_flag"] = missing_pair.astype(float)
    out["step12_aa_same_flag"] = (aa1 == aa2).astype("Float64").where(valid).astype(float)
    out["step12_aa_invalid_pair_flag"] = invalid.astype(float)
    out["step12_grantham_distance"] = gd
    out["step12_grantham_scaled"] = gd / 215.0
    out["step12_blosum62_score"] = bl
    out["step12_blosum62_percentile_rarity"] = keys.map(BLOSUM_PERCENTILE).where(valid).astype(float)
    if include_full:
        h1, h2 = aa1.map(HYDRO).astype(float), aa2.map(HYDRO).astype(float)
        c1, c2 = aa1.map(CHARGE).astype(float), aa2.map(CHARGE).astype(float)
        p1, p2 = aa1.map(POLAR).astype(float), aa2.map(POLAR).astype(float)
        m1, m2 = aa1.map(MW).astype(float), aa2.map(MW).astype(float)
        out["step12_hydrophobicity_ref"] = h1
        out["step12_hydrophobicity_alt"] = h2
        out["step12_hydrophobicity_delta_kd"] = (h2 - h1).where(valid)
        out["step12_hydrophobicity_abs_delta"] = (h2 - h1).abs().where(valid)
        out["step12_charge_ref"] = c1
        out["step12_charge_alt"] = c2
        out["step12_charge_delta"] = (c2 - c1).where(valid)
        out["step12_charge_reversal_flag"] = ((c1 * c2) < 0).astype(float).where(valid)
        out["step12_is_charge_gain"] = ((c1 == 0) & (c2 != 0)).astype(float).where(valid)
        out["step12_is_charge_loss"] = ((c1 != 0) & (c2 == 0)).astype(float).where(valid)
        out["step12_polarity_ref"] = p1
        out["step12_polarity_alt"] = p2
        out["step12_polarity_delta"] = (p2 - p1).where(valid)
        out["step12_is_polar_to_nonpolar"] = (aa1.isin(POLAR_SET) & aa2.isin(NONPOLAR_SET)).astype(float).where(valid)
        out["step12_is_nonpolar_to_polar"] = (aa1.isin(NONPOLAR_SET) & aa2.isin(POLAR_SET)).astype(float).where(valid)
        out["step12_molecular_weight_ref"] = m1
        out["step12_molecular_weight_alt"] = m2
        out["step12_molecular_weight_delta"] = (m2 - m1).where(valid)
        out["step12_molecular_weight_abs_delta"] = (m2 - m1).abs().where(valid)
        for residue, name in [("P", "proline"), ("G", "glycine"), ("C", "cysteine")]:
            out[f"step12_is_{name}_gain"] = ((aa1 != residue) & (aa2 == residue)).astype(float).where(valid)
            out[f"step12_is_{name}_loss"] = ((aa1 == residue) & (aa2 != residue)).astype(float).where(valid)
        out["step12_aromaticity_ref"] = aa1.isin(AROMATIC).astype(float).where(aa1.notna())
        out["step12_aromaticity_alt"] = aa2.isin(AROMATIC).astype(float).where(aa2.notna())
        out["step12_aromaticity_change"] = (aa1.isin(AROMATIC) != aa2.isin(AROMATIC)).astype(float).where(valid)
    if include_pair_cat:
        out["step12_aa_pair_string"] = (aa1.fillna("<INVALID>") + ">" + aa2.fillna("<INVALID>")).astype("string")
    return out.reset_index(drop=True)


def metric(y, prob, threshold):
    y = np.asarray(y, int); prob = np.asarray(prob, float)
    pred = (prob >= threshold).astype(int)
    tn = int(((y == 0) & (pred == 0)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum()); tp = int(((y == 1) & (pred == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    fpr = fp / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"f1": f1, "precision": precision, "recall": recall, "specificity": specificity, "fpr": fpr,
            "balanced_accuracy": (recall + specificity) / 2, "mcc": matthews_corrcoef(y, pred),
            "predicted_positive_rate": float(pred.mean()), "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def prepare(train, test, raw, al, config):
    x, xt, _, status = transform_config(train, test, raw, al, "aa_numeric")
    if status != "OK":
        raise RuntimeError(status)
    x = pd.concat([x.reset_index(drop=True), missingness_features(train, raw)], axis=1)
    xt = pd.concat([xt.reset_index(drop=True), missingness_features(test, raw)], axis=1)
    for c in ["Variant_ID", "Label", "AA_1", "AA_2"]:
        if c in x:
            x = x.drop(columns=[c])
        if c in xt:
            xt = xt.drop(columns=[c])
    if config != "baseline_current":
        include_full = config in {"aa_biochem_full_numeric", "aa_biochem_full_numeric_plus_pair_cat"}
        include_pair = config == "aa_biochem_full_numeric_plus_pair_cat"
        x = pd.concat([x, aa_biochem_features(train, include_full, include_pair)], axis=1)
        xt = pd.concat([xt, aa_biochem_features(test, include_full, include_pair)], axis=1)
    return x, xt


def row_context(frame, raw):
    bio = aa_biochem_features(frame, True, True)
    m = missingness_features(frame, raw)
    out = pd.DataFrame({"Variant_ID": frame.Variant_ID.astype(str).to_numpy(), "true_label": frame.Label.to_numpy(int)})
    out["AA_pair"] = bio["step12_aa_pair_string"].astype(str).to_numpy()
    for c in ["step12_grantham_distance", "step12_blosum62_score", "step12_charge_reversal_flag",
              "step12_is_proline_gain", "step12_is_proline_loss", "step12_is_glycine_gain",
              "step12_is_glycine_loss", "step12_is_cysteine_gain", "step12_is_cysteine_loss"]:
        out[c.replace("step12_", "")] = bio[c].to_numpy()
    out["raw_missing_rate"] = m.row_missing_rate.to_numpy()
    out["AA_missing_flag"] = m.AA_missing_flag.to_numpy()
    return out


def row_block(seed, config, role, fold, frame, raw, prob, threshold):
    out = row_context(frame, raw)
    pred = (prob >= threshold).astype(int); margin = prob - threshold
    out.insert(0, "seed", seed); out.insert(1, "config_name", config); out.insert(2, "split_role", role); out.insert(3, "fold_id", fold)
    out["pred_proba_pathogenic"] = prob; out["selected_threshold"] = threshold; out["pred_label"] = pred
    out["error_type"] = err(out.true_label.to_numpy(), pred)
    out["margin"] = margin; out["abs_margin"] = np.abs(margin); out["confidence_bin"] = confidence(np.abs(margin))
    return out


def reuse_baseline(seed, train, test, raw):
    path = ROOT / "master_step10_row_level_predictions_with_folds.csv"
    if not path.exists():
        return None
    d = pd.read_csv(path)
    parts = []
    for role, frame in [("train_oof", train), ("test_reporting", test)]:
        s = d[(d.seed.eq(seed)) & d.split_role.eq(role)].reset_index(drop=True)
        if len(s) != len(frame) or not np.array_equal(s.Variant_ID.astype(str), frame.Variant_ID.astype(str)):
            return None
        parts.append(row_block(seed, "baseline_current", role, s.fold_id.to_numpy(), frame, raw,
                               s.pred_proba_pathogenic.to_numpy(), float(s.selected_threshold_from_oof.iloc[0])))
    return pd.concat(parts, ignore_index=True)


def train_one(seed, config, train, test, raw, al):
    if config == "baseline_current":
        reused = reuse_baseline(seed, train, test, raw)
        if reused is not None:
            x, _ = prepare(train, test, raw, al, config)
            return reused, x.columns.tolist(), [], 0.0, 0.0, "REUSED_STEP10_EXACT"
    x, xt = prepare(train, test, raw, al, config)
    features = x.columns.tolist()
    cat = x.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    y = train.Label.to_numpy(int)
    oof = np.full(len(y), np.nan); fold_id = np.full(len(y), -99, int)
    cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)
    t0 = time.perf_counter()
    for fold, (fi, vi) in enumerate(cv.split(x, y)):
        model = make_model(seed)
        model.fit(catboost_frame(x.iloc[fi], cat), y[fi], cat_features=cat)
        oof[vi] = model.predict_proba(catboost_frame(x.iloc[vi], cat))[:, 1]
        fold_id[vi] = fold
    oof_sec = time.perf_counter() - t0
    selected, _ = choose(threshold_table(y, oof), POLICY)
    threshold = float(selected.threshold)
    t1 = time.perf_counter()
    model = make_model(seed)
    model.fit(catboost_frame(x, cat), y, cat_features=cat)
    ptest = model.predict_proba(catboost_frame(xt, cat))[:, 1]
    final_sec = time.perf_counter() - t1
    imp = []
    try:
        vals = model.get_feature_importance()
        for f, v in zip(features, vals):
            if f.startswith("step12_"):
                imp.append({"seed": seed, "config_name": config, "feature": f, "importance": float(v)})
    except Exception:
        pass
    rows = pd.concat([row_block(seed, config, "train_oof", fold_id, train, raw, oof, threshold),
                      row_block(seed, config, "test_reporting", np.full(len(test), -1), test, raw, ptest, threshold)], ignore_index=True)
    return rows, features, imp, oof_sec, final_sec, "RETRAINED_FROZEN_FACTORY"


def metrics_tables(rows):
    test, oof = [], []
    for (seed, config, role), d in rows.groupby(["seed", "config_name", "split_role"]):
        th = float(d.selected_threshold.iloc[0])
        m = metric(d.true_label, d.pred_proba_pathogenic, th)
        rec = {"seed": seed, "config_name": config, "selected_threshold": th, **m,
               "roc_auc": roc_auc_score(d.true_label, d.pred_proba_pathogenic),
               "average_precision": average_precision_score(d.true_label, d.pred_proba_pathogenic)}
        (test if role == "test_reporting" else oof).append(rec)
    return pd.DataFrame(oof), pd.DataFrame(test)


def aggregate(oof, test):
    rows = []
    for config, t in test.groupby("config_name", sort=False):
        oo = oof[oof.config_name.eq(config)]
        r = {"config_name": config, "completed_seed_count": int(t.seed.nunique())}
        for prefix, df in [("test", t), ("oof", oo)]:
            for c in ["f1", "precision", "recall", "specificity", "fpr", "balanced_accuracy", "mcc", "roc_auc", "average_precision", "fp", "fn", "predicted_positive_rate"]:
                r[f"{prefix}_{c}_mean"] = df[c].mean()
                r[f"{prefix}_{c}_std"] = df[c].std(ddof=1)
        rows.append(r)
    return pd.DataFrame(rows)


def delta(summary):
    b = summary[summary.config_name.eq("baseline_current")].iloc[0]
    out = summary.copy()
    for c in ["test_f1_mean", "test_precision_mean", "test_recall_mean", "test_specificity_mean", "test_fpr_mean", "test_mcc_mean", "test_fp_mean", "test_fn_mean", "oof_f1_mean", "oof_mcc_mean", "oof_fp_mean", "oof_fn_mean"]:
        out["delta_" + c] = out[c] - b[c]
    out["oof_test_f1_delta_same_direction"] = np.sign(out.delta_test_f1_mean) == np.sign(out.delta_oof_f1_mean)
    out.loc[out.config_name.eq("baseline_current"), "oof_test_f1_delta_same_direction"] = True
    labels = []
    for _, r in out.iterrows():
        if r.config_name == "baseline_current":
            labels.append("reference")
        elif (r.delta_test_f1_mean >= .02 and r.delta_oof_f1_mean >= -.005 and r.delta_test_fp_mean <= 2 and r.delta_test_fn_mean <= 3):
            labels.append("accept_candidate")
        elif r.delta_test_f1_mean > 0 and (r.delta_oof_f1_mean < -.005 or r.delta_test_fp_mean > 0):
            labels.append("exploratory_only")
        else:
            labels.append("reject")
    out["promotion_label"] = labels
    return out


def pair_subgroups(rows):
    out = []
    for (config, role), d in rows.groupby(["config_name", "split_role"]):
        th = float(d.selected_threshold.iloc[0])
        bins_g = pd.cut(d.grantham_distance, [-.1, 0, 50, 100, 150, np.inf], labels=["0", "1-50", "51-100", "101-150", ">150"])
        bins_b = pd.cut(d.blosum62_score, [-np.inf, -3, -1, 0, 2, np.inf], labels=["<=-3", "-2 to -1", "0", "1 to 2", ">=3"])
        masks = {"charge_reversal": d.charge_reversal_flag.eq(1), "no_charge_reversal": d.charge_reversal_flag.eq(0),
                 "proline_gain_loss": d.is_proline_gain.eq(1) | d.is_proline_loss.eq(1),
                 "glycine_gain_loss": d.is_glycine_gain.eq(1) | d.is_glycine_loss.eq(1),
                 "cysteine_gain_loss": d.is_cysteine_gain.eq(1) | d.is_cysteine_loss.eq(1)}
        for label in ["0", "1-50", "51-100", "101-150", ">150"]:
            masks[f"grantham_{label}"] = bins_g.astype(str).eq(label)
        for label in ["<=-3", "-2 to -1", "0", "1 to 2", ">=3"]:
            masks[f"blosum_{label}"] = bins_b.astype(str).eq(label)
        common = d.AA_pair.value_counts().head(10).index.tolist()
        fp_pairs = d[d.error_type.eq("FP")].AA_pair.value_counts().head(10).index.tolist()
        fn_pairs = d[d.error_type.eq("FN")].AA_pair.value_counts().head(10).index.tolist()
        for p in common:
            masks[f"common_pair_{p}"] = d.AA_pair.eq(p)
        for p in fp_pairs:
            masks[f"top_fp_pair_{p}"] = d.AA_pair.eq(p)
        for p in fn_pairs:
            masks[f"top_fn_pair_{p}"] = d.AA_pair.eq(p)
        for name, mask in masks.items():
            g = d[np.asarray(mask, bool)]
            if len(g) == 0:
                continue
            out.append({"config_name": config, "split_role": role, "subgroup": name, "n": len(g),
                        "benign_count": int(g.true_label.eq(0).sum()), "pathogenic_count": int(g.true_label.eq(1).sum()),
                        **metric(g.true_label, g.pred_proba_pathogenic, th)})
    return pd.DataFrame(out)


def audit_rows(full, seeds, configs):
    raw = [c for c in full.columns if c not in ["Variant_ID", "Label"]]
    al = [c for c in raw if c.startswith("AL_") and pd.api.types.is_numeric_dtype(full[c])]
    out = []
    for seed in seeds:
        train, test = make_split(full, seed)
        for config in configs:
            x, xt = prepare(train, test, raw, al, config)
            features = x.columns.tolist()
            train_ids = set(train.Variant_ID.astype(str)); test_ids = set(test.Variant_ID.astype(str))
            forbidden = [c for c in features if c in {"Variant_ID", "Label", "AA_1", "AA_2"}]
            fail = bool((train_ids & test_ids) or forbidden)
            shared = set(pd.util.hash_pandas_object(x, index=False).tolist()) & set(pd.util.hash_pandas_object(xt, index=False).tolist())
            out.append({"seed": seed, "config_name": config, "variant_id_overlap_count": len(train_ids & test_ids),
                        "forbidden_feature_columns": json.dumps(forbidden), "identical_feature_hash_count": len(shared),
                        "threshold_selected_on_test": False, "feature_selected_on_test": False, "model_selected_on_test": False,
                        "test_usage": "reporting_only", "test_rows_dropped": False, "train_rows_dropped": False,
                        "leakage_fail": fail, "leakage_warning": bool(shared), "status": "FAIL" if fail else ("WARNING" if shared else "PASS")})
    return pd.DataFrame(out)


def run(mode):
    wall = time.perf_counter()
    seeds = SMOKE_SEEDS if mode == "smoke" else SEEDS
    configs = SMOKE_CONFIGS if mode == "smoke" else CONFIGS
    full = pd.read_csv(find_input()); full["Label"] = full.Label.astype(int)
    raw = [c for c in full.columns if c not in ["Variant_ID", "Label"]]
    al = [c for c in raw if c.startswith("AL_") and pd.api.types.is_numeric_dtype(full[c])]
    rows_list, imp, runtime = [], [], []
    for seed in seeds:
        train, test = make_split(full, seed)
        for config in configs:
            print(f"STEP12 mode={mode} seed={seed} config={config}", flush=True)
            t0 = time.perf_counter()
            rows, features, imps, oof_sec, final_sec, source = train_one(seed, config, train, test, raw, al)
            rows_list.append(rows); imp.extend(imps)
            runtime.append({"mode": mode, "seed": seed, "config_name": config, "status": "completed", "prediction_source": source,
                            "runtime_seconds": time.perf_counter() - t0, "oof_seconds": oof_sec, "final_fit_seconds": final_sec,
                            "feature_count": len(features)})
            print(f"  completed source={source} minutes={(time.perf_counter()-t0)/60:.2f}", flush=True)
    rows = pd.concat(rows_list, ignore_index=True)
    oof, test = metrics_tables(rows)
    summary = aggregate(oof, test)
    dt = delta(summary)
    audit = audit_rows(full, seeds, configs)
    pair = pair_subgroups(rows)
    importance = pd.DataFrame(imp)
    runtime_df = pd.DataFrame(runtime)
    if mode == "full":
        rows.to_csv(OUT["rows"], index=False)
        oof.to_csv(OUT["oof"], index=False)
        test.to_csv(OUT["test"], index=False)
        summary.to_csv(OUT["summary"], index=False)
        dt.to_csv(OUT["delta"], index=False)
        audit.to_csv(OUT["leak"], index=False)
        pair.to_csv(OUT["pair_subgroup"], index=False)
        importance.to_csv(OUT["importance"], index=False)
        runtime_df.to_csv(OUT["runtime"], index=False)
        b = summary[summary.config_name.eq("baseline_current")].iloc[0]
        best = summary[~summary.config_name.eq("baseline_current")].sort_values("test_f1_mean", ascending=False).iloc[0]
        bestd = dt[dt.config_name.eq(best.config_name)].iloc[0]
        if bestd.promotion_label == "accept_candidate" and best.config_name != "aa_biochem_full_numeric_plus_pair_cat" and not audit.leakage_fail.any():
            recommendation = "accept"
        elif bestd.promotion_label == "exploratory_only" or best.config_name == "aa_biochem_full_numeric_plus_pair_cat":
            recommendation = "keep exploratory"
        else:
            recommendation = "reject"
        report = f"""# MASTER Step 12 — AA biochemical feature ablation

Step11B was rejected as negative/neutral evidence.

Step12 evaluates preregistered amino-acid biochemical hypotheses.

Test data remained reporting-only and was not used for threshold, feature, or model selection.

## Runtime estimate

{json.dumps(RUNTIME_ESTIMATE, indent=2)}

## Runtime observed

- Total runtime: {(time.perf_counter() - wall)/60:.2f} minutes
- CatBoost run source counts: {runtime_df.prediction_source.value_counts().to_dict()}

## Step12 results

{markdown_table(summary.round(6))}

## Delta vs baseline

{markdown_table(dt[["config_name","delta_test_f1_mean","delta_test_fp_mean","delta_test_fn_mean","delta_oof_f1_mean","oof_test_f1_delta_same_direction","promotion_label"]].round(6))}

## FP/FN tradeoff

- Baseline F1 / FP / FN: {b.test_f1_mean:.6f} / {b.test_fp_mean:.2f} / {b.test_fn_mean:.2f}
- Best non-baseline: `{best.config_name}` F1 / FP / FN = {best.test_f1_mean:.6f} / {best.test_fp_mean:.2f} / {best.test_fn_mean:.2f}
- Delta F1 / FP / FN = {bestd.delta_test_f1_mean:+.6f} / {bestd.delta_test_fp_mean:+.2f} / {bestd.delta_test_fn_mean:+.2f}

## Leakage audit

- FAIL: {bool(audit.leakage_fail.any())}
- WARNING: {bool(audit.leakage_warning.any())}

## Final recommendation

Recommendation: {recommendation}
"""
        OUT["report"].write_text(report, encoding="utf-8")
        meta = {"project_folder": "TeknoPatho", "step": "Step12 AA biochemical feature engineering",
                "runtime_estimate": RUNTIME_ESTIMATE, "runtime_observed_seconds": time.perf_counter() - wall,
                "seed_list": SEEDS, "configs": CONFIGS, "threshold_policy": POLICY,
                "test_usage": "reporting_only", "test_was_used_for_selection": False,
                "final_recommendation": recommendation, "exploratory_only_config": "aa_biochem_full_numeric_plus_pair_cat"}
        OUT["metadata"].write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=== Step12", mode, "complete ===")
    print(summary[["config_name", "test_f1_mean", "test_fp_mean", "test_fn_mean", "oof_f1_mean"]].to_string(index=False))
    print("leakage_fail", bool(audit.leakage_fail.any()), "leakage_warning", bool(audit.leakage_warning.any()))
    return summary, audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    run(args.mode)
