"""MASTER Step 12B: AA biochemical replacement ablation.

Tests whether preregistered biochemical AA features should replace the
existing aa_numeric feature block instead of being added on top.

Governance: test is reporting-only and is not used for threshold, feature, or
model selection. Threshold is selected only from train OOF predictions.
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
from master_step4_aa_al_ablation import transform_config
from master_step7_preprocessing_missingness_ablation import missingness_features
from master_step8_threshold_policy_refinement import choose, threshold_table
from master_step9_error_analysis import confidence, err
from master_step12_aa_biochemical_features import aa_biochem_features
from master_v02_catboost_threshold_stability import catboost_frame, make_model, markdown_table

ROOT = Path(__file__).resolve().parent
SEEDS = [7, 21, 42, 101, 2024]
SMOKE_SEEDS = [7]
CONFIGS = [
    "baseline_current",
    "remove_aa_numeric_only",
    "replace_aa_numeric_with_biochem_conservative",
    "replace_aa_numeric_with_biochem_full_numeric",
]
SMOKE_CONFIGS = ["baseline_current", "replace_aa_numeric_with_biochem_conservative"]
N_SPLITS = 5
POLICY = "weighted_f1_current"
OUT = {
    "report": ROOT / "master_step12b_report.md",
    "metadata": ROOT / "master_step12b_metadata.json",
    "leak": ROOT / "master_step12b_leakage_audit.csv",
    "oof": ROOT / "master_step12b_oof_metrics_by_seed_config.csv",
    "test": ROOT / "master_step12b_test_metrics_by_seed_config.csv",
    "summary": ROOT / "master_step12b_repeated_test_summary_by_config.csv",
    "delta": ROOT / "master_step12b_delta_vs_baseline.csv",
    "runtime": ROOT / "master_step12b_runtime_report.csv",
}
RUNTIME_ESTIMATE = {
    "expected_catboost_fit_count_theoretical": 120,
    "expected_catboost_fit_count_with_baseline_reuse": 90,
    "expected_fold_count_theoretical": 100,
    "expected_new_fold_count_with_baseline_reuse": 75,
    "smoke_minutes": {"optimistic": 3, "expected": 5, "pessimistic": 9},
    "seed7_full_minutes": {"optimistic": 7, "expected": 14, "pessimistic": 25},
    "full_grid_minutes": {"optimistic": 35, "expected": 70, "pessimistic": 110},
    "expected_shorter_than_step12": True,
}

AA_DERIVED_TOKENS = [
    "aa1_valid", "aa2_valid", "aa_pair_valid", "aa_same", "aa_blosum", "aa_grantham",
    "aa_hydrophobicity", "aa_volume", "aa_molecular_weight", "aa_polarity", "aa_charge",
    "aa_aromatic", "aa_sulfur", "aa_proline", "aa_glycine", "aa_group", "physicochemical",
    "blosum", "grantham", "hydrophobicity", "molecular_weight", "polarity", "charge_changed",
    "aromatic_changed", "sulfur_changed", "proline_changed", "glycine_changed",
]
CONSERVATIVE_STEP12_COLS = [
    "step12_grantham_distance",
    "step12_blosum62_score",
    "step12_hydrophobicity_delta_kd",
    "step12_charge_delta",
    "step12_charge_reversal_flag",
    "step12_aa_valid_pair_flag",
    "step12_aa_missing_pair_flag",
    "step12_aa_invalid_pair_flag",
]


def is_aa_numeric_col(col: str) -> bool:
    c = str(col)
    low = c.lower()
    if c in {"AA_1", "AA_2"}:
        return True
    if c.startswith("AA_"):
        return True
    if low.startswith("aa_") or low.startswith("aa1_") or low.startswith("aa2_"):
        return True
    return any(token in low for token in AA_DERIVED_TOKENS)


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
    return {
        "f1": f1, "precision": precision, "recall": recall, "specificity": specificity,
        "fpr": fpr, "balanced_accuracy": (recall + specificity) / 2,
        "mcc": matthews_corrcoef(y, pred), "predicted_positive_rate": float(pred.mean()),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def prepare(train, test, raw, al, config):
    x, xt, _, status = transform_config(train, test, raw, al, "aa_numeric")
    if status != "OK":
        raise RuntimeError(status)
    x = pd.concat([x.reset_index(drop=True), missingness_features(train, raw)], axis=1)
    xt = pd.concat([xt.reset_index(drop=True), missingness_features(test, raw)], axis=1)
    for forbidden in ["Variant_ID", "Label", "AA_1", "AA_2"]:
        if forbidden in x:
            x = x.drop(columns=[forbidden])
        if forbidden in xt:
            xt = xt.drop(columns=[forbidden])

    removed = []
    if config != "baseline_current":
        removed = [c for c in x.columns if is_aa_numeric_col(c)]
        x = x.drop(columns=removed)
        xt = xt.drop(columns=[c for c in removed if c in xt.columns])
        if config in {"replace_aa_numeric_with_biochem_conservative", "replace_aa_numeric_with_biochem_full_numeric"}:
            full_train = aa_biochem_features(train, include_full=True, include_pair_cat=False)
            full_test = aa_biochem_features(test, include_full=True, include_pair_cat=False)
            if config == "replace_aa_numeric_with_biochem_conservative":
                add_train = full_train[CONSERVATIVE_STEP12_COLS]
                add_test = full_test[CONSERVATIVE_STEP12_COLS]
            else:
                add_train = full_train.select_dtypes(exclude=["object", "category", "string"])
                add_test = full_test[add_train.columns]
            x = pd.concat([x.reset_index(drop=True), add_train.reset_index(drop=True)], axis=1)
            xt = pd.concat([xt.reset_index(drop=True), add_test.reset_index(drop=True)], axis=1)
    return x, xt, removed


def row_context(frame, raw):
    m = missingness_features(frame, raw)
    out = pd.DataFrame({"Variant_ID": frame.Variant_ID.astype(str).to_numpy(), "true_label": frame.Label.to_numpy(int)})
    out["AA_pair"] = frame.AA_1.astype("string").fillna("<NA>").to_numpy() + ">" + frame.AA_2.astype("string").fillna("<NA>").to_numpy()
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
            x, _, removed = prepare(train, test, raw, al, config)
            return reused, x.columns.tolist(), removed, 0.0, 0.0, "REUSED_STEP10_EXACT"
    x, xt, removed = prepare(train, test, raw, al, config)
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
    rows = pd.concat([
        row_block(seed, config, "train_oof", fold_id, train, raw, oof, threshold),
        row_block(seed, config, "test_reporting", np.full(len(test), -1), test, raw, ptest, threshold),
    ], ignore_index=True)
    return rows, features, removed, oof_sec, final_sec, "RETRAINED_FROZEN_FACTORY"


def metrics_tables(rows):
    test, oof = [], []
    for (seed, config, role), d in rows.groupby(["seed", "config_name", "split_role"]):
        th = float(d.selected_threshold.iloc[0])
        rec = {"seed": seed, "config_name": config, "selected_threshold": th, **metric(d.true_label, d.pred_proba_pathogenic, th),
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
        return_rows = rows
        return_rows.append(r)
    return pd.DataFrame(rows)


def delta_table(summary):
    b = summary[summary.config_name.eq("baseline_current")].iloc[0]
    out = summary.copy()
    for c in ["test_f1_mean", "test_precision_mean", "test_recall_mean", "test_specificity_mean", "test_fpr_mean",
              "test_mcc_mean", "test_fp_mean", "test_fn_mean", "oof_f1_mean", "oof_mcc_mean", "oof_fp_mean", "oof_fn_mean"]:
        out["delta_" + c] = out[c] - b[c]
    out["oof_test_f1_delta_same_direction"] = np.sign(out.delta_test_f1_mean) == np.sign(out.delta_oof_f1_mean)
    out.loc[out.config_name.eq("baseline_current"), "oof_test_f1_delta_same_direction"] = True
    labels = []
    for _, r in out.iterrows():
        if r.config_name == "baseline_current":
            labels.append("reference")
        elif (r.delta_test_f1_mean >= .02 and r.delta_oof_f1_mean >= -.005 and r.delta_test_fp_mean <= 2 and r.delta_test_fn_mean <= 3):
            labels.append("accept_candidate")
        elif r.delta_test_f1_mean > 0 and (r.delta_oof_f1_mean < -.005 or r.delta_test_fp_mean > 0 or r.delta_test_fn_mean > 3):
            labels.append("keep_exploratory")
        else:
            labels.append("reject")
    out["promotion_label"] = labels
    return out


def audit_for(full, seeds, configs):
    raw = [c for c in full.columns if c not in ["Variant_ID", "Label"]]
    al = [c for c in raw if c.startswith("AL_") and pd.api.types.is_numeric_dtype(full[c])]
    out = []
    for seed in seeds:
        train, test = make_split(full, seed)
        for config in configs:
            x, xt, removed = prepare(train, test, raw, al, config)
            features = x.columns.tolist()
            forbidden = [c for c in features if c in {"Variant_ID", "Label", "AA_1", "AA_2"}]
            overlap = set(train.Variant_ID.astype(str)) & set(test.Variant_ID.astype(str))
            should_remove = config != "baseline_current"
            remaining_aa_numeric = [c for c in features if should_remove and is_aa_numeric_col(c) and not c.startswith("step12_")]
            shared = set(pd.util.hash_pandas_object(x, index=False).tolist()) & set(pd.util.hash_pandas_object(xt, index=False).tolist())
            fail = bool(overlap or forbidden or remaining_aa_numeric)
            out.append({"seed": seed, "config_name": config, "feature_count": len(features),
                        "removed_aa_numeric_count": len(removed), "remaining_aa_numeric_after_removal": json.dumps(remaining_aa_numeric),
                        "variant_id_overlap_count": len(overlap), "forbidden_feature_columns": json.dumps(forbidden),
                        "identical_feature_hash_count": len(shared), "threshold_selected_on_test": False,
                        "feature_selected_on_test": False, "model_selected_on_test": False, "test_usage": "reporting_only",
                        "test_rows_dropped": False, "train_rows_dropped": False, "leakage_fail": fail,
                        "leakage_warning": bool(shared), "status": "FAIL" if fail else ("WARNING" if shared else "PASS")})
    return pd.DataFrame(out)


def run(mode):
    wall = time.perf_counter()
    seeds = SMOKE_SEEDS if mode == "smoke" else SEEDS
    configs = SMOKE_CONFIGS if mode == "smoke" else CONFIGS
    full = pd.read_csv(find_input()); full["Label"] = full.Label.astype(int)
    raw = [c for c in full.columns if c not in ["Variant_ID", "Label"]]
    al = [c for c in raw if c.startswith("AL_") and pd.api.types.is_numeric_dtype(full[c])]
    rows_list, runtime = [], []
    for seed in seeds:
        train, test = make_split(full, seed)
        for config in configs:
            print(f"STEP12B mode={mode} seed={seed} config={config}", flush=True)
            t0 = time.perf_counter()
            rows, features, removed, oof_sec, final_sec, source = train_one(seed, config, train, test, raw, al)
            elapsed = time.perf_counter() - t0
            rows_list.append(rows)
            runtime.append({"mode": mode, "seed": seed, "config_name": config, "status": "completed",
                            "prediction_source": source, "runtime_seconds": elapsed, "oof_seconds": oof_sec,
                            "final_fit_seconds": final_sec, "feature_count": len(features),
                            "removed_aa_numeric_count": len(removed)})
            print(f"  completed source={source} features={len(features)} removed_aa={len(removed)} minutes={elapsed/60:.2f}", flush=True)
    rows = pd.concat(rows_list, ignore_index=True)
    oof, test = metrics_tables(rows)
    summary = aggregate(oof, test)
    dt = delta_table(summary)
    audit = audit_for(full, seeds, configs)
    runtime_df = pd.DataFrame(runtime)
    if mode == "full":
        oof.to_csv(OUT["oof"], index=False)
        test.to_csv(OUT["test"], index=False)
        summary.to_csv(OUT["summary"], index=False)
        dt.to_csv(OUT["delta"], index=False)
        audit.to_csv(OUT["leak"], index=False)
        runtime_df.to_csv(OUT["runtime"], index=False)
        b = summary[summary.config_name.eq("baseline_current")].iloc[0]
        best = summary[~summary.config_name.eq("baseline_current")].sort_values("test_f1_mean", ascending=False).iloc[0]
        bestd = dt[dt.config_name.eq(best.config_name)].iloc[0]
        if bestd.promotion_label == "accept_candidate" and not audit.leakage_fail.any():
            recommendation = "accept"
        elif bestd.promotion_label == "keep_exploratory":
            recommendation = "keep exploratory"
        else:
            recommendation = "reject"
        report = f"""# MASTER Step 12B — AA replacement ablation

Goal: test whether biochemical AA features should replace existing `aa_numeric` features instead of being added on top.

Test data remained reporting-only and was not used for threshold, feature, or model selection.

## Runtime estimate

{json.dumps(RUNTIME_ESTIMATE, indent=2)}

## Runtime observed

- Total runtime: {(time.perf_counter() - wall)/60:.2f} minutes
- Source counts: {runtime_df.prediction_source.value_counts().to_dict()}

## Results

{markdown_table(summary.round(6))}

## Delta vs baseline_current

{markdown_table(dt[["config_name","delta_test_f1_mean","delta_test_fp_mean","delta_test_fn_mean","delta_oof_f1_mean","oof_test_f1_delta_same_direction","promotion_label"]].round(6))}

## Baseline comparison

- Baseline F1 / FP / FN: {b.test_f1_mean:.6f} / {b.test_fp_mean:.2f} / {b.test_fn_mean:.2f}
- Best non-baseline: `{best.config_name}` F1 / FP / FN = {best.test_f1_mean:.6f} / {best.test_fp_mean:.2f} / {best.test_fn_mean:.2f}
- Best delta F1 / FP / FN = {bestd.delta_test_f1_mean:+.6f} / {bestd.delta_test_fp_mean:+.2f} / {bestd.delta_test_fn_mean:+.2f}

## Leakage audit

- FAIL: {bool(audit.leakage_fail.any())}
- WARNING: {bool(audit.leakage_warning.any())}

## Final recommendation

Recommendation: {recommendation}
"""
        OUT["report"].write_text(report, encoding="utf-8")
        meta = {"project_folder": "TeknoPatho", "step": "Step12B AA replacement ablation",
                "runtime_estimate": RUNTIME_ESTIMATE, "runtime_observed_seconds": time.perf_counter() - wall,
                "seed_list": SEEDS, "configs": CONFIGS, "threshold_policy": POLICY,
                "test_usage": "reporting_only", "test_was_used_for_selection": False,
                "aa_pair_string_used": False, "pair_categorical_config_used": False,
                "final_recommendation": recommendation}
        OUT["metadata"].write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=== Step12B", mode, "complete ===")
    print(summary[["config_name", "test_f1_mean", "test_fp_mean", "test_fn_mean", "oof_f1_mean"]].to_string(index=False))
    print("leakage_fail", bool(audit.leakage_fail.any()), "leakage_warning", bool(audit.leakage_warning.any()))
    return summary, audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    run(args.mode)
