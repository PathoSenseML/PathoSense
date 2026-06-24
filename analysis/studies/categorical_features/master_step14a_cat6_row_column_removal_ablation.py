"""MASTER Step 14A: CAT_6 row-removal and column-removal ablation.

Step14A evaluates the teacher-requested CAT_6 row-removal and column-removal
strategy. The primary valid experiment removes CAT_6-present rows from
training only, drops CAT_6 from the feature set, and still predicts every test
row.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from master_step3b_repeated_split_catboost import find_input, make_split
from master_step4_aa_al_ablation import transform_config
from master_step7_preprocessing_missingness_ablation import missingness_features
from master_step8_threshold_policy_refinement import choose, threshold_table
from master_v02_catboost_threshold_stability import catboost_frame, make_model, markdown_table

ROOT = Path(__file__).resolve().parent
SEEDS = [7, 21, 42]
CONFIGS = [
    "active_reference_with_CAT6",
    "drop_CAT6_column_only",
    "train_drop_CAT6_rows_then_drop_CAT6_column",
]
MODELS = ["catboost", "lightgbm", "xgboost", "extratrees"]
WEIGHTS = {m: 0.25 for m in MODELS}
N_SPLITS = 5
POLICY = "weighted_f1_current"
CAT6_PRESENT_INVALID = {"", "NA", "NaN", "None", "null", "missing", "Missing", "UNKNOWN", "Unknown", "unknown", "-"}
OUT = {
    "report": ROOT / "master_step14a_cat6_report.md",
    "metadata": ROOT / "master_step14a_metadata.json",
    "leak": ROOT / "master_step14a_leakage_audit.csv",
    "oof": ROOT / "master_step14a_oof_metrics.csv",
    "test": ROOT / "master_step14a_test_metrics.csv",
    "summary": ROOT / "master_step14a_summary.csv",
    "delta": ROOT / "master_step14a_delta_vs_active_reference.csv",
    "removed": ROOT / "master_step14a_cat6_removed_rows_report.csv",
    "subgroup": ROOT / "master_step14a_cat6_subgroup_diagnostic.csv",
    "runtime": ROOT / "master_step14a_runtime_report.csv",
}
RUNTIME_ESTIMATE = {
    "screening_seed_list": SEEDS,
    "model_fit_configs": CONFIGS,
    "cv_fit_count": 180,
    "full_train_fit_count": 36,
    "total_expected_model_fit_count": 216,
    "diagnostic_config_D_additional_fits": 0,
    "runtime_minutes": {"optimistic": 35, "expected": 65, "pessimistic": 120},
}


def cat6_present_mask(frame: pd.DataFrame) -> pd.Series:
    if "CAT_6" not in frame:
        return pd.Series(False, index=frame.index)
    s = frame["CAT_6"]
    stripped = s.astype("string").str.strip()
    return s.notna() & (~stripped.isin(CAT6_PRESENT_INVALID))


def raw_for_config(full_columns, config):
    raw = [c for c in full_columns if c not in ["Variant_ID", "Label"]]
    if config in {"drop_CAT6_column_only", "train_drop_CAT6_rows_then_drop_CAT6_column"}:
        raw = [c for c in raw if c != "CAT_6"]
    return raw


def prepare(train, test, full_columns, config):
    raw = raw_for_config(full_columns, config)
    al = [c for c in raw if c.startswith("AL_") and c in train.columns and pd.api.types.is_numeric_dtype(train[c])]
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
    if config in {"drop_CAT6_column_only", "train_drop_CAT6_rows_then_drop_CAT6_column"}:
        forbidden = [c for c in x.columns if "CAT_6" in c or c == "CAT_6"]
        if forbidden:
            x = x.drop(columns=forbidden)
            xt = xt.drop(columns=[c for c in forbidden if c in xt.columns])
    return x, xt, raw


def numeric_matrix(x_fit, x_other, impute=False):
    x_fit = x_fit.copy(); x_other = x_other.copy()
    cat_cols = x_fit.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    for c in cat_cols:
        vals = pd.Series(x_fit[c].astype("string").fillna("<MISSING>").unique())
        mp = {v: i for i, v in enumerate(vals)}
        x_fit[c] = x_fit[c].astype("string").fillna("<MISSING>").map(mp).astype(float)
        x_other[c] = x_other[c].astype("string").fillna("<MISSING>").map(mp).fillna(-1).astype(float)
    x_fit = x_fit.apply(pd.to_numeric, errors="coerce")
    x_other = x_other.apply(pd.to_numeric, errors="coerce")
    if impute:
        imp = SimpleImputer(strategy="median")
        return imp.fit_transform(x_fit), imp.transform(x_other)
    return x_fit, x_other


def model_for(name, seed):
    if name == "catboost":
        return make_model(seed)
    if name == "lightgbm":
        return LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31, subsample=0.9,
                              colsample_bytree=0.9, random_state=seed, n_jobs=-1, verbose=-1)
    if name == "xgboost":
        return XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.9,
                             colsample_bytree=0.9, objective="binary:logistic", eval_metric="logloss",
                             tree_method="hist", random_state=seed, n_jobs=-1)
    if name == "extratrees":
        return ExtraTreesClassifier(n_estimators=500, max_features="sqrt", min_samples_leaf=2,
                                    random_state=seed, n_jobs=-1)
    raise ValueError(name)


def fit_predict(name, seed, x_train, y_train, x_val):
    model = model_for(name, seed)
    if name == "catboost":
        cat = x_train.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        model.fit(catboost_frame(x_train, cat), y_train, cat_features=cat)
        return model.predict_proba(catboost_frame(x_val, cat))[:, 1]
    xf, xv = numeric_matrix(x_train, x_val, impute=(name == "extratrees"))
    model.fit(xf, y_train)
    return model.predict_proba(xv)[:, 1]


def metric(y, prob, threshold):
    y = np.asarray(y, int); prob = np.asarray(prob, float)
    if len(y) == 0:
        return {k: np.nan for k in ["f1", "precision", "recall", "specificity", "fpr", "mcc", "roc_auc",
                                    "average_precision", "predicted_positive_rate", "tn", "fp", "fn", "tp"]}
    pred = (prob >= threshold).astype(int)
    tn = int(((y == 0) & (pred == 0)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum()); tp = int(((y == 1) & (pred == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    fpr = fp / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    auc = roc_auc_score(y, prob) if len(np.unique(y)) > 1 else np.nan
    ap = average_precision_score(y, prob) if len(np.unique(y)) > 1 else np.nan
    return {"f1": f1, "precision": precision, "recall": recall, "specificity": specificity,
            "fpr": fpr, "mcc": matthews_corrcoef(y, pred), "roc_auc": auc, "average_precision": ap,
            "predicted_positive_rate": float(pred.mean()), "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def run_config(seed, config, train0, test, full_columns):
    present_train = cat6_present_mask(train0)
    if config == "train_drop_CAT6_rows_then_drop_CAT6_column":
        fit_train = train0.loc[~present_train].reset_index(drop=True)
        removed = train0.loc[present_train].reset_index(drop=True)
    else:
        fit_train = train0.reset_index(drop=True)
        removed = train0.iloc[0:0].copy()
    x, xt, raw = prepare(fit_train, test, full_columns, config)
    y = fit_train.Label.to_numpy(int)
    yt = test.Label.to_numpy(int)
    cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)
    oof_models = {m: np.full(len(y), np.nan) for m in MODELS}
    test_models = {}
    runtime = []
    for fold, (tri, vai) in enumerate(cv.split(x, y)):
        for m in MODELS:
            t0 = time.perf_counter()
            oof_models[m][vai] = fit_predict(m, seed, x.iloc[tri], y[tri], x.iloc[vai])
            runtime.append({"seed": seed, "config_name": config, "stage": "oof_fold", "fold_id": fold,
                            "model": m, "runtime_seconds": time.perf_counter() - t0})
    poof = sum(WEIGHTS[m] * oof_models[m] for m in MODELS)
    selected, _ = choose(threshold_table(y, poof), POLICY)
    th = float(selected.threshold)
    for m in MODELS:
        t0 = time.perf_counter()
        test_models[m] = fit_predict(m, seed, x, y, xt)
        runtime.append({"seed": seed, "config_name": config, "stage": "full_train_test", "fold_id": -1,
                        "model": m, "runtime_seconds": time.perf_counter() - t0})
    ptest = sum(WEIGHTS[m] * test_models[m] for m in MODELS)
    oof_row = {"seed": seed, "config_name": config, "selected_threshold": th, "train_n_for_fit": len(fit_train),
               "original_train_n": len(train0), "removed_train_n": len(removed), "feature_count": len(x.columns),
               **metric(y, poof, th)}
    test_row = {"seed": seed, "config_name": config, "selected_threshold": th, "train_n_for_fit": len(fit_train),
                "original_train_n": len(train0), "removed_train_n": len(removed), "feature_count": len(x.columns),
                "test_cat6_present_rate": float(cat6_present_mask(test).mean()), **metric(yt, ptest, th)}
    removed_report = {"seed": seed, "config_name": config, "original_train_n": len(train0), "train_n_for_fit": len(fit_train),
                      "removed_train_n": len(removed), "removed_train_pct": len(removed) / max(len(train0), 1),
                      "removed_label_0": int(removed.Label.eq(0).sum()) if len(removed) else 0,
                      "removed_label_1": int(removed.Label.eq(1).sum()) if len(removed) else 0,
                      "kept_label_0": int(fit_train.Label.eq(0).sum()), "kept_label_1": int(fit_train.Label.eq(1).sum()),
                      "cat6_present_rule": "CAT_6 not null, stripped not empty, and not in {NA, NaN, None, null, missing, Missing, UNKNOWN, Unknown, unknown, -}"}
    return oof_row, test_row, pd.DataFrame(runtime), removed_report, ptest, th, fit_train, x.columns.tolist(), raw


def subgroup_diagnostics(seed, test, config_c_prob, threshold):
    present = cat6_present_mask(test)
    rows = []
    for name, mask in {"all_test_rows": np.ones(len(test), bool), "CAT_6_missing_test_rows": ~present,
                       "CAT_6_present_test_rows": present}.items():
        g = test.loc[np.asarray(mask, bool)]
        prob = config_c_prob[np.asarray(mask, bool)]
        rows.append({"seed": seed, "source_config": "train_drop_CAT6_rows_then_drop_CAT6_column",
                     "diagnostic_subgroup": name, "n": len(g), "cat6_present_n": int(present[np.asarray(mask, bool)].sum()),
                     **metric(g.Label.to_numpy(int), prob, threshold)})
    return rows


def aggregate(test, oof):
    rows = []
    for config, t in test.groupby("config_name", sort=False):
        oo = oof[oof.config_name.eq(config)]
        r = {"config_name": config, "completed_seed_count": int(t.seed.nunique()),
             "feature_count_mean": t.feature_count.mean(), "removed_train_n_mean": t.removed_train_n.mean(),
             "test_cat6_present_rate_mean": t.test_cat6_present_rate.mean()}
        for prefix, df in [("test", t), ("oof", oo)]:
            for c in ["f1", "precision", "recall", "specificity", "fpr", "mcc", "roc_auc", "average_precision",
                      "fp", "fn", "predicted_positive_rate", "selected_threshold"]:
                r[f"{prefix}_{c}_mean"] = df[c].mean()
                r[f"{prefix}_{c}_std"] = df[c].std(ddof=1)
        rows.append(r)
    return pd.DataFrame(rows)


def delta(summary):
    b = summary[summary.config_name.eq("active_reference_with_CAT6")].iloc[0]
    out = summary.copy()
    for c in ["test_f1_mean", "test_fp_mean", "test_fn_mean", "oof_f1_mean"]:
        out["delta_" + c] = out[c] - b[c]
    labels = []
    for _, r in out.iterrows():
        if r.config_name == "active_reference_with_CAT6":
            labels.append("reference")
        elif r.config_name == "train_drop_CAT6_rows_then_drop_CAT6_column" and r.delta_test_f1_mean >= .005 and r.delta_oof_f1_mean >= -.005 and r.delta_test_fp_mean <= 5 and r.delta_test_fn_mean <= 3:
            labels.append("promote_candidate_revalidate_5_seed")
        elif r.delta_test_f1_mean > 0:
            labels.append("keep_exploratory")
        else:
            labels.append("reject")
    out["promotion_label"] = labels
    return out


def audit(full, seeds):
    out = []
    for seed in seeds:
        train, test = make_split(full, seed)
        for config in CONFIGS:
            fit_train = train.loc[~cat6_present_mask(train)].reset_index(drop=True) if config == "train_drop_CAT6_rows_then_drop_CAT6_column" else train
            x, xt, raw = prepare(fit_train, test, full.columns, config)
            overlap = set(fit_train.Variant_ID.astype(str)) & set(test.Variant_ID.astype(str))
            forbidden = [c for c in x.columns if c in {"Variant_ID", "Label", "AA_1", "AA_2"}]
            cat6_features = [c for c in x.columns if config != "active_reference_with_CAT6" and "CAT_6" in c]
            fail = bool(overlap or forbidden or cat6_features)
            out.append({"seed": seed, "config_name": config, "variant_id_overlap_count": len(overlap),
                        "forbidden_feature_columns": json.dumps(forbidden), "cat6_feature_columns_after_removal": json.dumps(cat6_features),
                        "cat6_removed_from_raw_before_missingness": bool(config == "active_reference_with_CAT6" or "CAT_6" not in raw),
                        "test_rows_removed": False, "train_rows_removed": bool(config == "train_drop_CAT6_rows_then_drop_CAT6_column"),
                        "threshold_selected_on_test": False, "model_selected_on_test": False,
                        "test_usage": "reporting_only", "leakage_fail": fail, "leakage_warning": False,
                        "status": "FAIL" if fail else "PASS"})
    return pd.DataFrame(out)


def main():
    wall = time.perf_counter()
    full = pd.read_csv(find_input()); full["Label"] = full.Label.astype(int)
    oof_rows, test_rows, runtimes, removed_rows, subgroup_rows = [], [], [], [], []
    for seed in SEEDS:
        train, test = make_split(full, seed)
        c_prob = None; c_th = None
        for config in CONFIGS:
            print(f"STEP14A seed={seed} config={config}", flush=True)
            t0 = time.perf_counter()
            oo, te, rt, rem, ptest, th, fit_train, features, raw = run_config(seed, config, train, test, full.columns)
            oof_rows.append(oo); test_rows.append(te); runtimes.append(rt); removed_rows.append(rem)
            if config == "train_drop_CAT6_rows_then_drop_CAT6_column":
                c_prob, c_th = ptest, th
            print(f"  completed features={len(features)} train_fit={len(fit_train)} removed={rem['removed_train_n']} minutes={(time.perf_counter()-t0)/60:.2f}", flush=True)
        subgroup_rows.extend(subgroup_diagnostics(seed, test, c_prob, c_th))
    oof = pd.DataFrame(oof_rows); test = pd.DataFrame(test_rows)
    runtime = pd.concat(runtimes, ignore_index=True)
    removed = pd.DataFrame(removed_rows)
    sub = pd.DataFrame(subgroup_rows)
    summary = aggregate(test, oof)
    delt = delta(summary)
    leak = audit(full, SEEDS)
    oof.to_csv(OUT["oof"], index=False)
    test.to_csv(OUT["test"], index=False)
    summary.to_csv(OUT["summary"], index=False)
    delt.to_csv(OUT["delta"], index=False)
    removed.to_csv(OUT["removed"], index=False)
    sub.to_csv(OUT["subgroup"], index=False)
    runtime.to_csv(OUT["runtime"], index=False)
    leak.to_csv(OUT["leak"], index=False)
    ref = summary[summary.config_name.eq("active_reference_with_CAT6")].iloc[0]
    c = summary[summary.config_name.eq("train_drop_CAT6_rows_then_drop_CAT6_column")].iloc[0]
    cd = delt[delt.config_name.eq("train_drop_CAT6_rows_then_drop_CAT6_column")].iloc[0]
    rec = "promote_to_5_seed_revalidation" if cd.promotion_label == "promote_candidate_revalidate_5_seed" and not leak.leakage_fail.any() else ("keep exploratory" if cd.delta_test_f1_mean > 0 else "reject")
    report = f"""# MASTER Step 14A — CAT_6 row/column removal ablation

Step14A evaluates the teacher-requested CAT_6 row-removal and column-removal strategy.

The primary valid experiment removes CAT_6-present rows from training only, drops CAT_6 from the feature set, and still predicts every test row.

Test rows were not removed for the primary score.

CAT_6 was not used as a model feature after removal, including CAT_6-derived missingness features.

Any subset analysis that excludes CAT_6-present test rows is diagnostic only and is not used for model selection.

Step14A is a 3-seed screening run. Any promoted candidate should be revalidated with the full 5-seed protocol.

## Runtime estimate

{json.dumps(RUNTIME_ESTIMATE, indent=2)}

## CAT_6 present rule

CAT_6-present = CAT_6 is not null AND stripped value is not empty string AND not one of {sorted(CAT6_PRESENT_INVALID)}.

## Runtime observed

- Total runtime: {(time.perf_counter() - wall)/60:.2f} minutes
- Model fit rows: {len(runtime)}

## Summary

{markdown_table(summary.round(6))}

## Delta vs active reference

{markdown_table(delt[["config_name","delta_test_f1_mean","delta_test_fp_mean","delta_test_fn_mean","delta_oof_f1_mean","promotion_label"]].round(6))}

## Removed train rows

{markdown_table(removed.round(6))}

## Diagnostic CAT_6 test subgroups

{markdown_table(sub.round(6))}

## Final recommendation

- Active reference F1 / FP / FN: {ref.test_f1_mean:.6f} / {ref.test_fp_mean:.2f} / {ref.test_fn_mean:.2f}
- Teacher strategy F1 / FP / FN: {c.test_f1_mean:.6f} / {c.test_fp_mean:.2f} / {c.test_fn_mean:.2f}
- Teacher strategy delta F1 / FP / FN: {cd.delta_test_f1_mean:+.6f} / {cd.delta_test_fp_mean:+.2f} / {cd.delta_test_fn_mean:+.2f}
- Leakage FAIL/WARNING: {bool(leak.leakage_fail.any())}/{bool(leak.leakage_warning.any())}
- Recommendation: {rec}
"""
    OUT["report"].write_text(report, encoding="utf-8")
    OUT["metadata"].write_text(json.dumps({"project_folder": "TeknoPatho", "step": "Step14A CAT_6 row-column removal ablation",
        "runtime_estimate": RUNTIME_ESTIMATE, "runtime_observed_seconds": time.perf_counter() - wall,
        "seed_list": SEEDS, "active_model": "CatBoost+LightGBM+XGBoost+ExtraTrees equal probability average",
        "weights": WEIGHTS, "feature_config": "baseline_features", "cat6_present_invalid_values": sorted(CAT6_PRESENT_INVALID),
        "test_rows_removed_primary": False, "threshold_policy": POLICY, "test_usage": "reporting_only",
        "final_recommendation": rec}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=== Step14A complete ===")
    print(summary[["config_name","test_f1_mean","test_fp_mean","test_fn_mean","oof_f1_mean","removed_train_n_mean"]].to_string(index=False))
    print("leakage_fail", bool(leak.leakage_fail.any()), "leakage_warning", bool(leak.leakage_warning.any()))


if __name__ == "__main__":
    main()
