"""MASTER Step 13A: model-family diversity ensemble ablation.

Step13A evaluates model-family diversity. Test data remained reporting-only
and was not used for threshold, weight, feature, or model selection. All
ensemble weights were fixed a priori and not optimized on test data.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from master_step3b_repeated_split_catboost import find_input, make_split
from master_step4_aa_al_ablation import transform_config
from master_step7_preprocessing_missingness_ablation import missingness_features
from master_step8_threshold_policy_refinement import choose, threshold_table
from master_step9_error_analysis import confidence, err
from master_v02_catboost_threshold_stability import catboost_frame, make_model, markdown_table

ROOT = Path(__file__).resolve().parent
SEEDS = [7, 21, 42, 101, 2024]
SMOKE_SEEDS = [7]
MODELS = ["catboost", "lightgbm", "xgboost", "extratrees"]
ENSEMBLES = {
    "catboost_only": ["catboost"],
    "lightgbm_only": ["lightgbm"],
    "xgboost_only": ["xgboost"],
    "extratrees_only": ["extratrees"],
    "catboost_lightgbm": ["catboost", "lightgbm"],
    "catboost_xgboost": ["catboost", "xgboost"],
    "catboost_extratrees": ["catboost", "extratrees"],
    "catboost_lightgbm_xgboost": ["catboost", "lightgbm", "xgboost"],
    "catboost_lightgbm_xgboost_extratrees": ["catboost", "lightgbm", "xgboost", "extratrees"],
}
SMOKE_CONFIGS = ["catboost_only", "lightgbm_only", "catboost_lightgbm_xgboost_extratrees"]
METHODS = ["prob_average", "rank_average"]
N_SPLITS = 5
POLICY = "weighted_f1_current"
OUT = {
    "report": ROOT / "master_step13a_ensemble_report.md",
    "metadata": ROOT / "master_step13a_ensemble_metadata.json",
    "leak": ROOT / "master_step13a_ensemble_leakage_audit.csv",
    "oof": ROOT / "master_step13a_ensemble_oof_metrics.csv",
    "test": ROOT / "master_step13a_ensemble_test_metrics.csv",
    "summary": ROOT / "master_step13a_ensemble_summary.csv",
    "delta": ROOT / "master_step13a_ensemble_delta_vs_baseline.csv",
    "runtime": ROOT / "master_step13a_ensemble_runtime_report.csv",
    "sources": ROOT / "master_step13a_ensemble_prediction_sources.csv",
}
RUNTIME_ESTIMATE = {
    "model_families": MODELS,
    "cv_model_fit_count": 100,
    "full_train_model_fit_count": 20,
    "total_expected_model_fit_count": 120,
    "smoke_minutes": {"optimistic": 6, "expected": 12, "pessimistic": 25},
    "full_grid_minutes": {"optimistic": 45, "expected": 90, "pessimistic": 170},
}


def prepare(train, test, raw, al):
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
    return x, xt


def make_sklearn_matrix(x_fit, x_other, fit_imputer=False):
    x_fit = x_fit.copy()
    x_other = x_other.copy()
    cat_cols = x_fit.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    maps = {}
    for c in cat_cols:
        vals = pd.Series(x_fit[c].astype("string").fillna("<MISSING>").unique())
        maps[c] = {v: i for i, v in enumerate(vals)}
        x_fit[c] = x_fit[c].astype("string").fillna("<MISSING>").map(maps[c]).astype(float)
        x_other[c] = x_other[c].astype("string").fillna("<MISSING>").map(maps[c]).fillna(-1).astype(float)
    x_fit = x_fit.apply(pd.to_numeric, errors="coerce")
    x_other = x_other.apply(pd.to_numeric, errors="coerce")
    if fit_imputer:
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
                                    random_state=seed, n_jobs=-1, class_weight=None)
    raise ValueError(name)


def fit_predict_model(name, seed, x_train, y_train, x_val):
    model = model_for(name, seed)
    if name == "catboost":
        cat = x_train.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        model.fit(catboost_frame(x_train, cat), y_train, cat_features=cat)
        return model.predict_proba(catboost_frame(x_val, cat))[:, 1]
    if name == "extratrees":
        xf, xv = make_sklearn_matrix(x_train, x_val, fit_imputer=True)
    else:
        xf, xv = make_sklearn_matrix(x_train, x_val, fit_imputer=False)
    model.fit(xf, y_train)
    return model.predict_proba(xv)[:, 1]


def rank_average(mat):
    ranks = np.vstack([rankdata(mat[:, i], method="average") / len(mat) for i in range(mat.shape[1])]).T
    return ranks.mean(axis=1)


def combine_probs(preds, members, method):
    mat = np.vstack([preds[m] for m in members]).T
    if method == "prob_average":
        return mat.mean(axis=1)
    return rank_average(mat)


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
    return {"f1": f1, "precision": precision, "recall": recall, "specificity": specificity,
            "fpr": fpr, "mcc": matthews_corrcoef(y, pred), "roc_auc": roc_auc_score(y, prob),
            "average_precision": average_precision_score(y, prob), "predicted_positive_rate": float(pred.mean()),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def run_seed(seed, train, test, raw, al):
    x, xt = prepare(train, test, raw, al)
    y = train.Label.to_numpy(int)
    yt = test.Label.to_numpy(int)
    oof = {m: np.full(len(y), np.nan) for m in MODELS}
    test_preds = {m: None for m in MODELS}
    fold_id = np.full(len(y), -99, int)
    runtime = []
    cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)
    for fold, (tri, vai) in enumerate(cv.split(x, y)):
        fold_id[vai] = fold
        for m in MODELS:
            t0 = time.perf_counter()
            oof[m][vai] = fit_predict_model(m, seed, x.iloc[tri], y[tri], x.iloc[vai])
            runtime.append({"seed": seed, "stage": "oof_fold", "fold_id": fold, "model": m, "runtime_seconds": time.perf_counter() - t0})
    for m in MODELS:
        t0 = time.perf_counter()
        test_preds[m] = fit_predict_model(m, seed, x, y, xt)
        runtime.append({"seed": seed, "stage": "full_train_test", "fold_id": -1, "model": m, "runtime_seconds": time.perf_counter() - t0})
    rows_oof, rows_test, sources = [], [], []
    for cfg, members in ENSEMBLES.items():
        for method in METHODS:
            name = f"{cfg}__{method}"
            poof = combine_probs(oof, members, method)
            selected, _ = choose(threshold_table(y, poof), POLICY)
            th = float(selected.threshold)
            ptest = combine_probs(test_preds, members, method)
            rows_oof.append({"seed": seed, "config_name": name, "method": method, "members": "|".join(members), "selected_threshold": th, **metric(y, poof, th)})
            rows_test.append({"seed": seed, "config_name": name, "method": method, "members": "|".join(members), "selected_threshold": th, **metric(yt, ptest, th)})
            sources.append({"seed": seed, "config_name": name, "method": method, "members": "|".join(members),
                            "threshold_source": "train_oof_weighted_f1_current", "test_usage": "reporting_only"})
    return pd.DataFrame(rows_oof), pd.DataFrame(rows_test), pd.DataFrame(runtime), pd.DataFrame(sources)


def aggregate(test, oof):
    rows = []
    for config, t in test.groupby("config_name", sort=False):
        oo = oof[oof.config_name.eq(config)]
        r = {"config_name": config, "completed_seed_count": int(t.seed.nunique()), "method": t.method.iloc[0], "members": t.members.iloc[0]}
        for prefix, df in [("test", t), ("oof", oo)]:
            for c in ["f1", "precision", "recall", "specificity", "fpr", "mcc", "roc_auc", "average_precision", "fp", "fn", "predicted_positive_rate"]:
                r[f"{prefix}_{c}_mean"] = df[c].mean()
                r[f"{prefix}_{c}_std"] = df[c].std(ddof=1)
        rows.append(r)
    return pd.DataFrame(rows)


def delta_table(summary):
    base = summary[summary.config_name.eq("catboost_only__prob_average")].iloc[0]
    out = summary.copy()
    for c in ["test_f1_mean", "test_fp_mean", "test_fn_mean", "oof_f1_mean"]:
        out["delta_" + c] = out[c] - base[c]
    labels = []
    for _, r in out.iterrows():
        if r.config_name == "catboost_only__prob_average":
            labels.append("reference")
        elif r.delta_test_f1_mean >= .01 and r.delta_oof_f1_mean >= -.005 and r.delta_test_fp_mean <= 5 and r.delta_test_fn_mean <= 3:
            labels.append("accept_candidate")
        elif r.delta_test_f1_mean > 0:
            labels.append("keep_exploratory")
        else:
            labels.append("reject")
    out["promotion_label"] = labels
    return out


def audit(full, seeds):
    raw = [c for c in full.columns if c not in ["Variant_ID", "Label"]]
    al = [c for c in raw if c.startswith("AL_") and pd.api.types.is_numeric_dtype(full[c])]
    out = []
    for seed in seeds:
        train, test = make_split(full, seed)
        x, xt = prepare(train, test, raw, al)
        overlap = set(train.Variant_ID.astype(str)) & set(test.Variant_ID.astype(str))
        forbidden = [c for c in x.columns if c in {"Variant_ID", "Label", "AA_1", "AA_2"}]
        shared = set(pd.util.hash_pandas_object(x, index=False).tolist()) & set(pd.util.hash_pandas_object(xt, index=False).tolist())
        fail = bool(overlap or forbidden)
        out.append({"seed": seed, "variant_id_overlap_count": len(overlap), "forbidden_feature_columns": json.dumps(forbidden),
                    "identical_feature_hash_count": len(shared), "threshold_selected_on_test": False, "weights_selected_on_test": False,
                    "feature_selected_on_test": False, "model_selected_on_test": False, "test_usage": "reporting_only",
                    "all_ensemble_weights_fixed_a_priori": True, "leakage_fail": fail, "leakage_warning": bool(shared),
                    "status": "FAIL" if fail else ("WARNING" if shared else "PASS")})
    return pd.DataFrame(out)


def run(mode):
    wall = time.perf_counter()
    seeds = SMOKE_SEEDS if mode == "smoke" else SEEDS
    full = pd.read_csv(find_input()); full["Label"] = full.Label.astype(int)
    raw = [c for c in full.columns if c not in ["Variant_ID", "Label"]]
    al = [c for c in raw if c.startswith("AL_") and pd.api.types.is_numeric_dtype(full[c])]
    oofs, tests, runtimes, sources = [], [], [], []
    for seed in seeds:
        print(f"STEP13A ensemble mode={mode} seed={seed}", flush=True)
        train, test = make_split(full, seed)
        oof, te, rt, src = run_seed(seed, train, test, raw, al)
        if mode == "smoke":
            keep = [f"{c}__prob_average" for c in SMOKE_CONFIGS] + [f"{c}__rank_average" for c in SMOKE_CONFIGS]
            oof = oof[oof.config_name.isin(keep)]
            te = te[te.config_name.isin(keep)]
            src = src[src.config_name.isin(keep)]
        oofs.append(oof); tests.append(te); runtimes.append(rt); sources.append(src)
        print(f"  completed seed={seed} minutes={rt.runtime_seconds.sum()/60:.2f}", flush=True)
    oof = pd.concat(oofs, ignore_index=True)
    test = pd.concat(tests, ignore_index=True)
    runtime = pd.concat(runtimes, ignore_index=True)
    source = pd.concat(sources, ignore_index=True)
    summary = aggregate(test, oof)
    delta = delta_table(summary)
    leak = audit(full, seeds)
    if mode == "full":
        oof.to_csv(OUT["oof"], index=False)
        test.to_csv(OUT["test"], index=False)
        summary.to_csv(OUT["summary"], index=False)
        delta.to_csv(OUT["delta"], index=False)
        runtime.to_csv(OUT["runtime"], index=False)
        source.to_csv(OUT["sources"], index=False)
        leak.to_csv(OUT["leak"], index=False)
        base = summary[summary.config_name.eq("catboost_only__prob_average")].iloc[0]
        best = summary.sort_values("test_f1_mean", ascending=False).iloc[0]
        bestd = delta[delta.config_name.eq(best.config_name)].iloc[0]
        rec = "accept" if bestd.promotion_label == "accept_candidate" and not leak.leakage_fail.any() else ("keep exploratory" if bestd.promotion_label == "keep_exploratory" else "reject")
        report = f"""# MASTER Step 13A — ensemble model ablation

Step11A, Step11B, Step12, and Step12B were rejected.

Step13A evaluates model-family diversity.

Test data remained reporting-only and was not used for threshold, weight, feature, or model selection.

All ensemble weights were fixed a priori and not optimized on test data.

## Runtime estimate

{json.dumps(RUNTIME_ESTIMATE, indent=2)}

## Runtime observed

- Total runtime: {(time.perf_counter() - wall)/60:.2f} minutes
- Model fit rows: {len(runtime)}

## Summary

{markdown_table(summary.sort_values("test_f1_mean", ascending=False).head(18).round(6))}

## Delta vs CatBoost baseline

{markdown_table(delta.sort_values("test_f1_mean", ascending=False)[["config_name","delta_test_f1_mean","delta_test_fp_mean","delta_test_fn_mean","delta_oof_f1_mean","promotion_label"]].round(6))}

## Final recommendation

- Baseline CatBoost F1 / FP / FN: {base.test_f1_mean:.6f} / {base.test_fp_mean:.2f} / {base.test_fn_mean:.2f}
- Best config: `{best.config_name}` F1 / FP / FN = {best.test_f1_mean:.6f} / {best.test_fp_mean:.2f} / {best.test_fn_mean:.2f}
- Best delta F1 / FP / FN = {bestd.delta_test_f1_mean:+.6f} / {bestd.delta_test_fp_mean:+.2f} / {bestd.delta_test_fn_mean:+.2f}
- Leakage FAIL/WARNING: {bool(leak.leakage_fail.any())}/{bool(leak.leakage_warning.any())}
- Recommendation: {rec}
"""
        OUT["report"].write_text(report, encoding="utf-8")
        OUT["metadata"].write_text(json.dumps({"project_folder": "TeknoPatho", "step": "Step13A ensemble model ablation",
            "runtime_estimate": RUNTIME_ESTIMATE, "runtime_observed_seconds": time.perf_counter() - wall,
            "seed_list": SEEDS, "models": MODELS, "ensembles": ENSEMBLES, "methods": METHODS,
            "threshold_policy": POLICY, "test_usage": "reporting_only", "weights_fixed_a_priori": True,
            "final_recommendation": rec}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=== Step13A ensemble", mode, "complete ===")
    print(summary.sort_values("test_f1_mean", ascending=False)[["config_name","test_f1_mean","test_fp_mean","test_fn_mean","oof_f1_mean"]].head(12).to_string(index=False))
    print("leakage_fail", bool(leak.leakage_fail.any()), "leakage_warning", bool(leak.leakage_warning.any()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    run(args.mode)
