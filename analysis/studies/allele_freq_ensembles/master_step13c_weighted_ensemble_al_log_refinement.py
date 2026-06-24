"""MASTER Step 13C: weighted CatBoost/ExtraTrees + AL_log refinement.

Step13C evaluates preregistered CatBoost/ExtraTrees ensemble weights.
AL_log is tested only as a preregistered feature-family hypothesis.
Test data remained reporting-only and was not used for threshold, feature,
weight, or model selection. AA biochemical features were excluded because
Step12, Step12B, and Step13B did not support them.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from master_step3b_repeated_split_catboost import find_input, make_split
from master_step4_aa_al_ablation import al_log_features, transform_config
from master_step7_preprocessing_missingness_ablation import missingness_features
from master_step8_threshold_policy_refinement import choose, threshold_table
from master_v02_catboost_threshold_stability import catboost_frame, make_model, markdown_table

ROOT = Path(__file__).resolve().parent
SEEDS = [7, 21, 42, 101, 2024]
SMOKE_SEEDS = [7]
FEATURE_CONFIGS = ["baseline_features", "baseline_plus_al_log"]
WEIGHTS = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
SMOKE_WEIGHTS = [0.50, 0.70, 0.85]
N_SPLITS = 5
POLICY = "weighted_f1_current"
OUT = {
    "report": ROOT / "master_step13c_report.md",
    "metadata": ROOT / "master_step13c_metadata.json",
    "leak": ROOT / "master_step13c_leakage_audit.csv",
    "oof": ROOT / "master_step13c_oof_metrics.csv",
    "test": ROOT / "master_step13c_test_metrics.csv",
    "summary": ROOT / "master_step13c_summary.csv",
    "delta": ROOT / "master_step13c_delta_vs_baseline.csv",
    "curve": ROOT / "master_step13c_weight_curve.csv",
    "runtime": ROOT / "master_step13c_runtime_report.csv",
}
RUNTIME_ESTIMATE = {
    "cv_fit_count": 100,
    "full_train_fit_count": 20,
    "total_expected_model_fit_count": 120,
    "weight_combinations": len(FEATURE_CONFIGS) * len(WEIGHTS),
    "smoke_minutes": {"optimistic": 5, "expected": 12, "pessimistic": 25},
    "full_grid_minutes": {"optimistic": 25, "expected": 50, "pessimistic": 90},
}


def prepare(train, test, raw, al, feature_config):
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
    added = []
    if feature_config == "baseline_plus_al_log":
        tr = al_log_features(train, al).reset_index(drop=True)
        te = al_log_features(test, al).reset_index(drop=True)
        x = pd.concat([x, tr], axis=1)
        xt = pd.concat([xt, te], axis=1)
        added = tr.columns.tolist()
    return x, xt, added


def make_extra_matrix(x_fit, x_other):
    x_fit = x_fit.copy(); x_other = x_other.copy()
    cat_cols = x_fit.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    for c in cat_cols:
        vals = pd.Series(x_fit[c].astype("string").fillna("<MISSING>").unique())
        mp = {v: i for i, v in enumerate(vals)}
        x_fit[c] = x_fit[c].astype("string").fillna("<MISSING>").map(mp).astype(float)
        x_other[c] = x_other[c].astype("string").fillna("<MISSING>").map(mp).fillna(-1).astype(float)
    x_fit = x_fit.apply(pd.to_numeric, errors="coerce")
    x_other = x_other.apply(pd.to_numeric, errors="coerce")
    imp = SimpleImputer(strategy="median")
    return imp.fit_transform(x_fit), imp.transform(x_other)


def fit_catboost(seed, x_train, y_train, x_val):
    cat = x_train.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    model = make_model(seed)
    model.fit(catboost_frame(x_train, cat), y_train, cat_features=cat)
    return model.predict_proba(catboost_frame(x_val, cat))[:, 1]


def fit_extratrees(seed, x_train, y_train, x_val):
    xf, xv = make_extra_matrix(x_train, x_val)
    model = ExtraTreesClassifier(n_estimators=500, max_features="sqrt", min_samples_leaf=2,
                                 random_state=seed, n_jobs=-1)
    model.fit(xf, y_train)
    return model.predict_proba(xv)[:, 1]


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


def source_predictions(seed, feature_config, train, test, raw, al):
    x, xt, added = prepare(train, test, raw, al, feature_config)
    y = train.Label.to_numpy(int)
    oof_cat = np.full(len(y), np.nan); oof_ext = np.full(len(y), np.nan)
    cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)
    runtime = []
    for fold, (tri, vai) in enumerate(cv.split(x, y)):
        t0 = time.perf_counter()
        oof_cat[vai] = fit_catboost(seed, x.iloc[tri], y[tri], x.iloc[vai])
        runtime.append({"seed": seed, "feature_config": feature_config, "stage": "oof_fold", "fold_id": fold, "model": "catboost", "runtime_seconds": time.perf_counter() - t0})
        t0 = time.perf_counter()
        oof_ext[vai] = fit_extratrees(seed, x.iloc[tri], y[tri], x.iloc[vai])
        runtime.append({"seed": seed, "feature_config": feature_config, "stage": "oof_fold", "fold_id": fold, "model": "extratrees", "runtime_seconds": time.perf_counter() - t0})
    t0 = time.perf_counter()
    test_cat = fit_catboost(seed, x, y, xt)
    runtime.append({"seed": seed, "feature_config": feature_config, "stage": "full_train_test", "fold_id": -1, "model": "catboost", "runtime_seconds": time.perf_counter() - t0})
    t0 = time.perf_counter()
    test_ext = fit_extratrees(seed, x, y, xt)
    runtime.append({"seed": seed, "feature_config": feature_config, "stage": "full_train_test", "fold_id": -1, "model": "extratrees", "runtime_seconds": time.perf_counter() - t0})
    return {"cat": oof_cat, "ext": oof_ext, "test_cat": test_cat, "test_ext": test_ext,
            "feature_count": len(x.columns), "added_feature_count": len(added)}, pd.DataFrame(runtime)


def evaluate_weights(seed, feature_config, preds, y, yt, weights):
    oof_rows, test_rows = [], []
    for w in weights:
        poof = w * preds["cat"] + (1 - w) * preds["ext"]
        selected, _ = choose(threshold_table(y, poof), POLICY)
        th = float(selected.threshold)
        ptest = w * preds["test_cat"] + (1 - w) * preds["test_ext"]
        base = {"seed": seed, "feature_config": feature_config, "catboost_weight": w,
                "extratrees_weight": 1 - w, "selected_threshold": th,
                "feature_count": preds["feature_count"], "added_feature_count": preds["added_feature_count"]}
        oof_rows.append({**base, **metric(y, poof, th)})
        test_rows.append({**base, **metric(yt, ptest, th)})
    return oof_rows, test_rows


def aggregate(test, oof):
    rows = []
    for (cfg, w), t in test.groupby(["feature_config", "catboost_weight"], sort=False):
        oo = oof[(oof.feature_config.eq(cfg)) & (oof.catboost_weight.eq(w))]
        r = {"feature_config": cfg, "catboost_weight": w, "extratrees_weight": 1 - w,
             "completed_seed_count": int(t.seed.nunique()), "feature_count_mean": t.feature_count.mean(),
             "added_feature_count_mean": t.added_feature_count.mean()}
        for prefix, df in [("test", t), ("oof", oo)]:
            for c in ["f1", "precision", "recall", "specificity", "fpr", "mcc", "roc_auc", "average_precision", "fp", "fn", "predicted_positive_rate"]:
                r[f"{prefix}_{c}_mean"] = df[c].mean()
                r[f"{prefix}_{c}_std"] = df[c].std(ddof=1)
        rows.append(r)
    return pd.DataFrame(rows)


def delta_table(summary):
    base = summary[(summary.feature_config.eq("baseline_features")) & (summary.catboost_weight.eq(0.50))].iloc[0]
    out = summary.copy()
    for c in ["test_f1_mean", "test_fp_mean", "test_fn_mean", "oof_f1_mean"]:
        out["delta_" + c] = out[c] - base[c]
    labels = []
    for _, r in out.iterrows():
        if r.feature_config == "baseline_features" and abs(r.catboost_weight - 0.50) < 1e-9:
            labels.append("reference")
        elif r.delta_test_f1_mean >= .01 and r.delta_oof_f1_mean >= -.005 and r.delta_test_fp_mean <= 5 and r.delta_test_fn_mean <= 3:
            labels.append("accept_candidate")
        elif r.feature_config == "baseline_plus_al_log" and r.delta_test_fp_mean < 0 and r.delta_test_fn_mean > 0 and r.delta_test_f1_mean < .01:
            labels.append("conservative_exploratory")
        elif r.delta_test_f1_mean > 0:
            labels.append("keep_exploratory")
        else:
            labels.append("reject")
    out["promotion_label"] = labels
    return out


def audit(full, seeds, feature_configs):
    raw = [c for c in full.columns if c not in ["Variant_ID", "Label"]]
    al = [c for c in raw if c.startswith("AL_") and pd.api.types.is_numeric_dtype(full[c])]
    out = []
    for seed in seeds:
        train, test = make_split(full, seed)
        for cfg in feature_configs:
            x, xt, added = prepare(train, test, raw, al, cfg)
            overlap = set(train.Variant_ID.astype(str)) & set(test.Variant_ID.astype(str))
            forbidden = [c for c in x.columns if c in {"Variant_ID", "Label", "AA_1", "AA_2"}]
            shared = set(pd.util.hash_pandas_object(x, index=False).tolist()) & set(pd.util.hash_pandas_object(xt, index=False).tolist())
            fail = bool(overlap or forbidden)
            out.append({"seed": seed, "feature_config": cfg, "variant_id_overlap_count": len(overlap),
                        "forbidden_feature_columns": json.dumps(forbidden), "identical_feature_hash_count": len(shared),
                        "threshold_selected_on_test": False, "weights_selected_on_test": False,
                        "feature_selected_on_test": False, "model_selected_on_test": False, "test_usage": "reporting_only",
                        "ensemble_weights_fixed_a_priori": True, "aa_biochemical_features_used": False,
                        "leakage_fail": fail, "leakage_warning": bool(shared),
                        "status": "FAIL" if fail else ("WARNING" if shared else "PASS")})
    return pd.DataFrame(out)


def run(mode):
    wall = time.perf_counter()
    seeds = SMOKE_SEEDS if mode == "smoke" else SEEDS
    weights = SMOKE_WEIGHTS if mode == "smoke" else WEIGHTS
    full = pd.read_csv(find_input()); full["Label"] = full.Label.astype(int)
    raw = [c for c in full.columns if c not in ["Variant_ID", "Label"]]
    al = [c for c in raw if c.startswith("AL_") and pd.api.types.is_numeric_dtype(full[c])]
    oof_rows, test_rows, runtimes = [], [], []
    for seed in seeds:
        train, test = make_split(full, seed)
        y = train.Label.to_numpy(int); yt = test.Label.to_numpy(int)
        for cfg in FEATURE_CONFIGS:
            print(f"STEP13C mode={mode} seed={seed} feature_config={cfg}", flush=True)
            t0 = time.perf_counter()
            preds, rt = source_predictions(seed, cfg, train, test, raw, al)
            oo, te = evaluate_weights(seed, cfg, preds, y, yt, weights)
            oof_rows.extend(oo); test_rows.extend(te); runtimes.append(rt)
            print(f"  completed weights={len(weights)} features={preds['feature_count']} minutes={(time.perf_counter()-t0)/60:.2f}", flush=True)
    oof = pd.DataFrame(oof_rows)
    test = pd.DataFrame(test_rows)
    runtime = pd.concat(runtimes, ignore_index=True)
    summary = aggregate(test, oof)
    delta = delta_table(summary)
    leak = audit(full, seeds, FEATURE_CONFIGS)
    curve = summary.copy()
    if mode == "full":
        oof.to_csv(OUT["oof"], index=False)
        test.to_csv(OUT["test"], index=False)
        summary.to_csv(OUT["summary"], index=False)
        delta.to_csv(OUT["delta"], index=False)
        curve.to_csv(OUT["curve"], index=False)
        runtime.to_csv(OUT["runtime"], index=False)
        leak.to_csv(OUT["leak"], index=False)
        base = summary[(summary.feature_config.eq("baseline_features")) & (summary.catboost_weight.eq(0.50))].iloc[0]
        best = summary.sort_values("test_f1_mean", ascending=False).iloc[0]
        bestd = delta[(delta.feature_config.eq(best.feature_config)) & (delta.catboost_weight.eq(best.catboost_weight))].iloc[0]
        rec = "accept" if bestd.promotion_label == "accept_candidate" and not leak.leakage_fail.any() else ("keep exploratory" if bestd.promotion_label in ["keep_exploratory", "conservative_exploratory"] else "reject")
        report = f"""# MASTER Step 13C — weighted ensemble + AL_log refinement

Step13C evaluates preregistered CatBoost/ExtraTrees ensemble weights.

AL_log is tested only as a preregistered feature-family hypothesis.

Test data remained reporting-only and was not used for threshold, feature, weight, or model selection.

AA biochemical features were excluded because Step12, Step12B, and Step13B did not support them.

## Runtime estimate

{json.dumps(RUNTIME_ESTIMATE, indent=2)}

## Runtime observed

- Total runtime: {(time.perf_counter() - wall)/60:.2f} minutes
- Model fit rows: {len(runtime)}

## Weight curve

{markdown_table(summary.sort_values(["feature_config","catboost_weight"]).round(6))}

## Delta vs baseline_features weight=0.50

{markdown_table(delta.sort_values("test_f1_mean", ascending=False)[["feature_config","catboost_weight","delta_test_f1_mean","delta_test_fp_mean","delta_test_fn_mean","delta_oof_f1_mean","promotion_label"]].round(6))}

## Final recommendation

- Reference F1 / FP / FN: {base.test_f1_mean:.6f} / {base.test_fp_mean:.2f} / {base.test_fn_mean:.2f}
- Best config: `{best.feature_config}` weight={best.catboost_weight:.2f} F1 / FP / FN = {best.test_f1_mean:.6f} / {best.test_fp_mean:.2f} / {best.test_fn_mean:.2f}
- Best delta F1 / FP / FN = {bestd.delta_test_f1_mean:+.6f} / {bestd.delta_test_fp_mean:+.2f} / {bestd.delta_test_fn_mean:+.2f}
- Leakage FAIL/WARNING: {bool(leak.leakage_fail.any())}/{bool(leak.leakage_warning.any())}
- Recommendation: {rec}
"""
        OUT["report"].write_text(report, encoding="utf-8")
        OUT["metadata"].write_text(json.dumps({"project_folder": "TeknoPatho", "step": "Step13C weighted ensemble AL_log refinement",
            "runtime_estimate": RUNTIME_ESTIMATE, "runtime_observed_seconds": time.perf_counter() - wall,
            "seed_list": SEEDS, "feature_configs": FEATURE_CONFIGS, "catboost_weights": WEIGHTS,
            "ensemble": "CatBoost+ExtraTrees", "threshold_policy": POLICY, "test_usage": "reporting_only",
            "weights_fixed_a_priori": True, "aa_biochemical_features_used": False,
            "final_recommendation": rec}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=== Step13C", mode, "complete ===")
    print(summary.sort_values("test_f1_mean", ascending=False)[["feature_config","catboost_weight","test_f1_mean","test_fp_mean","test_fn_mean","oof_f1_mean"]].head(12).to_string(index=False))
    print("leakage_fail", bool(leak.leakage_fail.any()), "leakage_warning", bool(leak.leakage_warning.any()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    run(args.mode)
