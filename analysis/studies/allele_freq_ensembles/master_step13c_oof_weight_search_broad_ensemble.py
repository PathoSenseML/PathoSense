"""MASTER Step 13C: OOF-only broad ensemble weight search.

Step13C evaluates OOF-only broad ensemble weight search. Ensemble weights were
selected using OOF predictions only. The exact prior successful weights were
unknown, so old-inspired weights were reported only as references, not forced
as the main experiment. AL_log is tested only as a preregistered
feature-family hypothesis. Test data remained reporting-only and was not used
for threshold, feature, weight, or model selection. AA biochemical features
were excluded because Step12, Step12B, and Step13B did not support them.
"""
from __future__ import annotations

import argparse
import itertools
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
from master_step4_aa_al_ablation import al_log_features, transform_config
from master_step7_preprocessing_missingness_ablation import missingness_features
from master_step8_threshold_policy_refinement import choose, threshold_table
from master_v02_catboost_threshold_stability import catboost_frame, make_model, markdown_table

ROOT = Path(__file__).resolve().parent
SEEDS = [7, 21, 42]
SMOKE_SEEDS = [7]
FEATURE_CONFIGS = ["baseline_features", "baseline_plus_al_log"]
MODELS = ["catboost", "lightgbm", "xgboost", "extratrees"]
N_SPLITS = 5
POLICY = "weighted_f1_current"
REFERENCE_VALUES = {
    "catboost_only_baseline_f1": 0.548192,
    "catboost_extratrees_equal_baseline_f1": 0.551601,
    "broad_equal_step13a_f1": 0.554365,
}
REFERENCE_WEIGHTS = {
    "R0_catboost_only": (1.0, 0.0, 0.0, 0.0),
    "R1_equal_broad": (0.25, 0.25, 0.25, 0.25),
    "R2_catboost_extratrees_equal": (0.50, 0.0, 0.0, 0.50),
    "R3_extratrees_heavy_reference": (0.30, 0.10, 0.20, 0.40),
}
OUT = {
    "report": ROOT / "master_step13c_report.md",
    "metadata": ROOT / "master_step13c_metadata.json",
    "leak": ROOT / "master_step13c_leakage_audit.csv",
    "oof": ROOT / "master_step13c_oof_metrics.csv",
    "test": ROOT / "master_step13c_test_metrics.csv",
    "grid": ROOT / "master_step13c_weight_grid_oof_results.csv",
    "selected": ROOT / "master_step13c_selected_weights_by_seed.csv",
    "summary": ROOT / "master_step13c_summary.csv",
    "delta": ROOT / "master_step13c_delta_vs_references.csv",
    "al": ROOT / "master_step13c_al_log_effect.csv",
    "runtime": ROOT / "master_step13c_runtime_report.csv",
}


def simplex_grid(step=0.1):
    units = int(round(1 / step))
    rows = []
    for a in range(units + 1):
        for b in range(units - a + 1):
            for c in range(units - a - b + 1):
                d = units - a - b - c
                rows.append((a / units, b / units, c / units, d / units))
    return rows


WEIGHT_GRID = simplex_grid(0.1)
RUNTIME_ESTIMATE = {
    "base_model_fit_count": 144,
    "oof_weight_combinations": len(WEIGHT_GRID),
    "feature_config_count": 2,
    "smoke_weight_candidate_evaluations": 2 * len(WEIGHT_GRID),
    "full_weight_candidate_evaluations": 2 * 3 * len(WEIGHT_GRID),
    "smoke_minutes": {"optimistic": 8, "expected": 20, "pessimistic": 40},
    "full_grid_minutes": {"optimistic": 30, "expected": 55, "pessimistic": 105},
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
    cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)
    oof = {m: np.full(len(y), np.nan) for m in MODELS}
    testp = {}
    runtime = []
    for fold, (tri, vai) in enumerate(cv.split(x, y)):
        for m in MODELS:
            t0 = time.perf_counter()
            oof[m][vai] = fit_predict(m, seed, x.iloc[tri], y[tri], x.iloc[vai])
            runtime.append({"seed": seed, "feature_config": feature_config, "stage": "oof_fold", "fold_id": fold,
                            "model": m, "runtime_seconds": time.perf_counter() - t0})
    for m in MODELS:
        t0 = time.perf_counter()
        testp[m] = fit_predict(m, seed, x, y, xt)
        runtime.append({"seed": seed, "feature_config": feature_config, "stage": "full_train_test", "fold_id": -1,
                        "model": m, "runtime_seconds": time.perf_counter() - t0})
    return oof, testp, len(x.columns), len(added), pd.DataFrame(runtime)


def combine(preds, weights):
    return sum(weights[i] * preds[m] for i, m in enumerate(MODELS))


def evaluate_grid(seed, feature_config, oofp, testp, y, yt):
    grid_rows = []
    best = None
    for w in WEIGHT_GRID:
        prob = combine(oofp, w)
        selected, _ = choose(threshold_table(y, prob), POLICY)
        th = float(selected.threshold)
        m = metric(y, prob, th)
        nonzero = int(sum(v > 0 for v in w))
        row = {"seed": seed, "feature_config": feature_config, "catboost_weight": w[0], "lightgbm_weight": w[1],
               "xgboost_weight": w[2], "extratrees_weight": w[3], "selected_threshold": th,
               "nonzero_model_count": nonzero, **m}
        grid_rows.append(row)
        key = (m["f1"], m["mcc"], -m["predicted_positive_rate"], -nonzero)
        if best is None or key > best[0]:
            best = (key, w, th, m)
    _, w, th, moof = best
    ptest = combine(testp, w)
    mtest = metric(yt, ptest, th)
    base = {"seed": seed, "feature_config": feature_config, "selection_type": "oof_selected",
            "catboost_weight": w[0], "lightgbm_weight": w[1], "xgboost_weight": w[2], "extratrees_weight": w[3],
            "selected_threshold": th, "nonzero_model_count": int(sum(v > 0 for v in w))}
    oof_rows = [{**base, **moof}]
    test_rows = [{**base, **mtest}]
    for name, rw in REFERENCE_WEIGHTS.items():
        prob = combine(oofp, rw)
        selected, _ = choose(threshold_table(y, prob), POLICY)
        rth = float(selected.threshold)
        ro = metric(y, prob, rth)
        rt = metric(yt, combine(testp, rw), rth)
        b = {"seed": seed, "feature_config": feature_config, "selection_type": name,
             "catboost_weight": rw[0], "lightgbm_weight": rw[1], "xgboost_weight": rw[2], "extratrees_weight": rw[3],
             "selected_threshold": rth, "nonzero_model_count": int(sum(v > 0 for v in rw))}
        oof_rows.append({**b, **ro}); test_rows.append({**b, **rt})
    return pd.DataFrame(grid_rows), oof_rows, test_rows


def aggregate(test, oof):
    rows = []
    for (cfg, sel), t in test.groupby(["feature_config", "selection_type"], sort=False):
        oo = oof[(oof.feature_config.eq(cfg)) & (oof.selection_type.eq(sel))]
        r = {"feature_config": cfg, "selection_type": sel, "completed_seed_count": int(t.seed.nunique())}
        for c in ["catboost_weight", "lightgbm_weight", "xgboost_weight", "extratrees_weight", "nonzero_model_count"]:
            r[f"{c}_mean"] = t[c].mean(); r[f"{c}_std"] = t[c].std(ddof=1)
        for prefix, df in [("test", t), ("oof", oo)]:
            for c in ["f1", "precision", "recall", "specificity", "fpr", "mcc", "roc_auc", "average_precision", "fp", "fn", "predicted_positive_rate", "selected_threshold"]:
                r[f"{prefix}_{c}_mean"] = df[c].mean()
                r[f"{prefix}_{c}_std"] = df[c].std(ddof=1)
        rows.append(r)
    return pd.DataFrame(rows)


def delta_refs(summary):
    out = summary.copy()
    out["delta_vs_catboost_only_f1"] = out.test_f1_mean - REFERENCE_VALUES["catboost_only_baseline_f1"]
    out["delta_vs_catboost_extratrees_equal_f1"] = out.test_f1_mean - REFERENCE_VALUES["catboost_extratrees_equal_baseline_f1"]
    out["delta_vs_broad_equal_step13a_f1"] = out.test_f1_mean - REFERENCE_VALUES["broad_equal_step13a_f1"]
    ref = out[(out.feature_config.eq("baseline_features")) & (out.selection_type.eq("R0_catboost_only"))]
    b = ref.iloc[0] if len(ref) else out.iloc[0]
    out["delta_oof_f1_vs_catboost_reference"] = out.oof_f1_mean - b.oof_f1_mean
    out["delta_fp_vs_catboost_reference"] = out.test_fp_mean - b.test_fp_mean
    out["delta_fn_vs_catboost_reference"] = out.test_fn_mean - b.test_fn_mean
    out["weight_stability_l1"] = out[["catboost_weight_std", "lightgbm_weight_std", "xgboost_weight_std", "extratrees_weight_std"]].fillna(0).sum(axis=1)
    labels = []
    for _, r in out.iterrows():
        if r.feature_config == "baseline_features" and r.selection_type == "R0_catboost_only":
            labels.append("reference")
        elif (r.delta_vs_catboost_only_f1 >= .01 and r.delta_vs_catboost_extratrees_equal_f1 >= .005
              and r.delta_oof_f1_vs_catboost_reference >= -.005 and r.delta_fp_vs_catboost_reference <= 5
              and r.delta_fn_vs_catboost_reference <= 3 and r.weight_stability_l1 <= .80):
            labels.append("accept_candidate")
        elif r.delta_vs_catboost_only_f1 > 0:
            labels.append("keep_exploratory")
        else:
            labels.append("reject")
    out["promotion_label"] = labels
    return out


def al_effect(summary):
    rows = []
    for sel in summary.selection_type.unique():
        b = summary[(summary.feature_config.eq("baseline_features")) & (summary.selection_type.eq(sel))]
        a = summary[(summary.feature_config.eq("baseline_plus_al_log")) & (summary.selection_type.eq(sel))]
        if len(b) and len(a):
            b = b.iloc[0]; a = a.iloc[0]
            rows.append({"selection_type": sel, "delta_al_log_test_f1": a.test_f1_mean - b.test_f1_mean,
                         "delta_al_log_test_fp": a.test_fp_mean - b.test_fp_mean,
                         "delta_al_log_test_fn": a.test_fn_mean - b.test_fn_mean,
                         "delta_al_log_oof_f1": a.oof_f1_mean - b.oof_f1_mean})
    return pd.DataFrame(rows)


def audit(full, seeds):
    raw = [c for c in full.columns if c not in ["Variant_ID", "Label"]]
    al = [c for c in raw if c.startswith("AL_") and pd.api.types.is_numeric_dtype(full[c])]
    out = []
    for seed in seeds:
        train, test = make_split(full, seed)
        for cfg in FEATURE_CONFIGS:
            x, xt, _ = prepare(train, test, raw, al, cfg)
            overlap = set(train.Variant_ID.astype(str)) & set(test.Variant_ID.astype(str))
            forbidden = [c for c in x.columns if c in {"Variant_ID", "Label", "AA_1", "AA_2"}]
            shared = set(pd.util.hash_pandas_object(x, index=False).tolist()) & set(pd.util.hash_pandas_object(xt, index=False).tolist())
            fail = bool(overlap or forbidden)
            out.append({"seed": seed, "feature_config": cfg, "variant_id_overlap_count": len(overlap),
                        "forbidden_feature_columns": json.dumps(forbidden), "identical_feature_hash_count": len(shared),
                        "threshold_selected_on_test": False, "weights_selected_on_test": False,
                        "feature_selected_on_test": False, "model_selected_on_test": False, "test_usage": "reporting_only",
                        "weights_selected_using_oof_only": True, "aa_biochemical_features_used": False,
                        "leakage_fail": fail, "leakage_warning": bool(shared),
                        "status": "FAIL" if fail else ("WARNING" if shared else "PASS")})
    return pd.DataFrame(out)


def run(mode):
    wall = time.perf_counter()
    seeds = SMOKE_SEEDS if mode == "smoke" else SEEDS
    full = pd.read_csv(find_input()); full["Label"] = full.Label.astype(int)
    raw = [c for c in full.columns if c not in ["Variant_ID", "Label"]]
    al = [c for c in raw if c.startswith("AL_") and pd.api.types.is_numeric_dtype(full[c])]
    grid_parts, oof_rows, test_rows, runtimes = [], [], [], []
    for seed in seeds:
        train, test = make_split(full, seed)
        y = train.Label.to_numpy(int); yt = test.Label.to_numpy(int)
        for cfg in FEATURE_CONFIGS:
            print(f"STEP13C OOF-search mode={mode} seed={seed} feature_config={cfg}", flush=True)
            t0 = time.perf_counter()
            oofp, testp, nfeat, nadd, rt = source_predictions(seed, cfg, train, test, raw, al)
            grid, oo, te = evaluate_grid(seed, cfg, oofp, testp, y, yt)
            for d in oo + te:
                d["feature_count"] = nfeat; d["added_feature_count"] = nadd
            grid["feature_count"] = nfeat; grid["added_feature_count"] = nadd
            grid_parts.append(grid); oof_rows.extend(oo); test_rows.extend(te); runtimes.append(rt)
            print(f"  completed grid={len(WEIGHT_GRID)} features={nfeat} minutes={(time.perf_counter()-t0)/60:.2f}", flush=True)
    grid = pd.concat(grid_parts, ignore_index=True)
    oof = pd.DataFrame(oof_rows); test = pd.DataFrame(test_rows)
    runtime = pd.concat(runtimes, ignore_index=True)
    summary = aggregate(test, oof)
    delta = delta_refs(summary)
    aldf = al_effect(summary)
    leak = audit(full, seeds)
    selected = oof[oof.selection_type.eq("oof_selected")][["seed", "feature_config", "catboost_weight", "lightgbm_weight", "xgboost_weight", "extratrees_weight", "selected_threshold", "f1", "mcc", "predicted_positive_rate"]]
    if mode == "full":
        oof.to_csv(OUT["oof"], index=False)
        test.to_csv(OUT["test"], index=False)
        grid.to_csv(OUT["grid"], index=False)
        selected.to_csv(OUT["selected"], index=False)
        summary.to_csv(OUT["summary"], index=False)
        delta.to_csv(OUT["delta"], index=False)
        aldf.to_csv(OUT["al"], index=False)
        runtime.to_csv(OUT["runtime"], index=False)
        leak.to_csv(OUT["leak"], index=False)
        best = summary.sort_values("test_f1_mean", ascending=False).iloc[0]
        bestd = delta[(delta.feature_config.eq(best.feature_config)) & (delta.selection_type.eq(best.selection_type))].iloc[0]
        rec = "accept" if bestd.promotion_label == "accept_candidate" and not leak.leakage_fail.any() else ("keep exploratory" if bestd.promotion_label == "keep_exploratory" else "reject")
        report = f"""# MASTER Step 13C — OOF-only broad ensemble weight search

Step13C evaluates OOF-only broad ensemble weight search.

Ensemble weights were selected using OOF predictions only.

The exact prior successful weights were unknown, so old-inspired weights were reported only as references, not forced as the main experiment.

AL_log is tested only as a preregistered feature-family hypothesis.

Test data remained reporting-only and was not used for threshold, feature, weight, or model selection.

AA biochemical features were excluded because Step12, Step12B, and Step13B did not support them.

## Runtime estimate

{json.dumps(RUNTIME_ESTIMATE, indent=2)}

## Runtime observed

- Total runtime: {(time.perf_counter() - wall)/60:.2f} minutes
- Model fit rows: {len(runtime)}
- OOF weight grid rows: {len(grid)}

## Top results

{markdown_table(summary.sort_values("test_f1_mean", ascending=False).head(12).round(6))}

## Delta vs references

{markdown_table(delta.sort_values("test_f1_mean", ascending=False)[["feature_config","selection_type","delta_vs_catboost_only_f1","delta_vs_catboost_extratrees_equal_f1","delta_vs_broad_equal_step13a_f1","delta_oof_f1_vs_catboost_reference","delta_fp_vs_catboost_reference","delta_fn_vs_catboost_reference","weight_stability_l1","promotion_label"]].round(6))}

## AL_log effect

{markdown_table(aldf.round(6))}

## Final recommendation

- Best config: `{best.feature_config}` / `{best.selection_type}` F1 / FP / FN = {best.test_f1_mean:.6f} / {best.test_fp_mean:.2f} / {best.test_fn_mean:.2f}
- Best deltas vs references: CatBoost-only {bestd.delta_vs_catboost_only_f1:+.6f}; CatBoost+ExtraTrees equal {bestd.delta_vs_catboost_extratrees_equal_f1:+.6f}; broad equal Step13A {bestd.delta_vs_broad_equal_step13a_f1:+.6f}
- Leakage FAIL/WARNING: {bool(leak.leakage_fail.any())}/{bool(leak.leakage_warning.any())}
- Recommendation: {rec}
"""
        OUT["report"].write_text(report, encoding="utf-8")
        OUT["metadata"].write_text(json.dumps({"project_folder": "TeknoPatho", "step": "Step13C OOF-only broad ensemble weight search",
            "runtime_estimate": RUNTIME_ESTIMATE, "runtime_observed_seconds": time.perf_counter() - wall,
            "seed_list": SEEDS, "feature_configs": FEATURE_CONFIGS, "models": MODELS,
            "weight_grid_step": 0.1, "weight_grid_size": len(WEIGHT_GRID), "threshold_policy": POLICY,
            "test_usage": "reporting_only", "weights_selected_using_oof_only": True,
            "aa_biochemical_features_used": False, "reference_weights": REFERENCE_WEIGHTS,
            "final_recommendation": rec}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=== Step13C OOF-search", mode, "complete ===")
    print(summary.sort_values("test_f1_mean", ascending=False)[["feature_config","selection_type","test_f1_mean","test_fp_mean","test_fn_mean","oof_f1_mean","catboost_weight_mean","lightgbm_weight_mean","xgboost_weight_mean","extratrees_weight_mean"]].head(12).to_string(index=False))
    print("leakage_fail", bool(leak.leakage_fail.any()), "leakage_warning", bool(leak.leakage_warning.any()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    run(args.mode)
