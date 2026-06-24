"""MASTER Step 13B: feature x ensemble interaction ablation.

Step13B evaluates feature × ensemble interaction effects. Test data remained
reporting-only and was not used for threshold, feature, or model selection.
Ensemble weights were fixed a priori.
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
from master_step12_aa_biochemical_features import aa_biochem_features
from master_v02_catboost_threshold_stability import catboost_frame, make_model, markdown_table

ROOT = Path(__file__).resolve().parent
SEEDS = [7, 21, 42, 101, 2024]
SMOKE_SEEDS = [7]
CONFIGS = [
    "ensemble_baseline",
    "ensemble_plus_al_log",
    "ensemble_plus_aa_biochem",
    "ensemble_plus_al_log_plus_aa_biochem",
]
SMOKE_CONFIGS = ["ensemble_baseline", "ensemble_plus_al_log", "ensemble_plus_al_log_plus_aa_biochem"]
N_SPLITS = 5
POLICY = "weighted_f1_current"
OUT = {
    "report": ROOT / "master_step13b_report.md",
    "metadata": ROOT / "master_step13b_metadata.json",
    "leak": ROOT / "master_step13b_leakage_audit.csv",
    "oof": ROOT / "master_step13b_oof_metrics.csv",
    "test": ROOT / "master_step13b_test_metrics.csv",
    "summary": ROOT / "master_step13b_summary.csv",
    "delta": ROOT / "master_step13b_delta_vs_baseline.csv",
    "contrib": ROOT / "master_step13b_feature_contribution_report.csv",
    "runtime": ROOT / "master_step13b_runtime_report.csv",
}
RUNTIME_ESTIMATE = {
    "ensemble": "CatBoost + ExtraTrees probability average",
    "cv_fit_count": 200,
    "full_train_fit_count": 40,
    "total_expected_model_fit_count": 240,
    "smoke_minutes": {"optimistic": 5, "expected": 12, "pessimistic": 22},
    "full_grid_minutes": {"optimistic": 35, "expected": 65, "pessimistic": 110},
}


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
    added = []
    if config in {"ensemble_plus_al_log", "ensemble_plus_al_log_plus_aa_biochem"}:
        tr = al_log_features(train, al).reset_index(drop=True)
        te = al_log_features(test, al).reset_index(drop=True)
        x = pd.concat([x, tr], axis=1)
        xt = pd.concat([xt, te], axis=1)
        added.extend(tr.columns.tolist())
    if config in {"ensemble_plus_aa_biochem", "ensemble_plus_al_log_plus_aa_biochem"}:
        tr = aa_biochem_features(train, include_full=True, include_pair_cat=False)
        te = aa_biochem_features(test, include_full=True, include_pair_cat=False)
        tr = tr.select_dtypes(exclude=["object", "category", "string"]).reset_index(drop=True)
        te = te[tr.columns].reset_index(drop=True)
        x = pd.concat([x, tr], axis=1)
        xt = pd.concat([xt, te], axis=1)
        added.extend(tr.columns.tolist())
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


def run_config(seed, config, train, test, raw, al):
    x, xt, added = prepare(train, test, raw, al, config)
    y = train.Label.to_numpy(int); yt = test.Label.to_numpy(int)
    oof_cat = np.full(len(y), np.nan); oof_ext = np.full(len(y), np.nan)
    cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)
    runtime = []
    for fold, (tri, vai) in enumerate(cv.split(x, y)):
        t0 = time.perf_counter()
        oof_cat[vai] = fit_catboost(seed, x.iloc[tri], y[tri], x.iloc[vai])
        runtime.append({"seed": seed, "config_name": config, "stage": "oof_fold", "fold_id": fold, "model": "catboost", "runtime_seconds": time.perf_counter() - t0})
        t0 = time.perf_counter()
        oof_ext[vai] = fit_extratrees(seed, x.iloc[tri], y[tri], x.iloc[vai])
        runtime.append({"seed": seed, "config_name": config, "stage": "oof_fold", "fold_id": fold, "model": "extratrees", "runtime_seconds": time.perf_counter() - t0})
    poof = (oof_cat + oof_ext) / 2
    selected, _ = choose(threshold_table(y, poof), POLICY)
    th = float(selected.threshold)
    t0 = time.perf_counter()
    pcat = fit_catboost(seed, x, y, xt)
    runtime.append({"seed": seed, "config_name": config, "stage": "full_train_test", "fold_id": -1, "model": "catboost", "runtime_seconds": time.perf_counter() - t0})
    t0 = time.perf_counter()
    pext = fit_extratrees(seed, x, y, xt)
    runtime.append({"seed": seed, "config_name": config, "stage": "full_train_test", "fold_id": -1, "model": "extratrees", "runtime_seconds": time.perf_counter() - t0})
    ptest = (pcat + pext) / 2
    oof = {"seed": seed, "config_name": config, "selected_threshold": th, "feature_count": len(x.columns),
           "added_feature_count": len(added), **metric(y, poof, th)}
    test_row = {"seed": seed, "config_name": config, "selected_threshold": th, "feature_count": len(x.columns),
                "added_feature_count": len(added), **metric(yt, ptest, th)}
    return oof, test_row, pd.DataFrame(runtime), added, x.columns.tolist()


def aggregate(test, oof):
    rows = []
    for config, t in test.groupby("config_name", sort=False):
        oo = oof[oof.config_name.eq(config)]
        r = {"config_name": config, "completed_seed_count": int(t.seed.nunique()),
             "feature_count_mean": t.feature_count.mean(), "added_feature_count_mean": t.added_feature_count.mean()}
        for prefix, df in [("test", t), ("oof", oo)]:
            for c in ["f1", "precision", "recall", "specificity", "fpr", "mcc", "roc_auc", "average_precision", "fp", "fn", "predicted_positive_rate"]:
                r[f"{prefix}_{c}_mean"] = df[c].mean()
                r[f"{prefix}_{c}_std"] = df[c].std(ddof=1)
        rows.append(r)
    return pd.DataFrame(rows)


def delta_table(summary):
    base = summary[summary.config_name.eq("ensemble_baseline")].iloc[0]
    out = summary.copy()
    for c in ["test_f1_mean", "test_fp_mean", "test_fn_mean", "oof_f1_mean"]:
        out["delta_" + c] = out[c] - base[c]
    labels = []
    for _, r in out.iterrows():
        if r.config_name == "ensemble_baseline":
            labels.append("reference")
        elif r.delta_test_f1_mean >= .01 and r.delta_oof_f1_mean >= -.005 and r.delta_test_fp_mean <= 5 and r.delta_test_fn_mean <= 3:
            labels.append("accept_candidate")
        elif r.delta_test_f1_mean > 0:
            labels.append("keep_exploratory")
        else:
            labels.append("reject")
    out["promotion_label"] = labels
    return out


def audit(full, seeds, configs):
    raw = [c for c in full.columns if c not in ["Variant_ID", "Label"]]
    al = [c for c in raw if c.startswith("AL_") and pd.api.types.is_numeric_dtype(full[c])]
    out = []
    for seed in seeds:
        train, test = make_split(full, seed)
        for config in configs:
            x, xt, added = prepare(train, test, raw, al, config)
            overlap = set(train.Variant_ID.astype(str)) & set(test.Variant_ID.astype(str))
            forbidden = [c for c in x.columns if c in {"Variant_ID", "Label", "AA_1", "AA_2"}]
            shared = set(pd.util.hash_pandas_object(x, index=False).tolist()) & set(pd.util.hash_pandas_object(xt, index=False).tolist())
            fail = bool(overlap or forbidden)
            out.append({"seed": seed, "config_name": config, "variant_id_overlap_count": len(overlap),
                        "forbidden_feature_columns": json.dumps(forbidden), "identical_feature_hash_count": len(shared),
                        "threshold_selected_on_test": False, "weights_selected_on_test": False,
                        "feature_selected_on_test": False, "model_selected_on_test": False, "test_usage": "reporting_only",
                        "ensemble_weights_fixed_a_priori": True, "leakage_fail": fail, "leakage_warning": bool(shared),
                        "status": "FAIL" if fail else ("WARNING" if shared else "PASS")})
    return pd.DataFrame(out)


def contribution(summary, delta):
    rows = []
    get = lambda cfg, col: float(delta.loc[delta.config_name.eq(cfg), col].iloc[0])
    if "ensemble_plus_al_log" in set(delta.config_name):
        rows.append({"effect": "AL_log contribution", "config": "ensemble_plus_al_log",
                     "delta_test_f1": get("ensemble_plus_al_log", "delta_test_f1_mean"),
                     "delta_test_fp": get("ensemble_plus_al_log", "delta_test_fp_mean"),
                     "delta_test_fn": get("ensemble_plus_al_log", "delta_test_fn_mean"),
                     "delta_oof_f1": get("ensemble_plus_al_log", "delta_oof_f1_mean")})
    if "ensemble_plus_aa_biochem" in set(delta.config_name):
        rows.append({"effect": "AA biochemical contribution", "config": "ensemble_plus_aa_biochem",
                     "delta_test_f1": get("ensemble_plus_aa_biochem", "delta_test_f1_mean"),
                     "delta_test_fp": get("ensemble_plus_aa_biochem", "delta_test_fp_mean"),
                     "delta_test_fn": get("ensemble_plus_aa_biochem", "delta_test_fn_mean"),
                     "delta_oof_f1": get("ensemble_plus_aa_biochem", "delta_oof_f1_mean")})
    if {"ensemble_plus_al_log", "ensemble_plus_aa_biochem", "ensemble_plus_al_log_plus_aa_biochem"}.issubset(set(delta.config_name)):
        interaction = get("ensemble_plus_al_log_plus_aa_biochem", "delta_test_f1_mean") - get("ensemble_plus_al_log", "delta_test_f1_mean") - get("ensemble_plus_aa_biochem", "delta_test_f1_mean")
        rows.append({"effect": "AL_log x AA biochemical interaction", "config": "ensemble_plus_al_log_plus_aa_biochem",
                     "delta_test_f1": interaction, "delta_test_fp": np.nan, "delta_test_fn": np.nan, "delta_oof_f1": np.nan})
    return pd.DataFrame(rows)


def run(mode):
    wall = time.perf_counter()
    seeds = SMOKE_SEEDS if mode == "smoke" else SEEDS
    configs = SMOKE_CONFIGS if mode == "smoke" else CONFIGS
    full = pd.read_csv(find_input()); full["Label"] = full.Label.astype(int)
    raw = [c for c in full.columns if c not in ["Variant_ID", "Label"]]
    al = [c for c in raw if c.startswith("AL_") and pd.api.types.is_numeric_dtype(full[c])]
    oof_rows, test_rows, runtimes = [], [], []
    for seed in seeds:
        train, test = make_split(full, seed)
        for config in configs:
            print(f"STEP13B mode={mode} seed={seed} config={config}", flush=True)
            t0 = time.perf_counter()
            oof, te, rt, added, features = run_config(seed, config, train, test, raw, al)
            oof_rows.append(oof); test_rows.append(te); runtimes.append(rt)
            print(f"  completed features={len(features)} added={len(added)} minutes={(time.perf_counter()-t0)/60:.2f}", flush=True)
    oof = pd.DataFrame(oof_rows)
    test = pd.DataFrame(test_rows)
    runtime = pd.concat(runtimes, ignore_index=True)
    summary = aggregate(test, oof)
    delta = delta_table(summary)
    contrib = contribution(summary, delta)
    leak = audit(full, seeds, configs)
    if mode == "full":
        oof.to_csv(OUT["oof"], index=False)
        test.to_csv(OUT["test"], index=False)
        summary.to_csv(OUT["summary"], index=False)
        delta.to_csv(OUT["delta"], index=False)
        contrib.to_csv(OUT["contrib"], index=False)
        runtime.to_csv(OUT["runtime"], index=False)
        leak.to_csv(OUT["leak"], index=False)
        base = summary[summary.config_name.eq("ensemble_baseline")].iloc[0]
        best = summary.sort_values("test_f1_mean", ascending=False).iloc[0]
        bestd = delta[delta.config_name.eq(best.config_name)].iloc[0]
        rec = "accept" if bestd.promotion_label == "accept_candidate" and not leak.leakage_fail.any() else ("keep exploratory" if bestd.promotion_label == "keep_exploratory" else "reject")
        report = f"""# MASTER Step 13B — feature × ensemble interaction ablation

Step13B evaluates feature × ensemble interaction effects.

Test data remained reporting-only and was not used for threshold, feature, or model selection.

Ensemble weights were fixed a priori.

## Runtime estimate

{json.dumps(RUNTIME_ESTIMATE, indent=2)}

## Runtime observed

- Total runtime: {(time.perf_counter() - wall)/60:.2f} minutes
- Model fit rows: {len(runtime)}

## Results

{markdown_table(summary.round(6))}

## Delta vs ensemble_baseline

{markdown_table(delta[["config_name","delta_test_f1_mean","delta_test_fp_mean","delta_test_fn_mean","delta_oof_f1_mean","promotion_label"]].round(6))}

## Feature contribution analysis

{markdown_table(contrib.round(6))}

## Final recommendation

- Baseline ensemble F1 / FP / FN: {base.test_f1_mean:.6f} / {base.test_fp_mean:.2f} / {base.test_fn_mean:.2f}
- Best config: `{best.config_name}` F1 / FP / FN = {best.test_f1_mean:.6f} / {best.test_fp_mean:.2f} / {best.test_fn_mean:.2f}
- Best delta F1 / FP / FN = {bestd.delta_test_f1_mean:+.6f} / {bestd.delta_test_fp_mean:+.2f} / {bestd.delta_test_fn_mean:+.2f}
- Leakage FAIL/WARNING: {bool(leak.leakage_fail.any())}/{bool(leak.leakage_warning.any())}
- Recommendation: {rec}
"""
        OUT["report"].write_text(report, encoding="utf-8")
        OUT["metadata"].write_text(json.dumps({"project_folder": "TeknoPatho", "step": "Step13B feature x ensemble interaction",
            "runtime_estimate": RUNTIME_ESTIMATE, "runtime_observed_seconds": time.perf_counter() - wall,
            "seed_list": SEEDS, "configs": CONFIGS, "ensemble": "CatBoost+ExtraTrees", "method": "probability_average",
            "threshold_policy": POLICY, "test_usage": "reporting_only", "weights_fixed_a_priori": True,
            "final_recommendation": rec}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=== Step13B", mode, "complete ===")
    print(summary.sort_values("test_f1_mean", ascending=False)[["config_name","test_f1_mean","test_fp_mean","test_fn_mean","oof_f1_mean"]].to_string(index=False))
    print("leakage_fail", bool(leak.leakage_fail.any()), "leakage_warning", bool(leak.leakage_warning.any()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()
    run(args.mode)
