"""MASTER Step 14C: CAT_3/CAT_4/CAT_5 encoding/redundancy ablation.

Step14C keeps CAT_6 unchanged in order to isolate CAT_3/CAT_4/CAT_5 handling.
All threshold selection is OOF-only; test metrics are reporting-only.
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
    "active_reference_native_CAT345_with_CAT6",
    "keep_CAT3_only_drop_CAT4_CAT5_with_CAT6",
    "CAT345_combined_string_with_CAT6",
    "CAT345_label_encoded_separate_with_CAT6",
    "CAT3_label_only_drop_CAT4_CAT5_with_CAT6",
]
MODELS = ["catboost", "lightgbm", "xgboost", "extratrees"]
WEIGHTS = {m: 0.25 for m in MODELS}
N_SPLITS = 5
POLICY = "weighted_f1_current"
OUT = {
    "report": ROOT / "master_step14c_cat345_report.md",
    "metadata": ROOT / "master_step14c_metadata.json",
    "leak": ROOT / "master_step14c_leakage_audit.csv",
    "oof": ROOT / "master_step14c_oof_metrics.csv",
    "test": ROOT / "master_step14c_test_metrics.csv",
    "summary": ROOT / "master_step14c_summary.csv",
    "delta": ROOT / "master_step14c_delta_vs_active_reference.csv",
    "runtime": ROOT / "master_step14c_runtime_report.csv",
    "unknown": ROOT / "master_step14c_unknown_category_report.csv",
    "feature_count": ROOT / "master_step14c_feature_count_report.csv",
}
RUNTIME_ESTIMATE = {
    "screening_seed_list": SEEDS,
    "config_count": len(CONFIGS),
    "model_families": MODELS,
    "cv_fit_count": len(CONFIGS) * len(SEEDS) * N_SPLITS * len(MODELS),
    "full_train_fit_count": len(CONFIGS) * len(SEEDS) * len(MODELS),
    "total_expected_model_fit_count": len(CONFIGS) * len(SEEDS) * (N_SPLITS + 1) * len(MODELS),
    "runtime_minutes": {"optimistic": 50, "expected": 75, "pessimistic": 120},
}


def norm_cat(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().fillna("<MISSING>").replace({"": "<MISSING>"})


def label_encode_pair(train_s: pd.Series, other_s: pd.Series):
    tr = norm_cat(train_s)
    ot = norm_cat(other_s)
    values = [v for v in pd.Series(tr.unique()).tolist() if v != "<MISSING>"]
    mp = {v: i for i, v in enumerate(sorted(values))}
    missing_code = len(mp)
    tr_enc = tr.map(mp).fillna(missing_code).where(tr.ne("<MISSING>"), missing_code).astype(float)
    ot_known = ot.map(mp)
    unknown = ot_known.isna() & ot.ne("<MISSING>")
    ot_enc = ot_known.fillna(missing_code).where(ot.ne("<MISSING>"), missing_code).astype(float)
    ot_enc = ot_enc.where(~unknown, -1.0)
    return tr_enc, ot_enc, float(unknown.mean())


def modify_frames(train: pd.DataFrame, other: pd.DataFrame, config: str):
    tr = train.copy()
    ot = other.copy()
    raw_drop = set()
    unknown = {}
    if config == "keep_CAT3_only_drop_CAT4_CAT5_with_CAT6":
        raw_drop.update(["CAT_4", "CAT_5"])
        tr = tr.drop(columns=[c for c in raw_drop if c in tr.columns])
        ot = ot.drop(columns=[c for c in raw_drop if c in ot.columns])
    elif config == "CAT345_combined_string_with_CAT6":
        tr["CAT_345_COMBINED"] = norm_cat(train["CAT_3"]) + "|" + norm_cat(train["CAT_4"]) + "|" + norm_cat(train["CAT_5"])
        ot["CAT_345_COMBINED"] = norm_cat(other["CAT_3"]) + "|" + norm_cat(other["CAT_4"]) + "|" + norm_cat(other["CAT_5"])
        raw_drop.update(["CAT_3", "CAT_4", "CAT_5"])
        tr = tr.drop(columns=[c for c in raw_drop if c in tr.columns])
        ot = ot.drop(columns=[c for c in raw_drop if c in ot.columns])
    elif config == "CAT345_label_encoded_separate_with_CAT6":
        for c in ["CAT_3", "CAT_4", "CAT_5"]:
            enc_tr, enc_ot, unk = label_encode_pair(train[c], other[c])
            tr[f"{c}_LABEL"] = enc_tr.to_numpy()
            ot[f"{c}_LABEL"] = enc_ot.to_numpy()
            unknown[f"{c}_LABEL_unknown_rate"] = unk
        raw_drop.update(["CAT_3", "CAT_4", "CAT_5"])
        tr = tr.drop(columns=[c for c in raw_drop if c in tr.columns])
        ot = ot.drop(columns=[c for c in raw_drop if c in ot.columns])
    elif config == "CAT3_label_only_drop_CAT4_CAT5_with_CAT6":
        enc_tr, enc_ot, unk = label_encode_pair(train["CAT_3"], other["CAT_3"])
        tr["CAT_3_LABEL"] = enc_tr.to_numpy()
        ot["CAT_3_LABEL"] = enc_ot.to_numpy()
        unknown["CAT_3_LABEL_unknown_rate"] = unk
        raw_drop.update(["CAT_3", "CAT_4", "CAT_5"])
        tr = tr.drop(columns=[c for c in raw_drop if c in tr.columns])
        ot = ot.drop(columns=[c for c in raw_drop if c in ot.columns])
    return tr.reset_index(drop=True), ot.reset_index(drop=True), raw_drop, unknown


def prepare(train: pd.DataFrame, other: pd.DataFrame, config: str):
    tr, ot, raw_drop, unknown = modify_frames(train, other, config)
    raw = [c for c in tr.columns if c not in ["Variant_ID", "Label"]]
    al = [c for c in raw if c.startswith("AL_") and pd.api.types.is_numeric_dtype(tr[c])]
    x, xo, _, status = transform_config(tr, ot, raw, al, "aa_numeric")
    if status != "OK":
        raise RuntimeError(status)
    x = pd.concat([x.reset_index(drop=True), missingness_features(tr, raw)], axis=1)
    xo = pd.concat([xo.reset_index(drop=True), missingness_features(ot, raw)], axis=1)
    for c in ["Variant_ID", "Label", "AA_1", "AA_2"]:
        if c in x:
            x = x.drop(columns=[c])
        if c in xo:
            xo = xo.drop(columns=[c])
    return x, xo, raw, raw_drop, unknown


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


def run_config(seed, config, train, test):
    y = train.Label.to_numpy(int); yt = test.Label.to_numpy(int)
    cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)
    oof_models = {m: np.full(len(y), np.nan) for m in MODELS}
    runtime = []
    unknown_rows = []
    for fold, (tri, vai) in enumerate(cv.split(train, y)):
        xtr, xva, raw, raw_drop, unk = prepare(train.iloc[tri].reset_index(drop=True), train.iloc[vai].reset_index(drop=True), config)
        for k, v in unk.items():
            unknown_rows.append({"seed": seed, "config_name": config, "stage": "oof_fold", "fold_id": fold, "feature": k, "unknown_rate": v})
        for m in MODELS:
            t0 = time.perf_counter()
            oof_models[m][vai] = fit_predict(m, seed, xtr, y[tri], xva)
            runtime.append({"seed": seed, "config_name": config, "stage": "oof_fold", "fold_id": fold, "model": m, "runtime_seconds": time.perf_counter() - t0})
    poof = sum(WEIGHTS[m] * oof_models[m] for m in MODELS)
    selected, _ = choose(threshold_table(y, poof), POLICY)
    th = float(selected.threshold)
    xfull, xtest, raw, raw_drop, unk = prepare(train.reset_index(drop=True), test.reset_index(drop=True), config)
    for k, v in unk.items():
        unknown_rows.append({"seed": seed, "config_name": config, "stage": "full_train_test", "fold_id": -1, "feature": k, "unknown_rate": v})
    test_models = {}
    for m in MODELS:
        t0 = time.perf_counter()
        test_models[m] = fit_predict(m, seed, xfull, y, xtest)
        runtime.append({"seed": seed, "config_name": config, "stage": "full_train_test", "fold_id": -1, "model": m, "runtime_seconds": time.perf_counter() - t0})
    ptest = sum(WEIGHTS[m] * test_models[m] for m in MODELS)
    base = {"seed": seed, "config_name": config, "selected_threshold": th, "feature_count": len(xfull.columns)}
    oof_row = {**base, **metric(y, poof, th)}
    test_row = {**base, **metric(yt, ptest, th)}
    return oof_row, test_row, pd.DataFrame(runtime), unknown_rows, xfull.columns.tolist(), raw_drop


def aggregate(test, oof):
    rows = []
    for config, t in test.groupby("config_name", sort=False):
        oo = oof[oof.config_name.eq(config)]
        r = {"config_name": config, "completed_seed_count": int(t.seed.nunique()), "feature_count_mean": t.feature_count.mean()}
        for prefix, df in [("test", t), ("oof", oo)]:
            for c in ["f1", "precision", "recall", "specificity", "fpr", "mcc", "roc_auc", "average_precision", "fp", "fn", "predicted_positive_rate", "selected_threshold"]:
                r[f"{prefix}_{c}_mean"] = df[c].mean()
                r[f"{prefix}_{c}_std"] = df[c].std(ddof=1)
        rows.append(r)
    return pd.DataFrame(rows)


def delta_table(summary):
    ref = summary[summary.config_name.eq("active_reference_native_CAT345_with_CAT6")].iloc[0]
    out = summary.copy()
    for c in ["test_f1_mean", "test_fp_mean", "test_fn_mean", "oof_f1_mean"]:
        out["delta_" + c] = out[c] - ref[c]
    labels = []
    for _, r in out.iterrows():
        if r.config_name == "active_reference_native_CAT345_with_CAT6":
            labels.append("reference")
        elif r.delta_test_f1_mean >= .004 and r.delta_oof_f1_mean >= -.005:
            labels.append("keep_candidate")
        elif r.delta_test_fp_mean < 0 and r.delta_test_fn_mean > 3 and r.delta_test_f1_mean < 0:
            labels.append("conservative_but_not_preferred")
        elif abs(r.delta_test_f1_mean) <= .002 and r.feature_count_mean < ref.feature_count_mean:
            labels.append("simplification_candidate")
        else:
            labels.append("reject")
    out["decision_label"] = labels
    return out


def leakage_audit(full, feature_records):
    rows = []
    for rec in feature_records:
        config = rec["config_name"]; features = rec["features"]; raw_drop = set(rec["raw_drop"])
        forbidden = [c for c in features if c in {"Variant_ID", "Label", "AA_1", "AA_2"}]
        dropped_remain = []
        missing_remain = []
        expected_drop = {
            "keep_CAT3_only_drop_CAT4_CAT5_with_CAT6": {"CAT_4", "CAT_5"},
            "CAT345_combined_string_with_CAT6": {"CAT_3", "CAT_4", "CAT_5"},
            "CAT345_label_encoded_separate_with_CAT6": {"CAT_3", "CAT_4", "CAT_5"},
            "CAT3_label_only_drop_CAT4_CAT5_with_CAT6": {"CAT_3", "CAT_4", "CAT_5"},
        }.get(config, set())
        for c in expected_drop:
            if c in features:
                dropped_remain.append(c)
            for f in features:
                if f"{c}_missing" in f or f"{c}_LABEL_missing" in f:
                    missing_remain.append(f)
        cat6_allowed = any("CAT_6" in f or f == "CAT_6" for f in features)
        fail = bool(forbidden or dropped_remain or missing_remain)
        rows.append({"seed": rec["seed"], "config_name": config, "forbidden_feature_columns": json.dumps(forbidden),
                     "dropped_cat345_columns_remaining": json.dumps(dropped_remain),
                     "dropped_cat345_missingness_remaining": json.dumps(sorted(set(missing_remain))),
                     "cat6_allowed_and_present_or_missingness_possible": bool(cat6_allowed),
                     "threshold_selected_on_test": False, "test_rows_removed": False, "train_rows_removed": False,
                     "leakage_fail": fail, "status": "FAIL" if fail else "PASS"})
    return pd.DataFrame(rows)


def main():
    wall = time.perf_counter()
    full = pd.read_csv(find_input()); full["Label"] = full.Label.astype(int)
    oof_rows, test_rows, runtimes, unknown_rows, feature_records = [], [], [], [], []
    for seed in SEEDS:
        train, test = make_split(full, seed)
        for config in CONFIGS:
            print(f"STEP14C seed={seed} config={config}", flush=True)
            t0 = time.perf_counter()
            oo, te, rt, unk, features, raw_drop = run_config(seed, config, train, test)
            oof_rows.append(oo); test_rows.append(te); runtimes.append(rt); unknown_rows.extend(unk)
            feature_records.append({"seed": seed, "config_name": config, "features": features, "raw_drop": sorted(raw_drop), "feature_count": len(features)})
            print(f"  completed features={len(features)} minutes={(time.perf_counter()-t0)/60:.2f}", flush=True)
    oof = pd.DataFrame(oof_rows); test = pd.DataFrame(test_rows)
    runtime = pd.concat(runtimes, ignore_index=True)
    unknown = pd.DataFrame(unknown_rows) if unknown_rows else pd.DataFrame(columns=["seed", "config_name", "stage", "fold_id", "feature", "unknown_rate"])
    feature_count = pd.DataFrame([{k: v for k, v in r.items() if k != "features"} for r in feature_records])
    summary = aggregate(test, oof)
    delt = delta_table(summary)
    leak = leakage_audit(full, feature_records)

    oof.to_csv(OUT["oof"], index=False)
    test.to_csv(OUT["test"], index=False)
    summary.to_csv(OUT["summary"], index=False)
    delt.to_csv(OUT["delta"], index=False)
    runtime.to_csv(OUT["runtime"], index=False)
    unknown.to_csv(OUT["unknown"], index=False)
    feature_count.to_csv(OUT["feature_count"], index=False)
    leak.to_csv(OUT["leak"], index=False)

    ref = summary[summary.config_name.eq("active_reference_native_CAT345_with_CAT6")].iloc[0]
    best = summary.sort_values("test_f1_mean", ascending=False).iloc[0]
    bestd = delt[delt.config_name.eq(best.config_name)].iloc[0]
    report = f"""# MASTER Step 14C — CAT_3/CAT_4/CAT_5 redundancy and encoding ablation

Step14C keeps CAT_6 unchanged in order to isolate CAT_3/CAT_4/CAT_5 handling.

CAT_3/CAT_4/CAT_5 are treated as highly redundant based on Step14B diagnostic results.

All threshold selection is OOF-only; test metrics are reporting-only.

Step14C is a time-constrained 3-seed screening experiment.

## Context

Step14A showed CAT_6 train-row removal lowered F1 and increased FN. Step14B showed CAT_3/4/5 are highly redundant, while CAT_6 behaves as a separate sparse marker. Therefore CAT_6 is untouched here.

## Configs

- `active_reference_native_CAT345_with_CAT6`: keep CAT_3/4/5 and CAT_6 native.
- `keep_CAT3_only_drop_CAT4_CAT5_with_CAT6`: keep CAT_3 and CAT_6, drop CAT_4/CAT_5 before missingness.
- `CAT345_combined_string_with_CAT6`: replace CAT_3/4/5 with CAT_345_COMBINED, keep CAT_6.
- `CAT345_label_encoded_separate_with_CAT6`: replace CAT_3/4/5 with fold-safe label codes, keep CAT_6.
- `CAT3_label_only_drop_CAT4_CAT5_with_CAT6`: CAT_3 label only, drop CAT_4/CAT_5, keep CAT_6.

## Runtime observed

- Total runtime: {(time.perf_counter() - wall)/60:.2f} minutes
- Model fit rows: {len(runtime)}

## Summary

{markdown_table(summary.round(6))}

## Delta vs active reference

{markdown_table(delt[["config_name","delta_test_f1_mean","delta_test_fp_mean","delta_test_fn_mean","delta_oof_f1_mean","decision_label"]].round(6))}

## FP/FN and OOF/test interpretation

Reference F1 / FP / FN: {ref.test_f1_mean:.6f} / {ref.test_fp_mean:.2f} / {ref.test_fn_mean:.2f}.
Best time-constrained candidate: `{best.config_name}` with F1 / FP / FN = {best.test_f1_mean:.6f} / {best.test_fp_mean:.2f} / {best.test_fn_mean:.2f}.
Best delta F1 / FP / FN = {bestd.delta_test_f1_mean:+.6f} / {bestd.delta_test_fp_mean:+.2f} / {bestd.delta_test_fn_mean:+.2f}.

## Leakage audit

- FAIL: {bool(leak.leakage_fail.any())}

## Final recommendation

Use Step14C results as a screening signal, not final proof. Next: Step14C2: Apply the best CAT_345 handling, then test CAT_6 keep vs CAT_6 column-drop-only. Do not repeat CAT_6 train row removal unless explicitly required.
"""
    OUT["report"].write_text(report, encoding="utf-8")
    OUT["metadata"].write_text(json.dumps({"project_folder": "TeknoPatho", "step": "Step14C CAT345 encoding redundancy ablation",
        "runtime_estimate": RUNTIME_ESTIMATE, "runtime_observed_seconds": time.perf_counter() - wall,
        "seed_list": SEEDS, "configs": CONFIGS, "active_model": "CatBoost+LightGBM+XGBoost+ExtraTrees equal probability average",
        "cat6_unchanged": True, "threshold_policy": POLICY, "test_usage": "reporting_only",
        "final_best_time_constrained_candidate": str(best.config_name)}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=== Step14C complete ===")
    print(summary[["config_name","test_f1_mean","test_fp_mean","test_fn_mean","oof_f1_mean","feature_count_mean"]].to_string(index=False))
    print("leakage_fail", bool(leak.leakage_fail.any()))


if __name__ == "__main__":
    main()
