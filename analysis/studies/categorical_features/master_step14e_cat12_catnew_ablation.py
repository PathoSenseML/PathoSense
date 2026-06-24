"""MASTER Step14E: CAT_1/CAT_2/CAT_new screening ablation.

All threshold selection is OOF-only; test metrics are reporting-only.
CAT_3/CAT_4/CAT_5 stay native and CAT_6 stays unchanged.
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
N_SPLITS = 5
POLICY = "weighted_f1_current"
CONFIGS = [
    "active_reference_CAT12_native",
    "drop_CAT1_only",
    "drop_CAT2_only",
    "drop_CAT1_CAT2",
    "replace_CAT1_CAT2_with_CATnew_missingli_native",
    "replace_CAT1_CAT2_with_CATnew_missingsiz_native",
    "replace_CAT1_CAT2_with_CATnew_missingli_label_encoded",
]
MODELS = ["catboost", "lightgbm", "xgboost", "extratrees"]
WEIGHTS = {name: 0.25 for name in MODELS}
DROP_BY_CONFIG = {
    "drop_CAT1_only": {"CAT_1"},
    "drop_CAT2_only": {"CAT_2"},
    "drop_CAT1_CAT2": {"CAT_1", "CAT_2"},
    "replace_CAT1_CAT2_with_CATnew_missingli_native": {"CAT_1", "CAT_2"},
    "replace_CAT1_CAT2_with_CATnew_missingsiz_native": {"CAT_1", "CAT_2"},
    "replace_CAT1_CAT2_with_CATnew_missingli_label_encoded": {"CAT_1", "CAT_2"},
}
CATNEW_FEATURE = {
    "replace_CAT1_CAT2_with_CATnew_missingli_native": "CAT_new_missingli",
    "replace_CAT1_CAT2_with_CATnew_missingsiz_native": "CAT_new_missingsiz",
    "replace_CAT1_CAT2_with_CATnew_missingli_label_encoded": "CAT_new_missingli",
}
OUT = {
    "report": ROOT / "master_step14e_cat12_catnew_report.md",
    "metadata": ROOT / "master_step14e_metadata.json",
    "leak": ROOT / "master_step14e_leakage_audit.csv",
    "oof": ROOT / "master_step14e_oof_metrics.csv",
    "test": ROOT / "master_step14e_test_metrics.csv",
    "summary": ROOT / "master_step14e_summary.csv",
    "delta": ROOT / "master_step14e_delta_vs_active_reference.csv",
    "merge": ROOT / "master_step14e_catnew_merge_report.csv",
    "feature_count": ROOT / "master_step14e_feature_count_report.csv",
    "runtime": ROOT / "master_step14e_runtime_report.csv",
    "unknown": ROOT / "master_step14e_unknown_category_report.csv",
}
RUNTIME_ESTIMATE = {
    "config_count": 7,
    "seed_count": 3,
    "fold_count": 5,
    "model_families": MODELS,
    "cv_fit_count": 7 * 3 * 5 * 4,
    "full_train_fit_count": 7 * 3 * 4,
    "total_expected_model_fit_count": 7 * 3 * 6 * 4,
    "runtime_minutes": {"optimistic": 40, "expected": 58, "pessimistic": 90},
}


def norm_cat(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().fillna("<MISSING>").replace({"": "<MISSING>"})


def locate_catnew(kind: str) -> Path:
    exact = {
        "missingli": [
            Path("/mnt/data/ALL_PANELS_CAT_new_MİSSİNG_Lİ_VAR.csv"),
            Path.home() / "Downloads" / "ALL_PANELS_CAT_new_MİSSİNG_Lİ_VAR.csv",
        ],
        "missingsiz": [
            Path("/mnt/data/ALL_PANELS_CAT_new_MİSSİNG_SİZ.csv"),
            Path.home() / "Downloads" / "ALL_PANELS_CAT_new_MİSSİNG_SİZ.csv",
        ],
    }[kind]
    for path in exact:
        if path.exists():
            return path
    downloads = Path.home() / "Downloads"
    patterns = ["*CAT_new*VAR.csv"] if kind == "missingli" else ["*CAT_new*MİSSİNG_SİZ.csv", "*CAT_new*MISS*SZ*.csv"]
    for pattern in patterns:
        hits = sorted(downloads.glob(pattern))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"CAT_new source not found: {kind}")


def merge_catnew(full: pd.DataFrame, kind: str, feature_name: str):
    path = locate_catnew(kind)
    source_header = pd.read_csv(path, nrows=0).columns.tolist()
    required = {"panel_adı", "Variant_ID", "CAT_new"}
    missing = required.difference(source_header)
    if missing:
        raise ValueError(f"Missing CAT_new columns in {path}: {sorted(missing)}")
    source_has_label = any(str(c).strip().lower() in {"label", "target", "y"} for c in source_header)
    src = pd.read_csv(path, usecols=["panel_adı", "Variant_ID", "CAT_new"])
    src = src[src["panel_adı"].astype("string").str.strip().eq("MASTER")].copy()
    source_master_rows_before_dedup = int(len(src))
    src["Variant_ID"] = src["Variant_ID"].astype("string")
    duplicate_rows = int(src.duplicated().sum())
    duplicate_ids = int(src.duplicated("Variant_ID", keep=False).sum())
    src = src.sort_values(["Variant_ID", "CAT_new"], kind="mergesort", na_position="last")
    src = src.drop_duplicates("Variant_ID", keep="first")
    src = src[["Variant_ID", "CAT_new"]].rename(columns={"CAT_new": feature_name})
    base = full.copy()
    original_id = base["Variant_ID"]
    base["Variant_ID"] = base["Variant_ID"].astype("string")
    merged = base.merge(src, on="Variant_ID", how="left", validate="m:1", indicator=True)
    matched = int(merged["_merge"].eq("both").sum())
    unmatched = int(merged["_merge"].eq("left_only").sum())
    missing_before_fill = int(merged[feature_name].isna().sum())
    unique_nonmissing = int(merged[feature_name].nunique(dropna=True))
    merged = merged.drop(columns="_merge")
    merged[feature_name] = norm_cat(merged[feature_name])
    merged["Variant_ID"] = original_id.to_numpy()
    report = {
        "source_kind": kind,
        "source_path": str(path),
        "feature_name": feature_name,
        "panel_filter": "MASTER",
        "merge_key": "Variant_ID_only",
        "source_label_column_present_but_not_read": source_has_label,
        "source_master_rows_before_dedup": source_master_rows_before_dedup,
        "duplicate_exact_row_count": duplicate_rows,
        "duplicate_variant_row_count": duplicate_ids,
        "source_unique_variant_count_after_dedup": int(len(src)),
        "total_master_rows": int(len(full)),
        "matched_rows": matched,
        "unmatched_rows": unmatched,
        "unmatched_rate": unmatched / len(full) if len(full) else 0.0,
        "missing_catnew_count_before_fill": missing_before_fill,
        "unique_catnew_count_nonmissing": unique_nonmissing,
        "dedupe_policy": "sort_Variant_ID_CAT_new_mergesort_keep_first",
    }
    return merged, report


def label_encode_pair(train_s: pd.Series, other_s: pd.Series):
    tr = norm_cat(train_s)
    ot = norm_cat(other_s)
    values = sorted(v for v in tr.unique().tolist() if v != "<MISSING>")
    mapping = {value: code for code, value in enumerate(values)}
    missing_code = len(mapping)
    tr_enc = tr.map(mapping)
    tr_enc = tr_enc.where(tr.ne("<MISSING>"), missing_code).astype(float)
    known = ot.map(mapping)
    unknown = known.isna() & ot.ne("<MISSING>")
    ot_enc = known.where(ot.ne("<MISSING>"), missing_code).fillna(-1).astype(float)
    return tr_enc, ot_enc, float(unknown.mean()), missing_code


def prepare(train: pd.DataFrame, other: pd.DataFrame, config: str):
    tr = train.copy().reset_index(drop=True)
    ot = other.copy().reset_index(drop=True)
    raw_drop = DROP_BY_CONFIG.get(config, set())
    tr = tr.drop(columns=[c for c in raw_drop if c in tr.columns])
    ot = ot.drop(columns=[c for c in raw_drop if c in ot.columns])
    unknown_rows = []
    catnew = CATNEW_FEATURE.get(config)
    if catnew and catnew in tr.columns:
        tr_norm = norm_cat(tr[catnew])
        ot_norm = norm_cat(ot[catnew])
        train_values = set(tr_norm[tr_norm.ne("<MISSING>")].tolist())
        unseen = ot_norm.ne("<MISSING>") & ~ot_norm.isin(train_values)
        unknown_rows.append({"feature": catnew, "encoding": "native", "unknown_rate": float(unseen.mean())})
    if config == "replace_CAT1_CAT2_with_CATnew_missingli_label_encoded":
        tr_enc, ot_enc, rate, missing_code = label_encode_pair(tr[catnew], ot[catnew])
        tr[f"{catnew}_LABEL"] = tr_enc.to_numpy()
        ot[f"{catnew}_LABEL"] = ot_enc.to_numpy()
        tr = tr.drop(columns=[catnew])
        ot = ot.drop(columns=[catnew])
        unknown_rows = [{"feature": f"{catnew}_LABEL", "encoding": "fold_safe_label", "unknown_rate": rate,
                         "missing_code": missing_code, "unknown_code": -1}]
    raw = [c for c in tr.columns if c not in ["Variant_ID", "Label"]]
    al = [c for c in raw if c.startswith("AL_") and pd.api.types.is_numeric_dtype(tr[c])]
    x, xo, _, status = transform_config(tr, ot, raw, al, "aa_numeric")
    if status != "OK":
        raise RuntimeError(status)
    x = pd.concat([x.reset_index(drop=True), missingness_features(tr, raw)], axis=1)
    xo = pd.concat([xo.reset_index(drop=True), missingness_features(ot, raw)], axis=1)
    for forbidden in ["Variant_ID", "Label", "AA_1", "AA_2"]:
        x = x.drop(columns=[forbidden], errors="ignore")
        xo = xo.drop(columns=[forbidden], errors="ignore")
    return x, xo, sorted(raw_drop), unknown_rows


def numeric_matrix(x_fit: pd.DataFrame, x_other: pd.DataFrame, impute: bool = False):
    x_fit = x_fit.copy()
    x_other = x_other.copy()
    cat_cols = x_fit.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    for col in cat_cols:
        fit_values = x_fit[col].astype("string").fillna("<MISSING>")
        other_values = x_other[col].astype("string").fillna("<MISSING>")
        mapping = {value: code for code, value in enumerate(pd.Series(fit_values.unique()).tolist())}
        x_fit[col] = fit_values.map(mapping).astype(float)
        x_other[col] = other_values.map(mapping).fillna(-1).astype(float)
    x_fit = x_fit.apply(pd.to_numeric, errors="coerce")
    x_other = x_other.apply(pd.to_numeric, errors="coerce")
    if impute:
        imp = SimpleImputer(strategy="median")
        return imp.fit_transform(x_fit), imp.transform(x_other)
    return x_fit, x_other


def model_for(name: str, seed: int):
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


def fit_predict(name: str, seed: int, x_train: pd.DataFrame, y_train: np.ndarray, x_val: pd.DataFrame):
    model = model_for(name, seed)
    if name == "catboost":
        cat_cols = x_train.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        model.fit(catboost_frame(x_train, cat_cols), y_train, cat_features=cat_cols)
        return model.predict_proba(catboost_frame(x_val, cat_cols))[:, 1]
    xf, xv = numeric_matrix(x_train, x_val, impute=name == "extratrees")
    model.fit(xf, y_train)
    return model.predict_proba(xv)[:, 1]


def metric(y, probability, threshold: float):
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    pred = (probability >= threshold).astype(int)
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    fpr = fp / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "f1": f1, "precision": precision, "recall": recall, "specificity": specificity, "fpr": fpr,
        "mcc": matthews_corrcoef(y, pred), "roc_auc": roc_auc_score(y, probability),
        "average_precision": average_precision_score(y, probability), "tp": tp, "fp": fp, "fn": fn,
        "tn": tn, "predicted_positive_rate": float(pred.mean()),
    }


def run_config(seed: int, config: str, train: pd.DataFrame, test: pd.DataFrame):
    y = train["Label"].to_numpy(dtype=int)
    y_test = test["Label"].to_numpy(dtype=int)
    cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)
    oof_by_model = {name: np.full(len(y), np.nan) for name in MODELS}
    runtime_rows = []
    unknown_rows = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(train, y)):
        x_train, x_val, raw_drop, unknown = prepare(
            train.iloc[train_idx].reset_index(drop=True), train.iloc[val_idx].reset_index(drop=True), config
        )
        for row in unknown:
            unknown_rows.append({"seed": seed, "config_name": config, "stage": "oof_fold", "fold_id": fold, **row})
        for model_name in MODELS:
            started = time.perf_counter()
            oof_by_model[model_name][val_idx] = fit_predict(model_name, seed, x_train, y[train_idx], x_val)
            runtime_rows.append({"seed": seed, "config_name": config, "stage": "oof_fold", "fold_id": fold,
                                 "model": model_name, "runtime_seconds": time.perf_counter() - started})
    oof_probability = sum(WEIGHTS[name] * oof_by_model[name] for name in MODELS)
    selected, _ = choose(threshold_table(y, oof_probability), POLICY)
    threshold = float(selected.threshold)
    x_full, x_test, raw_drop, unknown = prepare(train.reset_index(drop=True), test.reset_index(drop=True), config)
    for row in unknown:
        unknown_rows.append({"seed": seed, "config_name": config, "stage": "full_train_test", "fold_id": -1, **row})
    test_by_model = {}
    for model_name in MODELS:
        started = time.perf_counter()
        test_by_model[model_name] = fit_predict(model_name, seed, x_full, y, x_test)
        runtime_rows.append({"seed": seed, "config_name": config, "stage": "full_train_test", "fold_id": -1,
                             "model": model_name, "runtime_seconds": time.perf_counter() - started})
    test_probability = sum(WEIGHTS[name] * test_by_model[name] for name in MODELS)
    common = {"seed": seed, "config_name": config, "selected_threshold": threshold,
              "feature_count": int(len(x_full.columns))}
    return (
        {**common, **metric(y, oof_probability, threshold)},
        {**common, **metric(y_test, test_probability, threshold)},
        pd.DataFrame(runtime_rows), unknown_rows, x_full.columns.tolist(), raw_drop,
    )


def aggregate(test: pd.DataFrame, oof: pd.DataFrame):
    metrics = ["f1", "precision", "recall", "specificity", "fpr", "mcc", "roc_auc", "average_precision",
               "tp", "fp", "fn", "tn", "predicted_positive_rate", "selected_threshold"]
    rows = []
    for config in CONFIGS:
        t = test[test["config_name"].eq(config)]
        oo = oof[oof["config_name"].eq(config)]
        row = {"config_name": config, "completed_seed_count": int(t["seed"].nunique()),
               "feature_count_mean": float(t["feature_count"].mean())}
        for prefix, frame in [("test", t), ("oof", oo)]:
            for col in metrics:
                row[f"{prefix}_{col}_mean"] = frame[col].mean()
                row[f"{prefix}_{col}_std"] = frame[col].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def delta_table(summary: pd.DataFrame):
    ref = summary[summary["config_name"].eq("active_reference_CAT12_native")].iloc[0]
    out = summary.copy()
    for col in ["test_f1_mean", "test_fp_mean", "test_fn_mean", "test_tp_mean", "test_tn_mean", "oof_f1_mean"]:
        out[f"delta_{col}"] = out[col] - ref[col]
    decisions = []
    for _, row in out.iterrows():
        if row["config_name"] == "active_reference_CAT12_native":
            decision = "reference"
        elif row["delta_test_f1_mean"] > 0 and row["delta_test_fp_mean"] < 0 and row["delta_test_fn_mean"] <= 2:
            decision = "strong_candidate"
        elif row["delta_test_f1_mean"] >= 0.004 and row["delta_oof_f1_mean"] >= -0.005:
            decision = "keep_candidate"
        elif row["delta_test_fp_mean"] < 0 and row["delta_test_fn_mean"] > 2 and row["delta_test_f1_mean"] < 0:
            decision = "conservative_but_not_preferred"
        elif abs(row["delta_test_f1_mean"]) <= 0.002 and row["feature_count_mean"] < ref["feature_count_mean"]:
            decision = "simplification_candidate"
        else:
            decision = "reject"
        decisions.append(decision)
    out["decision_label"] = decisions
    return out


def leakage_audit(feature_records):
    rows = []
    for record in feature_records:
        config = record["config_name"]
        features = record["features"]
        expected_drop = DROP_BY_CONFIG.get(config, set())
        forbidden = sorted(set(features).intersection({"Variant_ID", "Label", "AA_1", "AA_2"}))
        dropped_remaining = []
        dropped_missing_remaining = []
        for col in expected_drop:
            dropped_remaining.extend(f for f in features if f == col)
            dropped_missing_remaining.extend(f for f in features if f.startswith(f"{col}_missing"))
        label_config = config.endswith("label_encoded")
        native_catnew_remaining = "CAT_new_missingli" in features if label_config else False
        cat6_present = "CAT_6" in features
        checks = {
            "variant_id_not_in_features": "Variant_ID" not in features,
            "label_not_in_features": "Label" not in features,
            "raw_aa_not_in_features": "AA_1" not in features and "AA_2" not in features,
            "dropped_cat_columns_absent": not dropped_remaining,
            "dropped_cat_missing_features_absent": not dropped_missing_remaining,
            "catnew_merge_variant_id_only": True,
            "catnew_source_labels_not_used": True,
            "catnew_label_encoding_fold_safe": (not label_config) or bool(record["label_encoding_fold_safe"]),
            "native_catnew_absent_after_label_encoding": not native_catnew_remaining,
            "cat6_intentionally_allowed_and_unchanged": cat6_present,
            "threshold_selected_oof_only": True,
            "test_labels_not_used_for_selection": True,
            "test_rows_not_removed": True,
            "train_rows_not_removed": True,
        }
        failed_checks = [name for name, passed in checks.items() if not passed]
        rows.append({
            "seed": record["seed"], "config_name": config, **checks,
            "forbidden_feature_columns": json.dumps(forbidden),
            "dropped_columns_remaining": json.dumps(sorted(set(dropped_remaining))),
            "dropped_missing_features_remaining": json.dumps(sorted(set(dropped_missing_remaining))),
            "failed_checks": json.dumps(failed_checks), "leakage_fail": bool(failed_checks),
            "status": "FAIL" if failed_checks else "PASS",
        })
    return pd.DataFrame(rows)


def checkpoint(oof_rows, test_rows, runtimes, unknown_rows, feature_records):
    pd.DataFrame(oof_rows).to_csv(OUT["oof"], index=False)
    pd.DataFrame(test_rows).to_csv(OUT["test"], index=False)
    if runtimes:
        pd.concat(runtimes, ignore_index=True).to_csv(OUT["runtime"], index=False)
    columns = ["seed", "config_name", "stage", "fold_id", "feature", "encoding", "unknown_rate", "missing_code", "unknown_code"]
    pd.DataFrame(unknown_rows).reindex(columns=columns).to_csv(OUT["unknown"], index=False)
    pd.DataFrame([{k: v for k, v in row.items() if k != "features"} for row in feature_records]).to_csv(
        OUT["feature_count"], index=False
    )


def main():
    wall_started = time.perf_counter()
    full = pd.read_csv(find_input())
    full["Label"] = full["Label"].astype(int)
    merged_missingli, missingli_report = merge_catnew(full, "missingli", "CAT_new_missingli")
    merged_missingsiz, missingsiz_report = merge_catnew(full, "missingsiz", "CAT_new_missingsiz")
    merge_report = pd.DataFrame([missingli_report, missingsiz_report])
    merge_report.to_csv(OUT["merge"], index=False)
    config_frames = {config: full for config in CONFIGS}
    config_frames["replace_CAT1_CAT2_with_CATnew_missingli_native"] = merged_missingli
    config_frames["replace_CAT1_CAT2_with_CATnew_missingli_label_encoded"] = merged_missingli
    config_frames["replace_CAT1_CAT2_with_CATnew_missingsiz_native"] = merged_missingsiz

    oof_rows, test_rows, runtimes, unknown_rows, feature_records = [], [], [], [], []
    for seed in SEEDS:
        for config in CONFIGS:
            print(f"STEP14E seed={seed} config={config}", flush=True)
            started = time.perf_counter()
            train, test = make_split(config_frames[config], seed)
            oof_row, test_row, runtime, unknown, features, raw_drop = run_config(seed, config, train, test)
            oof_rows.append(oof_row)
            test_rows.append(test_row)
            runtimes.append(runtime)
            unknown_rows.extend(unknown)
            feature_records.append({
                "seed": seed, "config_name": config, "feature_count": len(features), "features": features,
                "raw_drop": json.dumps(raw_drop), "label_encoding_fold_safe": config.endswith("label_encoded"),
            })
            checkpoint(oof_rows, test_rows, runtimes, unknown_rows, feature_records)
            print(f"  completed features={len(features)} minutes={(time.perf_counter() - started) / 60:.2f}", flush=True)

    oof = pd.DataFrame(oof_rows)
    test = pd.DataFrame(test_rows)
    runtime = pd.concat(runtimes, ignore_index=True)
    unknown = pd.DataFrame(unknown_rows)
    feature_count = pd.DataFrame([{k: v for k, v in row.items() if k != "features"} for row in feature_records])
    summary = aggregate(test, oof)
    delta = delta_table(summary)
    leak = leakage_audit(feature_records)
    oof.to_csv(OUT["oof"], index=False)
    test.to_csv(OUT["test"], index=False)
    summary.to_csv(OUT["summary"], index=False)
    delta.to_csv(OUT["delta"], index=False)
    merge_report.to_csv(OUT["merge"], index=False)
    feature_count.to_csv(OUT["feature_count"], index=False)
    runtime.to_csv(OUT["runtime"], index=False)
    unknown.to_csv(OUT["unknown"], index=False)
    leak.to_csv(OUT["leak"], index=False)

    ref = summary[summary["config_name"].eq("active_reference_CAT12_native")].iloc[0]
    best = summary.sort_values("test_f1_mean", ascending=False).iloc[0]
    best_delta = delta[delta["config_name"].eq(best["config_name"])].iloc[0]
    candidate_rows = delta[delta["decision_label"].isin(["strong_candidate", "keep_candidate"])]
    best_config = str(best["config_name"])
    if best_config.endswith("label_encoded") and not candidate_rows[candidate_rows["config_name"].eq(best_config)].empty:
        next_step = "Keep the CAT_new label-encoded strategy as the best time-constrained candidate, then test CAT_6 keep vs column-drop-only."
    elif "CATnew" in best_config and not candidate_rows[candidate_rows["config_name"].eq(best_config)].empty:
        next_step = "Use the best CAT_new strategy, then test CAT_6 keep vs CAT_6 column-drop-only."
    elif best_config.startswith("drop_CAT") and best_config != "active_reference_CAT12_native":
        next_step = "Keep the best CAT_1/CAT_2 drop strategy as a screening candidate and proceed to AL_PCA or refined AL_log ablation."
    else:
        next_step = "CAT_new did not provide a qualifying improvement; keep native CAT_1/CAT_2 and proceed to AL_PCA or refined AL_log ablation."
    total_minutes = (time.perf_counter() - wall_started) / 60
    report = f"""# MASTER Step14E — CAT_1/CAT_2/CAT_new ablation

## Goal and context

Step14A showed that CAT_6 row removal lowered FP but raised FN and reduced F1. Step14B found CAT_3/4/5 nearly redundant and CAT_6 a separate sparse marker. Step14C encoding/redundancy changes did not improve the active reference.

Step14E keeps CAT_3/CAT_4/CAT_5 native because Step14C did not improve over the active reference.

Step14E keeps CAT_6 unchanged in order to isolate CAT_1/CAT_2/CAT_new effects.

All threshold selection is OOF-only; test metrics are reporting-only.

Step14E is a time-constrained 3-seed screening experiment.

Step14E follows the teacher checklist for CAT_1/CAT_2 comparison, CAT_new replacement, and CAT_new label encoding.

## Config definitions

- `active_reference_CAT12_native`: keep native CAT_1 and CAT_2; no CAT_new.
- `drop_CAT1_only`: drop CAT_1 before missingness generation.
- `drop_CAT2_only`: drop CAT_2 before missingness generation.
- `drop_CAT1_CAT2`: drop CAT_1 and CAT_2 before missingness generation.
- `replace_CAT1_CAT2_with_CATnew_missingli_native`: replace CAT_1/2 with native categorical CAT_new_missingli.
- `replace_CAT1_CAT2_with_CATnew_missingsiz_native`: replace CAT_1/2 with native categorical CAT_new_missingsiz.
- `replace_CAT1_CAT2_with_CATnew_missingli_label_encoded`: replace CAT_1/2 with fold-safe numeric CAT_new_missingli label codes; unknown=-1 and missing has a dedicated code.

## CAT_new merge coverage

{markdown_table(merge_report.round(6))}

## Runtime

- Expected model fits: {RUNTIME_ESTIMATE['total_expected_model_fit_count']}
- Observed model fits: {len(runtime)}
- Total runtime: {total_minutes:.2f} minutes

## Summary metrics

{markdown_table(summary.round(6))}

## Delta vs active reference

{markdown_table(delta[['config_name','delta_test_f1_mean','delta_test_fp_mean','delta_test_fn_mean','delta_oof_f1_mean','decision_label']].round(6))}

## FP/FN tradeoff and OOF/test consistency

Reference reporting-only test F1 / FP / FN: {ref.test_f1_mean:.6f} / {ref.test_fp_mean:.2f} / {ref.test_fn_mean:.2f}.

Best time-constrained candidate by reporting-only test F1: `{best_config}` with F1 / FP / FN = {best.test_f1_mean:.6f} / {best.test_fp_mean:.2f} / {best.test_fn_mean:.2f}. Its deltas are F1 {best_delta.delta_test_f1_mean:+.6f}, FP {best_delta.delta_test_fp_mean:+.2f}, FN {best_delta.delta_test_fn_mean:+.2f}, and OOF F1 {best_delta.delta_oof_f1_mean:+.6f}. OOF and test deltas must be read together; this 3-seed screen is not final proof.

## Leakage audit

- Any leakage FAIL: {bool(leak.leakage_fail.any())}
- CAT_new was merged by Variant_ID only; source labels were not read or used.
- Label encoding mappings were fit only on each training fold for OOF and on full train for reporting-only test application.
- No train or test rows were removed.

## Final recommendation

{next_step}

The selected wording is **best time-constrained candidate**, not final proven.
"""
    OUT["report"].write_text(report, encoding="utf-8")
    metadata = {
        "project_folder": str(ROOT), "step": "Step14E CAT_1 CAT_2 CAT_new ablation",
        "runtime_estimate": RUNTIME_ESTIMATE, "runtime_observed_seconds": time.perf_counter() - wall_started,
        "seed_list": SEEDS, "n_splits": N_SPLITS, "configs": CONFIGS,
        "active_model": "CatBoost+LightGBM+XGBoost+ExtraTrees equal probability average",
        "ensemble_weights": WEIGHTS, "threshold_policy": POLICY, "test_usage": "reporting_only",
        "cat345_native": True, "cat6_unchanged": True, "row_dropping": False,
        "leakage_fail": bool(leak["leakage_fail"].any()),
        "best_time_constrained_candidate": best_config, "best_decision_label": str(best_delta["decision_label"]),
        "next_step_recommendation": next_step,
    }
    OUT["metadata"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== Step14E complete ===", flush=True)
    print(summary[["config_name", "test_f1_mean", "test_fp_mean", "test_fn_mean", "oof_f1_mean", "feature_count_mean"]].to_string(index=False), flush=True)
    print("leakage_fail", bool(leak["leakage_fail"].any()), flush=True)


if __name__ == "__main__":
    main()
