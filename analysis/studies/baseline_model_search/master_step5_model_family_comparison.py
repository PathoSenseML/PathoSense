"""MASTER Step 5: five-seed model-family comparison on two AA feature sets."""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder

from master_step3b_repeated_split_catboost import error_types, find_input, leakage_audit, make_split
from master_step4_aa_al_ablation import transform_config
from master_v02_catboost_threshold_stability import (
    catboost_frame, classification_metrics, make_model, markdown_table,
    select_weighted_f1_threshold,
)


ROOT = Path(__file__).resolve().parent
SEEDS = [7, 21, 42, 101, 2024]
FEATURE_SETS = ["aa_numeric", "aa_numeric_plus_al_log"]
MODEL_FAMILIES = [
    "CatBoostClassifier", "LightGBMClassifier", "XGBoostClassifier",
    "ExtraTreesClassifier", "HistGradientBoostingClassifier",
]
N_SPLITS = 5
ESTIMATED_TOTAL_RUNTIME_MINUTES = 225
CHECKPOINT_SCHEMA_VERSION = 1
PARTIALS = {
    "oof": ROOT / ".master_step5_partial_oof.csv",
    "test": ROOT / ".master_step5_partial_test.csv",
    "leakage": ROOT / ".master_step5_partial_leakage.csv",
    "runtime": ROOT / ".master_step5_partial_runtime.csv",
    "all_nan": ROOT / ".master_step5_partial_all_nan.csv",
}
OUTPUTS = {
    "split": ROOT / "master_step5_seed_split_summary.csv",
    "oof": ROOT / "master_step5_oof_metrics_by_seed_feature_model.csv",
    "test": ROOT / "master_step5_test_metrics_by_seed_feature_model.csv",
    "summary": ROOT / "master_step5_repeated_test_summary_by_feature_model.csv",
    "delta": ROOT / "master_step5_delta_vs_step4_catboost.csv",
    "leakage": ROOT / "master_step5_leakage_audit.csv",
    "all_nan": ROOT / "master_step5_all_nan_analysis.csv",
    "runtime": ROOT / "master_step5_runtime_report.csv",
    "ranking": ROOT / "master_step5_model_ranking.csv",
    "report": ROOT / "master_step5_report.md",
    "metadata": ROOT / "master_step5_metadata.json",
}


def make_preprocessor(frame: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    categorical = frame.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [col for col in frame.columns if col not in categorical]
    cat_pipe = __import__("sklearn.pipeline", fromlist=["Pipeline"]).Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric),
        ("categorical", cat_pipe, categorical),
    ], remainder="drop", verbose_feature_names_out=False)
    return preprocessor, numeric, categorical


def make_sklearn_model(model_family: str, seed: int):
    if model_family == "LightGBMClassifier":
        if not importlib.util.find_spec("lightgbm"):
            raise ImportError("lightgbm not installed")
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=31, max_depth=-1,
            subsample=0.9, colsample_bytree=0.9, objective="binary",
            random_state=seed, n_jobs=-1, verbose=-1,
        )
    if model_family == "XGBoostClassifier":
        if not importlib.util.find_spec("xgboost"):
            raise ImportError("xgboost not installed")
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=4, subsample=0.9,
            colsample_bytree=0.9, objective="binary:logistic", eval_metric="logloss",
            random_state=seed, n_jobs=-1, tree_method="hist",
        )
    if model_family == "ExtraTreesClassifier":
        return ExtraTreesClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=2,
            class_weight=None, random_state=seed, n_jobs=-1,
        )
    if model_family == "HistGradientBoostingClassifier":
        return HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=500, max_leaf_nodes=31,
            l2_regularization=0.0, random_state=seed,
        )
    raise ValueError(f"Unknown sklearn model family: {model_family}")


def feature_model_leakage(base: dict, seed: int, feature_set: str, model_family: str,
                          features: list[str]) -> dict:
    suspicious_tokens = ("label", "target", "outcome", "variant_id", "variantid")
    suspicious = [col for col in features if any(token in col.lower() for token in suspicious_tokens)]
    label_in = "Label" in features
    id_in = "Variant_ID" in features
    aa_retained = any(col in features for col in ["AA_1", "AA_2"])
    failures = []
    if base["variant_id_overlap_count"] > 0:
        failures.append("Variant_ID overlap")
    if label_in:
        failures.append("Label in features")
    if id_in:
        failures.append("Variant_ID in features")
    if aa_retained:
        failures.append("AA_1/AA_2 retained in numeric feature set")
    status = "FAIL" if failures else ("WARNING" if base["status"] == "WARNING" or suspicious else "PASS")
    return {
        "seed": seed, "feature_set": feature_set, "model_family": model_family,
        "status": status, "variant_id_overlap_count": base["variant_id_overlap_count"],
        "train_duplicate_variant_id_count": base["train_duplicate_variant_id_count"],
        "test_duplicate_variant_id_count": base["test_duplicate_variant_id_count"],
        "full_duplicate_row_count": json.loads(base["details"])["full_duplicate_rows"],
        "identical_raw_feature_hash_count": base["identical_feature_hash_count"],
        "identical_hash_all_nan_explained": base["identical_hash_all_nan_explained"],
        "feature_count": len(features), "label_in_features": label_in,
        "variant_id_in_features": id_in, "aa_raw_columns_retained": aa_retained,
        "suspicious_feature_columns": json.dumps(suspicious),
        "transformation_used_label": False,
        "transformation_used_test_learned_statistics": False,
        "preprocessing_fitted_on_test": False,
        "threshold_selected_on_test": False, "model_selected_on_test": False,
        "details": json.dumps({"base_split_audit": base["details"], "failures": failures}),
    }


def load_partials() -> tuple[dict[str, list[dict]], set[tuple[int, str, str]]]:
    if not all(path.exists() for path in PARTIALS.values()):
        return {name: [] for name in PARTIALS}, set()
    runtime_frame = pd.read_csv(PARTIALS["runtime"])
    if "checkpoint_schema" not in runtime_frame or not runtime_frame.checkpoint_schema.eq(CHECKPOINT_SCHEMA_VERSION).all():
        return {name: [] for name in PARTIALS}, set()
    rows = {name: pd.read_csv(path).to_dict("records") for name, path in PARTIALS.items()}
    key_sets = [{(int(row["seed"]), str(row["feature_set"]), str(row["model_family"])) for row in values}
                for values in rows.values()]
    completed = set.intersection(*key_sets) if key_sets else set()
    rows = {name: [row for row in values if (int(row["seed"]), str(row["feature_set"]),
                                               str(row["model_family"])) in completed]
            for name, values in rows.items()}
    return rows, completed


def save_partials(rows: dict[str, list[dict]]) -> None:
    for name, path in PARTIALS.items():
        pd.DataFrame(rows[name]).to_csv(path, index=False)


def bootstrap_catboost(rows: dict[str, list[dict]], completed: set[tuple[int, str, str]],
                       full: pd.DataFrame, raw_features: list[str], full_duplicates: int) -> None:
    paths = {
        "oof": ROOT / "master_step4_oof_metrics_by_seed_config.csv",
        "test": ROOT / "master_step4_test_metrics_by_seed_config.csv",
        "leakage": ROOT / "master_step4_leakage_audit.csv",
        "all_nan": ROOT / "master_step4_all_nan_analysis.csv",
    }
    if not all(path.exists() for path in paths.values()):
        return
    source = {name: pd.read_csv(path) for name, path in paths.items()}
    for seed in SEEDS:
        train, test = make_split(full, seed)
        base, _ = leakage_audit(full, train, test, raw_features, seed, full_duplicates)
        for feature_set in FEATURE_SETS:
            key = (seed, feature_set, "CatBoostClassifier")
            if key in completed:
                continue
            oof_match = source["oof"].query("seed == @seed and config_name == @feature_set")
            test_match = source["test"].query("seed == @seed and config_name == @feature_set")
            nan_match = source["all_nan"].query("seed == @seed and config_name == @feature_set")
            if len(oof_match) != 1 or len(test_match) != 1 or len(nan_match) != 1:
                continue
            oof_row = oof_match.iloc[0].to_dict()
            oof_row.pop("config_name", None)
            oof_row.update({"seed": seed, "feature_set": feature_set,
                            "model_family": "CatBoostClassifier", "status": "OK"})
            test_row = test_match.iloc[0].to_dict()
            test_row.pop("config_name", None)
            test_row.update({"seed": seed, "feature_set": feature_set,
                             "model_family": "CatBoostClassifier", "status": "OK"})
            features_count = int(oof_row["feature_count"])
            synthetic_features = [f"feature_{i}" for i in range(features_count)]
            # Use the real transformed feature list for leakage semantics.
            al_cols = [c for c in raw_features if c.startswith("AL_") and pd.api.types.is_numeric_dtype(full[c])]
            _, _, actual_features, _ = transform_config(train, test, raw_features, al_cols, feature_set)
            rows["oof"].append(oof_row)
            rows["test"].append(test_row)
            rows["leakage"].append(feature_model_leakage(
                base, seed, feature_set, "CatBoostClassifier", actual_features
            ))
            nan_row = nan_match.iloc[0].to_dict()
            nan_row.pop("config_name", None)
            nan_row.update({"seed": seed, "feature_set": feature_set,
                            "model_family": "CatBoostClassifier"})
            rows["all_nan"].append(nan_row)
            rows["runtime"].append({
                "seed": seed, "feature_set": feature_set, "model_family": "CatBoostClassifier",
                "actual_runtime_seconds": 0.0, "split_seconds": 0.0,
                "feature_transform_seconds": 0.0, "preprocessing_seconds": 0.0,
                "leakage_seconds": 0.0, "oof_training_seconds": 0.0,
                "final_fit_test_seconds": 0.0, "reporting_seconds": 0.0,
                "status": "REUSED_STEP4_VERIFIED", "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
            })
            completed.add(key)
    save_partials(rows)


def aggregate(test: pd.DataFrame, oof: pd.DataFrame, runtime: pd.DataFrame) -> pd.DataFrame:
    result = []
    ok = test[test.status.eq("OK")]
    for (feature_set, model), group in ok.groupby(["feature_set", "model_family"], sort=False):
        thresholds = oof[(oof.feature_set.eq(feature_set)) & (oof.model_family.eq(model)) & oof.status.eq("OK")]
        run = runtime[(runtime.feature_set.eq(feature_set)) & (runtime.model_family.eq(model))]
        result.append({
            "feature_set": feature_set, "model_family": model, "n_successful_seeds": len(group),
            "test_f1_mean": group.test_f1.mean(), "test_f1_std": group.test_f1.std(ddof=1),
            "test_f1_min": group.test_f1.min(), "test_f1_q25": group.test_f1.quantile(.25),
            "test_f1_median": group.test_f1.median(), "test_f1_q75": group.test_f1.quantile(.75),
            "test_f1_max": group.test_f1.max(),
            "test_precision_mean": group.test_precision.mean(), "test_precision_std": group.test_precision.std(ddof=1),
            "test_recall_mean": group.test_recall.mean(), "test_recall_std": group.test_recall.std(ddof=1),
            "test_fpr_mean": group.test_fpr.mean(), "test_fpr_std": group.test_fpr.std(ddof=1),
            "test_balanced_accuracy_mean": group.test_balanced_accuracy.mean(),
            "test_balanced_accuracy_std": group.test_balanced_accuracy.std(ddof=1),
            "test_fp_mean": group.test_fp.mean(), "test_fp_std": group.test_fp.std(ddof=1),
            "test_fn_mean": group.test_fn.mean(), "test_fn_std": group.test_fn.std(ddof=1),
            "selected_threshold_mean": thresholds.selected_threshold.mean(),
            "selected_threshold_std": thresholds.selected_threshold.std(ddof=1),
            "feature_count_mean": thresholds.feature_count.mean(),
            "runtime_mean_minutes": run.actual_runtime_seconds.mean()/60,
        })
    return pd.DataFrame(result)


def append_failure(rows: dict[str, list[dict]], seed: int, feature_set: str,
                   model: str, status: str, leak_row: dict, runtime_row: dict) -> None:
    base = {"seed": seed, "feature_set": feature_set, "model_family": model, "status": status}
    rows["oof"].append(base.copy())
    rows["test"].append(base.copy())
    fail_leak = leak_row.copy(); fail_leak["status"] = "SKIPPED/FAILED"
    rows["leakage"].append(fail_leak)
    rows["all_nan"].append(base.copy())
    rows["runtime"].append(runtime_row)


def main() -> None:
    wall_start = time.perf_counter()
    print("=== MASTER Step 5 estimated runtime ===", flush=True)
    print("5 seeds x 2 feature sets x 5 models = 50 runs", flush=True)
    print("CatBoost 3-5, LightGBM 1-3, XGBoost 1-4, ExtraTrees 1-3, HistGB 1-3 min/run", flush=True)
    print("Estimated total: 2.5-5 hours; slow machines may approach 6 hours", flush=True)
    print("Every run is checkpointed; verified Step 4 CatBoost runs are reused.", flush=True)

    input_path = find_input()
    full = pd.read_csv(input_path); full["Label"] = full["Label"].astype(int)
    raw_features = [c for c in full.columns if c not in ["Variant_ID", "Label"]]
    al_columns = [c for c in raw_features if c.startswith("AL_") and pd.api.types.is_numeric_dtype(full[c])]
    full_duplicates = int(full.duplicated().sum())
    rows, completed = load_partials()
    bootstrap_catboost(rows, completed, full, raw_features, full_duplicates)
    print(f"Resumed/reused runs: {len(completed)}", flush=True)
    split_rows = []

    for seed in SEEDS:
        split_start = time.perf_counter(); train, test = make_split(full, seed)
        split_seconds = time.perf_counter()-split_start
        split_rows.append({
            "seed": seed, "train_n": len(train), "test_n": len(test),
            "train_benign": int(train.Label.eq(0).sum()), "train_pathogenic": int(train.Label.eq(1).sum()),
            "test_benign": int(test.Label.eq(0).sum()), "test_pathogenic": int(test.Label.eq(1).sum()),
            "train_benign_share": train.Label.eq(0).mean(), "test_benign_share": test.Label.eq(0).mean(),
        })
        base_audit, _ = leakage_audit(full, train, test, raw_features, seed, full_duplicates)
        for feature_set in FEATURE_SETS:
            transform_start = time.perf_counter()
            x_train_raw, x_test_raw, features, transform_status = transform_config(
                train, test, raw_features, al_columns, feature_set
            )
            transform_seconds = time.perf_counter()-transform_start
            y_train, y_test = train.Label.to_numpy(int), test.Label.to_numpy(int)
            shares = pd.Series(y_train).value_counts(normalize=True).to_dict()
            weights = np.where(y_train == 0, .85/shares[0], .15/shares[1])
            for model_family in MODEL_FAMILIES:
                key = (seed, feature_set, model_family)
                if key in completed:
                    continue
                run_start = time.perf_counter(); pre_seconds = 0.0
                oof_seconds = 0.0; final_seconds = 0.0; reporting_seconds = 0.0
                leak_start = time.perf_counter()
                leak_row = feature_model_leakage(base_audit, seed, feature_set, model_family, features)
                leak_seconds = time.perf_counter()-leak_start
                print(f"\nseed={seed} feature={feature_set} model={model_family}", flush=True)
                try:
                    if transform_status.startswith("SKIPPED"):
                        raise RuntimeError(transform_status)
                    oof_start = time.perf_counter()
                    oof_prob = np.full(len(train), np.nan)
                    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
                    categorical = x_train_raw.select_dtypes(include=["object", "category"]).columns.tolist()
                    if model_family == "CatBoostClassifier":
                        x_train_cb = catboost_frame(x_train_raw, categorical)
                        for fold, (fit_idx, valid_idx) in enumerate(cv.split(x_train_cb, y_train), 1):
                            mdl = make_model(seed)
                            mdl.fit(x_train_cb.iloc[fit_idx], y_train[fit_idx], cat_features=categorical)
                            oof_prob[valid_idx] = mdl.predict_proba(x_train_cb.iloc[valid_idx])[:, 1]
                    else:
                        for fold, (fit_idx, valid_idx) in enumerate(cv.split(x_train_raw, y_train), 1):
                            pre, _, _ = make_preprocessor(x_train_raw)
                            tick = time.perf_counter()
                            fit_x = pre.fit_transform(x_train_raw.iloc[fit_idx])
                            valid_x = pre.transform(x_train_raw.iloc[valid_idx])
                            pre_seconds += time.perf_counter()-tick
                            mdl = make_sklearn_model(model_family, seed)
                            mdl.fit(fit_x, y_train[fit_idx])
                            oof_prob[valid_idx] = mdl.predict_proba(valid_x)[:, 1]
                        print("  5 OOF folds complete", flush=True)
                    threshold, weighted = select_weighted_f1_threshold(y_train, oof_prob, weights)
                    oof_raw = classification_metrics(y_train, oof_prob, threshold)
                    oof_seconds = time.perf_counter()-oof_start

                    final_start = time.perf_counter()
                    if model_family == "CatBoostClassifier":
                        x_train_final = catboost_frame(x_train_raw, categorical)
                        x_test_final = catboost_frame(x_test_raw, categorical)
                        final_model = make_model(seed)
                        final_model.fit(x_train_final, y_train, cat_features=categorical)
                        test_prob = final_model.predict_proba(x_test_final)[:, 1]
                    else:
                        pre, _, _ = make_preprocessor(x_train_raw)
                        tick = time.perf_counter()
                        fit_x = pre.fit_transform(x_train_raw); test_x = pre.transform(x_test_raw)
                        pre_seconds += time.perf_counter()-tick
                        final_model = make_sklearn_model(model_family, seed)
                        final_model.fit(fit_x, y_train); test_prob = final_model.predict_proba(test_x)[:, 1]
                    test_raw = classification_metrics(y_test, test_prob, threshold)
                    final_seconds = time.perf_counter()-final_start
                    report_start = time.perf_counter()
                    rows["oof"].append({
                        "seed": seed, "feature_set": feature_set, "model_family": model_family,
                        "status": "OK", "feature_count": len(features), "selected_threshold": threshold,
                        "oof_auc": roc_auc_score(y_train, oof_prob),
                        "oof_average_precision": average_precision_score(y_train, oof_prob),
                        "oof_f1": oof_raw["f1"], "oof_precision": oof_raw["precision"],
                        "oof_recall": oof_raw["recall"], "oof_specificity": oof_raw["specificity"],
                        "oof_fpr": oof_raw["fpr"], "oof_balanced_accuracy": oof_raw["balanced_accuracy"],
                        "oof_predicted_positive_rate": oof_raw["predicted_positive_rate"], **weighted,
                        "oof_tn": oof_raw["tn"], "oof_fp": oof_raw["fp"],
                        "oof_fn": oof_raw["fn"], "oof_tp": oof_raw["tp"],
                    })
                    rows["test"].append({
                        "seed": seed, "feature_set": feature_set, "model_family": model_family,
                        "status": "OK", "selected_threshold_from_oof": threshold,
                        "threshold_source": "OOF train CV only", "test_usage": "repeated split reporting only",
                        "test_auc": roc_auc_score(y_test, test_prob),
                        "test_average_precision": average_precision_score(y_test, test_prob),
                        "test_f1": test_raw["f1"], "test_precision": test_raw["precision"],
                        "test_recall": test_raw["recall"], "test_specificity": test_raw["specificity"],
                        "test_fpr": test_raw["fpr"], "test_balanced_accuracy": test_raw["balanced_accuracy"],
                        "test_predicted_positive_rate": test_raw["predicted_positive_rate"],
                        "test_tn": test_raw["tn"], "test_fp": test_raw["fp"],
                        "test_fn": test_raw["fn"], "test_tp": test_raw["tp"],
                    })
                    rows["leakage"].append(leak_row)
                    all_nan = x_test_raw.isna().all(axis=1).to_numpy(); pred = (test_prob >= threshold).astype(int)
                    errors = pd.Series(error_types(y_test[all_nan], pred[all_nan])).value_counts().to_dict()
                    rows["all_nan"].append({
                        "seed": seed, "feature_set": feature_set, "model_family": model_family,
                        "all_nan_test_count": int(all_nan.sum()),
                        "all_nan_test_benign": int(((y_test == 0) & all_nan).sum()),
                        "all_nan_test_pathogenic": int(((y_test == 1) & all_nan).sum()),
                        "all_nan_test_tn": int(errors.get("TN", 0)), "all_nan_test_fp": int(errors.get("FP", 0)),
                        "all_nan_test_fn": int(errors.get("FN", 0)), "all_nan_test_tp": int(errors.get("TP", 0)),
                    })
                    reporting_seconds = time.perf_counter()-report_start
                    status = "COMPLETED"
                except Exception as exc:
                    status = f"SKIPPED/FAILED: {type(exc).__name__}: {exc}"
                    runtime_row = {
                        "seed": seed, "feature_set": feature_set, "model_family": model_family,
                        "actual_runtime_seconds": time.perf_counter()-run_start,
                        "split_seconds": split_seconds, "feature_transform_seconds": transform_seconds,
                        "preprocessing_seconds": pre_seconds, "leakage_seconds": leak_seconds,
                        "oof_training_seconds": oof_seconds, "final_fit_test_seconds": final_seconds,
                        "reporting_seconds": reporting_seconds, "status": status,
                        "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
                    }
                    append_failure(rows, seed, feature_set, model_family, status, leak_row, runtime_row)
                if status == "COMPLETED":
                    rows["runtime"].append({
                        "seed": seed, "feature_set": feature_set, "model_family": model_family,
                        "actual_runtime_seconds": time.perf_counter()-run_start,
                        "split_seconds": split_seconds, "feature_transform_seconds": transform_seconds,
                        "preprocessing_seconds": pre_seconds, "leakage_seconds": leak_seconds,
                        "oof_training_seconds": oof_seconds, "final_fit_test_seconds": final_seconds,
                        "reporting_seconds": reporting_seconds, "status": status,
                        "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
                    })
                save_partials(rows); completed.add(key)
                trained_times = [float(r["actual_runtime_seconds"]) for r in rows["runtime"] if r["status"] == "COMPLETED"]
                eta = np.mean(trained_times)*(len(SEEDS)*len(FEATURE_SETS)*len(MODEL_FAMILIES)-len(completed))/60 if trained_times else np.nan
                print(f"status={status}; elapsed={(time.perf_counter()-run_start)/60:.2f} min; ETA={eta:.1f} min", flush=True)

    seed_order = {s:i for i,s in enumerate(SEEDS)}; feat_order = {f:i for i,f in enumerate(FEATURE_SETS)}
    model_order = {m:i for i,m in enumerate(MODEL_FAMILIES)}
    for values in rows.values():
        values.sort(key=lambda r: (seed_order[int(r["seed"])], feat_order[str(r["feature_set"])],
                                   model_order[str(r["model_family"])]))
    oof, test = pd.DataFrame(rows["oof"]), pd.DataFrame(rows["test"])
    leakage, all_nan, runtime = pd.DataFrame(rows["leakage"]), pd.DataFrame(rows["all_nan"]), pd.DataFrame(rows["runtime"])
    split = pd.DataFrame(split_rows); summary = aggregate(test, oof, runtime)
    refs = summary[summary.model_family.eq("CatBoostClassifier")].set_index("feature_set")
    delta_rows = []
    for _, row in summary.iterrows():
        ref = refs.loc[row.feature_set]
        d = row.test_f1_mean-ref.test_f1_mean
        delta_rows.append({
            "feature_set": row.feature_set, "model_family": row.model_family,
            "reference_catboost_f1_mean": ref.test_f1_mean, "model_f1_mean": row.test_f1_mean,
            "delta_f1_mean": d, "delta_precision_mean": row.test_precision_mean-ref.test_precision_mean,
            "delta_recall_mean": row.test_recall_mean-ref.test_recall_mean,
            "delta_fpr_mean": row.test_fpr_mean-ref.test_fpr_mean,
            "delta_balanced_accuracy_mean": row.test_balanced_accuracy_mean-ref.test_balanced_accuracy_mean,
            "delta_fp_mean": row.test_fp_mean-ref.test_fp_mean,
            "delta_fn_mean": row.test_fn_mean-ref.test_fn_mean,
            "interpretation": "reference" if row.model_family == "CatBoostClassifier" else
                ("promising 5-seed model-family evidence" if d >= .01 else
                 "not better than CatBoost in 5-seed aggregate" if d <= -.01 else "close/inconclusive 5-seed difference"),
        })
    delta = pd.DataFrame(delta_rows)
    ranking = summary.sort_values(
        ["test_f1_mean", "test_f1_std", "test_fpr_mean", "test_balanced_accuracy_mean"],
        ascending=[False, True, True, False],
    ).reset_index(drop=True)
    ranking.insert(0, "aggregate_rank", np.arange(1, len(ranking)+1))
    ranking["ranking_scope"] = "aggregate repeated split ranking; not single-seed selection"
    split.to_csv(OUTPUTS["split"], index=False); oof.to_csv(OUTPUTS["oof"], index=False)
    test.to_csv(OUTPUTS["test"], index=False); summary.to_csv(OUTPUTS["summary"], index=False)
    delta.to_csv(OUTPUTS["delta"], index=False); leakage.to_csv(OUTPUTS["leakage"], index=False)
    all_nan.to_csv(OUTPUTS["all_nan"], index=False); runtime.to_csv(OUTPUTS["runtime"], index=False)
    ranking.to_csv(OUTPUTS["ranking"], index=False)

    actual_minutes = runtime.loc[runtime.status.eq("COMPLETED"), "actual_runtime_seconds"].sum()/60
    wall_minutes = (time.perf_counter()-wall_start)/60; best = ranking.iloc[0]
    best_by_feature = ranking.sort_values("aggregate_rank").groupby("feature_set", sort=False).first().reset_index()
    non_catboost = ranking[~ranking.model_family.eq("CatBoostClassifier")]
    family_gap = best.test_f1_mean-non_catboost.test_f1_mean.max() if len(non_catboost) else np.inf
    if best.model_family == "CatBoostClassifier" and family_gap >= .02:
        next_step = "Refine the OOF-only threshold policy for the leading single model."
    else:
        next_step = "Several aggregate results are close; test OOF probability blending/ensemble without test-driven weights."
    metadata = {
        "project_folder": "TeknoPatho",
        "step": "Step 5 model-family comparison on AA numeric and AA numeric plus AL-log",
        "seed_list": SEEDS, "seed_count": 5, "feature_sets": FEATURE_SETS,
        "model_families": MODEL_FAMILIES,
        "split_rule": "same automatic prior-controlled split per seed across all feature sets/models",
        "target_train_prior": {"benign":.15,"pathogenic":.85},
        "target_test_prior": {"benign":.85,"pathogenic":.15},
        "threshold_policy": "weighted_final_like_oof_f1_threshold",
        "test_usage": "repeated split reporting only", "no_single_seed_selection": True,
        "eight_seed_pilot": False, "full_20_seed_confirmation": False,
        "random_forest_included": False,
        "estimated_total_runtime_minutes": ESTIMATED_TOTAL_RUNTIME_MINUTES,
        "actual_total_runtime_minutes": actual_minutes,
        "created_files": [Path(__file__).name, *[p.name for p in OUTPUTS.values()], *[p.name for p in PARTIALS.values()]],
    }
    OUTPUTS["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report = f"""# MASTER Step 5 five-seed model-family comparison

## Purpose and protocol

Step 4 found `aa_numeric` promising and `aa_numeric_plus_al_log` positive but with higher FP/FPR. The latter is retained because AL-log was specifically recommended and deserves family-level comparison rather than premature removal.

- Five seeds: {SEEDS}; the 8-seed pilot and 20-seed confirmation were not run.
- Feature sets: {FEATURE_SETS}; raw AA columns are removed in both. AA uses embedded BLOSUM62/Grantham and Kyte-Doolittle; AL-log is deterministic signed log.
- Models: {MODEL_FAMILIES}. RandomForest is excluded; no class weights, ensemble, SHAP, or tuning.
- All sklearn preprocessing is fitted only on each train fold (median imputation + safe ordinal encoding). Final preprocessing is fitted on full train only; test is transform-only.
- Every threshold comes from train OOF weighted final-like F1. Test is repeated-split reporting only.
- Verified Step 4 CatBoost results were reused; other model families were trained from scratch.
- Estimated runtime={ESTIMATED_TOTAL_RUNTIME_MINUTES} minutes; accumulated new-model runtime={actual_minutes:.2f}; current wall time={wall_minutes:.2f} minutes.

## Leakage summary

- FAIL present: {leakage.status.eq('FAIL').any()}; WARNING present: {leakage.status.eq('WARNING').any()}.
- Status counts: {leakage.status.value_counts().to_dict()}.

## Aggregate feature/model results

{markdown_table(summary)}

## Delta versus Step 4 CatBoost references

{markdown_table(delta)}

## Aggregate ranking

{markdown_table(ranking)}

This is aggregate repeated-split ranking, not single-seed selection. It is five-seed model-family comparison evidence, not a definitive final model.

## Interpretation

- Best aggregate pair: {best.feature_set} / {best.model_family}, F1={best.test_f1_mean:.6f} ± {best.test_f1_std:.6f}; precision={best.test_precision_mean:.6f}, recall={best.test_recall_mean:.6f}, FP={best.test_fp_mean:.2f}, FN={best.test_fn_mean:.2f}.
- Gap from the best CatBoost result to the best non-CatBoost family: {family_gap:.6f}.
- Best per feature set:

{markdown_table(best_by_feature)}

- `aa_numeric_plus_al_log` remains a secondary candidate if it raises recall but its FP/FPR must be weighed explicitly; an FPR-limited OOF threshold policy is the appropriate next controlled test when that pattern appears.

## Next step

{next_step}

No single seed is final evidence; aggregate mean ± std drives this diagnostic ranking.
"""
    OUTPUTS["report"].write_text(report, encoding="utf-8")
    print("\n=== MASTER Step 5 final summary ===")
    print(f"Seeds/features/models/expected runs: {len(SEEDS)}/{len(FEATURE_SETS)}/{len(MODEL_FAMILIES)}/{len(SEEDS)*len(FEATURE_SETS)*len(MODEL_FAMILIES)}")
    print(f"Successful runs: {len(test[test.status.eq('OK')])}")
    print(f"Leakage FAIL/WARNING: {leakage.status.eq('FAIL').any()}/{leakage.status.eq('WARNING').any()}")
    print(f"Best: {best.feature_set}/{best.model_family}; F1={best.test_f1_mean:.6f} ± {best.test_f1_std:.6f}")
    for _, row in best_by_feature.iterrows():
        ref = refs.loc[row.feature_set]
        print(f"{row.feature_set} best={row.model_family}, F1={row.test_f1_mean:.6f}, delta vs Step4 CatBoost={row.test_f1_mean-ref.test_f1_mean:+.6f}")
    print(f"Best precision/recall/FP/FN: {best.test_precision_mean:.6f}/{best.test_recall_mean:.6f}/{best.test_fp_mean:.2f}/{best.test_fn_mean:.2f}")
    completed_runtime = runtime[runtime.status.eq('COMPLETED')]
    print(f"Accumulated runtime / average new run: {actual_minutes:.2f} / {completed_runtime.actual_runtime_seconds.mean()/60:.2f} min")
    print(f"Estimated vs actual: {ESTIMATED_TOTAL_RUNTIME_MINUTES} / {actual_minutes:.2f} min")
    print("8-seed pilot performed: False; 20-seed confirmation performed: False")
    print(f"Next step: {next_step}")


if __name__ == "__main__":
    main()
