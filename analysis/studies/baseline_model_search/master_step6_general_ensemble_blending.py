"""MASTER Step 6: five-seed leakage-safe OOF-only probability blending."""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from master_step3b_repeated_split_catboost import error_types, find_input, leakage_audit, make_split
from master_step4_aa_al_ablation import transform_config
from master_step5_model_family_comparison import (
    make_preprocessor, make_sklearn_model, feature_model_leakage,
)
from master_v02_catboost_threshold_stability import (
    catboost_frame, classification_metrics, make_model, markdown_table,
    weighted_metrics,
)


ROOT = Path(__file__).resolve().parent
SEEDS = [7, 21, 42, 101, 2024]
FEATURE_SETS = ["aa_numeric", "aa_numeric_plus_al_log"]
MODELS = [
    "CatBoostClassifier", "ExtraTreesClassifier", "LightGBMClassifier",
    "XGBoostClassifier", "HistGradientBoostingClassifier",
]
N_SPLITS = 5
ESTIMATED_TOTAL_RUNTIME_MINUTES = 240
CHECKPOINT_SCHEMA = 1
PARTIALS = {
    "base_oof": ROOT / ".master_step6_partial_base_oof_predictions.csv",
    "base_test": ROOT / ".master_step6_partial_base_test_predictions.csv",
    "base_metrics": ROOT / ".master_step6_partial_base_metrics.csv",
    "ensemble_oof": ROOT / ".master_step6_partial_ensemble_oof.csv",
    "ensemble_test": ROOT / ".master_step6_partial_ensemble_test.csv",
    "leakage": ROOT / ".master_step6_partial_leakage.csv",
    "runtime": ROOT / ".master_step6_partial_runtime.csv",
    "all_nan": ROOT / ".master_step6_partial_all_nan.csv",
}
OUTPUTS = {
    "split": ROOT / "master_step6_seed_split_summary.csv",
    "base_oof": ROOT / "master_step6_base_oof_predictions.csv",
    "base_test": ROOT / "master_step6_base_test_predictions.csv",
    "base_metrics": ROOT / "master_step6_base_model_metrics.csv",
    "ensemble_oof": ROOT / "master_step6_ensemble_oof_metrics_by_seed_strategy.csv",
    "ensemble_test": ROOT / "master_step6_ensemble_test_metrics_by_seed_strategy.csv",
    "summary": ROOT / "master_step6_repeated_test_summary_by_strategy.csv",
    "delta": ROOT / "master_step6_delta_vs_step5_catboost.csv",
    "leakage": ROOT / "master_step6_leakage_audit.csv",
    "all_nan": ROOT / "master_step6_all_nan_analysis.csv",
    "runtime": ROOT / "master_step6_runtime_report.csv",
    "ranking": ROOT / "master_step6_ensemble_ranking.csv",
    "report": ROOT / "master_step6_report.md",
    "metadata": ROOT / "master_step6_metadata.json",
}

SINGLE_STRATEGIES = {
    "catboost_single_reference": "CatBoostClassifier",
    "extratrees_single_reference": "ExtraTreesClassifier",
    "lightgbm_single_reference": "LightGBMClassifier",
    "xgboost_single_reference": "XGBoostClassifier",
    "histgb_single_reference": "HistGradientBoostingClassifier",
}
FIXED_STRATEGIES = {
    "fixed_teknofest_safe_weights": {
        "CatBoostClassifier": .30, "ExtraTreesClassifier": .40,
        "LightGBMClassifier": .10, "XGBoostClassifier": .20,
        "HistGradientBoostingClassifier": 0.0,
    },
    "simple_mean_top4": {
        "CatBoostClassifier": .25, "ExtraTreesClassifier": .25,
        "LightGBMClassifier": .25, "XGBoostClassifier": .25,
        "HistGradientBoostingClassifier": 0.0,
    },
    "simple_mean_top5": {model: .20 for model in MODELS},
    "catboost_heavy_top4": {
        "CatBoostClassifier": .50, "ExtraTreesClassifier": .20,
        "LightGBMClassifier": .15, "XGBoostClassifier": .15,
        "HistGradientBoostingClassifier": 0.0,
    },
    "tree_diverse_fixed_top5": {
        "CatBoostClassifier": .35, "ExtraTreesClassifier": .25,
        "LightGBMClassifier": .15, "XGBoostClassifier": .15,
        "HistGradientBoostingClassifier": .10,
    },
}
WITHIN_STRATEGIES = [
    *SINGLE_STRATEGIES, *FIXED_STRATEGIES,
    "oof_grid_blend_top4", "oof_grid_blend_top5",
    "oof_grid_blend_catboost_extratrees", "oof_grid_blend_catboost_xgboost",
    "oof_grid_blend_catboost_lightgbm", "oof_sparse_top2_blend",
    "oof_sparse_top3_blend", "rank_average_top4", "rank_average_top5",
    "median_probability_top4", "median_probability_top5",
]
CROSS_STRATEGIES = [
    "cross_featureset_catboost_mean", "cross_featureset_best_fixed_mean",
    "cross_featureset_oof_sparse_blend",
]
ALL_STRATEGIES = WITHIN_STRATEGIES + CROSS_STRATEGIES


def empty_state() -> dict[str, list[dict]]:
    return {name: [] for name in PARTIALS}


def load_state() -> dict[str, list[dict]]:
    if not all(path.exists() for path in PARTIALS.values()):
        return empty_state()
    runtime = pd.read_csv(PARTIALS["runtime"])
    if "checkpoint_schema" not in runtime or not runtime.checkpoint_schema.eq(CHECKPOINT_SCHEMA).all():
        return empty_state()
    return {name: pd.read_csv(path).to_dict("records") for name, path in PARTIALS.items()}


def save_state(state: dict[str, list[dict]]) -> None:
    for name, path in PARTIALS.items():
        pd.DataFrame(state[name]).to_csv(path, index=False)


def base_key(row: dict) -> tuple[int, str, str]:
    return int(row["seed"]), str(row["feature_set"]), str(row["model_family"])


def ensemble_key(row: dict) -> tuple[int, str, str]:
    return int(row["seed"]), str(row["feature_set"]), str(row["ensemble_strategy"])


def fast_select_threshold(y: np.ndarray, prob: np.ndarray,
                          sample_weight: np.ndarray) -> tuple[float, float]:
    order = np.argsort(-prob, kind="mergesort")
    score, label, weight = prob[order], y[order], sample_weight[order]
    tp = np.cumsum(weight * (label == 1))
    fp = np.cumsum(weight * (label == 0))
    total_pos = float((sample_weight * (y == 1)).sum())
    group_end = np.r_[score[1:] != score[:-1], True]
    tp, fp, thresholds = tp[group_end], fp[group_end], score[group_end]
    precision = tp / np.maximum(tp + fp, 1e-15)
    recall = tp / max(total_pos, 1e-15)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-15)
    best_value = np.nanmax(f1)
    # Last tied index is the lowest threshold / highest-recall deterministic tie-break.
    index = int(np.flatnonzero(np.isclose(f1, best_value, rtol=0, atol=1e-15))[-1])
    return float(thresholds[index]), float(f1[index])


def compositions(total: int, parts: int, positive: bool = False):
    if parts == 1:
        if (not positive) or total > 0:
            yield (total,)
        return
    start = 1 if positive else 0
    for first in range(start, total + 1):
        remaining = total - first
        minimum = (parts - 1) if positive else 0
        if remaining < minimum:
            break
        for rest in compositions(remaining, parts - 1, positive):
            yield (first, *rest)


def weighted_blend(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return matrix @ weights


def search_candidates(y: np.ndarray, matrix: np.ndarray, names: list[str],
                      candidates, sample_weight: np.ndarray) -> tuple[np.ndarray, float, dict, float]:
    best = None
    for weights in candidates:
        weights = np.asarray(weights, dtype=float)
        prob = weighted_blend(matrix, weights)
        threshold, score = fast_select_threshold(y, prob, sample_weight)
        key = (score, -threshold, tuple(-weights))
        if best is None or key > best[0]:
            best = (key, prob, threshold, dict(zip(names, weights)), score)
    return best[1], best[2], best[3], best[4]


def empirical_percentiles(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    sorted_ref = np.sort(reference)
    return np.searchsorted(sorted_ref, values, side="right") / len(sorted_ref)


def rank_blend(oof_matrix: np.ndarray, test_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    oof_rank = np.column_stack([
        pd.Series(oof_matrix[:, i]).rank(method="average", pct=True).to_numpy()
        for i in range(oof_matrix.shape[1])
    ])
    test_rank = np.column_stack([
        empirical_percentiles(oof_matrix[:, i], test_matrix[:, i])
        for i in range(oof_matrix.shape[1])
    ])
    return oof_rank.mean(axis=1), test_rank.mean(axis=1)


def metric_rows(seed: int, feature_set: str, strategy: str, y_oof: np.ndarray,
                p_oof: np.ndarray, y_test: np.ndarray, p_test: np.ndarray,
                threshold: float, weights: dict, members: list[str],
                weight_source: str, sample_weight: np.ndarray) -> tuple[dict, dict]:
    raw_oof = classification_metrics(y_oof, p_oof, threshold)
    weighted = weighted_metrics(y_oof, p_oof, threshold, sample_weight)
    raw_test = classification_metrics(y_test, p_test, threshold)
    common = {
        "seed": seed, "feature_set": feature_set, "ensemble_strategy": strategy,
        "status": "OK", "selected_weights_json": json.dumps(weights, sort_keys=True),
        "selected_model_members_json": json.dumps(members),
    }
    oof = {
        **common, "selected_threshold": threshold,
        "oof_auc": roc_auc_score(y_oof, p_oof),
        "oof_average_precision": average_precision_score(y_oof, p_oof),
        "oof_f1": raw_oof["f1"], "oof_precision": raw_oof["precision"],
        "oof_recall": raw_oof["recall"], "oof_specificity": raw_oof["specificity"],
        "oof_fpr": raw_oof["fpr"], "oof_balanced_accuracy": raw_oof["balanced_accuracy"],
        "oof_predicted_positive_rate": raw_oof["predicted_positive_rate"], **weighted,
        "oof_tn": raw_oof["tn"], "oof_fp": raw_oof["fp"],
        "oof_fn": raw_oof["fn"], "oof_tp": raw_oof["tp"],
    }
    test = {
        **common, "selected_threshold_from_oof": threshold,
        "threshold_source": "OOF ensemble train CV only", "weight_source": weight_source,
        "test_usage": "repeated split reporting only",
        "test_auc": roc_auc_score(y_test, p_test),
        "test_average_precision": average_precision_score(y_test, p_test),
        "test_f1": raw_test["f1"], "test_precision": raw_test["precision"],
        "test_recall": raw_test["recall"], "test_specificity": raw_test["specificity"],
        "test_fpr": raw_test["fpr"], "test_balanced_accuracy": raw_test["balanced_accuracy"],
        "test_predicted_positive_rate": raw_test["predicted_positive_rate"],
        "test_tn": raw_test["tn"], "test_fp": raw_test["fp"],
        "test_fn": raw_test["fn"], "test_tp": raw_test["tp"],
    }
    return oof, test


def ensemble_leakage(base: dict, seed: int, feature_set: str, name: str,
                     status: str | None = None) -> dict:
    fail = base["variant_id_overlap_count"] > 0
    if status is None:
        status = "WARNING" if base.get("status") == "WARNING" else "PASS"
    return {
        "seed": seed, "feature_set": feature_set, "component_type": "ensemble_strategy",
        "name": name, "status": "FAIL" if fail else status,
        "variant_id_overlap_count": base["variant_id_overlap_count"],
        "identical_raw_feature_hash_count": base["identical_feature_hash_count"],
        "identical_hash_all_nan_explained": base["identical_hash_all_nan_explained"],
        "label_in_features": False, "variant_id_in_features": False,
        "aa_raw_columns_retained": False, "preprocessing_fitted_on_test": False,
        "threshold_selected_on_test": False, "ensemble_weights_selected_on_test": False,
        "model_selected_on_test": False,
        "details": base["details"],
    }


def base_training(full: pd.DataFrame, raw_features: list[str], al_columns: list[str],
                  state: dict[str, list[dict]], splits: dict[int, tuple[pd.DataFrame, pd.DataFrame]],
                  audits: dict[int, dict], split_seconds: dict[int, float]) -> None:
    completed = {base_key(row) for row in state["base_metrics"]}
    for seed in SEEDS:
        train, test = splits[seed]
        y_train, y_test = train.Label.to_numpy(int), test.Label.to_numpy(int)
        for feature_set in FEATURE_SETS:
            transform_start = time.perf_counter()
            x_train_raw, x_test_raw, features, status = transform_config(
                train, test, raw_features, al_columns, feature_set
            )
            transform_seconds = time.perf_counter() - transform_start
            categorical = x_train_raw.select_dtypes(include=["object", "category"]).columns.tolist()
            for model_family in MODELS:
                key = (seed, feature_set, model_family)
                if key in completed:
                    continue
                run_start = time.perf_counter(); pre_seconds = 0.0
                oof_seconds = final_seconds = 0.0
                print(f"BASE seed={seed} feature={feature_set} model={model_family}", flush=True)
                try:
                    if status.startswith("SKIPPED"):
                        raise RuntimeError(status)
                    oof_start = time.perf_counter(); oof_prob = np.full(len(train), np.nan)
                    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
                    if model_family == "CatBoostClassifier":
                        x_cb = catboost_frame(x_train_raw, categorical)
                        for fit_idx, valid_idx in cv.split(x_cb, y_train):
                            model = make_model(seed)
                            model.fit(x_cb.iloc[fit_idx], y_train[fit_idx], cat_features=categorical)
                            oof_prob[valid_idx] = model.predict_proba(x_cb.iloc[valid_idx])[:, 1]
                    else:
                        for fit_idx, valid_idx in cv.split(x_train_raw, y_train):
                            pre, _, _ = make_preprocessor(x_train_raw)
                            tick = time.perf_counter()
                            fit_x = pre.fit_transform(x_train_raw.iloc[fit_idx])
                            valid_x = pre.transform(x_train_raw.iloc[valid_idx])
                            pre_seconds += time.perf_counter() - tick
                            model = make_sklearn_model(model_family, seed)
                            model.fit(fit_x, y_train[fit_idx])
                            oof_prob[valid_idx] = model.predict_proba(valid_x)[:, 1]
                    oof_seconds = time.perf_counter() - oof_start
                    final_start = time.perf_counter()
                    if model_family == "CatBoostClassifier":
                        x_cb, test_cb = catboost_frame(x_train_raw, categorical), catboost_frame(x_test_raw, categorical)
                        final_model = make_model(seed); final_model.fit(x_cb, y_train, cat_features=categorical)
                        test_prob = final_model.predict_proba(test_cb)[:, 1]
                    else:
                        pre, _, _ = make_preprocessor(x_train_raw)
                        tick = time.perf_counter()
                        fit_x, test_x = pre.fit_transform(x_train_raw), pre.transform(x_test_raw)
                        pre_seconds += time.perf_counter() - tick
                        final_model = make_sklearn_model(model_family, seed)
                        final_model.fit(fit_x, y_train); test_prob = final_model.predict_proba(test_x)[:, 1]
                    final_seconds = time.perf_counter() - final_start
                    for index in range(len(train)):
                        state["base_oof"].append({
                            "seed": seed, "feature_set": feature_set, "model_family": model_family,
                            "row_index": index, "Variant_ID": train.Variant_ID.iloc[index],
                            "true_label": y_train[index], "oof_proba_pathogenic": oof_prob[index],
                        })
                    for index in range(len(test)):
                        state["base_test"].append({
                            "seed": seed, "feature_set": feature_set, "model_family": model_family,
                            "row_index": index, "Variant_ID": test.Variant_ID.iloc[index],
                            "true_label": y_test[index], "test_proba_pathogenic": test_prob[index],
                        })
                    state["base_metrics"].append({
                        "seed": seed, "feature_set": feature_set, "model_family": model_family,
                        "status": "OK", "feature_count": len(features),
                        "base_oof_auc": roc_auc_score(y_train, oof_prob),
                        "base_oof_average_precision": average_precision_score(y_train, oof_prob),
                        "base_test_auc": roc_auc_score(y_test, test_prob),
                        "base_test_average_precision": average_precision_score(y_test, test_prob),
                        "runtime": time.perf_counter() - run_start,
                    })
                    leak = feature_model_leakage(audits[seed], seed, feature_set, model_family, features)
                    leak.update({"component_type": "base_model", "name": model_family})
                    state["leakage"].append(leak)
                    runtime_status = "COMPLETED"
                except Exception as exc:
                    runtime_status = f"SKIPPED/FAILED: {type(exc).__name__}: {exc}"
                    state["base_metrics"].append({
                        "seed": seed, "feature_set": feature_set, "model_family": model_family,
                        "status": runtime_status, "feature_count": len(features),
                    })
                    state["leakage"].append({
                        "seed": seed, "feature_set": feature_set, "component_type": "base_model",
                        "name": model_family, "status": "SKIPPED/FAILED", "details": runtime_status,
                    })
                state["runtime"].append({
                    "seed": seed, "feature_set": feature_set, "component_type": "base_model",
                    "name": model_family, "actual_runtime_seconds": time.perf_counter()-run_start,
                    "split_seconds": split_seconds[seed], "feature_transform_seconds": transform_seconds,
                    "preprocessing_seconds": pre_seconds, "leakage_seconds": 0.0,
                    "oof_training_seconds": oof_seconds, "final_fit_test_seconds": final_seconds,
                    "ensemble_weight_search_seconds": 0.0, "threshold_selection_seconds": 0.0,
                    "reporting_seconds": 0.0, "status": runtime_status,
                    "checkpoint_schema": CHECKPOINT_SCHEMA,
                })
                save_state(state); completed.add(key)
                done = len(completed); remaining = len(SEEDS)*len(FEATURE_SETS)*len(MODELS)-done
                successful_times = [float(r["actual_runtime_seconds"]) for r in state["runtime"]
                                    if r["component_type"] == "base_model" and r["status"] == "COMPLETED"]
                eta = np.mean(successful_times)*remaining/60 if successful_times else np.nan
                print(f"  {runtime_status}; elapsed={(time.perf_counter()-run_start)/60:.2f} min; base ETA={eta:.1f} min", flush=True)


def prediction_matrices(state: dict[str, list[dict]], seed: int, feature_set: str):
    oof = pd.DataFrame([r for r in state["base_oof"] if int(r["seed"]) == seed and r["feature_set"] == feature_set])
    test = pd.DataFrame([r for r in state["base_test"] if int(r["seed"]) == seed and r["feature_set"] == feature_set])
    if oof.empty or test.empty:
        raise RuntimeError("base probabilities unavailable")
    oof_pivot = oof.pivot(index="row_index", columns="model_family", values="oof_proba_pathogenic")
    test_pivot = test.pivot(index="row_index", columns="model_family", values="test_proba_pathogenic")
    if not set(MODELS).issubset(oof_pivot.columns) or not set(MODELS).issubset(test_pivot.columns):
        raise RuntimeError("one or more base model probabilities missing")
    y_oof = oof.drop_duplicates("row_index").sort_values("row_index").true_label.to_numpy(int)
    y_test = test.drop_duplicates("row_index").sort_values("row_index").true_label.to_numpy(int)
    return oof_pivot[MODELS].to_numpy(), test_pivot[MODELS].to_numpy(), y_oof, y_test


def within_strategy(strategy: str, y: np.ndarray, oof: np.ndarray, test: np.ndarray,
                    sample_weight: np.ndarray):
    index = {name: i for i, name in enumerate(MODELS)}
    if strategy in SINGLE_STRATEGIES:
        model = SINGLE_STRATEGIES[strategy]; weights = {model: 1.0}
        return oof[:, index[model]], test[:, index[model]], weights, [model], "fixed prior"
    if strategy in FIXED_STRATEGIES:
        weights = FIXED_STRATEGIES[strategy]; vector = np.array([weights[m] for m in MODELS])
        return oof @ vector, test @ vector, weights, [m for m in MODELS if weights[m] > 0], "fixed prior"
    top4 = MODELS[:4]
    if strategy == "oof_grid_blend_top4":
        candidates = (np.array(c)/10 for c in compositions(10, 4, False))
        p, _, w, _ = search_candidates(y, oof[:, :4], top4, candidates, sample_weight)
    elif strategy == "oof_grid_blend_top5":
        candidates = (np.array(c)/10 for c in compositions(10, 5, False))
        p, _, w, _ = search_candidates(y, oof, MODELS, candidates, sample_weight)
    elif strategy.startswith("oof_grid_blend_catboost_"):
        other_key = strategy.removeprefix("oof_grid_blend_catboost_")
        other = {"extratrees":"ExtraTreesClassifier", "xgboost":"XGBoostClassifier",
                 "lightgbm":"LightGBMClassifier"}[other_key]
        names = ["CatBoostClassifier", other]
        matrix = oof[:, [index[n] for n in names]]
        candidates = (np.array([i, 20-i])/20 for i in range(21))
        p, _, w, _ = search_candidates(y, matrix, names, candidates, sample_weight)
    elif strategy == "oof_sparse_top2_blend":
        best = None
        for members in itertools.combinations(MODELS, 2):
            matrix = oof[:, [index[n] for n in members]]
            candidates = (np.array([i, 20-i])/20 for i in range(1,20))
            candidate = search_candidates(y, matrix, list(members), candidates, sample_weight)
            if best is None or candidate[3] > best[3]: best = candidate
        p, _, w, _ = best
    elif strategy == "oof_sparse_top3_blend":
        best = None
        for members in itertools.combinations(MODELS, 3):
            matrix = oof[:, [index[n] for n in members]]
            candidates = (np.array(c)/10 for c in compositions(10, 3, True))
            candidate = search_candidates(y, matrix, list(members), candidates, sample_weight)
            if best is None or candidate[3] > best[3]: best = candidate
        p, _, w, _ = best
    elif strategy in {"rank_average_top4", "rank_average_top5"}:
        count = 4 if strategy.endswith("top4") else 5
        po, pt = rank_blend(oof[:, :count], test[:, :count])
        names = MODELS[:count]; weights = {n:1/count for n in names}
        return po, pt, weights, names, "fixed prior"
    elif strategy in {"median_probability_top4", "median_probability_top5"}:
        count = 4 if strategy.endswith("top4") else 5
        names = MODELS[:count]
        return np.median(oof[:,:count], axis=1), np.median(test[:,:count], axis=1), \
               {"method":"median"}, names, "fixed prior"
    else:
        raise ValueError(strategy)
    vector = np.array([w.get(m, 0.0) for m in MODELS])
    test_prob = test @ vector
    return p, test_prob, w, [m for m,v in w.items() if v > 0], "OOF train CV only"


def ensemble_evaluation(state: dict[str, list[dict]], splits, audits,
                        raw_features: list[str], al_columns: list[str]) -> None:
    completed = {ensemble_key(row) for row in state["ensemble_test"]}
    matrices = {}
    for seed in SEEDS:
        for feature_set in FEATURE_SETS:
            matrices[(seed,feature_set)] = prediction_matrices(state, seed, feature_set)
    for seed in SEEDS:
        for feature_set in FEATURE_SETS:
            oof_matrix, test_matrix, y_oof, y_test = matrices[(seed,feature_set)]
            shares = pd.Series(y_oof).value_counts(normalize=True).to_dict()
            sample_weight = np.where(y_oof == 0, .85/shares[0], .15/shares[1])
            for strategy in WITHIN_STRATEGIES:
                key = (seed, feature_set, strategy)
                if key in completed: continue
                run_start = time.perf_counter(); search_start = time.perf_counter()
                try:
                    p_oof, p_test, weights, members, source = within_strategy(
                        strategy, y_oof, oof_matrix, test_matrix, sample_weight
                    )
                    search_seconds = time.perf_counter()-search_start
                    threshold_start = time.perf_counter()
                    threshold, _ = fast_select_threshold(y_oof, p_oof, sample_weight)
                    threshold_seconds = time.perf_counter()-threshold_start
                    oof_row, test_row = metric_rows(
                        seed, feature_set, strategy, y_oof, p_oof, y_test, p_test,
                        threshold, weights, members, source, sample_weight,
                    )
                    state["ensemble_oof"].append(oof_row); state["ensemble_test"].append(test_row)
                    state["leakage"].append(ensemble_leakage(audits[seed],seed,feature_set,strategy))
                    _, transformed_test, _, _ = transform_config(
                        splits[seed][0], splits[seed][1], raw_features, al_columns, feature_set
                    )
                    all_nan = transformed_test.isna().all(axis=1).to_numpy()
                    pred = (p_test >= threshold).astype(int)
                    errors = pd.Series(error_types(y_test[all_nan],pred[all_nan])).value_counts().to_dict()
                    state["all_nan"].append({
                        "seed":seed,"feature_set":feature_set,"ensemble_strategy":strategy,
                        "all_nan_test_count":int(all_nan.sum()),
                        "all_nan_test_benign":int(((y_test==0)&all_nan).sum()),
                        "all_nan_test_pathogenic":int(((y_test==1)&all_nan).sum()),
                        "all_nan_test_tn":int(errors.get("TN",0)),"all_nan_test_fp":int(errors.get("FP",0)),
                        "all_nan_test_fn":int(errors.get("FN",0)),"all_nan_test_tp":int(errors.get("TP",0)),
                    })
                    status="COMPLETED"
                except Exception as exc:
                    status=f"SKIPPED/FAILED: {type(exc).__name__}: {exc}"
                    search_seconds=threshold_seconds=0.0
                    common={"seed":seed,"feature_set":feature_set,"ensemble_strategy":strategy,"status":status}
                    state["ensemble_oof"].append(common.copy()); state["ensemble_test"].append(common.copy())
                    state["leakage"].append(ensemble_leakage(audits[seed],seed,feature_set,strategy,"SKIPPED/FAILED"))
                    state["all_nan"].append(common.copy())
                state["runtime"].append({
                    "seed":seed,"feature_set":feature_set,"component_type":"ensemble_strategy","name":strategy,
                    "actual_runtime_seconds":time.perf_counter()-run_start,"split_seconds":0.0,
                    "feature_transform_seconds":0.0,"preprocessing_seconds":0.0,"leakage_seconds":0.0,
                    "oof_training_seconds":0.0,"final_fit_test_seconds":0.0,
                    "ensemble_weight_search_seconds":search_seconds,"threshold_selection_seconds":threshold_seconds,
                    "reporting_seconds":0.0,"status":status,"checkpoint_schema":CHECKPOINT_SCHEMA,
                })
                save_state(state); completed.add(key)

        # Cross-feature-set strategies use aligned row orders from the same seed split.
        for strategy in CROSS_STRATEGIES:
            key=(seed,"cross_feature_set",strategy)
            if key in completed: continue
            run_start=time.perf_counter(); search_start=time.perf_counter()
            try:
                oof_a,test_a,y_oof,y_test=matrices[(seed,"aa_numeric")]
                oof_b,test_b,_,_=matrices[(seed,"aa_numeric_plus_al_log")]
                names=[f"aa_numeric::{m}" for m in MODELS]+[f"aa_numeric_plus_al_log::{m}" for m in MODELS]
                pool_oof=np.column_stack([oof_a,oof_b]); pool_test=np.column_stack([test_a,test_b])
                shares=pd.Series(y_oof).value_counts(normalize=True).to_dict()
                sw=np.where(y_oof==0,.85/shares[0],.15/shares[1])
                if strategy=="cross_featureset_catboost_mean":
                    ids=[0,5]; weights={names[i]:.5 for i in ids}
                    p_oof=pool_oof[:,ids].mean(1);p_test=pool_test[:,ids].mean(1);source="fixed prior"
                elif strategy=="cross_featureset_best_fixed_mean":
                    ids=[0,1,5,6];weights={names[i]:.25 for i in ids}
                    p_oof=pool_oof[:,ids].mean(1);p_test=pool_test[:,ids].mean(1);source="fixed prior"
                else:
                    best=None
                    # At most 3 members, strictly-positive 0.10-grid weights.
                    for size in [1,2,3]:
                        for ids_tuple in itertools.combinations(range(10),size):
                            candidates=(np.array(c)/10 for c in compositions(10,size,True))
                            candidate=search_candidates(y_oof,pool_oof[:,ids_tuple],
                                                        [names[i] for i in ids_tuple],candidates,sw)
                            if best is None or candidate[3] > best[0][3]:
                                best = (candidate, ids_tuple)
                    (p_oof,_,weights,_),_=best
                    vector=np.array([weights.get(n,0.0) for n in names]);p_test=pool_test@vector;source="OOF train CV only"
                search_seconds=time.perf_counter()-search_start
                threshold_start=time.perf_counter();threshold,_=fast_select_threshold(y_oof,p_oof,sw)
                threshold_seconds=time.perf_counter()-threshold_start
                members=[n for n,v in weights.items() if isinstance(v,(float,int)) and v>0]
                oo,tt=metric_rows(seed,"cross_feature_set",strategy,y_oof,p_oof,y_test,p_test,
                                  threshold,weights,members,source,sw)
                state["ensemble_oof"].append(oo);state["ensemble_test"].append(tt)
                state["leakage"].append(ensemble_leakage(audits[seed],seed,"cross_feature_set",strategy))
                state["all_nan"].append({"seed":seed,"feature_set":"cross_feature_set",
                                          "ensemble_strategy":strategy,"all_nan_test_count":0,
                                          "all_nan_test_benign":0,"all_nan_test_pathogenic":0,
                                          "all_nan_test_tn":0,"all_nan_test_fp":0,"all_nan_test_fn":0,"all_nan_test_tp":0})
                status="COMPLETED"
            except Exception as exc:
                status=f"SKIPPED/FAILED: {type(exc).__name__}: {exc}";search_seconds=threshold_seconds=0.0
                common={"seed":seed,"feature_set":"cross_feature_set","ensemble_strategy":strategy,"status":status}
                state["ensemble_oof"].append(common.copy());state["ensemble_test"].append(common.copy())
                state["leakage"].append(ensemble_leakage(audits[seed],seed,"cross_feature_set",strategy,"SKIPPED/FAILED"))
                state["all_nan"].append(common.copy())
            state["runtime"].append({
                "seed":seed,"feature_set":"cross_feature_set","component_type":"ensemble_strategy","name":strategy,
                "actual_runtime_seconds":time.perf_counter()-run_start,"split_seconds":0.0,
                "feature_transform_seconds":0.0,"preprocessing_seconds":0.0,"leakage_seconds":0.0,
                "oof_training_seconds":0.0,"final_fit_test_seconds":0.0,
                "ensemble_weight_search_seconds":search_seconds,"threshold_selection_seconds":threshold_seconds,
                "reporting_seconds":0.0,"status":status,"checkpoint_schema":CHECKPOINT_SCHEMA,
            })
            save_state(state);completed.add(key)
        print(f"ENSEMBLES seed={seed} complete",flush=True)


def aggregate(test: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for (feature,strategy),g in test[test.status.eq("OK")].groupby(["feature_set","ensemble_strategy"],sort=False):
        rows.append({
            "feature_set":feature,"ensemble_strategy":strategy,"n_successful_seeds":len(g),
            "test_f1_mean":g.test_f1.mean(),"test_f1_std":g.test_f1.std(ddof=1),
            "test_f1_min":g.test_f1.min(),"test_f1_q25":g.test_f1.quantile(.25),
            "test_f1_median":g.test_f1.median(),"test_f1_q75":g.test_f1.quantile(.75),"test_f1_max":g.test_f1.max(),
            "test_precision_mean":g.test_precision.mean(),"test_precision_std":g.test_precision.std(ddof=1),
            "test_recall_mean":g.test_recall.mean(),"test_recall_std":g.test_recall.std(ddof=1),
            "test_fpr_mean":g.test_fpr.mean(),"test_fpr_std":g.test_fpr.std(ddof=1),
            "test_balanced_accuracy_mean":g.test_balanced_accuracy.mean(),"test_balanced_accuracy_std":g.test_balanced_accuracy.std(ddof=1),
            "test_fp_mean":g.test_fp.mean(),"test_fp_std":g.test_fp.std(ddof=1),
            "test_fn_mean":g.test_fn.mean(),"test_fn_std":g.test_fn.std(ddof=1),
            "selected_threshold_mean":g.selected_threshold_from_oof.mean(),
            "selected_threshold_std":g.selected_threshold_from_oof.std(ddof=1),
        })
    return pd.DataFrame(rows)


def main() -> None:
    wall_start=time.perf_counter()
    print("=== MASTER Step 6 estimated runtime ===",flush=True)
    print("5 seeds x 2 feature sets x 5 base models = 50 base runs",flush=True)
    print("Estimated 2.5-5.5 hours; up to 6 hours on slow machines",flush=True)
    print("Step 5 has no row-level probabilities, so base models are retrained with checkpoints.",flush=True)
    input_path=find_input();full=pd.read_csv(input_path);full["Label"]=full.Label.astype(int)
    raw_features=[c for c in full.columns if c not in ["Variant_ID","Label"]]
    al_columns=[c for c in raw_features if c.startswith("AL_") and pd.api.types.is_numeric_dtype(full[c])]
    full_duplicates=int(full.duplicated().sum())
    splits={};audits={};split_rows=[];split_seconds={}
    for seed in SEEDS:
        split_start=time.perf_counter()
        train,test=make_split(full,seed);splits[seed]=(train,test)
        audit,_=leakage_audit(full,train,test,raw_features,seed,full_duplicates);audits[seed]=audit
        split_seconds[seed]=time.perf_counter()-split_start
        split_rows.append({"seed":seed,"train_n":len(train),"test_n":len(test),
                           "train_benign":int(train.Label.eq(0).sum()),"train_pathogenic":int(train.Label.eq(1).sum()),
                           "test_benign":int(test.Label.eq(0).sum()),"test_pathogenic":int(test.Label.eq(1).sum()),
                           "train_benign_share":train.Label.eq(0).mean(),"test_benign_share":test.Label.eq(0).mean()})
    state=load_state();base_training(full,raw_features,al_columns,state,splits,audits,split_seconds)
    ensemble_evaluation(state,splits,audits,raw_features,al_columns)
    base_oof=pd.DataFrame(state["base_oof"]);base_test=pd.DataFrame(state["base_test"])
    base_metrics=pd.DataFrame(state["base_metrics"]);ens_oof=pd.DataFrame(state["ensemble_oof"])
    ens_test=pd.DataFrame(state["ensemble_test"]);leakage=pd.DataFrame(state["leakage"])
    runtime=pd.DataFrame(state["runtime"]);all_nan=pd.DataFrame(state["all_nan"])
    summary=aggregate(ens_test)
    refs={
        "aa_numeric":summary.query("feature_set=='aa_numeric' and ensemble_strategy=='catboost_single_reference'").iloc[0],
        "aa_numeric_plus_al_log":summary.query("feature_set=='aa_numeric_plus_al_log' and ensemble_strategy=='catboost_single_reference'").iloc[0],
    }
    delta_rows=[]
    for _,row in summary.iterrows():
        ref_key="aa_numeric" if row.feature_set=="cross_feature_set" else row.feature_set
        ref=refs[ref_key];d=row.test_f1_mean-ref.test_f1_mean
        delta_rows.append({
            "feature_set":row.feature_set,"ensemble_strategy":row.ensemble_strategy,
            "reference_name":f"{ref_key} CatBoost single reference","reference_f1_mean":ref.test_f1_mean,
            "ensemble_f1_mean":row.test_f1_mean,"delta_f1_mean":d,
            "delta_precision_mean":row.test_precision_mean-ref.test_precision_mean,
            "delta_recall_mean":row.test_recall_mean-ref.test_recall_mean,
            "delta_fpr_mean":row.test_fpr_mean-ref.test_fpr_mean,
            "delta_balanced_accuracy_mean":row.test_balanced_accuracy_mean-ref.test_balanced_accuracy_mean,
            "delta_fp_mean":row.test_fp_mean-ref.test_fp_mean,"delta_fn_mean":row.test_fn_mean-ref.test_fn_mean,
            "interpretation":"reference" if row.ensemble_strategy=="catboost_single_reference" else
                ("promising 5-seed ensemble evidence" if d>=.01 else
                 "below CatBoost reference" if d<=-.01 else "close/inconclusive 5-seed difference"),
        })
    delta=pd.DataFrame(delta_rows)
    ranking=summary.sort_values(["test_f1_mean","test_f1_std","test_fpr_mean","test_balanced_accuracy_mean"],
                                ascending=[False,True,True,False]).reset_index(drop=True)
    ranking.insert(0,"aggregate_rank",np.arange(1,len(ranking)+1))
    ranking["ranking_scope"]="aggregate repeated split ranking; not single-seed selection"
    pd.DataFrame(split_rows).to_csv(OUTPUTS["split"],index=False)
    base_oof.to_csv(OUTPUTS["base_oof"],index=False);base_test.to_csv(OUTPUTS["base_test"],index=False)
    base_metrics.to_csv(OUTPUTS["base_metrics"],index=False);ens_oof.to_csv(OUTPUTS["ensemble_oof"],index=False)
    ens_test.to_csv(OUTPUTS["ensemble_test"],index=False);summary.to_csv(OUTPUTS["summary"],index=False)
    delta.to_csv(OUTPUTS["delta"],index=False);leakage.to_csv(OUTPUTS["leakage"],index=False)
    all_nan.to_csv(OUTPUTS["all_nan"],index=False);runtime.to_csv(OUTPUTS["runtime"],index=False)
    ranking.to_csv(OUTPUTS["ranking"],index=False)
    base_ok=base_metrics.status.eq("OK").sum();ens_ok=ens_test.status.eq("OK").sum()
    actual_minutes=runtime.loc[runtime.status.eq("COMPLETED"),"actual_runtime_seconds"].sum()/60
    base_times=runtime[(runtime.component_type=="base_model")&runtime.status.eq("COMPLETED")]
    ens_times=runtime[(runtime.component_type=="ensemble_strategy")&runtime.status.eq("COMPLETED")]
    blends=ranking[~ranking.ensemble_strategy.str.endswith("single_reference")]
    best_blend=blends.iloc[0];best_ref=refs["aa_numeric"]
    beats=best_blend.test_f1_mean>=best_ref.test_f1_mean+.01
    old=summary.query("ensemble_strategy=='fixed_teknofest_safe_weights'").sort_values("test_f1_mean",ascending=False).iloc[0]
    cross=summary.query("feature_set=='cross_feature_set'").sort_values("test_f1_mean",ascending=False).iloc[0]
    next_step=("Refine the OOF-only threshold policy for the leading ensemble." if beats else
               "Ensembles did not clearly beat aa_numeric CatBoost; refine its OOF-only threshold policy.")
    metadata={
        "project_folder":"TeknoPatho","step":"Step 6 general OOF-only ensemble blending",
        "seed_list":SEEDS,"seed_count":5,"feature_sets":FEATURE_SETS,"base_models":MODELS,
        "ensemble_strategies":ALL_STRATEGIES,
        "split_rule":"same automatic prior-controlled split per seed across all models/strategies",
        "target_train_prior":{"benign":.15,"pathogenic":.85},"target_test_prior":{"benign":.85,"pathogenic":.15},
        "threshold_policy":"weighted_final_like_oof_f1_threshold","test_usage":"repeated split reporting only",
        "no_single_seed_selection":True,"eight_seed_pilot":False,"full_20_seed_confirmation":False,
        "old_teknofest_weights_included_as_baseline":True,"general_ensemble_search":True,
        "estimated_total_runtime_minutes":ESTIMATED_TOTAL_RUNTIME_MINUTES,"actual_total_runtime_minutes":actual_minutes,
        "created_files":[Path(__file__).name,*[p.name for p in OUTPUTS.values()],*[p.name for p in PARTIALS.values()]],
    }
    OUTPUTS["metadata"].write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    report=f"""# MASTER Step 6 general OOF-only ensemble blending

## Purpose and protocol

Step 5 showed CatBoost leading both feature sets. Step 6 tests general fixed, grid-selected, sparse, rank, median, and cross-feature probability blends. The old Teknofest health weights are only one fixed baseline strategy; this study is not limited to them.

- Five seeds={SEEDS}; no 8-seed pilot or 20-seed confirmation.
- Base models={MODELS}; feature sets={FEATURE_SETS}.
- Step 5 lacked row-level probabilities, so all base models were retrained and OOF/test probabilities saved.
- Fold preprocessing is train-fold-only. Ensemble weights and thresholds are OOF-only. Test is reporting-only.
- OOF-selected blends are not test-driven, but may overfit OOF and remain five-seed evidence rather than final proof.
- Estimated runtime={ESTIMATED_TOTAL_RUNTIME_MINUTES} min; accumulated actual runtime={actual_minutes:.2f} min; wall={((time.perf_counter()-wall_start)/60):.2f} min.

## Leakage summary

- FAIL={leakage.status.eq('FAIL').any()}; WARNING={leakage.status.eq('WARNING').any()}; statuses={leakage.status.value_counts().to_dict()}.

## Aggregate strategy results

{markdown_table(summary)}

## Delta versus Step 5 CatBoost

{markdown_table(delta)}

## Ranking

{markdown_table(ranking)}

## Key diagnostics

- Best non-single ensemble: {best_blend.feature_set}/{best_blend.ensemble_strategy}, F1={best_blend.test_f1_mean:.6f} ± {best_blend.test_f1_std:.6f}; precision={best_blend.test_precision_mean:.6f}, recall={best_blend.test_recall_mean:.6f}, FP={best_blend.test_fp_mean:.2f}, FN={best_blend.test_fn_mean:.2f}.
- Best aa_numeric CatBoost reference: F1={best_ref.test_f1_mean:.6f}. Ensemble clearly beats reference by predeclared +0.01 rule: {beats}.
- Old Teknofest weights best feature-set result: {old.feature_set}, F1={old.test_f1_mean:.6f}.
- Best cross-feature result: {cross.ensemble_strategy}, F1={cross.test_f1_mean:.6f}.
- AL-log remains useful only if an aa_numeric_plus_al_log/cross blend improves aggregate behavior without unacceptable FP/FPR; otherwise it remains secondary.

## Next step

{next_step}

This is five-seed general ensemble evidence, not a definitive final model. No single seed or test-driven weight was selected.
"""
    OUTPUTS["report"].write_text(report,encoding="utf-8")
    print("\n=== MASTER Step 6 final summary ===")
    print(f"Seeds/features/base models/strategies: {len(SEEDS)}/{len(FEATURE_SETS)}/{len(MODELS)}/{len(ALL_STRATEGIES)}")
    print(f"Successful base/ensemble runs: {base_ok}/{ens_ok}")
    print(f"Leakage FAIL/WARNING: {leakage.status.eq('FAIL').any()}/{leakage.status.eq('WARNING').any()}")
    print(f"Best ensemble: {best_blend.feature_set}/{best_blend.ensemble_strategy}; F1={best_blend.test_f1_mean:.6f} ± {best_blend.test_f1_std:.6f}")
    print(f"CatBoost reference={best_ref.test_f1_mean:.6f}; ensemble beats by +0.01={beats}")
    print(f"Old Teknofest fixed weights={old.test_f1_mean:.6f}; best cross={cross.ensemble_strategy}/{cross.test_f1_mean:.6f}")
    for feature in FEATURE_SETS:
        row=blends[blends.feature_set.eq(feature)].iloc[0]
        print(f"{feature} best ensemble={row.ensemble_strategy}/{row.test_f1_mean:.6f}")
    print(f"Best precision/recall/FP/FN={best_blend.test_precision_mean:.6f}/{best_blend.test_recall_mean:.6f}/{best_blend.test_fp_mean:.2f}/{best_blend.test_fn_mean:.2f}")
    print(f"Runtime={actual_minutes:.2f} min; avg base={base_times.actual_runtime_seconds.mean()/60:.2f}; avg ensemble={ens_times.actual_runtime_seconds.mean():.3f} sec")
    print(f"Estimated vs actual={ESTIMATED_TOTAL_RUNTIME_MINUTES}/{actual_minutes:.2f} min")
    print("8-seed pilot=False; 20-seed confirmation=False")
    print(f"Next step: {next_step}")


if __name__=="__main__":
    main()
