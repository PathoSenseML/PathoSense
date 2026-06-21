"""Cached CANCER OOF probabilities and lightweight blend scans.

Scope is intentionally narrow:
  * CANCER primary preprocessing stays no_rowdrop.
  * PCA stays rejected / ablation only.
  * No model search is performed.
  * Models are trained only once per fold to create validation-fold OOF
    probabilities; blend scans then run only on cached probabilities.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from experiment_logger import log_experiment_result, write_decision_log
from pca_experiments import (
    OUTPUT_DIR,
    apply_platt,
    bootstrap_f1_mcc,
    f1_at_prior,
    fit_platt,
    get_model_params,
    load_updated_panel,
    mcc_at_prior_analytic,
    prepare_fold,
    tpr_fpr,
)


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent / "denemeler" / "feature_selection"
PIPELINES_DIR = PROJECT_ROOT / "pipelines"
if str(PIPELINES_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINES_DIR))

from utils import TEST_PRIORS  # noqa: E402


OUT_DIR = OUTPUT_DIR / "cancer_primary"
CACHE_PATH = OUT_DIR / "cached_oof_probs.csv"
COARSE_PATH = OUT_DIR / "blend_scan_coarse.csv"
REFINED_PATH = OUT_DIR / "blend_scan_refined.csv"
SUMMARY_PATH = OUT_DIR / "blend_scan_best_summary.md"
META_PATH = OUT_DIR / "cached_oof_metadata.json"
TARGET_COL = "Label"
FINALS_N = 600
BOOTSTRAP_N = 300
COARSE_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
COARSE_THRESHOLDS = [0.70, 0.75, 0.80, 0.82, 0.85, 0.90]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def metric_row(y_true, probs, threshold, prior):
    preds = (probs >= threshold).astype(int)
    tpr, fpr = tpr_fpr(y_true, preds)
    f1 = f1_at_prior(tpr, fpr, prior)
    mcc = mcc_at_prior_analytic(tpr, fpr, prior)
    f1b = _fast_finals_adjusted_f1(tpr, fpr, prior)
    return {
        "f1_star": float(f1),
        "finals_adjusted_f1": float(f1b),
        "finals_adjusted_f1_method": "binomial_bootstrap_fast",
        "auc": float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else 0.5,
        "average_precision": float(average_precision_score(y_true, probs)) if len(np.unique(y_true)) > 1 else float(np.mean(y_true)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "mcc_star": float(mcc),
        "pred_positive_mean": float(preds.mean()),
    }


def _fast_finals_adjusted_f1(tpr: float, fpr: float, prior: float, seed: int = 42) -> float:
    n_pos = max(1, int(round(FINALS_N * prior)))
    n_neg = max(1, FINALS_N - n_pos)
    rng = np.random.RandomState(seed)
    tp = rng.binomial(n_pos, min(max(tpr, 0.0), 1.0), size=BOOTSTRAP_N)
    fp = rng.binomial(n_neg, min(max(fpr, 0.0), 1.0), size=BOOTSTRAP_N)
    fn = n_pos - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)
    return float(f1.mean())


def _train_own_fold(train_df: pd.DataFrame, test_df: pd.DataFrame):
    prep = prepare_fold(train_df, test_df, impute=True, outlier_control=True, drop_rows=False)
    Xtr, ytr, Xte, yte = prep.X_train, prep.y_train, prep.X_test, prep.y_test
    fit_idx, cal_idx = train_test_split(
        np.arange(len(ytr)),
        test_size=0.25,
        random_state=42,
        stratify=ytr,
    )
    spw = (ytr.iloc[fit_idx] == 0).sum() / max(int((ytr.iloc[fit_idx] == 1).sum()), 1)
    params = get_model_params(len(fit_idx), spw)
    params["n_jobs"] = -1
    params["n_estimators"] = min(int(params.get("n_estimators", 300)), 120)
    params["learning_rate"] = max(float(params.get("learning_rate", 0.03)), 0.05)
    model = LGBMClassifier(**params, random_state=42)
    model.fit(
        Xtr.iloc[fit_idx],
        ytr.iloc[fit_idx],
        eval_set=[(Xtr.iloc[cal_idx], ytr.iloc[cal_idx])],
        callbacks=[lgb.early_stopping(15, verbose=False), lgb.log_evaluation(0)],
    )
    cal_raw = model.predict_proba(Xtr.iloc[cal_idx])[:, 1]
    test_raw = model.predict_proba(Xte)[:, 1]
    platt = fit_platt(cal_raw, ytr.iloc[cal_idx].values)
    return {
        "own_cal_raw": cal_raw,
        "own_calibrated": apply_platt(test_raw, platt),
        "own_raw": test_raw,
        "y_cal": ytr.iloc[cal_idx].reset_index(drop=True),
        "y_test": yte.reset_index(drop=True),
        "cal_original_index": cal_idx,
        "metadata": prep.metadata,
        "fit_size": int(len(fit_idx)),
        "calibration_size": int(len(cal_idx)),
        "output_features": int(Xtr.shape[1]),
        "dropped_features": int(train_df.shape[1] - 1 - Xtr.shape[1]),
    }


def _transfer_scores_lgbm(cancer: pd.DataFrame):
    master = load_updated_panel("MASTER")
    source = master[~master["Variant_ID"].isin(set(cancer["Variant_ID"]))].reset_index(drop=True)
    prep = prepare_fold(source, cancer.reset_index(drop=True), impute=True, outlier_control=True, drop_rows=False)
    Xs, ys, Xt = prep.X_train, prep.y_train, prep.X_test
    spw = (ys == 0).sum() / max(int((ys == 1).sum()), 1)
    params = get_model_params(len(Xs), spw)
    params["n_jobs"] = -1
    params["n_estimators"] = min(int(params.get("n_estimators", 300)), 160)
    params["learning_rate"] = max(float(params.get("learning_rate", 0.03)), 0.05)
    fit_idx, es_idx = train_test_split(np.arange(len(ys)), test_size=0.15, random_state=42, stratify=ys)
    model = LGBMClassifier(**params, random_state=42)
    model.fit(
        Xs.iloc[fit_idx],
        ys.iloc[fit_idx],
        eval_set=[(Xs.iloc[es_idx], ys.iloc[es_idx])],
        callbacks=[lgb.early_stopping(15, verbose=False), lgb.log_evaluation(0)],
    )
    info = {
        "transfer_scorer": "LGBM source-only updated MASTER excluding all CANCER IDs",
        "source_rows": int(len(source)),
        "source_features": int(Xs.shape[1]),
        "excluded_cancer_ids": int(cancer["Variant_ID"].nunique()),
    }
    return model.predict_proba(Xt)[:, 1], info


def _completed_fold_ids(cache_path: Path, n_splits: int, n_repeats: int) -> set[int]:
    if not cache_path.exists():
        return set()
    cache = pd.read_csv(cache_path)
    if cache.empty:
        return set()
    same_plan = (
        cache.get("n_splits", pd.Series(dtype=int)).astype(int).eq(n_splits).all()
        and cache.get("n_repeats", pd.Series(dtype=int)).astype(int).eq(n_repeats).all()
    )
    if not same_plan:
        raise ValueError(f"{cache_path} exists but was generated with a different CV plan.")
    complete = cache[cache["status"].eq("complete")]
    return set(complete["fold_id"].astype(int).unique())


def generate_cached_oof(n_splits: int, n_repeats: int, max_seconds: float | None = None) -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cancer = load_updated_panel("KANSER").reset_index(drop=True)
    y = cancer[TARGET_COL].astype(int).reset_index(drop=True)
    total_folds = n_splits * n_repeats
    completed = _completed_fold_ids(CACHE_PATH, n_splits, n_repeats)
    if len(completed) == total_folds:
        print(f"Using existing complete cache: {CACHE_PATH}", flush=True)
        return pd.read_csv(CACHE_PATH)

    g_raw, transfer_info = _transfer_scores_lgbm(cancer)
    META_PATH.write_text(json.dumps({
        "created_at": _now(),
        "cache_path": str(CACHE_PATH),
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "preprocess_mode": "no_rowdrop",
        "pca_enabled": False,
        "calibration_method": "Platt/sigmoid",
        "transfer_info": transfer_info,
    }, indent=2), encoding="utf-8")
    print(f"Transfer scorer ready: {transfer_info}", flush=True)

    started = time.time()
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    for fold_id, (tr, te) in enumerate(splitter.split(cancer, y), start=1):
        if fold_id in completed:
            print(f"Fold {fold_id}/{total_folds} already cached; skipping.", flush=True)
            continue
        if max_seconds is not None and time.time() - started > max_seconds:
            print(f"Stopping cache generation after {round(time.time() - started, 1)}s; partial cache saved.", flush=True)
            break

        repeat_id = (fold_id - 1) // n_splits + 1
        split_id = (fold_id - 1) % n_splits + 1
        print(f"Training cache fold {fold_id}/{total_folds}", flush=True)
        own = _train_own_fold(cancer.iloc[tr].reset_index(drop=True), cancer.iloc[te].reset_index(drop=True))
        cal_global = tr[own["cal_original_index"]]
        g_platt = fit_platt(g_raw[cal_global], own["y_cal"].values)
        g_test_calibrated = apply_platt(g_raw[te], g_platt)
        rows = pd.DataFrame({
            "fold_id": fold_id,
            "repeat_id": repeat_id,
            "split_id": split_id,
            "sample_id": te,
            "variant_id": cancer.iloc[te]["Variant_ID"].to_numpy() if "Variant_ID" in cancer.columns else te,
            "y_true": own["y_test"].to_numpy(),
            "cancer_own_prob": own["own_calibrated"],
            "general_transfer_prob": g_test_calibrated,
            "cancer_own_prob_raw": own["own_raw"],
            "general_transfer_prob_raw": g_raw[te],
            "cancer_own_prob_calibrated": own["own_calibrated"],
            "general_transfer_prob_calibrated": g_test_calibrated,
            "n_splits": n_splits,
            "n_repeats": n_repeats,
            "train_size": len(tr),
            "validation_size": len(te),
            "model_fit_size": own["fit_size"],
            "calibration_size": own["calibration_size"],
            "preprocess_mode": "no_rowdrop",
            "row_drop_enabled": False,
            "pca_enabled": False,
            "outlier_control_enabled": True,
            "imputation_method": "median",
            "encoding_method": "safe ordinal / label encoding",
            "calibration_method": "Platt/sigmoid",
            "output_features": own["output_features"],
            "dropped_features": own["dropped_features"],
            "status": "complete",
            "created_at": _now(),
        })
        rows.to_csv(CACHE_PATH, mode="a", header=not CACHE_PATH.exists(), index=False)
        completed.add(fold_id)
        print(f"Saved fold {fold_id} rows to {CACHE_PATH}", flush=True)

    return pd.read_csv(CACHE_PATH)


def _existing_complete_combos(path: Path) -> set[tuple[float, float]]:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if df.empty or "status" not in df:
        return set()
    done = df[df["status"].eq("complete")]
    return {(round(float(r.weight), 4), round(float(r.threshold), 4)) for r in done.itertuples(index=False)}


def _append_scan_row(path: Path, row: dict) -> None:
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def _evaluate_combo(cache: pd.DataFrame, weight: float, threshold: float, prior: float) -> dict:
    rows = []
    for fold_id, fold in cache.groupby("fold_id", sort=True):
        probs = weight * fold["general_transfer_prob"].astype(float).to_numpy()
        probs += (1.0 - weight) * fold["cancer_own_prob"].astype(float).to_numpy()
        metrics = metric_row(fold["y_true"].astype(int).to_numpy(), probs, threshold, prior)
        metrics["fold_id"] = fold_id
        rows.append(metrics)
    fold_metrics = pd.DataFrame(rows)
    out = {
        "weight": weight,
        "threshold": threshold,
        "f1_star": float(fold_metrics["f1_star"].mean()),
        "finals_adjusted_f1": float(fold_metrics["finals_adjusted_f1"].mean()),
        "auc": float(fold_metrics["auc"].mean()),
        "average_precision": float(fold_metrics["average_precision"].mean()),
        "precision": float(fold_metrics["precision"].mean()),
        "recall": float(fold_metrics["recall"].mean()),
        "mcc_star": float(fold_metrics["mcc_star"].mean()),
        "pred_positive_mean": float(fold_metrics["pred_positive_mean"].mean()),
        "n_folds": int(fold_metrics.shape[0]),
        "status": "complete",
        "scan_version": "cached_fast_v3",
        "created_at": _now(),
    }
    return out


def _prepare_scan_file(path: Path) -> None:
    if not path.exists():
        return
    existing = pd.read_csv(path, nrows=5)
    if "scan_version" in existing.columns and existing["scan_version"].eq("cached_fast_v3").all():
        return
    backup = path.with_name(f"{path.stem}_legacy_partial_{datetime.now().strftime('%Y%m%d_%H%M%S')}{path.suffix}")
    path.replace(backup)
    print(f"Moved legacy partial scan to {backup}", flush=True)


def _central_log_row(row: dict, scan_name: str, output_file: Path, baseline_f1: float) -> dict:
    weight = float(row["weight"])
    return {
        "panel": "CANCER",
        "experiment_name": f"cancer_cached_oof_{scan_name}_w{weight:.2f}_t{float(row['threshold']):.2f}",
        "pipeline_role": "sanity_check",
        "preprocess_mode": "no_rowdrop",
        "row_drop_enabled": False,
        "pca_enabled": False,
        "pca_variant": "none",
        "scaling_enabled": False,
        "outlier_control_enabled": True,
        "imputation_method": "median",
        "encoding_method": "safe ordinal / label encoding",
        "feature_selection_method": "none",
        "model_name": "fixed current LightGBM cancer-own + general transfer",
        "calibration_method": "Platt/sigmoid",
        "threshold_strategy": f"fixed_{float(row['threshold']):.2f}",
        "selected_threshold": float(row["threshold"]),
        "blend_enabled": True,
        "blend_weight": weight,
        "general_transfer_weight": weight,
        "cancer_own_weight": round(1.0 - weight, 4),
        "cv_strategy": "RepeatedStratifiedKFold cached validation-fold OOF",
        "n_folds": row.get("n_folds", "n/a"),
        "input_rows": 388,
        "output_rows": 388,
        "dropped_rows": 0,
        "input_features": 364,
        "output_features": "fold-local",
        "dropped_features": "fold-local",
        "positive_ratio_before": 0.690722,
        "positive_ratio_after": 0.690722,
        "auc": row.get("auc", "n/a"),
        "f1_star": row.get("f1_star", "n/a"),
        "finals_adjusted_f1": row.get("finals_adjusted_f1", "n/a"),
        "precision": row.get("precision", "n/a"),
        "recall": row.get("recall", "n/a"),
        "pred_positive_mean": row.get("pred_positive_mean", "n/a"),
        "delta_f1_vs_baseline": float(row["f1_star"]) - baseline_f1,
        "notes": "Cached OOF fixed-threshold blend scan; no PCA, no model search, no retraining during scan. finals_adjusted_f1 uses fast binomial bootstrap.",
        "output_file": output_file,
    }


def scan_cached_probs(cache: pd.DataFrame, weights: list[float], thresholds: list[float], path: Path, scan_name: str) -> pd.DataFrame:
    prior = TEST_PRIORS["cancer"]
    complete_cache = cache[cache["status"].eq("complete")].copy()
    _prepare_scan_file(path)
    done = _existing_complete_combos(path)
    baseline_f1 = _baseline_f1_for_scan(complete_cache, weights, thresholds, path, prior)
    for weight in weights:
        for threshold in thresholds:
            combo = (round(float(weight), 4), round(float(threshold), 4))
            if combo in done:
                continue
            try:
                row = _evaluate_combo(complete_cache, float(weight), float(threshold), prior)
                row["delta_f1_vs_cancer_own"] = float(row["f1_star"]) - baseline_f1
                row["scan_name"] = scan_name
                _append_scan_row(path, row)
                log_experiment_result(_central_log_row(row, scan_name, path, baseline_f1))
                done.add(combo)
            except Exception as exc:  # keep scans resumable
                _append_scan_row(path, {
                    "weight": weight,
                    "threshold": threshold,
                    "f1_star": "n/a",
                    "finals_adjusted_f1": "n/a",
                    "auc": "n/a",
                    "precision": "n/a",
                    "recall": "n/a",
                    "pred_positive_mean": "n/a",
                    "delta_f1_vs_cancer_own": "n/a",
                    "status": "failed",
                    "error": str(exc),
                    "scan_name": scan_name,
                    "created_at": _now(),
                })
    return pd.read_csv(path)


def _baseline_f1_for_scan(cache: pd.DataFrame, weights: list[float], thresholds: list[float], path: Path, prior: float) -> float:
    if path.exists():
        existing = pd.read_csv(path)
        own = existing[(existing.get("status", "") == "complete") & np.isclose(existing["weight"].astype(float), 0.0)]
        if not own.empty:
            return float(own["f1_star"].astype(float).max())
    if 0.0 in weights:
        return max(_evaluate_combo(cache, 0.0, float(t), prior)["f1_star"] for t in thresholds)
    return max(_evaluate_combo(cache, 0.0, float(t), prior)["f1_star"] for t in COARSE_THRESHOLDS)


def refined_grid_from(coarse: pd.DataFrame) -> tuple[list[float], list[float]]:
    complete = coarse[coarse["status"].eq("complete")].copy()
    best = complete.sort_values(["f1_star", "finals_adjusted_f1", "auc"], ascending=False).iloc[0]
    best_w = float(best["weight"])
    best_t = float(best["threshold"])
    weights = [round(x, 2) for x in np.arange(max(0.0, best_w - 0.10), min(1.0, best_w + 0.10) + 0.0001, 0.05)]
    thresholds = [round(x, 2) for x in np.arange(max(0.70, best_t - 0.03), min(0.90, best_t + 0.03) + 0.0001, 0.01)]
    return weights, thresholds


def write_summary(coarse: pd.DataFrame, refined: pd.DataFrame | None, path: Path) -> None:
    sections = [
        "# CANCER Cached OOF Blend Scan",
        "",
        "Fixed decisions: `no_rowdrop`, safe encoding, median imputation, outlier control, Platt/sigmoid calibration, no PCA.",
        "",
        "The scan uses cached validation-fold probabilities. Model training is skipped whenever `cached_oof_probs.csv` is already complete.",
        "",
        "## Coarse Best",
        "",
        _markdown_table(coarse[coarse["status"].eq("complete")].sort_values(["f1_star", "finals_adjusted_f1", "auc"], ascending=False).head(10)),
    ]
    if refined is not None and not refined.empty:
        sections.extend([
            "",
            "## Refined Best",
            "",
            _markdown_table(refined[refined["status"].eq("complete")].sort_values(["f1_star", "finals_adjusted_f1", "auc"], ascending=False).head(10)),
            "",
            "## Conservative 0.80-0.82",
            "",
            _markdown_table(refined[refined["threshold"].astype(float).between(0.80, 0.82)]),
        ])
    path.write_text("\n".join(sections) + "\n", encoding="utf-8")


def _markdown_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_No rows._"
    cols = list(df.columns)
    rows = [[str(v) for v in row] for row in df.astype(object).itertuples(index=False, name=None)]
    widths = [max(len(str(c)), *(len(r[i]) for r in rows)) for i, c in enumerate(cols)]

    def fmt(vals):
        return "| " + " | ".join(str(v).replace("\n", " ").ljust(widths[i]) for i, v in enumerate(vals)) + " |"

    return "\n".join([fmt(cols), "| " + " | ".join("-" * w for w in widths) + " |"] + [fmt(r) for r in rows])


def run(n_splits: int, n_repeats: int, max_seconds: float | None = None) -> None:
    cache = generate_cached_oof(n_splits, n_repeats, max_seconds=max_seconds)
    expected_folds = n_splits * n_repeats
    complete_folds = cache[cache["status"].eq("complete")]["fold_id"].nunique()
    if complete_folds < expected_folds:
        print(f"Cache is partial: {complete_folds}/{expected_folds} folds. Blend scan will wait for a complete cache.", flush=True)
        return

    coarse = scan_cached_probs(cache, COARSE_WEIGHTS, COARSE_THRESHOLDS, COARSE_PATH, "coarse")
    refined_weights, refined_thresholds = refined_grid_from(coarse)
    refined = scan_cached_probs(cache, refined_weights, refined_thresholds, REFINED_PATH, "refined")
    write_summary(coarse, refined, SUMMARY_PATH)
    best = refined[refined["status"].eq("complete")].sort_values(["f1_star", "finals_adjusted_f1", "auc"], ascending=False).iloc[0]
    write_decision_log([
        "Strict row-drop was rejected as the primary CANCER preprocessing because it dropped 158/388 rows and changed the positive class ratio from 69.07% to 59.13%.",
        "CANCER primary preprocessing is no_rowdrop.",
        "PCA remains rejected for the primary CANCER pipeline.",
        "Primary CANCER pipeline remains updated CAT features + no_rowdrop + safe encoding + median imputation + outlier control + no PCA.",
        "Calibration strategy remains Platt/sigmoid for now.",
        "Threshold strategy remains aligned with the conservative 0.80-0.82 range while cached blend scans are reviewed.",
        f"Latest cached OOF blend candidate: weight={best['weight']}, threshold={best['threshold']}, F1*={best['f1_star']}, finals-adjusted F1={best['finals_adjusted_f1']}.",
    ])
    print(f"Saved {CACHE_PATH}", flush=True)
    print(f"Saved {COARSE_PATH}", flush=True)
    print(f"Saved {REFINED_PATH}", flush=True)
    print(f"Saved {SUMMARY_PATH}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate cached CANCER OOF probabilities and scan blend weights from cache.")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--max-seconds", type=float, default=None)
    args = parser.parse_args()
    run(args.splits, args.repeats, max_seconds=args.max_seconds)


if __name__ == "__main__":
    main()
