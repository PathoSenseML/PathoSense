"""Leakage-safe PCA experiment scaffolding for updated MASTER/CANCER data.

This script prepares the three requested PCA variants without running blend
experiments or searching new model families. The fixed evaluator is the existing
project LightGBM configuration from pipelines/run_pipeline.py, reused through the
honest threshold/metric helpers from analysis/harness.py.

Run examples:
    python pca_experiments.py --list
    python pca_experiments.py --panel MASTER --experiment baseline --folds 3 --repeats 1
    python pca_experiments.py --panel KANSER --experiment exp2 --components 0.95
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from preprocess_updated_cats import (
    COMBINED_CAT_FILE,
    OUTPUT_DIR,
    PANEL_MAP,
    TARGET_COL,
    UpdatedCategoricalPreprocessor,
    load_combined_categorical,
    merge_updated_categorical_features,
    split_master_cancer,
)
from experiment_logger import log_experiment_result


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent / "denemeler" / "feature_selection"
PIPELINES_DIR = PROJECT_ROOT / "pipelines"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
for path in (str(PIPELINES_DIR), str(ANALYSIS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from run_pipeline import get_model_params  # noqa: E402
from harness import (  # noqa: E402
    FINALS_N,
    bootstrap_f1_mcc,
    f1_at_prior,
    lgb_oof_and_test,
    mcc_at_prior_analytic,
    pick_threshold_at_prior,
    tpr_fpr,
)
from calibration import apply_platt, fit_platt  # noqa: E402
from utils import TEST_PRIORS  # noqa: E402


PCA_OUTPUT_DIR = OUTPUT_DIR / "pca_experiments"


@dataclass
class FoldPrepared:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    metadata: dict[str, Any]


class QuantileOutlierClipper:
    """Train-only p1/p99 clipping, matching the project's percentile clipping style."""

    def __init__(self, columns: list[str], lower: float = 0.01, upper: float = 0.99):
        self.columns = list(columns)
        self.lower = lower
        self.upper = upper
        self.bounds_: dict[str, tuple[float, float]] = {}

    def fit(self, X: pd.DataFrame) -> "QuantileOutlierClipper":
        for col in self.columns:
            if col not in X.columns:
                continue
            s = pd.to_numeric(X[col], errors="coerce").dropna()
            if len(s) < 10:
                continue
            lo, hi = float(s.quantile(self.lower)), float(s.quantile(self.upper))
            if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                self.bounds_[col] = (lo, hi)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, (lo, hi) in self.bounds_.items():
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce").clip(lo, hi)
        return X


def _json_safe(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def load_updated_panel(panel: str, combined_path: Path | str = COMBINED_CAT_FILE) -> pd.DataFrame:
    panel = panel.upper()
    if panel not in PANEL_MAP:
        raise ValueError(f"Unsupported panel {panel!r}; expected MASTER or KANSER")
    combined = load_combined_categorical(combined_path)
    parts = split_master_cancer(combined, OUTPUT_DIR / "intermediate")
    return merge_updated_categorical_features(panel, parts[panel])


def _align_y(df: pd.DataFrame, mask: pd.Series) -> pd.Series:
    return df.loc[mask.to_numpy(), TARGET_COL].reset_index(drop=True).astype(int)


def _median_impute_from_preprocessor(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    pre: UpdatedCategoricalPreprocessor,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train = X_train.copy()
    X_test = X_test.copy()
    for col, val in pre.numeric_medians_.items():
        if col in X_train.columns:
            X_train[col] = X_train[col].fillna(val)
        if col in X_test.columns:
            X_test[col] = X_test[col].fillna(val)
    # Categorical columns are encoded already; keep any defensive NaN fill finite.
    return X_train.fillna(0.0), X_test.fillna(0.0)


def prepare_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    impute: bool,
    outlier_control: bool,
    drop_rows: bool,
    no_imputation_pca_safe: bool = False,
) -> FoldPrepared | None:
    pre = UpdatedCategoricalPreprocessor(missing_threshold=0.80, impute_numeric=False)
    X_train = pre.fit_transform(train_df, drop_rows=drop_rows)
    y_train = _align_y(train_df, pre.last_row_keep_mask_)
    train_rows_dropped = int((~pre.last_row_keep_mask_).sum())

    X_test = pre.transform(test_df, drop_rows=drop_rows)
    y_test = _align_y(test_df, pre.last_row_keep_mask_)
    test_rows_dropped = int((~pre.last_row_keep_mask_).sum())

    metadata = {
        "preprocess_mode": "strict" if drop_rows else "no_rowdrop",
        "row_drop_enabled": drop_rows,
        "original_train_shape": train_df.shape,
        "original_test_shape": test_df.shape,
        "shape_after_missing_filter_train": X_train.shape,
        "shape_after_missing_filter_test": X_test.shape,
        "dropped_columns": pre.dropped_columns_,
        "dropped_train_rows": train_rows_dropped,
        "dropped_test_rows": test_rows_dropped,
        "numeric_columns": pre.numeric_columns_,
        "categorical_columns": pre.categorical_columns_,
    }

    if no_imputation_pca_safe:
        train_nan_cols = X_train.columns[X_train.isna().any()].tolist()
        if train_nan_cols:
            X_train = X_train.drop(columns=train_nan_cols)
            X_test = X_test.drop(columns=[c for c in train_nan_cols if c in X_test.columns])
            metadata["pca_without_imputation_train_nan_columns_dropped"] = train_nan_cols
        if X_test.isna().any().any():
            keep = ~X_test.isna().any(axis=1)
            metadata["pca_without_imputation_test_rows_dropped_after_train_col_drop"] = int((~keep).sum())
            X_test = X_test.loc[keep].reset_index(drop=True)
            y_test = y_test.loc[keep.to_numpy()].reset_index(drop=True)
        if X_train.empty or X_test.empty or X_train.shape[1] == 0:
            metadata["skip_reason"] = "PCA without imputation left no complete feature matrix"
            return None
    else:
        if outlier_control:
            clipper = QuantileOutlierClipper(pre.numeric_columns_).fit(X_train)
            X_train = clipper.transform(X_train)
            X_test = clipper.transform(X_test)
            metadata["outlier_bounds"] = clipper.bounds_
        if impute:
            X_train, X_test = _median_impute_from_preprocessor(X_train, X_test, pre)
            metadata["imputation_values"] = pre.numeric_medians_

    return FoldPrepared(X_train, y_train, X_test, y_test, metadata)


def apply_pca(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    components: int | float,
    scale: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    Xtr = X_train.astype(float)
    Xte = X_test.astype(float)
    scaler = None
    if scale:
        scaler = StandardScaler()
        Xtr_arr = scaler.fit_transform(Xtr)
        Xte_arr = scaler.transform(Xte)
    else:
        Xtr_arr = Xtr.to_numpy()
        Xte_arr = Xte.to_numpy()

    max_components = min(Xtr_arr.shape[0], Xtr_arr.shape[1])
    n_components: int | float = components
    if isinstance(components, int):
        n_components = min(components, max_components)
    pca = PCA(n_components=n_components, random_state=42)
    Ztr = pca.fit_transform(Xtr_arr)
    Zte = pca.transform(Xte_arr)
    cols = [f"PC{i + 1}" for i in range(Ztr.shape[1])]
    info = {
        "scaling": "StandardScaler" if scale else None,
        "requested_components": components,
        "n_components": int(pca.n_components_),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "explained_variance_total": float(np.sum(pca.explained_variance_ratio_)),
    }
    if scaler is not None:
        info["scaler_mean_features"] = int(len(scaler.mean_))
    return pd.DataFrame(Ztr, columns=cols), pd.DataFrame(Zte, columns=cols), info


def evaluate_fixed_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    panel_project_name: str,
    n_seeds: int,
) -> dict[str, float]:
    spw = (y_train == 0).sum() / max(int((y_train == 1).sum()), 1)
    params = get_model_params(len(X_train), spw)
    n_folds = min(5, max(2, int(y_train.sum()), int((y_train == 0).sum())))
    test_probs, train_oof = lgb_oof_and_test(
        X_train,
        y_train,
        X_test,
        params,
        n_seeds=n_seeds,
        n_folds=n_folds,
    )
    platt = fit_platt(train_oof, y_train.values)
    train_oof = apply_platt(train_oof, platt)
    test_probs = apply_platt(test_probs, platt)
    prior = TEST_PRIORS.get(panel_project_name, 0.5)
    threshold = pick_threshold_at_prior(y_train.values, train_oof, prior)
    preds = (test_probs >= threshold).astype(int)
    tpr, fpr = tpr_fpr(y_test.values, preds)
    f1 = f1_at_prior(tpr, fpr, prior)
    mcc = mcc_at_prior_analytic(tpr, fpr, prior)
    finals_n = FINALS_N.get(panel_project_name, 0)
    f1_boot, _ = bootstrap_f1_mcc(y_test.values, preds, prior, finals_n) if finals_n else (f1, mcc)
    try:
        auc = roc_auc_score(y_test, test_probs)
        ap = average_precision_score(y_test, test_probs)
    except ValueError:
        auc = 0.5
        ap = float(y_test.mean())
    return {
        "threshold": float(threshold),
        "f1_at_prior": float(f1),
        "mcc_at_prior": float(mcc),
        "f1_finals_bootstrap": float(f1_boot),
        "auc": float(auc),
        "average_precision": float(ap),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "tpr": float(tpr),
        "fpr": float(fpr),
    }


EXPERIMENTS = {
    "baseline": {
        "description": "Missing filtering + outlier control + median imputation, no PCA",
        "impute": True,
        "outlier_control": True,
        "pca": False,
        "scale": False,
        "no_imputation_pca_safe": False,
    },
    "exp1": {
        "description": "PCA before imputation; safely drops train-missing columns and logs if impossible",
        "impute": False,
        "outlier_control": False,
        "pca": True,
        "scale": False,
        "no_imputation_pca_safe": True,
    },
    "exp2": {
        "description": "Missing filtering + outlier control + median imputation + PCA",
        "impute": True,
        "outlier_control": True,
        "pca": True,
        "scale": False,
        "no_imputation_pca_safe": False,
    },
    "exp3": {
        "description": "Best previous PCA setup plus StandardScaler before PCA",
        "impute": True,
        "outlier_control": True,
        "pca": True,
        "scale": True,
        "no_imputation_pca_safe": False,
    },
}

PCA_VARIANT_NAMES = {
    "baseline": "none",
    "exp1": "pca_before_imputation",
    "exp2": "impute_pca",
    "exp3": "scale_pca",
}


def run_experiment(
    panel: str,
    experiment: str,
    *,
    components: int | float = 0.95,
    preprocess_mode: str = "auto",
    folds: int = 5,
    repeats: int = 1,
    n_seeds: int = 1,
    output_dir: Path | str = PCA_OUTPUT_DIR,
) -> dict[str, Any]:
    panel = panel.upper()
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment {experiment!r}; expected {sorted(EXPERIMENTS)}")
    if preprocess_mode == "auto":
        preprocess_mode = default_preprocess_mode(panel)
    if preprocess_mode not in {"strict", "no_rowdrop"}:
        raise ValueError("preprocess_mode must be 'auto', 'strict', or 'no_rowdrop'")
    cfg = EXPERIMENTS[experiment]
    drop_rows = preprocess_mode == "strict"
    pipeline_role = pipeline_role_for_mode(panel, preprocess_mode)
    df = load_updated_panel(panel)
    project_panel = PANEL_MAP[panel]["project_panel"]
    y = df[TARGET_COL].astype(int).reset_index(drop=True)

    rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=42)
    fold_rows = []
    skipped = []
    t0 = time.time()

    for fold_id, (tr, te) in enumerate(rskf.split(df, y), start=1):
        train_df = df.iloc[tr].reset_index(drop=True)
        test_df = df.iloc[te].reset_index(drop=True)
        prepared = prepare_fold(
            train_df,
            test_df,
            impute=cfg["impute"],
            outlier_control=cfg["outlier_control"],
            drop_rows=drop_rows,
            no_imputation_pca_safe=cfg["no_imputation_pca_safe"],
        )
        if prepared is None:
            skipped.append({"fold": fold_id, "reason": "PCA could not be applied without handling missing values"})
            continue

        Xtr, Xte = prepared.X_train, prepared.X_test
        pca_info = None
        if cfg["pca"]:
            if Xtr.isna().any().any() or Xte.isna().any().any():
                skipped.append({
                    "fold": fold_id,
                    "reason": "standard PCA cannot handle NaN values",
                    "train_nan_cells": int(Xtr.isna().sum().sum()),
                    "test_nan_cells": int(Xte.isna().sum().sum()),
                })
                continue
            Xtr, Xte, pca_info = apply_pca(Xtr, Xte, components=components, scale=cfg["scale"])

        metrics = evaluate_fixed_lgbm(
            Xtr,
            prepared.y_train,
            Xte,
            prepared.y_test,
            panel_project_name=project_panel,
            n_seeds=n_seeds,
        )
        fold_rows.append({
            "fold": fold_id,
            "panel": panel,
            "experiment": experiment,
            "preprocess_mode": preprocess_mode,
            "pipeline_role": pipeline_role,
            "dataset_original_shape": df.shape,
            "shape_after_missing_filter_train": prepared.metadata["shape_after_missing_filter_train"],
            "shape_after_missing_filter_test": prepared.metadata["shape_after_missing_filter_test"],
            "model_train_shape": Xtr.shape,
            "model_test_shape": Xte.shape,
            "pca": pca_info,
            "metrics": metrics,
            "preprocessing": prepared.metadata,
        })
        n_comp = pca_info["n_components"] if pca_info else "none"
        print(
            f"[{panel}][{experiment}][fold {fold_id}] "
            f"train={Xtr.shape} test={Xte.shape} PCA={n_comp} "
            f"F1*={metrics['f1_at_prior']:.4f}"
        )

    if not fold_rows:
        summary = {
            "panel": panel,
            "experiment": experiment,
            "preprocess_mode": preprocess_mode,
            "pipeline_role": pipeline_role,
            "status": "skipped",
            "skipped_folds": skipped,
            "description": cfg["description"],
        }
    else:
        f1s = np.array([r["metrics"]["f1_at_prior"] for r in fold_rows], dtype=float)
        mccs = np.array([r["metrics"]["mcc_at_prior"] for r in fold_rows], dtype=float)
        f1_finals = np.array([r["metrics"]["f1_finals_bootstrap"] for r in fold_rows], dtype=float)
        aucs = np.array([r["metrics"]["auc"] for r in fold_rows], dtype=float)
        precisions = np.array([r["metrics"]["precision"] for r in fold_rows], dtype=float)
        recalls = np.array([r["metrics"]["recall"] for r in fold_rows], dtype=float)
        pca_components = [
            r["pca"]["n_components"]
            for r in fold_rows
            if r.get("pca") is not None
        ]
        explained_totals = [
            r["pca"]["explained_variance_total"]
            for r in fold_rows
            if r.get("pca") is not None
        ]
        summary = {
            "panel": panel,
            "experiment": experiment,
            "preprocess_mode": preprocess_mode,
            "pipeline_role": pipeline_role,
            "description": cfg["description"],
            "dataset_original_shape": df.shape,
            "n_folds_completed": len(fold_rows),
            "n_folds_skipped": len(skipped),
            "f1_mean": float(f1s.mean()),
            "f1_se": float(f1s.std(ddof=1) / np.sqrt(len(f1s))) if len(f1s) > 1 else 0.0,
            "f1_finals_mean": float(f1_finals.mean()),
            "mcc_mean": float(mccs.mean()),
            "auc_mean": float(aucs.mean()),
            "precision_mean": float(precisions.mean()),
            "recall_mean": float(recalls.mean()),
            "pca_components_mean": float(np.mean(pca_components)) if pca_components else None,
            "pca_components_per_fold": pca_components,
            "explained_variance_total_mean": float(np.mean(explained_totals)) if explained_totals else None,
            "seconds": round(time.time() - t0, 1),
            "skipped_folds": skipped,
        }

    out = {"summary": summary, "folds": fold_rows}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{panel}_{experiment}_{preprocess_mode}_pca_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=_json_safe)
    log_experiment_result({
        "panel": "CANCER" if panel == "KANSER" else panel,
        "experiment_name": f"{panel.lower()}_{experiment}_{preprocess_mode}",
        "pipeline_role": summary.get("pipeline_role", "primary"),
        "preprocess_mode": preprocess_mode,
        "row_drop_enabled": preprocess_mode == "strict",
        "pca_enabled": bool(EXPERIMENTS[experiment]["pca"]),
        "pca_variant": PCA_VARIANT_NAMES.get(experiment, experiment),
        "scaling_enabled": bool(EXPERIMENTS[experiment]["scale"]),
        "outlier_control_enabled": bool(EXPERIMENTS[experiment]["outlier_control"]),
        "imputation_method": "median" if EXPERIMENTS[experiment]["impute"] else "none",
        "encoding_method": "safe ordinal / label encoding",
        "feature_selection_method": "none",
        "model_name": "fixed LightGBM",
        "calibration_method": "Platt/sigmoid",
        "threshold_strategy": "train-fold threshold optimized at test prior",
        "selected_threshold": "per_fold",
        "blend_enabled": False,
        "blend_weight": "n/a",
        "general_transfer_weight": "n/a",
        "cancer_own_weight": 1.0,
        "cv_strategy": f"RepeatedStratifiedKFold {folds}x{repeats}",
        "n_folds": summary.get("n_folds_completed", 0),
        "input_rows": df.shape[0],
        "output_rows": summary.get("dataset_original_shape", [df.shape[0]])[0],
        "dropped_rows": 0 if preprocess_mode == "no_rowdrop" else "per_fold",
        "input_features": df.shape[1] - (1 if TARGET_COL in df.columns else 0),
        "output_features": summary.get("pca_components_mean") or "344",
        "dropped_features": "per_fold",
        "positive_ratio_before": float(df[TARGET_COL].mean()) if TARGET_COL in df.columns else "n/a",
        "positive_ratio_after": float(df[TARGET_COL].mean()) if TARGET_COL in df.columns and preprocess_mode == "no_rowdrop" else "per_fold",
        "auc": summary.get("auc_mean", "n/a"),
        "f1_star": summary.get("f1_mean", "n/a"),
        "finals_adjusted_f1": summary.get("f1_finals_mean", "n/a"),
        "precision": summary.get("precision_mean", "n/a"),
        "recall": summary.get("recall_mean", "n/a"),
        "pred_positive_mean": summary.get("pred_positive_mean", "n/a"),
        "delta_f1_vs_baseline": "n/a",
        "notes": summary.get("description", ""),
        "output_file": path,
    })
    print(f"[{panel}][{experiment}] saved results: {path}")
    print(json.dumps(summary, indent=2, default=_json_safe))
    return out


def default_preprocess_mode(panel: str) -> str:
    """Primary preprocessing mode by panel.

    CANCER/KANSER defaults to no_rowdrop because strict row dropping removed
    158/388 rows and shifted the positive prior by -9.94 percentage points.
    Strict remains available for CANCER only as an explicit ablation.
    """
    return "no_rowdrop" if panel.upper() in {"CANCER", "KANSER"} else "strict"


def pipeline_role_for_mode(panel: str, preprocess_mode: str) -> str:
    if panel.upper() in {"CANCER", "KANSER"} and preprocess_mode == "strict":
        return "ablation"
    return "primary"


def _parse_components(value: str) -> int | float:
    if "." in value:
        return float(value)
    return int(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prepared PCA experiments on updated MASTER/CANCER data")
    parser.add_argument("--list", action="store_true", help="List experiments and exit")
    parser.add_argument("--panel", choices=["MASTER", "KANSER"], help="Panel to evaluate")
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), help="Experiment to run")
    parser.add_argument("--components", default="0.95", help="PCA components: int or variance fraction, default 0.95")
    parser.add_argument(
        "--preprocess-mode",
        choices=["auto", "strict", "no_rowdrop"],
        default="auto",
        help=(
            "auto uses no_rowdrop for CANCER and strict for other panels. "
            "For CANCER, strict is an ablation. Fit still happens inside each train fold."
        ),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--n-seeds", type=int, default=1, help="Fixed LightGBM seeds; keep small for smoke runs")
    parser.add_argument("--output-dir", default=str(PCA_OUTPUT_DIR))
    args = parser.parse_args()

    if args.list:
        print("Available PCA experiments:")
        for name, cfg in EXPERIMENTS.items():
            print(f"  {name}: {cfg['description']}")
        return

    if not args.panel or not args.experiment:
        parser.error("--panel and --experiment are required unless --list is used")

    run_experiment(
        args.panel,
        args.experiment,
        components=_parse_components(args.components),
        preprocess_mode=args.preprocess_mode,
        folds=args.folds,
        repeats=args.repeats,
        n_seeds=args.n_seeds,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
