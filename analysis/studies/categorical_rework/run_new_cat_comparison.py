"""Compare missingsiz new CAT variants under the fixed CANCER pipeline.

Runs two dataset variants by default:
  * new_cat_only
  * all_cats_kept

Frozen choices:
  * no row drop
  * fold-local median imputation
  * fold-local p1/p99 outlier clipping
  * fold-local one-hot encoding for categorical columns
  * CAT_new missing is represented as CAT_new_is_missing, not as a region
  * Platt/sigmoid calibration
  * no PCA
  * fixed threshold 0.82
  * CANCER own model + MASTER general-transfer blend
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder

from experiment_logger import log_experiment_result
from pca_experiments import (
    TEST_PRIORS,
    apply_platt,
    f1_at_prior,
    fit_platt,
    get_model_params,
    mcc_at_prior_analytic,
    tpr_fpr,
)


ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "outputs" / "new_cat_datasets" / "missingsiz"
OUT_ROOT = ROOT / "outputs" / "new_cat_comparison"
TARGET_COL = "Label"
ID_COL = "Variant_ID"
CAT_NEW_COL = "CAT_new"
CAT_NEW_MISSING_COL = "CAT_new_is_missing"
FINALS_N = 600
BOOTSTRAP_N = 300


@dataclass
class PreparedFold:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    categorical_columns: list[str]
    output_features: int


class QuantileClipper:
    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper
        self.bounds_: dict[str, tuple[float, float]] = {}

    def fit(self, X: pd.DataFrame) -> "QuantileClipper":
        self.bounds_ = {}
        for col in X.columns:
            s = pd.to_numeric(X[col], errors="coerce").dropna()
            if len(s) < 10:
                continue
            lo = float(s.quantile(self.lower))
            hi = float(s.quantile(self.upper))
            if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                self.bounds_[col] = (lo, hi)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col, (lo, hi) in self.bounds_.items():
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").clip(lo, hi)
        return out


class FoldPreprocessor:
    def __init__(self) -> None:
        self.feature_columns_: list[str] = []
        self.numeric_columns_: list[str] = []
        self.other_categorical_columns_: list[str] = []
        self.numeric_medians_: dict[str, float] = {}
        self.clipper_ = QuantileClipper()
        self.other_encoder_: OneHotEncoder | None = None
        self.cat_new_encoder_: OneHotEncoder | None = None
        self.other_feature_names_: list[str] = []
        self.cat_new_feature_names_: list[str] = []

    def fit(self, df: pd.DataFrame) -> "FoldPreprocessor":
        X = self._feature_frame(df)
        self.feature_columns_ = list(X.columns)
        object_cols = X.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
        self.other_categorical_columns_ = [c for c in object_cols if c != CAT_NEW_COL]
        self.numeric_columns_ = [c for c in self.feature_columns_ if c not in object_cols]

        numeric = self._numeric_frame(X)
        med = numeric.median()
        self.numeric_medians_ = {
            c: (float(v) if pd.notna(v) and np.isfinite(v) else 0.0)
            for c, v in med.items()
        }
        numeric = numeric.fillna(pd.Series(self.numeric_medians_)).fillna(0.0)
        self.clipper_.fit(numeric)

        if self.other_categorical_columns_:
            other = self._other_cat_frame(X)
            self.other_encoder_ = _onehot()
            self.other_encoder_.fit(other)
            self.other_feature_names_ = list(self.other_encoder_.get_feature_names_out(self.other_categorical_columns_))

        if CAT_NEW_COL in X.columns:
            cat_new = self._cat_new_series(X)
            non_missing = cat_new[cat_new.notna()].to_frame(CAT_NEW_COL)
            if not non_missing.empty:
                self.cat_new_encoder_ = _onehot()
                self.cat_new_encoder_.fit(non_missing)
                self.cat_new_feature_names_ = list(self.cat_new_encoder_.get_feature_names_out([CAT_NEW_COL]))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._feature_frame(df)
        for col in self.feature_columns_:
            if col not in X.columns:
                X[col] = np.nan
        X = X[self.feature_columns_]

        pieces: list[pd.DataFrame] = []
        numeric = self._numeric_frame(X)
        if not numeric.empty:
            numeric = numeric.fillna(pd.Series(self.numeric_medians_)).fillna(0.0)
            numeric = self.clipper_.transform(numeric)
            numeric = numeric.fillna(pd.Series(self.numeric_medians_)).fillna(0.0)
            pieces.append(numeric.astype(float).reset_index(drop=True))

        if self.other_categorical_columns_ and self.other_encoder_ is not None:
            other = self._other_cat_frame(X)
            enc = self.other_encoder_.transform(other)
            if not isinstance(enc, np.ndarray):
                enc = enc.toarray()
            pieces.append(pd.DataFrame(enc, columns=self.other_feature_names_).astype(float))

        if CAT_NEW_COL in X.columns:
            cat_new = self._cat_new_series(X)
            missing = cat_new.isna()
            pieces.append(pd.DataFrame({CAT_NEW_MISSING_COL: missing.astype(int).to_numpy()}))
            if self.cat_new_encoder_ is not None:
                safe_values = cat_new.fillna("__CAT_NEW_MISSING__").to_frame(CAT_NEW_COL)
                enc = self.cat_new_encoder_.transform(safe_values)
                if not isinstance(enc, np.ndarray):
                    enc = enc.toarray()
                # Missing/unknown values are all-zero because handle_unknown="ignore".
                pieces.append(pd.DataFrame(enc, columns=self.cat_new_feature_names_).astype(float))

        return pd.concat(pieces, axis=1) if pieces else pd.DataFrame(index=df.index)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    @staticmethod
    def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
        return df[[c for c in df.columns if c not in {ID_COL, TARGET_COL}]].copy()

    def _numeric_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.numeric_columns_ if c in X.columns]
        return X[cols].apply(pd.to_numeric, errors="coerce") if cols else pd.DataFrame(index=X.index)

    def _other_cat_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        cats = X[self.other_categorical_columns_].copy()
        for col in self.other_categorical_columns_:
            cats[col] = cats[col].astype("string").fillna("__MISSING__").replace("", "__MISSING__").astype(str)
        return cats

    @staticmethod
    def _cat_new_series(X: pd.DataFrame) -> pd.Series:
        s = X[CAT_NEW_COL].astype("string")
        s = s.where(s.notna() & s.str.strip().ne(""), pd.NA)
        return s.str.strip()


def _onehot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_variant(kind: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    paths = {
        "master": DATASET_ROOT / kind / f"YARISMA_TRAIN_MASTER_new_cat_missingsiz_{kind}.csv",
        "cancer": DATASET_ROOT / kind / f"YARISMA_TRAIN_KANSER_new_cat_missingsiz_{kind}.csv",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    return pd.read_csv(paths["master"]), pd.read_csv(paths["cancer"]), paths


def prepare_fold(train_df: pd.DataFrame, test_df: pd.DataFrame) -> PreparedFold:
    prep = FoldPreprocessor()
    X_train = prep.fit_transform(train_df)
    X_test = prep.transform(test_df)
    return PreparedFold(
        X_train=X_train,
        y_train=train_df[TARGET_COL].astype(int).reset_index(drop=True),
        X_test=X_test,
        y_test=test_df[TARGET_COL].astype(int).reset_index(drop=True),
        categorical_columns=prep.other_categorical_columns_ + ([CAT_NEW_COL] if CAT_NEW_COL in prep.feature_columns_ else []),
        output_features=int(X_train.shape[1]),
    )


def train_lgbm(X_train: pd.DataFrame, y_train: pd.Series, eval_fraction: float, max_estimators: int):
    idx = np.arange(len(y_train))
    fit_idx, eval_idx = train_test_split(idx, test_size=eval_fraction, random_state=42, stratify=y_train)
    spw = (y_train.iloc[fit_idx] == 0).sum() / max(int((y_train.iloc[fit_idx] == 1).sum()), 1)
    params = get_model_params(len(fit_idx), spw)
    params["n_jobs"] = -1
    params["n_estimators"] = min(int(params.get("n_estimators", 300)), max_estimators)
    params["learning_rate"] = max(float(params.get("learning_rate", 0.03)), 0.05)
    params["verbose"] = -1
    model = LGBMClassifier(**params, random_state=42)
    model.fit(
        X_train.iloc[fit_idx],
        y_train.iloc[fit_idx],
        eval_set=[(X_train.iloc[eval_idx], y_train.iloc[eval_idx])],
        callbacks=[lgb.early_stopping(15, verbose=False), lgb.log_evaluation(0)],
    )
    return model, fit_idx, eval_idx


def transfer_scores(master: pd.DataFrame, cancer: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    source = master[~master[ID_COL].isin(set(cancer[ID_COL]))].reset_index(drop=True)
    prepared = prepare_fold(source, cancer.reset_index(drop=True))
    model, _, _ = train_lgbm(prepared.X_train, prepared.y_train, eval_fraction=0.15, max_estimators=160)
    return model.predict_proba(prepared.X_test)[:, 1], {
        "source_rows": int(len(source)),
        "source_features": int(prepared.X_train.shape[1]),
        "target_features": int(prepared.X_test.shape[1]),
    }


def cancer_own_fold(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, Any]:
    prepared = prepare_fold(train_df, test_df)
    idx = np.arange(len(prepared.y_train))
    fit_idx, cal_idx = train_test_split(idx, test_size=0.25, random_state=42, stratify=prepared.y_train)
    spw = (prepared.y_train.iloc[fit_idx] == 0).sum() / max(int((prepared.y_train.iloc[fit_idx] == 1).sum()), 1)
    params = get_model_params(len(fit_idx), spw)
    params["n_jobs"] = -1
    params["n_estimators"] = min(int(params.get("n_estimators", 300)), 120)
    params["learning_rate"] = max(float(params.get("learning_rate", 0.03)), 0.05)
    params["verbose"] = -1
    model = LGBMClassifier(**params, random_state=42)
    model.fit(
        prepared.X_train.iloc[fit_idx],
        prepared.y_train.iloc[fit_idx],
        eval_set=[(prepared.X_train.iloc[cal_idx], prepared.y_train.iloc[cal_idx])],
        callbacks=[lgb.early_stopping(15, verbose=False), lgb.log_evaluation(0)],
    )
    cal_raw = model.predict_proba(prepared.X_train.iloc[cal_idx])[:, 1]
    test_raw = model.predict_proba(prepared.X_test)[:, 1]
    platt = fit_platt(cal_raw, prepared.y_train.iloc[cal_idx].values)
    return {
        "prob": apply_platt(test_raw, platt),
        "y_test": prepared.y_test,
        "y_cal": prepared.y_train.iloc[cal_idx].reset_index(drop=True),
        "cal_original_index": cal_idx,
        "output_features": prepared.output_features,
        "categorical_columns": prepared.categorical_columns,
    }


def finals_f1(tpr: float, fpr: float, prior: float, seed: int = 42) -> float:
    n_pos = max(1, int(round(FINALS_N * prior)))
    n_neg = max(1, FINALS_N - n_pos)
    rng = np.random.RandomState(seed)
    tp = rng.binomial(n_pos, min(max(tpr, 0.0), 1.0), size=BOOTSTRAP_N)
    fp = rng.binomial(n_neg, min(max(fpr, 0.0), 1.0), size=BOOTSTRAP_N)
    fn = n_pos - tp
    denom = 2 * tp + fp + fn
    return float(np.divide(2 * tp, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0).mean())


def metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float, prior: float) -> dict[str, float]:
    pred = (probs >= threshold).astype(int)
    tpr, fpr = tpr_fpr(y_true, pred)
    return {
        "f1_star": float(f1_at_prior(tpr, fpr, prior)),
        "finals_adjusted_f1": finals_f1(tpr, fpr, prior),
        "auc": float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else 0.5,
        "average_precision": float(average_precision_score(y_true, probs)) if len(np.unique(y_true)) > 1 else float(np.mean(y_true)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "mcc_star": float(mcc_at_prior_analytic(tpr, fpr, prior)),
        "pred_positive_mean": float(pred.mean()),
    }


def run_variant(kind: str, threshold: float, blend_weight: float, splits: int, repeats: int) -> dict[str, Any]:
    master, cancer, paths = load_variant(kind)
    prior = TEST_PRIORS["cancer"]
    y = cancer[TARGET_COL].astype(int).reset_index(drop=True)
    transfer_raw, transfer_info = transfer_scores(master, cancer)

    fold_rows: list[dict[str, Any]] = []
    oof_rows: list[pd.DataFrame] = []
    splitter = RepeatedStratifiedKFold(n_splits=splits, n_repeats=repeats, random_state=42)
    total = splits * repeats
    for fold_id, (tr, te) in enumerate(splitter.split(cancer, y), start=1):
        print(f"[{kind}] fold {fold_id}/{total}", flush=True)
        own = cancer_own_fold(cancer.iloc[tr].reset_index(drop=True), cancer.iloc[te].reset_index(drop=True))
        cal_global = tr[own["cal_original_index"]]
        transfer_platt = fit_platt(transfer_raw[cal_global], own["y_cal"].values)
        transfer_prob = apply_platt(transfer_raw[te], transfer_platt)
        prob = blend_weight * transfer_prob + (1.0 - blend_weight) * own["prob"]
        y_true = own["y_test"].astype(int).to_numpy()
        row = metrics(y_true, prob, threshold, prior)
        row.update({
            "fold_id": fold_id,
            "repeat_id": (fold_id - 1) // splits + 1,
            "split_id": (fold_id - 1) % splits + 1,
            "output_features": own["output_features"],
            "categorical_columns": ";".join(own["categorical_columns"]),
        })
        fold_rows.append(row)
        oof_rows.append(pd.DataFrame({
            "fold_id": fold_id,
            "sample_id": te,
            "variant_id": cancer.iloc[te][ID_COL].to_numpy(),
            "y_true": y_true,
            "prob": prob,
            "pred": (prob >= threshold).astype(int),
        }))

    folds = pd.DataFrame(fold_rows)
    metric_cols = ["f1_star", "finals_adjusted_f1", "auc", "average_precision", "precision", "recall", "mcc_star", "pred_positive_mean"]
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_kind": kind,
        "source": "missingsiz",
        "threshold": threshold,
        "blend_weight": blend_weight,
        "cancer_own_weight": round(1.0 - blend_weight, 6),
        "splits": splits,
        "repeats": repeats,
        "n_folds": int(len(folds)),
        "input_rows": int(len(cancer)),
        "input_features": int(cancer.shape[1] - 2),
        "positive_ratio": float(y.mean()),
        "metrics": {col: float(folds[col].mean()) for col in metric_cols},
        "output_features_mean": float(folds["output_features"].mean()),
        "categorical_columns": sorted(set(";".join(folds["categorical_columns"]).split(";")) - {""}),
        "transfer_info": transfer_info,
        "dataset_paths": {k: str(v) for k, v in paths.items()},
    }

    out_dir = OUT_ROOT / f"missingsiz_{kind}_w{blend_weight:.2f}_t{threshold:.2f}".replace(".", "p")
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_path = out_dir / "fold_metrics.csv"
    oof_path = out_dir / "oof_predictions.csv"
    summary_path = out_dir / "summary.json"
    folds.to_csv(fold_path, index=False)
    pd.concat(oof_rows, ignore_index=True).to_csv(oof_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log_experiment_result({
        "panel": "CANCER",
        "experiment_name": f"new_cat_missingsiz_{kind}_fixed_t{threshold:.2f}_w{blend_weight:.2f}",
        "pipeline_role": "primary" if kind == "new_cat_only" else "ablation",
        "preprocess_mode": "no_rowdrop",
        "row_drop_enabled": False,
        "pca_enabled": False,
        "pca_variant": "none",
        "scaling_enabled": False,
        "outlier_control_enabled": True,
        "imputation_method": "median",
        "encoding_method": "fold-local one-hot; CAT_new missing indicator with region one-hots all zero",
        "feature_selection_method": "none",
        "model_name": "fixed LightGBM cancer-own + MASTER general-transfer blend",
        "calibration_method": "Platt/sigmoid",
        "threshold_strategy": f"fixed_{threshold:.2f}",
        "selected_threshold": threshold,
        "blend_enabled": True,
        "blend_weight": blend_weight,
        "general_transfer_weight": blend_weight,
        "cancer_own_weight": round(1.0 - blend_weight, 6),
        "cv_strategy": "RepeatedStratifiedKFold",
        "n_folds": summary["n_folds"],
        "input_rows": summary["input_rows"],
        "output_rows": summary["input_rows"],
        "dropped_rows": 0,
        "input_features": summary["input_features"],
        "output_features": summary["output_features_mean"],
        "dropped_features": 0,
        "positive_ratio_before": summary["positive_ratio"],
        "positive_ratio_after": summary["positive_ratio"],
        "auc": summary["metrics"]["auc"],
        "f1_star": summary["metrics"]["f1_star"],
        "finals_adjusted_f1": summary["metrics"]["finals_adjusted_f1"],
        "precision": summary["metrics"]["precision"],
        "recall": summary["metrics"]["recall"],
        "pred_positive_mean": summary["metrics"]["pred_positive_mean"],
        "delta_f1_vs_baseline": "n/a",
        "notes": f"missingsiz {kind}; no PCA, no row drop, fixed threshold 0.82, leakage-safe fold preprocessing.",
        "output_file": summary_path,
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run missingsiz new CAT comparison.")
    parser.add_argument("--kinds", nargs="+", choices=["new_cat_only", "all_cats_kept"], default=["new_cat_only", "all_cats_kept"])
    parser.add_argument("--threshold", type=float, default=0.82)
    parser.add_argument("--blend-weight", type=float, default=0.30)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=6)
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = [run_variant(kind, args.threshold, args.blend_weight, args.splits, args.repeats) for kind in args.kinds]
    comparison_path = OUT_ROOT / "comparison_summary.json"
    comparison_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print("\nComparison:")
    for s in summaries:
        m = s["metrics"]
        print(
            f"{s['dataset_kind']}: "
            f"F1*={m['f1_star']:.6f}, finals={m['finals_adjusted_f1']:.6f}, "
            f"AUC={m['auc']:.6f}, precision={m['precision']:.6f}, recall={m['recall']:.6f}"
        )
    print(f"Saved {comparison_path}")


if __name__ == "__main__":
    main()
