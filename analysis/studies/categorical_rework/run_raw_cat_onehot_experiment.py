"""Run the fixed CANCER pipeline on raw old CAT datasets with CAT one-hot encoding.

Uses only:
  * data/YARISMA_TRAIN_MASTER.csv
  * data/YARISMA_TRAIN_KANSER.csv

Frozen choices:
  * no row drop
  * fold-local median imputation
  * fold-local p1/p99 outlier clipping
  * fold-local one-hot for old CAT_1..CAT_6
  * each old CAT_i gets CAT_i_is_missing; missing does not activate a category
  * Platt/sigmoid calibration
  * no PCA
  * fixed threshold 0.82
  * 50/50 MASTER general-transfer + KANSER own blend by default
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
DATA_DIR = ROOT / "data"
OUT_ROOT = ROOT / "outputs" / "raw_cat_onehot"
MASTER_PATH = DATA_DIR / "YARISMA_TRAIN_MASTER.csv"
CANCER_PATH = DATA_DIR / "YARISMA_TRAIN_KANSER.csv"
TARGET_COL = "Label"
ID_COL = "Variant_ID"
OLD_CAT_RE = re.compile(r"^CAT_\d+$")
FINALS_N = 600
BOOTSTRAP_N = 300


@dataclass
class PreparedFold:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    output_features: int
    old_cat_columns: list[str]
    other_categorical_columns: list[str]


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
        self.old_cat_columns_: list[str] = []
        self.other_categorical_columns_: list[str] = []
        self.numeric_columns_: list[str] = []
        self.numeric_medians_: dict[str, float] = {}
        self.clipper_ = QuantileClipper()
        self.old_cat_encoders_: dict[str, OneHotEncoder] = {}
        self.old_cat_feature_names_: dict[str, list[str]] = {}
        self.other_encoder_: OneHotEncoder | None = None
        self.other_feature_names_: list[str] = []

    def fit(self, df: pd.DataFrame) -> "FoldPreprocessor":
        X = self._feature_frame(df)
        self.feature_columns_ = list(X.columns)
        self.old_cat_columns_ = [c for c in self.feature_columns_ if OLD_CAT_RE.match(c)]
        object_cols = X.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
        self.other_categorical_columns_ = [c for c in object_cols if c not in self.old_cat_columns_]
        self.numeric_columns_ = [c for c in self.feature_columns_ if c not in object_cols]

        numeric = self._numeric_frame(X)
        med = numeric.median()
        self.numeric_medians_ = {
            c: (float(v) if pd.notna(v) and np.isfinite(v) else 0.0)
            for c, v in med.items()
        }
        numeric = numeric.fillna(pd.Series(self.numeric_medians_)).fillna(0.0)
        self.clipper_.fit(numeric)

        for col in self.old_cat_columns_:
            s = self._old_cat_series(X, col)
            non_missing = s[s.notna()].to_frame(col)
            if non_missing.empty:
                continue
            enc = _onehot()
            enc.fit(non_missing)
            self.old_cat_encoders_[col] = enc
            self.old_cat_feature_names_[col] = list(enc.get_feature_names_out([col]))

        if self.other_categorical_columns_:
            other = self._other_cat_frame(X)
            self.other_encoder_ = _onehot()
            self.other_encoder_.fit(other)
            self.other_feature_names_ = list(self.other_encoder_.get_feature_names_out(self.other_categorical_columns_))
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

        for col in self.old_cat_columns_:
            s = self._old_cat_series(X, col)
            missing = s.isna()
            pieces.append(pd.DataFrame({f"{col}_is_missing": missing.astype(int).to_numpy()}))
            enc = self.old_cat_encoders_.get(col)
            if enc is None:
                continue
            safe = s.fillna("__CAT_MISSING__").to_frame(col)
            encoded = enc.transform(safe)
            if not isinstance(encoded, np.ndarray):
                encoded = encoded.toarray()
            # Missing and unseen categories are all-zero because handle_unknown="ignore".
            pieces.append(pd.DataFrame(encoded, columns=self.old_cat_feature_names_[col]).astype(float))

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
    def _old_cat_series(X: pd.DataFrame, col: str) -> pd.Series:
        s = X[col].astype("string")
        s = s.where(s.notna() & s.str.strip().ne(""), pd.NA)
        return s.str.strip()


def _onehot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def prepare_fold(train_df: pd.DataFrame, test_df: pd.DataFrame) -> PreparedFold:
    pre = FoldPreprocessor()
    X_train = pre.fit_transform(train_df)
    X_test = pre.transform(test_df)
    return PreparedFold(
        X_train=X_train,
        y_train=train_df[TARGET_COL].astype(int).reset_index(drop=True),
        X_test=X_test,
        y_test=test_df[TARGET_COL].astype(int).reset_index(drop=True),
        output_features=int(X_train.shape[1]),
        old_cat_columns=pre.old_cat_columns_,
        other_categorical_columns=pre.other_categorical_columns_,
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
        "old_cat_columns": prepared.old_cat_columns,
        "other_categorical_columns": prepared.other_categorical_columns,
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


def run(threshold: float, blend_weight: float, splits: int, repeats: int) -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    master = pd.read_csv(MASTER_PATH)
    cancer = pd.read_csv(CANCER_PATH)
    prior = TEST_PRIORS["cancer"]
    y = cancer[TARGET_COL].astype(int).reset_index(drop=True)
    transfer_raw, transfer_info = transfer_scores(master, cancer)

    fold_rows: list[dict[str, Any]] = []
    oof_rows: list[pd.DataFrame] = []
    splitter = RepeatedStratifiedKFold(n_splits=splits, n_repeats=repeats, random_state=42)
    total = splits * repeats
    for fold_id, (tr, te) in enumerate(splitter.split(cancer, y), start=1):
        print(f"[raw_old_cat_onehot] fold {fold_id}/{total}", flush=True)
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
            "old_cat_columns": ";".join(own["old_cat_columns"]),
            "other_categorical_columns": ";".join(own["other_categorical_columns"]),
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
        "dataset_kind": "raw_old_cat_onehot",
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
        "old_cat_columns": sorted(set(";".join(folds["old_cat_columns"]).split(";")) - {""}),
        "other_categorical_columns": sorted(set(";".join(folds["other_categorical_columns"]).split(";")) - {""}),
        "transfer_info": transfer_info,
        "dataset_paths": {"master": str(MASTER_PATH), "cancer": str(CANCER_PATH)},
    }

    out_dir = OUT_ROOT / f"raw_old_cat_onehot_w{blend_weight:.2f}_t{threshold:.2f}".replace(".", "p")
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_path = out_dir / "fold_metrics.csv"
    oof_path = out_dir / "oof_predictions.csv"
    summary_path = out_dir / "summary.json"
    folds.to_csv(fold_path, index=False)
    pd.concat(oof_rows, ignore_index=True).to_csv(oof_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log_experiment_result({
        "panel": "CANCER",
        "experiment_name": f"raw_old_cat_onehot_fixed_t{threshold:.2f}_w{blend_weight:.2f}",
        "pipeline_role": "baseline_ablation",
        "preprocess_mode": "no_rowdrop",
        "row_drop_enabled": False,
        "pca_enabled": False,
        "pca_variant": "none",
        "scaling_enabled": False,
        "outlier_control_enabled": True,
        "imputation_method": "median",
        "encoding_method": "fold-local one-hot for old CAT_1..CAT_6; per-CAT missing indicators; missing CAT one-hots all zero",
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
        "notes": "Raw old MASTER/KANSER datasets only; no new CAT files used.",
        "output_file": summary_path,
    })
    print(
        f"raw_old_cat_onehot: F1*={summary['metrics']['f1_star']:.6f}, "
        f"finals={summary['metrics']['finals_adjusted_f1']:.6f}, "
        f"AUC={summary['metrics']['auc']:.6f}, "
        f"precision={summary['metrics']['precision']:.6f}, "
        f"recall={summary['metrics']['recall']:.6f}"
    )
    print(f"Saved {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run raw old CAT one-hot experiment.")
    parser.add_argument("--threshold", type=float, default=0.82)
    parser.add_argument("--blend-weight", type=float, default=0.50)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=6)
    args = parser.parse_args()
    run(args.threshold, args.blend_weight, args.splits, args.repeats)


if __name__ == "__main__":
    main()
