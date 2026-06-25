from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


TARGET_COL = "Label"
ID_COLS = {"Variant_ID"}
MISSING_TOKEN = "__MISSING__"
UNKNOWN_TOKEN = "__UNKNOWN__"
RARE_TOKEN = "__RARE__"


BLOSUM62: dict[tuple[str, str], int] = {}
_BLOSUM62_ROWS = {
    "A": [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
    "R": [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],
    "N": [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],
    "D": [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],
    "C": [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
    "Q": [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],
    "E": [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],
    "G": [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],
    "H": [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],
    "I": [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],
    "L": [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],
    "K": [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],
    "M": [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],
    "F": [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],
    "P": [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],
    "S": [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],
    "T": [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],
    "W": [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],
    "Y": [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],
    "V": [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],
}
_AA_ORDER = list("ARNDCQEGHILKMFPSTWYV")
for aa1, scores in _BLOSUM62_ROWS.items():
    for aa2, score in zip(_AA_ORDER, scores):
        BLOSUM62[(aa1, aa2)] = score
        BLOSUM62[(aa2, aa1)] = score


@dataclass(frozen=True)
class CandidateConfig:
    config_id: str
    categorical_mode: str
    impute_strategy: str
    missing_indicators: bool
    use_blosum: bool
    numeric_transform: str


class FoldPreprocessor:
    def __init__(
        self,
        config: CandidateConfig,
        rare_min_count: int = 5,
        rare_min_fraction: float = 0.01,
    ) -> None:
        self.config = config
        self.rare_min_count = rare_min_count
        self.rare_min_fraction = rare_min_fraction

    def fit(self, x: pd.DataFrame) -> "FoldPreprocessor":
        self.feature_cols_ = [c for c in x.columns if c not in ID_COLS and c != TARGET_COL]
        engineered = self._add_engineered_features(x[self.feature_cols_].copy())

        self.categorical_cols_ = [
            c
            for c in engineered.columns
            if pd.api.types.is_object_dtype(engineered[c])
            or pd.api.types.is_string_dtype(engineered[c])
            or isinstance(engineered[c].dtype, pd.CategoricalDtype)
        ]
        self.numeric_cols_ = [c for c in engineered.columns if c not in self.categorical_cols_]

        numeric = engineered[self.numeric_cols_].apply(pd.to_numeric, errors="coerce")
        categorical = (
            engineered[self.categorical_cols_].astype("string")
            if self.categorical_cols_
            else pd.DataFrame(index=engineered.index)
        )

        if self.config.numeric_transform in {"clip", "clip_log"}:
            self.clip_bounds_ = {
                c: (float(numeric[c].quantile(0.01)), float(numeric[c].quantile(0.99)))
                for c in self.numeric_cols_
            }
        else:
            self.clip_bounds_ = {}

        clipped_numeric = numeric.copy()
        for c, (lo, hi) in self.clip_bounds_.items():
            clipped_numeric[c] = clipped_numeric[c].clip(lo, hi)

        if self.config.numeric_transform in {"log", "clip_log"}:
            self.log_cols_ = []
            for c in self.numeric_cols_:
                series = clipped_numeric[c].dropna()
                if series.empty:
                    continue
                if series.min() >= 0 and series.nunique() > 10 and abs(series.skew()) >= 1.0:
                    self.log_cols_.append(c)
        else:
            self.log_cols_ = []

        numeric_for_impute = self._with_numeric_transforms(numeric)
        self.missing_indicator_cols_ = (
            [c for c in numeric_for_impute.columns if numeric_for_impute[c].isna().any()]
            if self.config.missing_indicators
            else []
        )

        self.impute_values_: dict[str, float] = {}
        if self.config.impute_strategy != "native_nan":
            for c in numeric_for_impute.columns:
                series = numeric_for_impute[c]
                if self.config.impute_strategy == "median":
                    value = series.median()
                elif self.config.impute_strategy == "mean":
                    value = series.mean()
                elif self.config.impute_strategy == "zero":
                    value = 0.0
                else:
                    raise ValueError(f"Unknown impute strategy: {self.config.impute_strategy}")
                self.impute_values_[c] = 0.0 if pd.isna(value) else float(value)

        self.keep_categorical_cols_ = [] if self.config.categorical_mode == "none" else list(self.categorical_cols_)
        self.allowed_categories_: dict[str, set[str]] = {}
        self.category_levels_: dict[str, list[str]] = {}
        for c in self.keep_categorical_cols_:
            values = categorical[c].fillna(MISSING_TOKEN).astype(str)
            if self.config.categorical_mode == "native_rare":
                counts = values.value_counts(dropna=False)
                min_count = max(self.rare_min_count, int(np.ceil(len(values) * self.rare_min_fraction)))
                allowed = set(counts[counts >= min_count].index.astype(str))
            else:
                allowed = set(values.unique().astype(str))
            levels = sorted(allowed | {MISSING_TOKEN, UNKNOWN_TOKEN, RARE_TOKEN})
            self.allowed_categories_[c] = allowed
            self.category_levels_[c] = levels

        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        engineered = self._add_engineered_features(x[self.feature_cols_].copy())
        numeric = engineered[self.numeric_cols_].apply(pd.to_numeric, errors="coerce")
        numeric = self._with_numeric_transforms(numeric)

        frames = [numeric]

        if self.missing_indicator_cols_:
            flags = pd.DataFrame(
                {f"{c}__is_missing": numeric[c].isna().astype(np.int8) for c in self.missing_indicator_cols_},
                index=numeric.index,
            )
            frames.append(flags)

        if self.impute_values_:
            numeric = numeric.fillna(self.impute_values_)
            frames[0] = numeric

        if self.keep_categorical_cols_:
            categorical = pd.DataFrame(index=engineered.index)
            for c in self.keep_categorical_cols_:
                values = engineered[c].astype("string").fillna(MISSING_TOKEN).astype(str)
                if self.config.categorical_mode == "native_rare":
                    values = values.where(values.isin(self.allowed_categories_[c]), RARE_TOKEN)
                else:
                    values = values.where(values.isin(self.allowed_categories_[c]), UNKNOWN_TOKEN)
                categorical[c] = pd.Categorical(values, categories=self.category_levels_[c])
            frames.append(categorical)

        return pd.concat(frames, axis=1)

    def fit_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        return self.fit(x).transform(x)

    def _add_engineered_features(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.config.use_blosum or not {"AA_1", "AA_2"}.issubset(data.columns):
            return data

        aa1 = data["AA_1"].astype("string").str.upper()
        aa2 = data["AA_2"].astype("string").str.upper()
        scores = []
        valid = []
        same = []
        stop_involved = []
        for left, right in zip(aa1, aa2):
            left_s = None if pd.isna(left) else str(left)
            right_s = None if pd.isna(right) else str(right)
            score = BLOSUM62.get((left_s, right_s))
            scores.append(np.nan if score is None else score)
            valid.append(0 if score is None else 1)
            same.append(1 if left_s is not None and left_s == right_s else 0)
            stop_involved.append(1 if left_s == "*" or right_s == "*" else 0)

        data = data.copy()
        data["BLOSUM62_AA1_AA2"] = scores
        data["BLOSUM62_valid"] = valid
        data["AA_same"] = same
        data["AA_stop_involved"] = stop_involved
        return data

    def _with_numeric_transforms(self, numeric: pd.DataFrame) -> pd.DataFrame:
        numeric = numeric.copy()
        for c, (lo, hi) in self.clip_bounds_.items():
            numeric[c] = numeric[c].clip(lo, hi)
        if self.log_cols_:
            logs = pd.DataFrame(
                {f"{c}__log1p": np.log1p(numeric[c].clip(lower=0)) for c in self.log_cols_},
                index=numeric.index,
            )
            numeric = pd.concat([numeric, logs], axis=1)
        return numeric


def read_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"{path} icinde {TARGET_COL} yok.")
    return df


def make_candidates(candidate_set: str) -> list[CandidateConfig]:
    categorical_modes = ["none", "native", "native_rare"]
    imputers = ["native_nan", "median", "mean", "zero"]
    missing_options = [False, True]
    blosum_options = [False, True]
    numeric_transforms = ["none"] if candidate_set == "core" else ["none", "clip", "log"]

    configs = []
    for categorical_mode in categorical_modes:
        for impute_strategy in imputers:
            for missing_indicators in missing_options:
                for use_blosum in blosum_options:
                    for numeric_transform in numeric_transforms:
                        parts = [
                            f"cat={categorical_mode}",
                            f"imp={impute_strategy}",
                            f"miss={int(missing_indicators)}",
                            f"blosum={int(use_blosum)}",
                            f"num={numeric_transform}",
                        ]
                        configs.append(
                            CandidateConfig(
                                config_id("|".join(parts)),
                                categorical_mode,
                                impute_strategy,
                                missing_indicators,
                                use_blosum,
                                numeric_transform,
                            )
                        )
    return configs


def config_id(text: str) -> str:
    return text.replace("=", "_").replace("|", "__")


def model(seed: int, n_estimators: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=10,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


def fit_predict(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    config: CandidateConfig,
    seed: int,
    n_estimators: int,
) -> tuple[np.ndarray, np.ndarray, int, pd.Series]:
    y_train = train_df[TARGET_COL].astype(int)
    preprocessor = FoldPreprocessor(config)
    x_train = preprocessor.fit_transform(train_df)
    x_valid = preprocessor.transform(valid_df)
    clf = model(seed, n_estimators)
    clf.fit(x_train, y_train)
    proba = clf.predict_proba(x_valid)[:, 1]
    pred = (proba >= 0.5).astype(int)
    importance = pd.Series(clf.feature_importances_, index=x_train.columns, dtype=float)
    return pred, proba, x_train.shape[1], importance


def score_predictions(y_true: pd.Series, pred: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    try:
        auc = roc_auc_score(y_true, proba)
    except ValueError:
        auc = np.nan
    return {
        "f1_macro": f1_score(y_true, pred, average="macro", zero_division=0),
        "f1_label_0": f1_score(y_true, pred, pos_label=0, zero_division=0),
        "f1_label_1": f1_score(y_true, pred, pos_label=1, zero_division=0),
        "precision_label_1": precision_score(y_true, pred, zero_division=0),
        "recall_label_1": recall_score(y_true, pred, zero_division=0),
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "roc_auc": auc,
    }


def evaluate_config_cv(
    df: pd.DataFrame,
    row_indices: np.ndarray,
    config: CandidateConfig,
    inner_folds: int,
    seed: int,
    n_estimators: int,
) -> list[dict[str, object]]:
    y = df.iloc[row_indices][TARGET_COL].astype(int).to_numpy()
    cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    rows = []
    for inner_fold, (train_pos, valid_pos) in enumerate(cv.split(np.zeros(len(y)), y), start=1):
        train_idx = row_indices[train_pos]
        valid_idx = row_indices[valid_pos]
        train_df = df.iloc[train_idx].reset_index(drop=True)
        valid_df = df.iloc[valid_idx].reset_index(drop=True)
        pred, proba, n_features, _ = fit_predict(train_df, valid_df, config, seed + inner_fold, n_estimators)
        metrics = score_predictions(valid_df[TARGET_COL].astype(int), pred, proba)
        rows.append(
            {
                **asdict(config),
                "inner_fold": inner_fold,
                "n_features": n_features,
                **metrics,
            }
        )
    return rows


def choose_best_config(inner_scores: pd.DataFrame) -> pd.Series:
    grouped = (
        inner_scores.groupby(
            [
                "config_id",
                "categorical_mode",
                "impute_strategy",
                "missing_indicators",
                "use_blosum",
                "numeric_transform",
            ],
            as_index=False,
        )
        .agg(
            f1_macro_mean=("f1_macro", "mean"),
            f1_macro_std=("f1_macro", "std"),
            f1_label_0_mean=("f1_label_0", "mean"),
            f1_label_1_mean=("f1_label_1", "mean"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            roc_auc_mean=("roc_auc", "mean"),
            n_features_mean=("n_features", "mean"),
        )
        .sort_values(["f1_macro_mean", "balanced_accuracy_mean", "n_features_mean"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    return grouped.iloc[0]


def nested_experiment(
    df: pd.DataFrame,
    candidates: list[CandidateConfig],
    outer_folds: int,
    inner_folds: int,
    seed: int,
    n_estimators: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = df[TARGET_COL].astype(int).to_numpy()
    outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    all_inner_rows = []
    outer_rows = []
    importance_rows = []

    for outer_fold, (outer_train_idx, outer_valid_idx) in enumerate(outer_cv.split(np.zeros(len(y)), y), start=1):
        for i, config in enumerate(candidates, start=1):
            if i % 25 == 0:
                print(f"outer {outer_fold}/{outer_folds}: {i}/{len(candidates)} configs")
            all_inner_rows.extend(
                {
                    "outer_fold": outer_fold,
                    **row,
                }
                for row in evaluate_config_cv(
                    df,
                    outer_train_idx,
                    config,
                    inner_folds=inner_folds,
                    seed=seed + outer_fold * 1000 + i,
                    n_estimators=n_estimators,
                )
            )

        inner_df = pd.DataFrame([r for r in all_inner_rows if r["outer_fold"] == outer_fold])
        best = choose_best_config(inner_df)
        best_config = CandidateConfig(
            config_id=best["config_id"],
            categorical_mode=best["categorical_mode"],
            impute_strategy=best["impute_strategy"],
            missing_indicators=bool(best["missing_indicators"]),
            use_blosum=bool(best["use_blosum"]),
            numeric_transform=best["numeric_transform"],
        )

        train_df = df.iloc[outer_train_idx].reset_index(drop=True)
        valid_df = df.iloc[outer_valid_idx].reset_index(drop=True)
        pred, proba, n_features, importance = fit_predict(
            train_df,
            valid_df,
            best_config,
            seed + outer_fold * 100,
            n_estimators,
        )
        metrics = score_predictions(valid_df[TARGET_COL].astype(int), pred, proba)
        outer_rows.append(
            {
                "outer_fold": outer_fold,
                **asdict(best_config),
                "inner_selected_f1_macro_mean": best["f1_macro_mean"],
                "inner_selected_f1_macro_std": best["f1_macro_std"],
                "n_train": len(train_df),
                "n_valid": len(valid_df),
                "n_features": n_features,
                **metrics,
            }
        )
        for feature, value in importance.items():
            importance_rows.append(
                {
                    "outer_fold": outer_fold,
                    "config_id": best_config.config_id,
                    "feature": feature,
                    "importance": value,
                }
            )

    return pd.DataFrame(outer_rows), pd.DataFrame(all_inner_rows), pd.DataFrame(importance_rows)


def summarize_inner_all(inner_scores: pd.DataFrame) -> pd.DataFrame:
    return (
        inner_scores.groupby(
            [
                "config_id",
                "categorical_mode",
                "impute_strategy",
                "missing_indicators",
                "use_blosum",
                "numeric_transform",
            ],
            as_index=False,
        )
        .agg(
            f1_macro_mean=("f1_macro", "mean"),
            f1_macro_std=("f1_macro", "std"),
            f1_label_0_mean=("f1_label_0", "mean"),
            f1_label_1_mean=("f1_label_1", "mean"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            roc_auc_mean=("roc_auc", "mean"),
            n_features_mean=("n_features", "mean"),
        )
        .sort_values(["f1_macro_mean", "balanced_accuracy_mean"], ascending=False)
    )


def summarize_outer(outer_scores: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "f1_macro",
        "f1_label_0",
        "f1_label_1",
        "balanced_accuracy",
        "roc_auc",
        "accuracy",
        "precision_label_1",
        "recall_label_1",
    ]
    rows = []
    for metric in metrics:
        rows.append(
            {
                "metric": metric,
                "mean": outer_scores[metric].mean(),
                "std": outer_scores[metric].std(),
                "min": outer_scores[metric].min(),
                "max": outer_scores[metric].max(),
            }
        )
    return pd.DataFrame(rows)


def profile_dataset(df: pd.DataFrame, candidate_count: int) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in ID_COLS and c != TARGET_COL]
    categorical_cols = [
        c
        for c in feature_cols
        if pd.api.types.is_object_dtype(df[c])
        or pd.api.types.is_string_dtype(df[c])
        or isinstance(df[c].dtype, pd.CategoricalDtype)
    ]
    y = df[TARGET_COL].astype(int)
    all_positive = np.ones(len(y), dtype=int)
    return pd.DataFrame(
        [
            {
                "rows": len(df),
                "columns": df.shape[1],
                "feature_columns": len(feature_cols),
                "numeric_features": len(feature_cols) - len(categorical_cols),
                "categorical_features": len(categorical_cols),
                "label_counts": json.dumps({str(k): int(v) for k, v in y.value_counts().to_dict().items()}),
                "missing_cells": int(df[feature_cols].isna().sum().sum()),
                "missing_cell_rate": float(df[feature_cols].isna().sum().sum() / (len(df) * len(feature_cols))),
                "variant_id_duplicates": int(df["Variant_ID"].duplicated().sum()) if "Variant_ID" in df else np.nan,
                "full_row_duplicates": int(df.duplicated().sum()),
                "feature_duplicates_excluding_id": int(df.drop(columns=list(ID_COLS & set(df.columns))).duplicated().sum()),
                "all_positive_f1_label_1": float(f1_score(y, all_positive, zero_division=0)),
                "all_positive_f1_macro": float(f1_score(y, all_positive, average="macro", zero_division=0)),
                "candidate_count": candidate_count,
            }
        ]
    )


def write_report(
    output: Path,
    df: pd.DataFrame,
    candidates: list[CandidateConfig],
    outer_scores: pd.DataFrame,
    inner_scores: pd.DataFrame,
    importances: pd.DataFrame,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    outer_summary = summarize_outer(outer_scores)
    inner_leaderboard = summarize_inner_all(inner_scores)
    candidate_grid = pd.DataFrame([asdict(c) for c in candidates])
    profile = profile_dataset(df, len(candidates))
    top_importance = (
        importances.groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
        .head(75)
    )

    notes = pd.DataFrame(
        [
            {
                "topic": "primary_metric",
                "note": "Preprocess secimi inner CV f1_macro ile yapildi; outer CV sonucu raporlanan daha az optimistik skordur.",
            },
            {
                "topic": "leakage_control",
                "note": "Imputation, rare category, clipping, log feature secimi ve BLOSUM feature olusturma her fold icinde train'e fit edilip validation'a transform edilir.",
            },
            {
                "topic": "blosum",
                "note": "AA_1 ve AA_2 tek harf amino asitse BLOSUM62 skoru eklenir; bilinmeyen/stop/kompleks degerlerde skor NaN kalir ve valid bayragi eklenir.",
            },
        ]
    )

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        outer_summary.to_excel(writer, sheet_name="outer_summary", index=False)
        outer_scores.to_excel(writer, sheet_name="outer_fold_scores", index=False)
        inner_leaderboard.to_excel(writer, sheet_name="inner_leaderboard", index=False)
        inner_scores.to_excel(writer, sheet_name="inner_all_scores", index=False)
        top_importance.to_excel(writer, sheet_name="top_importance", index=False)
        candidate_grid.to_excel(writer, sheet_name="candidate_grid", index=False)
        profile.to_excel(writer, sheet_name="dataset_profile", index=False)
        notes.to_excel(writer, sheet_name="notes", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kanser verisi icin LightGBM preprocess kombinasyonlarini nested CV ile raporlar.")
    parser.add_argument("--data", type=Path, default=Path("YARISMA_TRAIN_KANSER.csv"))
    parser.add_argument("--output", type=Path, default=Path("lightgbm_nested_preprocess_kANSER_ONLY.xlsx"))
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=350)
    parser.add_argument(
        "--candidate-set",
        choices=["core", "full"],
        default="core",
        help="core: 48 kombinasyon. full: clip/log varyantlariyla 144 kombinasyon.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    df = read_dataset(args.data)
    candidates = make_candidates(args.candidate_set)
    print(f"dataset={args.data} rows={len(df)} candidates={len(candidates)}")
    outer_scores, inner_scores, importances = nested_experiment(
        df=df,
        candidates=candidates,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        seed=args.seed,
        n_estimators=args.n_estimators,
    )
    write_report(args.output, df, candidates, outer_scores, inner_scores, importances)
    print(f"Excel raporu yazildi: {args.output.resolve()}")
