"""Leakage-safe BLOSUM62 amino-acid-substitution feature experiment.

BLOSUM62 features in this script are biological proxy features computed only
from AA_1 and AA_2. They do not use Label, Variant_ID, validation labels, or
test labels, so the same function can be applied to future unlabeled test data
as long as AA_1 and AA_2 exist. These are not CADD, REVEL, AlphaMissense, or
other real in-silico prediction scores.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ID_COLUMN = "Variant_ID"
TARGET_COLUMN = "Label"
AA_COLUMNS = ("AA_1", "AA_2")
NON_FEATURE_COLUMNS = {ID_COLUMN, TARGET_COLUMN}
PANEL_NAMES = ("MASTER", "KANSER", "PAH", "CFTR")
BLOSUM62_FEATURES = [
    "blosum62_score",
    "blosum62_negative_flag",
    "blosum62_zero_flag",
    "blosum62_positive_flag",
    "blosum62_abs_score",
    "aa_same",
    "aa_unknown_flag",
]
OUTPUT_FILENAMES = [
    "blosum62_feature_results.csv",
    "blosum62_delta_vs_baseline.csv",
    "blosum62_feature_list.txt",
    "blosum62_summary.md",
]

AMINO_ACIDS = ("A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V")
BLOSUM62_ROWS = {
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
BLOSUM62 = {
    row_aa: {col_aa: score for col_aa, score in zip(AMINO_ACIDS, scores)}
    for row_aa, scores in BLOSUM62_ROWS.items()
}


def clean_aa(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    if text in {"", "-", ".", "NA", "NAN", "NONE", "NULL", "UNKNOWN"}:
        return None
    if len(text) != 1 or text not in BLOSUM62:
        return None
    return text


def add_blosum62_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append leakage-safe BLOSUM62 features from AA_1 and AA_2 only.

    This function never reads Label or Variant_ID. It can be used on future test
    data without labels, and the appended columns are amino-acid-substitution
    biological proxy features, not real CADD/REVEL/AlphaMissense scores.
    """
    out = df.copy()
    if not set(AA_COLUMNS).issubset(out.columns):
        for col in BLOSUM62_FEATURES:
            out[col] = 0
        out["aa_unknown_flag"] = 1
        return out

    aa1 = out["AA_1"].map(clean_aa)
    aa2 = out["AA_2"].map(clean_aa)
    valid = aa1.notna() & aa2.notna()

    scores = pd.Series(np.zeros(len(out), dtype=int), index=out.index)
    for idx in out.index[valid]:
        scores.loc[idx] = BLOSUM62[aa1.loc[idx]][aa2.loc[idx]]

    out["blosum62_score"] = scores.astype(int)
    out["blosum62_negative_flag"] = (out["blosum62_score"] < 0).astype(int)
    out["blosum62_zero_flag"] = (out["blosum62_score"] == 0).astype(int)
    out["blosum62_positive_flag"] = (out["blosum62_score"] > 0).astype(int)
    out["blosum62_abs_score"] = out["blosum62_score"].abs().astype(int)
    out["aa_same"] = (valid & (aa1 == aa2)).astype(int)
    out["aa_unknown_flag"] = (~valid).astype(int)
    return out


def get_original_feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in NON_FEATURE_COLUMNS]


def split_train_validation(y: pd.Series, validation_size: float, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    train_parts = []
    valid_parts = []
    for _, labels in y.groupby(y, sort=False):
        indices = labels.index.to_numpy().copy()
        rng.shuffle(indices)
        n_valid = max(1, int(round(len(indices) * validation_size)))
        valid_parts.append(indices[:n_valid])
        train_parts.append(indices[n_valid:])
    train_idx = np.concatenate(train_parts)
    valid_idx = np.concatenate(valid_parts)
    rng.shuffle(train_idx)
    rng.shuffle(valid_idx)
    return train_idx, valid_idx


@dataclass
class NumpyPreprocessor:
    numeric_cols: list[str]
    categorical_cols: list[str]
    medians: pd.Series
    means: pd.Series
    stds: pd.Series
    categories: dict[str, list[str]]

    @classmethod
    def fit(cls, df: pd.DataFrame, columns: list[str]) -> "NumpyPreprocessor":
        x = df[columns].copy()
        numeric_cols = x.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [col for col in columns if col not in numeric_cols]
        numeric = x[numeric_cols].apply(pd.to_numeric, errors="coerce") if numeric_cols else pd.DataFrame(index=x.index)
        medians = numeric.median(axis=0).fillna(0.0)
        filled = numeric.fillna(medians)
        means = filled.mean(axis=0).fillna(0.0)
        stds = filled.std(axis=0, ddof=0).replace(0, 1.0).fillna(1.0)
        categories = {col: sorted(x[col].fillna("__MISSING__").astype(str).unique().tolist()) for col in categorical_cols}
        return cls(numeric_cols, categorical_cols, medians, means, stds, categories)

    def transform(self, df: pd.DataFrame, columns: list[str]) -> np.ndarray:
        x = df.reindex(columns=columns).copy()
        blocks = []
        if self.numeric_cols:
            numeric = x[self.numeric_cols].apply(pd.to_numeric, errors="coerce")
            numeric = numeric.fillna(self.medians)
            numeric = (numeric - self.means) / self.stds
            blocks.append(numeric.to_numpy(dtype=float))
        for col in self.categorical_cols:
            values = x[col].fillna("__MISSING__").astype(str)
            cats = self.categories[col]
            block = np.zeros((len(values), len(cats)), dtype=float)
            positions = {cat: idx for idx, cat in enumerate(cats)}
            for row_idx, value in enumerate(values):
                cat_idx = positions.get(value)
                if cat_idx is not None:
                    block[row_idx, cat_idx] = 1.0
            blocks.append(block)
        if not blocks:
            return np.zeros((len(df), 0), dtype=float)
        return np.hstack(blocks)


class NumpyLogisticRegression:
    def __init__(self, learning_rate: float = 0.05, epochs: int = 900, l2: float = 0.001):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.weights: np.ndarray | None = None
        self.bias = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NumpyLogisticRegression":
        self.weights = np.zeros(x.shape[1], dtype=float)
        pos_rate = np.clip(y.mean(), 1e-5, 1 - 1e-5)
        self.bias = float(np.log(pos_rate / (1 - pos_rate)))
        n_rows = max(1, x.shape[0])
        for _ in range(self.epochs):
            logits = np.clip(x @ self.weights + self.bias, -35, 35)
            probs = 1.0 / (1.0 + np.exp(-logits))
            error = probs - y
            grad_w = (x.T @ error) / n_rows + self.l2 * self.weights
            grad_b = float(error.mean())
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Model has not been fitted.")
        logits = np.clip(x @ self.weights + self.bias, -35, 35)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - probs, probs])


def invalid_aa_values(df: pd.DataFrame) -> dict[str, list[str]]:
    invalid = {}
    for col in AA_COLUMNS:
        if col not in df.columns:
            invalid[col] = ["<missing column>"]
            continue
        values = df[col].dropna().astype(str).str.strip()
        bad = sorted({value for value in values if clean_aa(value) is None})
        invalid[col] = bad
    return invalid


def find_repo_root(start_dir: Path) -> Path:
    for candidate in [start_dir, *start_dir.parents]:
        if (candidate / "data" / "panels").exists():
            return candidate
    return start_dir


def discover_datasets(base_dir: Path) -> dict[str, Path]:
    repo_root = find_repo_root(base_dir)
    data_dir = repo_root / "data" / "panels"
    found = {}
    for panel in PANEL_NAMES:
        candidates = []
        candidates.extend(data_dir.glob(f"YARISMA_TRAIN_{panel}.csv"))
        candidates.extend(data_dir.glob(f"YARISMA_TRAIN_{panel}*.csv"))
        if candidates:
            found[panel] = sorted(candidates, key=lambda path: (len(str(path)), str(path)))[0]
    return found


def resolve_output_dir(base_dir: Path, requested_output_dir: str) -> Path:
    output_dir = base_dir / requested_output_dir
    if not any((output_dir / filename).exists() for filename in OUTPUT_FILENAMES):
        return output_dir

    stem = output_dir.name
    for run_number in range(2, 1000):
        candidate = output_dir.with_name(f"{stem}_run{run_number}")
        if not any((candidate / filename).exists() for filename in OUTPUT_FILENAMES):
            return candidate
    raise RuntimeError("Could not find a non-overwriting output directory.")


def average_precision_score_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    positives = y_sorted.sum()
    if positives == 0:
        return float("nan")
    precision_at_k = np.cumsum(y_sorted) / (np.arange(len(y_sorted)) + 1)
    return float((precision_at_k * y_sorted).sum() / positives)


def roc_auc_score_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positives = y_true == 1
    negatives = y_true == 0
    n_pos = positives.sum()
    n_neg = negatives.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    sorted_scores = y_score[order]
    start = 0
    while start < len(y_score):
        end = start + 1
        while end < len(y_score) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    rank_sum_pos = ranks[positives].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def log_loss_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    probs = np.clip(y_score, 1e-15, 1 - 1e-15)
    return float(-(y_true * np.log(probs) + (1 - y_true) * np.log(1 - probs)).mean())


def metrics_np(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    tp = int(((predictions == 1) & (y_true == 1)).sum())
    tn = int(((predictions == 0) & (y_true == 0)).sum())
    fp = int(((predictions == 1) & (y_true == 0)).sum())
    fn = int(((predictions == 0) & (y_true == 1)).sum())
    accuracy = (tp + tn) / max(1, len(y_true))
    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    precision = tp / max(1, tp + fp)
    f1 = 2 * precision * sensitivity / max(1e-12, precision + sensitivity)
    return {
        "auc": roc_auc_score_np(y_true, probabilities),
        "average_precision": average_precision_score_np(y_true, probabilities),
        "log_loss": log_loss_np(y_true, probabilities),
        "accuracy": accuracy,
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
        "f1": f1,
    }


def evaluate_model(
    df: pd.DataFrame,
    y: pd.Series,
    feature_columns: list[str],
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    random_state: int,
) -> tuple[str, dict[str, float]]:
    if importlib.util.find_spec("sklearn") is not None:
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, log_loss, roc_auc_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder

        x_train = df.iloc[train_idx][feature_columns]
        x_valid = df.iloc[valid_idx][feature_columns]
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]
        numeric_cols = x_train.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [col for col in feature_columns if col not in numeric_cols]
        preprocess = ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="median"), numeric_cols),
                ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols),
            ]
        )
        if importlib.util.find_spec("lightgbm") is not None:
            from lightgbm import LGBMClassifier

            model_name = "lightgbm_pipeline"
            model = LGBMClassifier(random_state=random_state, class_weight="balanced", verbose=-1)
        else:
            model_name = "sklearn_pipeline"
            model_cls = HistGradientBoostingClassifier if not categorical_cols else RandomForestClassifier
            model = model_cls(random_state=random_state)
        pipeline = Pipeline([("preprocess", preprocess), ("model", model)])
        pipeline.fit(x_train, y_train)
        probs = pipeline.predict_proba(x_valid)[:, 1]
        preds = (probs >= 0.5).astype(int)
        return model_name, {
            "auc": roc_auc_score(y_valid, probs),
            "average_precision": average_precision_score(y_valid, probs),
            "log_loss": log_loss(y_valid, probs),
            "accuracy": accuracy_score(y_valid, preds),
            "balanced_accuracy": balanced_accuracy_score(y_valid, preds),
            "f1": f1_score(y_valid, preds, zero_division=0),
        }

    preprocessor = NumpyPreprocessor.fit(df.iloc[train_idx], feature_columns)
    x_train = preprocessor.transform(df.iloc[train_idx], feature_columns)
    x_valid = preprocessor.transform(df.iloc[valid_idx], feature_columns)
    y_train = y.iloc[train_idx].to_numpy(dtype=float)
    y_valid = y.iloc[valid_idx].to_numpy(dtype=int)
    model = NumpyLogisticRegression()
    model.fit(x_train, y_train)
    probs = model.predict_proba(x_valid)[:, 1]
    return "numpy_logistic_fallback", metrics_np(y_valid, probs)


def metric_direction(metric: str) -> int:
    return -1 if metric == "log_loss" else 1


def build_delta(results: pd.DataFrame) -> pd.DataFrame:
    metrics = ["auc", "average_precision", "log_loss", "accuracy", "balanced_accuracy", "f1"]
    rows = []
    for dataset, group in results.groupby("dataset", sort=False):
        baseline = group[group["experiment"] == "baseline"].iloc[0]
        blosum62 = group[group["experiment"] == "blosum62"].iloc[0]
        row = {"dataset": dataset, "panel": baseline["panel"]}
        for metric in metrics:
            delta = float(blosum62[metric] - baseline[metric])
            direction = metric_direction(metric)
            if delta == 0:
                verdict = "unchanged"
            elif delta * direction > 0:
                verdict = "improved"
            else:
                verdict = "worsened"
            row[f"{metric}_baseline"] = baseline[metric]
            row[f"{metric}_blosum62"] = blosum62[metric]
            row[f"{metric}_delta"] = delta
            row[f"{metric}_verdict"] = verdict
        rows.append(row)
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    headers = ["panel", "metric", "baseline", "blosum62", "delta", "verdict"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    metrics = ["auc", "average_precision", "log_loss", "accuracy", "balanced_accuracy", "f1"]
    for _, row in df.iterrows():
        for metric in metrics:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["panel"]),
                        metric,
                        f"{row[f'{metric}_baseline']:.6f}",
                        f"{row[f'{metric}_blosum62']:.6f}",
                        f"{row[f'{metric}_delta']:.6f}",
                        str(row[f"{metric}_verdict"]),
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


def write_summary(output_path: Path, results: pd.DataFrame, delta: pd.DataFrame) -> None:
    lines = [
        "# BLOSUM62 Feature Experiment",
        "",
        "BLOSUM62 features use only AA_1 and AA_2.",
        "They are leakage-safe because they do not use Label, validation labels, or test labels.",
        "They can be generated for future test data as long as AA_1 and AA_2 exist.",
        "They are not CADD, REVEL, AlphaMissense, or other real in-silico scores.",
        "They are amino-acid-substitution biological proxy features.",
        "",
        "## Baseline vs BLOSUM62",
        "",
        markdown_table(delta),
        "",
        "## Raw Results",
        "",
        results.to_markdown(index=False),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_experiment(args: argparse.Namespace) -> pd.DataFrame:
    base_dir = Path(__file__).resolve().parents[1]
    output_dir = resolve_output_dir(base_dir, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = discover_datasets(base_dir)
    missing = [panel for panel in PANEL_NAMES if panel not in datasets]
    if missing:
        repo_root = find_repo_root(base_dir)
        expected_dir = repo_root / "data" / "panels"
        expected_files = ", ".join(f"YARISMA_TRAIN_{panel}.csv" for panel in missing)
        raise FileNotFoundError(
            "Missing repository-local panel dataset(s). "
            f"Expected under {expected_dir}: {expected_files}. "
            "This experiment does not search Downloads by default."
        )

    rows = []
    for panel in PANEL_NAMES:
        input_path = datasets[panel]
        df = pd.read_csv(input_path)
        print(f"{panel}: path={input_path}")
        print(f"{panel}: shape={df.shape}")
        print(f"{panel}: AA_1 exists={'AA_1' in df.columns}, AA_2 exists={'AA_2' in df.columns}")
        print(f"{panel}: invalid AA values={invalid_aa_values(df)}")
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"{input_path} must contain {TARGET_COLUMN!r} for evaluation.")

        y = df[TARGET_COLUMN].astype(int)
        original_feature_columns = get_original_feature_columns(df)
        train_fe = add_blosum62_features(df)
        blosum62_feature_columns = original_feature_columns + [
            col for col in BLOSUM62_FEATURES if col not in original_feature_columns
        ]
        train_idx, valid_idx = split_train_validation(y, args.validation_size, args.random_state)

        for experiment_name, columns in [
            ("baseline", original_feature_columns),
            ("blosum62", blosum62_feature_columns),
        ]:
            model_name, metrics = evaluate_model(train_fe, y, columns, train_idx, valid_idx, args.random_state)
            rows.append(
                {
                    "dataset": input_path.name,
                    "panel": panel,
                    "experiment": experiment_name,
                    "model": model_name,
                    "n_rows": len(train_fe),
                    "n_train": len(train_idx),
                    "n_validation": len(valid_idx),
                    "n_features": len(columns),
                    "variant_id_used_as_feature": ID_COLUMN in columns,
                    **metrics,
                }
            )

    results = pd.DataFrame(rows)
    delta = build_delta(results)
    results.to_csv(output_dir / "blosum62_feature_results.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    delta.to_csv(output_dir / "blosum62_delta_vs_baseline.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    with (output_dir / "blosum62_feature_list.txt").open("w", encoding="utf-8", newline="") as file:
        file.write("# BLOSUM62-based amino-acid-substitution biological proxy features.\n")
        file.write("# Leakage-safe: computed only from AA_1 and AA_2, without Label or test labels.\n")
        file.write("# Not CADD, REVEL, AlphaMissense, or any real in-silico score.\n")
        for col in BLOSUM62_FEATURES:
            file.write(f"{col}\n")
    write_summary(output_dir / "blosum62_summary.md", results, delta)
    print(f"Saved outputs to {output_dir}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline vs BLOSUM62 feature experiment for PathoSense panels.")
    parser.add_argument("--output-dir", default="blosum62_outputs", help="Output directory relative to the repository root.")
    parser.add_argument("--validation-size", type=float, default=0.2, help="Stratified validation fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


if __name__ == "__main__":
    experiment_results = run_experiment(parse_args())
    print(experiment_results.to_string(index=False))
