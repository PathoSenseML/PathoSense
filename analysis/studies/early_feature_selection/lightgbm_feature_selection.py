"""
Balanced LightGBM experiments with three feature selection methods.

For each encoded dataset this script runs:
1. LightGBM with all features, no feature selection.
2. LightGBM Feature Importance selection, then LightGBM again.
3. SelectKBest selection, then LightGBM again.
4. Mutual Information selection, then LightGBM again.

Class balancing is applied only to the training split to avoid leakage.
Validation and test splits keep the real class distribution.
"""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split

    ML_IMPORT_ERROR = None
except ImportError as exc:
    lgb = None
    SelectKBest = None
    f_classif = None
    mutual_info_classif = None
    ML_IMPORT_ERROR = exc


DATA_DIR = Path(__file__).resolve().parent
ENCODED_DATA_DIR = DATA_DIR / "encoded_datasets"
OUTPUT_DIR = DATA_DIR / "lightgbm_outputs"

TARGET_COLUMN = "Label"
ID_COLUMNS = ["Variant_ID"]

DATASETS = {
    "CFTR_panel": ENCODED_DATA_DIR / "YARISMA_TRAIN_CFTR_numeric_encoded.csv",
    "KANSER_panel": ENCODED_DATA_DIR / "YARISMA_TRAIN_KANSER_numeric_encoded.csv",
    "MASTER_main": ENCODED_DATA_DIR / "YARISMA_TRAIN_MASTER_numeric_encoded.csv",
    "PAH_panel": ENCODED_DATA_DIR / "YARISMA_TRAIN_PAH_numeric_encoded.csv",
}

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALID_SIZE_FROM_TRAIN_VALID = 0.25
EARLY_STOPPING_ROUNDS = 50
TOP_K_FEATURES = 80


def require_ml_packages() -> None:
    """Fail with a clear message if LightGBM/sklearn are not installed."""
    if ML_IMPORT_ERROR is not None:
        raise ImportError(
            "LightGBM training needs lightgbm and scikit-learn. "
            "Install them with: pip install -r requirements_eda.txt"
        ) from ML_IMPORT_ERROR


def clean_output_dir(output_dir: Path = OUTPUT_DIR) -> None:
    """Remove old LightGBM outputs as much as Windows file locks allow."""
    resolved_output = output_dir.resolve()
    resolved_data_dir = DATA_DIR.resolve()

    if resolved_output.name != "lightgbm_outputs":
        raise ValueError(f"Refusing to clean unexpected output directory: {output_dir}")
    if resolved_output.parent != resolved_data_dir:
        raise ValueError(f"Refusing to clean output outside data directory: {output_dir}")

    resolved_output.mkdir(parents=True, exist_ok=True)

    skipped_locked_paths = []
    for child in resolved_output.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        except PermissionError:
            skipped_locked_paths.append(child)

    if skipped_locked_paths:
        print("Could not remove locked old output files:")
        for path in skipped_locked_paths:
            print(f"  {path}")
        print("Close these files in the editor if you want a fully clean folder.")


def load_dataset(path: Path) -> pd.DataFrame:
    """Load one encoded + imputed dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def get_feature_columns(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    id_columns: Iterable[str] = ID_COLUMNS,
) -> List[str]:
    """Return model feature columns."""
    excluded = set(id_columns) | {target_column}
    return [col for col in df.columns if col not in excluded]


def make_train_valid_test_split(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create stratified train / validation / test split."""
    x = df[feature_columns]
    y = df[target_column].astype(int)

    x_train_valid, x_test, y_train_valid, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train_valid,
        y_train_valid,
        test_size=VALID_SIZE_FROM_TRAIN_VALID,
        random_state=RANDOM_STATE,
        stratify=y_train_valid,
    )

    return x_train, x_valid, x_test, y_train, y_valid, y_test


def balance_training_data(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Balance only the training split by oversampling minority classes."""
    train_df = x_train.copy()
    train_df[TARGET_COLUMN] = y_train.values

    before_counts = y_train.value_counts().sort_index()
    max_count = int(before_counts.max())

    balanced_parts = []
    for class_value, class_df in train_df.groupby(TARGET_COLUMN):
        balanced_parts.append(
            class_df.sample(
                n=max_count,
                replace=len(class_df) < max_count,
                random_state=random_state + int(class_value),
            )
        )

    balanced_df = (
        pd.concat(balanced_parts, axis=0)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )

    y_balanced = balanced_df[TARGET_COLUMN].astype(int)
    x_balanced = balanced_df.drop(columns=[TARGET_COLUMN])
    after_counts = y_balanced.value_counts().sort_index()

    balance_summary = pd.DataFrame(
        {
            "class_value": sorted(set(before_counts.index) | set(after_counts.index)),
        }
    )
    balance_summary["train_count_before_balance"] = balance_summary[
        "class_value"
    ].map(before_counts).fillna(0).astype(int)
    balance_summary["train_count_after_balance"] = balance_summary[
        "class_value"
    ].map(after_counts).fillna(0).astype(int)

    return x_balanced, y_balanced, balance_summary


def train_lightgbm_classifier(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
):
    """Train LightGBM classifier with early stopping."""
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        force_col_wise=True,
        verbose=-1,
    )

    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return model


def safe_probability_metric(
    metric_name: str,
    y_true: pd.Series,
    y_score: np.ndarray,
) -> float:
    """Calculate probability metrics that can fail on degenerate splits."""
    try:
        if metric_name == "roc_auc":
            return float(roc_auc_score(y_true, y_score))
        if metric_name == "average_precision":
            return float(average_precision_score(y_true, y_score))
        if metric_name == "log_loss":
            return float(log_loss(y_true, y_score, labels=[0, 1]))
    except ValueError:
        return np.nan
    raise ValueError(f"Unknown metric: {metric_name}")


def evaluate_classifier(
    model,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    dataset_name: str,
    method_name: str,
    feature_count: int,
) -> Dict[str, object]:
    """Evaluate one model on the untouched test split."""
    y_score = model.predict_proba(x_test)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)

    return {
        "dataset_name": dataset_name,
        "method_name": method_name,
        "feature_count": feature_count,
        "best_iteration": getattr(model, "best_iteration_", np.nan),
        "roc_auc": safe_probability_metric("roc_auc", y_test, y_score),
        "average_precision": safe_probability_metric(
            "average_precision",
            y_test,
            y_score,
        ),
        "log_loss": safe_probability_metric("log_loss", y_test, y_score),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
    }


def get_lightgbm_feature_scores(model, feature_names: List[str]) -> pd.DataFrame:
    """Return LightGBM gain and split importance."""
    booster = model.booster_
    scores = pd.DataFrame(
        {
            "feature": feature_names,
            "score": booster.feature_importance(importance_type="gain"),
            "split_importance": booster.feature_importance(importance_type="split"),
        }
    )
    return scores.sort_values(
        ["score", "split_importance"],
        ascending=False,
    ).reset_index(drop=True)


def select_by_lightgbm_importance(
    baseline_model,
    feature_columns: List[str],
    top_k: int,
) -> Tuple[List[str], pd.DataFrame]:
    """Select top features using baseline LightGBM gain importance."""
    scores = get_lightgbm_feature_scores(baseline_model, feature_columns)
    selected_features = scores["feature"].head(top_k).tolist()
    scores["selection_method"] = "lightgbm_importance"
    scores["selected"] = scores["feature"].isin(selected_features)
    return selected_features, scores


def select_by_select_k_best(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    top_k: int,
) -> Tuple[List[str], pd.DataFrame]:
    """Select top features using SelectKBest with ANOVA F-score."""
    selector = SelectKBest(score_func=f_classif, k=top_k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        selector.fit(x_train, y_train)

    scores = pd.DataFrame(
        {
            "feature": x_train.columns,
            "score": selector.scores_,
            "p_value": selector.pvalues_,
        }
    ).replace([np.inf, -np.inf], np.nan)
    scores["score"] = scores["score"].fillna(-np.inf)
    scores = scores.sort_values("score", ascending=False).reset_index(drop=True)

    selected_features = scores["feature"].head(top_k).tolist()
    scores["selection_method"] = "select_k_best_f_classif"
    scores["selected"] = scores["feature"].isin(selected_features)
    return selected_features, scores


def select_by_mutual_information(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    top_k: int,
) -> Tuple[List[str], pd.DataFrame]:
    """Select top features using mutual information with the target."""
    mi_scores = mutual_info_classif(
        x_train,
        y_train,
        random_state=RANDOM_STATE,
        discrete_features=False,
    )
    scores = pd.DataFrame(
        {
            "feature": x_train.columns,
            "score": mi_scores,
        }
    )
    scores = scores.sort_values("score", ascending=False).reset_index(drop=True)

    selected_features = scores["feature"].head(top_k).tolist()
    scores["selection_method"] = "mutual_information"
    scores["selected"] = scores["feature"].isin(selected_features)
    return selected_features, scores


def build_delta_vs_all_features(metrics: pd.DataFrame) -> pd.DataFrame:
    """Compare every feature-selection method against the all-features baseline."""
    baseline = metrics.loc[metrics["method_name"] == "all_features"].iloc[0]
    metric_columns = [
        "roc_auc",
        "average_precision",
        "log_loss",
        "accuracy",
        "balanced_accuracy",
        "f1",
        "precision",
        "recall",
    ]
    records = []

    for _, row in metrics.iterrows():
        if row["method_name"] == "all_features":
            continue

        record = {
            "dataset_name": row["dataset_name"],
            "method_name": row["method_name"],
            "baseline_feature_count": int(baseline["feature_count"]),
            "selected_feature_count": int(row["feature_count"]),
            "feature_count_reduction_pct": round(
                (1 - row["feature_count"] / baseline["feature_count"]) * 100,
                2,
            ),
            "baseline_best_iteration": baseline["best_iteration"],
            "selected_best_iteration": row["best_iteration"],
        }

        for col in metric_columns:
            record[f"baseline_{col}"] = baseline[col]
            record[f"selected_{col}"] = row[col]
            record[f"delta_{col}"] = (
                row[col] - baseline[col]
                if pd.notna(row[col]) and pd.notna(baseline[col])
                else np.nan
            )

        records.append(record)

    return pd.DataFrame(records)


def save_dataset_outputs(
    dataset_output_dir: Path,
    metrics: pd.DataFrame,
    delta_metrics: pd.DataFrame,
    balance_summary: pd.DataFrame,
    feature_scores: Dict[str, pd.DataFrame],
    selected_features: Dict[str, List[str]],
) -> None:
    """Save clear per-dataset outputs."""
    feature_scores_dir = dataset_output_dir / "feature_scores"
    selected_features_dir = dataset_output_dir / "selected_features"
    feature_scores_dir.mkdir(parents=True, exist_ok=True)
    selected_features_dir.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(dataset_output_dir / "metrics.csv", index=False)
    delta_metrics.to_csv(dataset_output_dir / "delta_vs_all_features.csv", index=False)
    balance_summary.to_csv(dataset_output_dir / "class_balance_summary.csv", index=False)

    for method_name, scores in feature_scores.items():
        scores.to_csv(feature_scores_dir / f"{method_name}.csv", index=False)

    for method_name, features in selected_features.items():
        pd.DataFrame({"feature": features}).to_csv(
            selected_features_dir / f"{method_name}.csv",
            index=False,
        )


def train_compare_for_dataset(
    dataset_name: str,
    dataset_path: Path,
    output_dir: Path = OUTPUT_DIR,
) -> Dict[str, object]:
    """Train all balanced LightGBM experiments for one dataset."""
    require_ml_packages()

    df = load_dataset(dataset_path)
    feature_columns = get_feature_columns(df)
    top_k = min(TOP_K_FEATURES, len(feature_columns))

    x_train, x_valid, x_test, y_train, y_valid, y_test = make_train_valid_test_split(
        df,
        feature_columns,
    )
    x_train_balanced, y_train_balanced, balance_summary = balance_training_data(
        x_train,
        y_train,
    )

    metrics_records = []
    feature_scores = {}
    selected_features = {}

    all_features_model = train_lightgbm_classifier(
        x_train_balanced,
        y_train_balanced,
        x_valid,
        y_valid,
    )
    metrics_records.append(
        evaluate_classifier(
            all_features_model,
            x_test,
            y_test,
            dataset_name,
            "all_features",
            len(feature_columns),
        )
    )

    selected_features["all_features"] = feature_columns

    selection_methods = {
        "lightgbm_importance": lambda: select_by_lightgbm_importance(
            all_features_model,
            feature_columns,
            top_k,
        ),
        "select_k_best": lambda: select_by_select_k_best(
            x_train_balanced,
            y_train_balanced,
            top_k,
        ),
        "mutual_information": lambda: select_by_mutual_information(
            x_train_balanced,
            y_train_balanced,
            top_k,
        ),
    }

    for method_name, selector_fn in selection_methods.items():
        method_features, method_scores = selector_fn()
        selected_features[method_name] = method_features
        feature_scores[method_name] = method_scores

        method_model = train_lightgbm_classifier(
            x_train_balanced[method_features],
            y_train_balanced,
            x_valid[method_features],
            y_valid,
        )
        metrics_records.append(
            evaluate_classifier(
                method_model,
                x_test[method_features],
                y_test,
                dataset_name,
                method_name,
                len(method_features),
            )
        )

    metrics = pd.DataFrame(metrics_records)
    delta_metrics = build_delta_vs_all_features(metrics)

    save_dataset_outputs(
        output_dir / dataset_name,
        metrics,
        delta_metrics,
        balance_summary,
        feature_scores,
        selected_features,
    )

    print(f"\n{dataset_name}")
    print("Class balance used on train split:")
    print(balance_summary.to_string(index=False))
    print("\nMetrics:")
    print(metrics.to_string(index=False))
    print("\nDelta vs all_features:")
    print(
        delta_metrics[
            [
                "method_name",
                "feature_count_reduction_pct",
                "delta_roc_auc",
                "delta_balanced_accuracy",
                "delta_f1",
                "delta_log_loss",
            ]
        ].to_string(index=False)
    )

    return {
        "dataset_name": dataset_name,
        "metrics": metrics,
        "delta_metrics": delta_metrics,
        "balance_summary": balance_summary,
        "feature_scores": feature_scores,
        "selected_features": selected_features,
    }


def train_compare_all_datasets(
    datasets: Dict[str, Path] = DATASETS,
    output_dir: Path = OUTPUT_DIR,
) -> Dict[str, Dict[str, object]]:
    """Run balanced baseline and three feature selection methods."""
    require_ml_packages()
    clean_output_dir(output_dir)

    results = {}
    all_metrics = []
    all_delta_metrics = []
    all_balance_summaries = []

    for dataset_name, dataset_path in datasets.items():
        result = train_compare_for_dataset(dataset_name, dataset_path, output_dir)
        results[dataset_name] = result
        all_metrics.append(result["metrics"])
        all_delta_metrics.append(result["delta_metrics"])

        balance_summary = result["balance_summary"].copy()
        balance_summary.insert(0, "dataset_name", dataset_name)
        all_balance_summaries.append(balance_summary)

    combined_metrics = pd.concat(all_metrics, ignore_index=True)
    combined_delta_metrics = pd.concat(all_delta_metrics, ignore_index=True)
    combined_balance = pd.concat(all_balance_summaries, ignore_index=True)

    combined_metrics.to_csv(output_dir / "combined_metrics.csv", index=False)
    combined_delta_metrics.to_csv(
        output_dir / "combined_delta_vs_all_features.csv",
        index=False,
    )
    combined_balance.to_csv(output_dir / "combined_class_balance_summary.csv", index=False)

    return results


if __name__ == "__main__":
    train_compare_all_datasets()
