"""Leakage-safe, reproducible baseline comparison for the MASTER panel.

The test set is never used for model, threshold, feature, or hyperparameter
selection. Run from any directory with Python 3.
"""

from __future__ import annotations

import importlib.util
import json
import math
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


RANDOM_STATE = 42
N_SPLITS = 5
ROOT = Path(__file__).resolve().parent
EXPECTED_INPUT = Path(r"C:\Users\AET\Documents\Teknofest sağlık\data\panels\YARISMA_TRAIN_MASTER.csv")
SEARCH_ROOTS = [
    Path(r"C:\Users\AET\Documents\Teknofest sağlık"),
    Path(r"C:\Users\AET\Desktop\PathoSense"),
    Path(r"C:\Users\AET\Documents\TeknoPatho"),
]
EXCLUDED = ["Variant_ID", "Label"]

OUTPUTS = {
    "train": ROOT / "master_train_set.csv",
    "test": ROOT / "master_test_set.csv",
    "split_report": ROOT / "master_split_report.md",
    "comparison": ROOT / "master_baseline_model_comparison.csv",
    "test_report": ROOT / "master_baseline_test_report.csv",
    "test_predictions": ROOT / "master_baseline_predictions_test.csv",
    "oof_predictions": ROOT / "master_baseline_predictions_oof.csv",
    "leakage": ROOT / "master_baseline_leakage_report.csv",
    "report": ROOT / "master_baseline_report.md",
    "model": ROOT / "master_baseline_best_model.pkl",
    "metadata": ROOT / "master_baseline_metadata.json",
}


def find_input() -> Path:
    if EXPECTED_INPUT.exists():
        return EXPECTED_INPUT.resolve()
    matches = []
    for search_root in SEARCH_ROOTS:
        if search_root.exists():
            matches.extend(search_root.rglob("YARISMA_TRAIN_MASTER.csv"))
    matches = sorted(set(matches))
    if not matches:
        raise FileNotFoundError("YARISMA_TRAIN_MASTER.csv was not found under the workspace")
    return matches[0].resolve()


def optimal_split_counts(n_benign: int, n_pathogenic: int) -> tuple[int, int]:
    """Find integer train counts closest to 15% benign train / 85% benign test.

    The continuous solution gives a unique target. Searching nearby integers
    avoids hard-coding dataset-specific counts while making the intent explicit.
    """
    # Solve b/(b+p)=.15 and (B-b)/((B-b)+(P-p))=.85.
    matrix = np.array([[0.85, -0.15], [-0.15, 0.85]], dtype=float)
    rhs = np.array([0.0, 0.85 * n_pathogenic - 0.15 * n_benign], dtype=float)
    target_b, target_p = np.linalg.solve(matrix, rhs)
    candidates = []
    for b in range(max(1, math.floor(target_b) - 5), min(n_benign, math.ceil(target_b) + 6)):
        for p in range(max(1, math.floor(target_p) - 5), min(n_pathogenic, math.ceil(target_p) + 6)):
            train_ratio = b / (b + p)
            test_ratio = (n_benign - b) / ((n_benign - b) + (n_pathogenic - p))
            ratio_error = (train_ratio - 0.15) ** 2 + (test_ratio - 0.85) ** 2
            distance = (b - target_b) ** 2 + (p - target_p) ** 2
            candidates.append((ratio_error, distance, b, p))
    _, _, b_train, p_train = min(candidates)
    return b_train, p_train


def make_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = df["Label"].value_counts().to_dict()
    if set(counts) != {0, 1}:
        raise ValueError(f"Label must contain exactly 0 and 1; found {counts}")
    b_train, p_train = optimal_split_counts(int(counts[0]), int(counts[1]))
    rng = np.random.RandomState(RANDOM_STATE)
    benign_idx = df.index[df["Label"].eq(0)].to_numpy().copy()
    pathogenic_idx = df.index[df["Label"].eq(1)].to_numpy().copy()
    rng.shuffle(benign_idx)
    rng.shuffle(pathogenic_idx)
    train_idx = np.concatenate([benign_idx[:b_train], pathogenic_idx[:p_train]])
    test_idx = np.concatenate([benign_idx[b_train:], pathogenic_idx[p_train:]])
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return df.loc[train_idx].reset_index(drop=True), df.loc[test_idx].reset_index(drop=True)


def dist(df: pd.DataFrame) -> dict[str, int]:
    vc = df["Label"].value_counts().reindex([0, 1], fill_value=0)
    return {"0": int(vc[0]), "1": int(vc[1])}


def quality_checks(full: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> dict:
    train_ids = set(train["Variant_ID"].dropna().astype(str))
    test_ids = set(test["Variant_ID"].dropna().astype(str))
    overlap_ids = train_ids & test_ids
    train_dup_ids = int(train["Variant_ID"].duplicated(keep=False).sum())
    test_dup_ids = int(test["Variant_ID"].duplicated(keep=False).sum())
    # Hashing is stable within this run and handles mixed columns/NaNs efficiently.
    train_hash = set(pd.util.hash_pandas_object(train[features], index=False).tolist())
    test_hash = set(pd.util.hash_pandas_object(test[features], index=False).tolist())
    suspicious_tokens = ("label", "target", "outcome", "variant_id", "variantid")
    suspicious_columns = [c for c in features if any(t in c.lower() for t in suspicious_tokens)]
    return {
        "full_duplicate_rows": int(full.duplicated().sum()),
        "train_duplicate_rows": int(train.duplicated().sum()),
        "test_duplicate_rows": int(test.duplicated().sum()),
        "variant_id_overlap_count": len(overlap_ids),
        "variant_id_overlap_sample": sorted(overlap_ids)[:20],
        "train_rows_with_duplicated_variant_id": train_dup_ids,
        "test_rows_with_duplicated_variant_id": test_dup_ids,
        "identical_feature_hashes_across_train_test": len(train_hash & test_hash),
        "suspicious_feature_columns": suspicious_columns,
        "global_preprocessing_fit": False,
        "threshold_selected_on_test": False,
        "model_selected_on_test": False,
    }


def leakage_rows(checks: dict) -> list[dict[str, str]]:
    def row(name: str, ok: bool, details: str) -> dict[str, str]:
        return {"check_name": name, "status": "PASS" if ok else "WARNING", "details": details}

    return [
        row("train_test_variant_id_intersection", checks["variant_id_overlap_count"] == 0,
            f"overlap_count={checks['variant_id_overlap_count']}; sample={checks['variant_id_overlap_sample']}"),
        row("same_variant_id_in_train_and_test", checks["variant_id_overlap_count"] == 0,
            f"overlap_count={checks['variant_id_overlap_count']}"),
        row("duplicate_variant_id_train", checks["train_rows_with_duplicated_variant_id"] == 0,
            f"rows_with_duplicate_id={checks['train_rows_with_duplicated_variant_id']}"),
        row("duplicate_variant_id_test", checks["test_rows_with_duplicated_variant_id"] == 0,
            f"rows_with_duplicate_id={checks['test_rows_with_duplicated_variant_id']}"),
        row("identical_features_across_train_test", checks["identical_feature_hashes_across_train_test"] == 0,
            f"shared_unique_feature_hashes={checks['identical_feature_hashes_across_train_test']}"),
        row("full_duplicate_rows", checks["full_duplicate_rows"] == 0,
            f"duplicate_rows={checks['full_duplicate_rows']}"),
        row("suspicious_feature_column_names", not checks["suspicious_feature_columns"],
            f"candidates={checks['suspicious_feature_columns']}"),
        row("global_preprocessing_fit", True,
            "No global fit detected; preprocessing is inside CV pipeline where applicable"),
        row("threshold_selection", True,
            "Threshold selected only on OOF train CV, not on test set"),
        row("model_selection", True,
            "Model selected by OOF/CV only, not by test performance"),
    ]


def sklearn_preprocessor(numeric: list[str], categorical: list[str], native_numeric_nan: bool) -> ColumnTransformer:
    numeric_transformer = "passthrough" if native_numeric_nan else SimpleImputer(strategy="median")
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    return ColumnTransformer(
        [("num", numeric_transformer, numeric), ("cat", categorical_transformer, categorical)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def model_specs(numeric: list[str], categorical: list[str]) -> list[dict]:
    specs: list[dict] = []
    if importlib.util.find_spec("catboost"):
        from catboost import CatBoostClassifier
        specs.append({"name": "CatBoostClassifier", "kind": "catboost", "model": CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.05, loss_function="Logloss",
            random_seed=RANDOM_STATE, verbose=False, allow_writing_files=False, thread_count=-1,
        )})
    else:
        specs.append({"name": "CatBoostClassifier", "skip": "catboost not installed"})

    if importlib.util.find_spec("lightgbm"):
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                               random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
        specs.append({"name": "LGBMClassifier", "kind": "pipeline", "model": Pipeline([
            ("preprocess", sklearn_preprocessor(numeric, categorical, native_numeric_nan=True)), ("model", model)
        ])})
    else:
        specs.append({"name": "LGBMClassifier", "skip": "lightgbm not installed"})

    if importlib.util.find_spec("xgboost"):
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                              subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE,
                              n_jobs=-1, eval_metric="logloss", tree_method="hist")
        specs.append({"name": "XGBClassifier", "kind": "pipeline", "model": Pipeline([
            ("preprocess", sklearn_preprocessor(numeric, categorical, native_numeric_nan=True)), ("model", model)
        ])})
    else:
        specs.append({"name": "XGBClassifier", "skip": "xgboost not installed"})

    regular_pre = sklearn_preprocessor(numeric, categorical, native_numeric_nan=False)
    specs.extend([
        {"name": "ExtraTreesClassifier", "kind": "pipeline", "model": Pipeline([
            ("preprocess", clone(regular_pre)), ("model", ExtraTreesClassifier(
                n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"))])},
        {"name": "RandomForestClassifier", "kind": "pipeline", "model": Pipeline([
            ("preprocess", clone(regular_pre)), ("model", RandomForestClassifier(
                n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced_subsample"))])},
        {"name": "HistGradientBoostingClassifier", "kind": "pipeline", "model": Pipeline([
            ("preprocess", clone(regular_pre)), ("model", HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=300, random_state=RANDOM_STATE))])},
        {"name": "LogisticRegression", "kind": "pipeline", "model": Pipeline([
            ("preprocess", clone(regular_pre)), ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE, class_weight="balanced"))])},
    ])
    return specs


def catboost_frame(x: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    result = x.copy()
    for col in categorical:
        result[col] = result[col].where(result[col].notna(), "__MISSING__").astype(str)
    return result


def fit_predict(spec: dict, x_train: pd.DataFrame, y_train: pd.Series,
                x_valid: pd.DataFrame, categorical: list[str]) -> tuple[object, np.ndarray]:
    model = clone(spec["model"])
    if spec["kind"] == "catboost":
        model.fit(catboost_frame(x_train, categorical), y_train, cat_features=categorical)
        prob = model.predict_proba(catboost_frame(x_valid, categorical))[:, 1]
    else:
        model.fit(x_train, y_train)
        prob = model.predict_proba(x_valid)[:, 1]
    return model, np.asarray(prob, dtype=float)


def best_threshold(y: np.ndarray, prob: np.ndarray) -> tuple[float, float, float, float]:
    precision, recall, thresholds = precision_recall_curve(y, prob)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    # First maximum is deterministic; PR thresholds are ascending.
    idx = int(np.nanargmax(f1))
    return float(thresholds[idx]), float(f1[idx]), float(precision[idx]), float(recall[idx])


def metrics(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y, prob)),
        "average_precision": float(average_precision_score(y, prob)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "log_loss": float(log_loss(y, np.clip(prob, 1e-15, 1 - 1e-15), labels=[0, 1])),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def error_types(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.select(
        [(y == 1) & (pred == 1), (y == 0) & (pred == 0), (y == 0) & (pred == 1), (y == 1) & (pred == 0)],
        ["TP", "TN", "FP", "FN"], default="UNKNOWN",
    )


def format_table(df: pd.DataFrame) -> str:
    shown = df.copy()
    for col in shown.select_dtypes(include=["float"]).columns:
        shown[col] = shown[col].map(lambda x: "" if pd.isna(x) else f"{x:.6f}")
    headers = "| " + " | ".join(shown.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(shown.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in shown.itertuples(index=False, name=None)]
    return "\n".join([headers, sep] + rows)


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    input_path = find_input()
    full = pd.read_csv(input_path)
    required = set(EXCLUDED)
    if not required.issubset(full.columns):
        raise ValueError(f"Missing required columns: {sorted(required - set(full.columns))}")
    if full["Label"].isna().any():
        raise ValueError("Label contains missing values")
    full["Label"] = full["Label"].astype(int)
    features = [c for c in full.columns if c not in EXCLUDED]
    categorical = full[features].select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [c for c in features if c not in categorical]

    train, test = make_split(full)
    train.to_csv(OUTPUTS["train"], index=False)
    test.to_csv(OUTPUTS["test"], index=False)
    checks = quality_checks(full, train, test, features)
    leakage = pd.DataFrame(leakage_rows(checks), columns=["check_name", "status", "details"])
    leakage.to_csv(OUTPUTS["leakage"], index=False)
    split_text = f"""# MASTER train/test split report

- Source: `{input_path}`
- Random state: {RANDOM_STATE}
- Split rule: integer counts nearest to 15% benign / 85% pathogenic in train and 85% benign / 15% pathogenic in test.
- Train distribution (0=benign, 1=pathogenic): `{dist(train)}`
- Test distribution (0=benign, 1=pathogenic): `{dist(test)}`
- Train benign share: {train['Label'].eq(0).mean():.6f}
- Test benign share: {test['Label'].eq(0).mean():.6f}
- Quality checks: `{json.dumps(checks, ensure_ascii=False)}`

**The test set was not used for model selection, threshold selection, feature selection, or hyperparameter selection.**
"""
    OUTPUTS["split_report"].write_text(split_text, encoding="utf-8")

    x_train, y_train = train[features], train["Label"].to_numpy()
    x_test, y_test = test[features], test["Label"].to_numpy()
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    comparison_rows, successful, skipped = [], {}, []

    for spec in model_specs(numeric, categorical):
        name = spec["name"]
        print(f"OOF: {name}", flush=True)
        if "skip" in spec:
            reason = spec["skip"]
            skipped.append(f"{name}: {reason}")
            comparison_rows.append({"model_name": name, "status": f"SKIPPED: {reason}"})
            continue
        try:
            oof = np.full(len(train), np.nan, dtype=float)
            for fold, (fit_idx, valid_idx) in enumerate(cv.split(x_train, y_train), 1):
                _, fold_prob = fit_predict(spec, x_train.iloc[fit_idx], pd.Series(y_train[fit_idx]),
                                           x_train.iloc[valid_idx], categorical)
                oof[valid_idx] = fold_prob
                print(f"  fold {fold}/{N_SPLITS} complete", flush=True)
            if np.isnan(oof).any():
                raise RuntimeError("OOF prediction array contains NaN")
            at_half = metrics(y_train, oof, 0.5)
            threshold, best_f1, best_precision, best_recall = best_threshold(y_train, oof)
            at_best = metrics(y_train, oof, threshold)
            row = {
                "model_name": name, "status": "OK", "oof_auc": at_half["auc"],
                "oof_average_precision": at_half["average_precision"],
                "oof_f1_at_0_5": at_half["f1"], "oof_precision_at_0_5": at_half["precision"],
                "oof_recall_at_0_5": at_half["recall"],
                "oof_balanced_accuracy_at_0_5": at_half["balanced_accuracy"],
                "oof_log_loss": at_half["log_loss"], "best_oof_threshold": threshold,
                "best_oof_f1": best_f1, "best_oof_precision": best_precision,
                "best_oof_recall": best_recall,
                "oof_tn": at_best["tn"], "oof_fp": at_best["fp"],
                "oof_fn": at_best["fn"], "oof_tp": at_best["tp"],
            }
            comparison_rows.append(row)
            successful[name] = {"spec": spec, "oof": oof, "row": row, "confusion": at_best}
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            skipped.append(f"{name}: {reason}")
            comparison_rows.append({"model_name": name, "status": f"SKIPPED: {reason}"})
            print(f"  SKIPPED: {reason}", flush=True)

    columns = [
        "model_name", "status", "oof_auc", "oof_average_precision", "oof_f1_at_0_5",
        "oof_precision_at_0_5", "oof_recall_at_0_5", "oof_balanced_accuracy_at_0_5",
        "oof_log_loss", "best_oof_threshold", "best_oof_f1", "best_oof_precision", "best_oof_recall",
        "oof_tn", "oof_fp", "oof_fn", "oof_tp",
    ]
    comparison = pd.DataFrame(comparison_rows).reindex(columns=columns)
    comparison.to_csv(OUTPUTS["comparison"], index=False)
    if not successful:
        raise RuntimeError("No model completed successfully")

    # Selection is exclusively by train OOF metrics, with required tie-breaks.
    winner_name = max(successful, key=lambda n: (
        successful[n]["row"]["best_oof_f1"], successful[n]["row"]["oof_auc"],
        successful[n]["row"]["oof_average_precision"],
        successful[n]["row"]["oof_balanced_accuracy_at_0_5"],
    ))
    winner = successful[winner_name]
    threshold = winner["row"]["best_oof_threshold"]
    print(f"Final fit: {winner_name}", flush=True)
    final_model, test_prob = fit_predict(winner["spec"], x_train, pd.Series(y_train), x_test, categorical)
    test_metrics = metrics(y_test, test_prob, threshold)
    test_pred = (test_prob >= threshold).astype(int)

    test_row = {
        "selected_model": winner_name, "selected_by": "best_oof_f1",
        "threshold_source": "OOF train CV", "threshold": threshold,
        "test_auc": test_metrics["auc"], "test_average_precision": test_metrics["average_precision"],
        "test_f1": test_metrics["f1"], "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"], "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_log_loss": test_metrics["log_loss"], "test_tn": test_metrics["tn"],
        "test_fp": test_metrics["fp"], "test_fn": test_metrics["fn"], "test_tp": test_metrics["tp"],
    }
    pd.DataFrame([test_row]).to_csv(OUTPUTS["test_report"], index=False)
    pd.DataFrame({
        "Variant_ID": test["Variant_ID"], "true_label": y_test,
        "pred_proba_pathogenic": test_prob, "pred_label": test_pred,
        "error_type": error_types(y_test, test_pred),
    }).to_csv(OUTPUTS["test_predictions"], index=False)

    oof = winner["oof"]
    oof_pred = (oof >= threshold).astype(int)
    pd.DataFrame({
        "Variant_ID": train["Variant_ID"], "true_label": y_train,
        "oof_proba_pathogenic": oof, "oof_pred_label": oof_pred,
        "model_name": winner_name, "error_type": error_types(y_train, oof_pred),
    }).to_csv(OUTPUTS["oof_predictions"], index=False)

    bundle = {
        "model": final_model, "model_name": winner_name, "threshold": threshold,
        "feature_columns": features, "categorical_columns": categorical,
        "numeric_columns": numeric, "catboost_string_fill": winner["spec"]["kind"] == "catboost",
    }
    joblib.dump(bundle, OUTPUTS["model"])
    metadata = {
        "project_folder": "TeknoPatho",
        "selected_model": winner_name, "selected_by": "best_oof_f1", "threshold": threshold,
        "threshold_source": "OOF train CV", "random_state": RANDOM_STATE,
        "feature_columns": features, "excluded_columns": EXCLUDED,
        "train_label_distribution": dist(train), "test_label_distribution": dist(test),
        "leakage_checks_summary": leakage["status"].value_counts().to_dict(),
        "created_files": [Path(__file__).name] + [p.name for p in OUTPUTS.values()],
    }
    OUTPUTS["metadata"].write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    ok_comparison = comparison[comparison["status"].eq("OK")]
    oof_confusions = pd.DataFrame([
        {
            "model_name": name,
            "threshold": result["row"]["best_oof_threshold"],
            "tn": result["confusion"]["tn"],
            "fp": result["confusion"]["fp"],
            "fn": result["confusion"]["fn"],
            "tp": result["confusion"]["tp"],
        }
        for name, result in successful.items()
    ])
    report = f"""# MASTER baseline v0 report

## Data and split

- Data file: `{input_path}`
- Working folder: `{ROOT}` (TeknoPatho)
- Full data shape: {full.shape[0]} rows × {full.shape[1]} columns
- Train label distribution (0=benign, 1=pathogenic): `{dist(train)}`
- Test label distribution (0=benign, 1=pathogenic): `{dist(test)}`
- Features used: {len(features)} ({len(numeric)} numeric, {len(categorical)} categorical)
- Excluded columns: `{EXCLUDED}`
- Random state: {RANDOM_STATE}

## Leakage controls

- **The test set was not used for model selection, threshold selection, feature selection, or hyperparameter selection.**
- The winner was selected only from 5-fold StratifiedKFold OOF predictions on train.
- The final threshold was selected only by train OOF F1 and transferred unchanged to test.
- All learned imputers and categorical encoders were fitted inside each fold through pipelines.
- CatBoost categorical missing replacement is a fixed, non-learned mapping; numeric missing values remain native.
- Variant_ID and Label were excluded from every model input.
- No AA numeric conversion, AL log transform, SHAP, ensemble, or engineered feature was used.
- No global fit detected; preprocessing is inside CV pipeline where applicable
- Threshold selected only on OOF train CV, not on test set
- Model selected by OOF/CV only, not by test performance

## Quality checks

`{json.dumps(checks, ensure_ascii=False)}`

{format_table(leakage)}

Warnings: {('none' if not any([checks['variant_id_overlap_count'], checks['train_rows_with_duplicated_variant_id'], checks['test_rows_with_duplicated_variant_id'], checks['identical_feature_hashes_across_train_test'], checks['full_duplicate_rows']]) else 'one or more duplicate/overlap checks are non-zero; inspect the values above')}

## Models tried

{', '.join(comparison['model_name'].tolist())}

Skipped models: {('none' if not skipped else '; '.join(skipped))}

## OOF results

{format_table(ok_comparison)}

### OOF confusion matrices at each model's best OOF-F1 threshold

{format_table(oof_confusions)}

Winner: **{winner_name}**, selected by highest Best OOF F1 ({winner['row']['best_oof_f1']:.6f}); ties, if any, were broken by ROC-AUC, Average Precision, and Balanced Accuracy. Its OOF threshold is {threshold:.12g}. Confusion matrix at that threshold: TN={winner['confusion']['tn']}, FP={winner['confusion']['fp']}, FN={winner['confusion']['fn']}, TP={winner['confusion']['tp']}.

## One-time test result

{format_table(pd.DataFrame([test_row]))}

**The test set was not used for model selection.** It was evaluated once after the model and threshold were selected from train OOF CV.
"""
    OUTPUTS["report"].write_text(report, encoding="utf-8")

    print("\n=== MASTER baseline summary ===")
    print(f"Working folder: {ROOT}")
    print(f"Data path: {input_path}")
    print(f"Train distribution: {dist(train)}")
    print(f"Test distribution: {dist(test)}")
    print(f"Tried models: {', '.join(comparison['model_name'])}")
    print(f"Skipped models: {('none' if not skipped else '; '.join(skipped))}")
    print(f"Best OOF model: {winner_name}")
    print(f"OOF F1: {winner['row']['best_oof_f1']:.6f}")
    print(f"Threshold: {threshold:.12g} (OOF train CV)")
    print(f"Test F1: {test_metrics['f1']:.6f}")
    print(f"Test precision: {test_metrics['precision']:.6f}")
    print(f"Test recall: {test_metrics['recall']:.6f}")
    print(f"Test confusion matrix: TN={test_metrics['tn']} FP={test_metrics['fp']} FN={test_metrics['fn']} TP={test_metrics['tp']}")
    print("Test set was not used for model selection.")


if __name__ == "__main__":
    main()

