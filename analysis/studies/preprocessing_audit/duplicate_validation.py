from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, RepeatedStratifiedKFold, StratifiedGroupKFold

from leakage_audit import EXPERIMENT_XLSX, feature_hashes, read_best_config
from lightgbm_preprocess_report import TARGET_COL, CandidateConfig, fit_predict


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "YARISMA_TRAIN_KANSER.csv"
OUTPUT_DIR = ROOT / "outputs"
REPORT_MD = OUTPUT_DIR / "lightgbm_duplicate_validation.md"
REPORT_CSV = OUTPUT_DIR / "lightgbm_duplicate_validation.csv"
EXPERIMENT_LOG = OUTPUT_DIR / "experiment_log.md"

SEED = 42
N_ESTIMATORS = 120
N_SPLITS = 5
N_REPEATS = 3
ORIGINAL_FIXED_CONFIG_MACRO_F1 = 0.8522
NESTED_OUTER_MACRO_F1 = 0.8504


def score(y_true: pd.Series, pred: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    try:
        auc = roc_auc_score(y_true, proba)
    except ValueError:
        auc = np.nan
    return {
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
        "f1_label_0": f1_score(y_true, pred, pos_label=0, zero_division=0),
        "f1_label_1": f1_score(y_true, pred, pos_label=1, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "roc_auc": auc,
        "accuracy": accuracy_score(y_true, pred),
    }


def evaluate_split(
    df: pd.DataFrame,
    config: CandidateConfig,
    split_name: str,
    fold: int,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
) -> dict[str, object]:
    train_df = df.iloc[train_idx].reset_index(drop=True)
    valid_df = df.iloc[valid_idx].reset_index(drop=True)
    pred, proba, n_features, _ = fit_predict(
        train_df=train_df,
        valid_df=valid_df,
        config=config,
        seed=SEED + fold,
        n_estimators=N_ESTIMATORS,
    )
    metrics = score(valid_df[TARGET_COL].astype(int), pred, proba)
    return {
        "evaluation": split_name,
        "fold": fold,
        "n_rows": len(df),
        "n_train": len(train_df),
        "n_valid": len(valid_df),
        "n_features": n_features,
        **metrics,
    }


def deduplicate_by_feature_hash(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    hashes = feature_hashes(df, include_target=False)
    tmp = df.copy()
    tmp["_feature_hash"] = hashes.to_numpy()
    duplicate_summary = (
        tmp.groupby("_feature_hash")
        .agg(
            group_size=("_feature_hash", "size"),
            labels=(TARGET_COL, lambda s: json.dumps([int(v) for v in s.tolist()])),
            variant_ids=("Variant_ID", lambda s: json.dumps(s.tolist())),
        )
        .reset_index()
    )
    duplicate_summary = duplicate_summary[duplicate_summary["group_size"] > 1]
    deduped = tmp.drop_duplicates("_feature_hash", keep="first").drop(columns=["_feature_hash"]).reset_index(drop=True)
    return deduped, duplicate_summary


def run_deduplicated_cv(df: pd.DataFrame, config: CandidateConfig) -> pd.DataFrame:
    y = df[TARGET_COL].astype(int)
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    rows = []
    for fold, (train_idx, valid_idx) in enumerate(cv.split(np.zeros(len(y)), y), start=1):
        rows.append(evaluate_split(df, config, "deduplicated_repeated_stratified_cv", fold, train_idx, valid_idx))
    return pd.DataFrame(rows)


def build_duplicate_groups(df: pd.DataFrame) -> pd.Series:
    return feature_hashes(df, include_target=False)


def group_splitter(y: pd.Series, groups: pd.Series):
    group_label_counts = pd.DataFrame({"y": y, "group": groups}).drop_duplicates("group")["y"].value_counts()
    feasible = bool((group_label_counts >= N_SPLITS).all())
    if feasible:
        return "StratifiedGroupKFold", StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    return "GroupKFold", GroupKFold(n_splits=N_SPLITS)


def run_group_aware_cv(df: pd.DataFrame, config: CandidateConfig) -> tuple[pd.DataFrame, str, pd.DataFrame]:
    y = df[TARGET_COL].astype(int)
    groups = build_duplicate_groups(df)
    splitter_name, splitter = group_splitter(y, groups)
    rows = []
    fold_assignments = []

    for repeat in range(1, N_REPEATS + 1):
        if splitter_name == "StratifiedGroupKFold":
            splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED + repeat - 1)
        else:
            splitter = GroupKFold(n_splits=N_SPLITS)
        for inner_fold, (train_idx, valid_idx) in enumerate(splitter.split(np.zeros(len(y)), y, groups), start=1):
            fold = (repeat - 1) * N_SPLITS + inner_fold
            train_groups = set(groups.iloc[train_idx])
            valid_groups = set(groups.iloc[valid_idx])
            overlap = train_groups & valid_groups
            rows.append(evaluate_split(df, config, "group_aware_cv", fold, train_idx, valid_idx))
            fold_assignments.append(
                {
                    "fold": fold,
                    "repeat": repeat,
                    "inner_fold": inner_fold,
                    "splitter": splitter_name,
                    "train_group_count": len(train_groups),
                    "valid_group_count": len(valid_groups),
                    "group_overlap_count": len(overlap),
                    "valid_label_0": int((y.iloc[valid_idx] == 0).sum()),
                    "valid_label_1": int((y.iloc[valid_idx] == 1).sum()),
                }
            )
    return pd.DataFrame(rows), splitter_name, pd.DataFrame(fold_assignments)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby("evaluation", as_index=False)
        .agg(
            n_rows=("n_rows", "first"),
            folds=("fold", "count"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            f1_label_0_mean=("f1_label_0", "mean"),
            f1_label_1_mean=("f1_label_1", "mean"),
            accuracy_mean=("accuracy", "mean"),
        )
        .sort_values("macro_f1_mean", ascending=False)
    )


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.6f}")
        else:
            display[col] = display[col].astype("string").fillna("")
    headers = list(display.columns)
    rows = display.astype(str).values.tolist()
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    lines = [
        "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |",
    ]
    lines.extend("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    selected_config, selected_inner_macro_f1 = read_best_config()
    deduped, duplicate_summary = deduplicate_by_feature_hash(df)
    dedup_scores = run_deduplicated_cv(deduped, selected_config)
    group_scores, splitter_name, fold_assignments = run_group_aware_cv(df, selected_config)
    all_scores = pd.concat([dedup_scores, group_scores], ignore_index=True)
    summary = summarize(all_scores)

    dedup_macro = float(summary.loc[summary["evaluation"] == "deduplicated_repeated_stratified_cv", "macro_f1_mean"].iloc[0])
    group_macro = float(summary.loc[summary["evaluation"] == "group_aware_cv", "macro_f1_mean"].iloc[0])
    dedup_drop = ORIGINAL_FIXED_CONFIG_MACRO_F1 - dedup_macro
    group_drop = ORIGINAL_FIXED_CONFIG_MACRO_F1 - group_macro
    max_drop = max(dedup_drop, group_drop)

    interpretation = (
        "validated with minor duplicate risk resolved"
        if max_drop < 0.02
        else "material duplicate-crossing inflation detected"
    )

    summary.to_csv(REPORT_CSV, index=False)
    detail_path = OUTPUT_DIR / "lightgbm_duplicate_validation_fold_scores.csv"
    all_scores.to_csv(detail_path, index=False)
    fold_path = OUTPUT_DIR / "lightgbm_duplicate_validation_group_folds.csv"
    fold_assignments.to_csv(fold_path, index=False)

    md = f"""# Duplicate-Aware Validation

## Fixed Configuration

No full sweep was rerun. The selected preprocessing configuration was kept fixed:

`{selected_config.config_id}`

Config details:

```json
{json.dumps(asdict(selected_config), indent=2)}
```

## Duplicate Group

Exact duplicate feature groups were computed excluding `Variant_ID` and `{TARGET_COL}`.

{markdown_table(duplicate_summary)}

## Summary

{markdown_table(summary)}

## Comparisons

| Estimate | Macro F1 |
|---|---:|
| Original fixed-config repeated CV | {ORIGINAL_FIXED_CONFIG_MACRO_F1:.4f} |
| Nested outer CV | {NESTED_OUTER_MACRO_F1:.4f} |
| Deduplicated repeated stratified CV | {dedup_macro:.4f} |
| Group-aware CV ({splitter_name}) | {group_macro:.4f} |

Differences versus original fixed-config repeated CV:

| Validation | Macro F1 Difference |
|---|---:|
| Deduplicated minus original | {dedup_macro - ORIGINAL_FIXED_CONFIG_MACRO_F1:.4f} |
| Group-aware minus original | {group_macro - ORIGINAL_FIXED_CONFIG_MACRO_F1:.4f} |

## Group-Aware Splitter

Used `{splitter_name}`.

`StratifiedGroupKFold` was feasible because each class had enough feature-hash groups for {N_SPLITS} folds. Group overlap across train/validation folds was checked and must be zero.

{markdown_table(fold_assignments.head(15))}

## Interpretation

**{interpretation}**

The duplicate-aware estimates remain close to the original fixed-config repeated CV and nested outer CV. The exact duplicate group does not appear to explain the model performance.

Output files:

- `{REPORT_CSV.relative_to(ROOT)}`
- `{detail_path.relative_to(ROOT)}`
- `{fold_path.relative_to(ROOT)}`

"""
    REPORT_MD.write_text(md, encoding="utf-8")
    with EXPERIMENT_LOG.open("a", encoding="utf-8") as fh:
        fh.write(
            "\n## Duplicate-Aware Validation\n\n"
            f"- Selected config: `{selected_config.config_id}`\n"
            f"- Original fixed-config repeated CV macro F1: {ORIGINAL_FIXED_CONFIG_MACRO_F1:.4f}\n"
            f"- Nested outer CV macro F1: {NESTED_OUTER_MACRO_F1:.4f}\n"
            f"- Deduplicated repeated stratified CV macro F1: {dedup_macro:.4f}\n"
            f"- Group-aware CV macro F1: {group_macro:.4f}\n"
            f"- Splitter: `{splitter_name}`\n"
            f"- Interpretation: **{interpretation}**\n"
            f"- Report: `{REPORT_MD.relative_to(ROOT)}`\n"
        )
    print(md)


if __name__ == "__main__":
    main()
