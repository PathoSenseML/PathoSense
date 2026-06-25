from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from lightgbm_preprocess_report import CandidateConfig, TARGET_COL, fit_predict


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "YARISMA_TRAIN_KANSER.csv"
OUTPUT_DIR = ROOT / "outputs"
OUT_CSV = OUTPUT_DIR / "lightgbm_historical_f1star_alignment.csv"
OUT_MD = OUTPUT_DIR / "lightgbm_historical_f1star_alignment.md"
OOF_CSV = OUTPUT_DIR / "lightgbm_historical_f1star_oof_predictions.csv"
EXPERIMENT_LOG = OUTPUT_DIR / "experiment_log.md"

SEED = 42
N_ESTIMATORS = 120
OUTER_SPLITS = 5
OUTER_REPEATS = 3
INNER_SPLITS = 5
CANCER_TEST_PRIOR = 100 / 600
FINALS_N = 600

SELECTED_CONFIG = CandidateConfig(
    config_id="cat_none__imp_zero__miss_0__blosum_1__num_none",
    categorical_mode="none",
    impute_strategy="zero",
    missing_indicators=False,
    use_blosum=True,
    numeric_transform="none",
)


def tpr_fpr(y_true: np.ndarray, preds: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(preds).astype(int)
    pos = y == 1
    neg = y == 0
    tpr = float(p[pos].mean()) if pos.any() else 0.0
    fpr = float(p[neg].mean()) if neg.any() else 0.0
    return tpr, fpr


def f1_at_prior(tpr: float, fpr: float, prior: float) -> float:
    tp = prior * tpr
    fp = (1 - prior) * fpr
    fn = prior * (1 - tpr)
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else 0.0


def pick_threshold_at_prior(
    y_train: np.ndarray,
    oof_probs: np.ndarray,
    prior: float,
    step: float = 0.005,
) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.02, 0.98, step):
        tpr, fpr = tpr_fpr(y_train, (oof_probs >= t).astype(int))
        f1 = f1_at_prior(tpr, fpr, prior)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def bootstrap_f1(
    y_true: np.ndarray,
    preds: np.ndarray,
    prior: float,
    n_target: int,
    n_boot: int = 300,
    seed: int = SEED,
) -> float:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(preds).astype(int)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    n_pos = max(1, int(round(n_target * prior)))
    n_neg = max(1, n_target - n_pos)
    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n_boot):
        idx = np.concatenate(
            [
                rng.choice(pos, n_pos, replace=True),
                rng.choice(neg, n_neg, replace=True),
            ]
        )
        scores.append(f1_score(y[idx], p[idx], zero_division=0))
    return float(np.mean(scores))


def train_oof_probs(train_df: pd.DataFrame, outer_fold: int) -> np.ndarray:
    y = train_df[TARGET_COL].astype(int).to_numpy()
    n_splits = min(INNER_SPLITS, int(y.sum()), int((y == 0).sum()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED + outer_fold)
    oof = np.zeros(len(train_df), dtype=float)
    for inner_fold, (inner_train_idx, inner_valid_idx) in enumerate(cv.split(np.zeros(len(y)), y), start=1):
        inner_train = train_df.iloc[inner_train_idx].reset_index(drop=True)
        inner_valid = train_df.iloc[inner_valid_idx].reset_index(drop=True)
        _, proba, _, _ = fit_predict(
            inner_train,
            inner_valid,
            SELECTED_CONFIG,
            seed=SEED + outer_fold * 1000 + inner_fold,
            n_estimators=N_ESTIMATORS,
        )
        oof[inner_valid_idx] = proba
    return oof


def run_alignment() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(DATA_PATH)
    y = df[TARGET_COL].astype(int)
    cv = RepeatedStratifiedKFold(n_splits=OUTER_SPLITS, n_repeats=OUTER_REPEATS, random_state=SEED)
    rows = []
    oof_rows = []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(np.zeros(len(y)), y), start=1):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        valid_df = df.iloc[valid_idx].reset_index(drop=True)
        threshold_oof = train_oof_probs(train_df, fold)
        threshold = pick_threshold_at_prior(
            train_df[TARGET_COL].astype(int).to_numpy(),
            threshold_oof,
            CANCER_TEST_PRIOR,
        )

        _, proba, n_features, _ = fit_predict(
            train_df,
            valid_df,
            SELECTED_CONFIG,
            seed=SEED + fold,
            n_estimators=N_ESTIMATORS,
        )
        pred = (proba >= threshold).astype(int)
        y_valid = valid_df[TARGET_COL].astype(int).to_numpy()
        tpr, fpr = tpr_fpr(y_valid, pred)

        rows.append(
            {
                "fold": fold,
                "n_train": len(train_df),
                "n_valid": len(valid_df),
                "n_features": n_features,
                "selected_threshold": threshold,
                "macro_f1": f1_score(y_valid, pred, average="macro", zero_division=0),
                "label_0_f1": f1_score(y_valid, pred, pos_label=0, zero_division=0),
                "label_1_f1": f1_score(y_valid, pred, pos_label=1, zero_division=0),
                "historical_f1_star": f1_at_prior(tpr, fpr, CANCER_TEST_PRIOR),
                "finals_adjusted_f1": bootstrap_f1(y_valid, pred, CANCER_TEST_PRIOR, FINALS_N),
                "precision": precision_score(y_valid, pred, zero_division=0),
                "recall": recall_score(y_valid, pred, zero_division=0),
                "predicted_positive_rate": float(pred.mean()),
                "roc_auc": roc_auc_score(y_valid, proba),
                "tpr": tpr,
                "fpr": fpr,
            }
        )

        for local_i, original_i in enumerate(valid_idx):
            oof_rows.append(
                {
                    "fold": fold,
                    "row_index": int(original_i),
                    "variant_id": valid_df.iloc[local_i].get("Variant_ID"),
                    "y_true": int(y_valid[local_i]),
                    "probability": float(proba[local_i]),
                    "prediction": int(pred[local_i]),
                    "selected_threshold": threshold,
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(oof_rows)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "macro_f1",
        "label_0_f1",
        "label_1_f1",
        "historical_f1_star",
        "finals_adjusted_f1",
        "selected_threshold",
        "precision",
        "recall",
        "predicted_positive_rate",
        "roc_auc",
    ]
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "mean": rows[metric].mean(),
                "std": rows[metric].std(),
                "min": rows[metric].min(),
                "max": rows[metric].max(),
            }
            for metric in metrics
        ]
    )


def markdown_table(frame: pd.DataFrame) -> str:
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
    fold_rows, oof_rows = run_alignment()
    summary = summarize(fold_rows)
    fold_rows.to_csv(OUT_CSV, index=False)
    oof_rows.to_csv(OOF_CSV, index=False)

    summary_lookup = dict(zip(summary["metric"], summary["mean"]))
    md = f"""# Historical F1* Metric Alignment

Fixed selected LightGBM preprocessing config:

```json
{json.dumps(asdict(SELECTED_CONFIG), indent=2)}
```

No preprocessing sweep or model search was run. Metrics were computed on repeated 5x3 OOF held-out predictions for the fixed selected config.

Historical cancer-style settings copied from the previous harness:

- cancer test prior: `{CANCER_TEST_PRIOR:.6f}` (`100 / 600`)
- finals size: `{FINALS_N}`
- threshold grid: `np.arange(0.02, 0.98, 0.005)`
- threshold selected per outer fold on train-only inner OOF probabilities
- F1*: analytic `f1_at_prior(tpr, fpr, prior)`
- finals-adjusted F1: bootstrap to finals size with the target prior

## Summary

{markdown_table(summary)}

## Per-Fold Metrics

{markdown_table(fold_rows)}

## Compact Results

- macro F1: `{summary_lookup["macro_f1"]:.6f}`
- label 0 F1: `{summary_lookup["label_0_f1"]:.6f}`
- label 1 F1: `{summary_lookup["label_1_f1"]:.6f}`
- historical-style F1*: `{summary_lookup["historical_f1_star"]:.6f}`
- finals-adjusted F1: `{summary_lookup["finals_adjusted_f1"]:.6f}`
- selected threshold mean: `{summary_lookup["selected_threshold"]:.6f}`
- precision: `{summary_lookup["precision"]:.6f}`
- recall: `{summary_lookup["recall"]:.6f}`
- predicted positive rate: `{summary_lookup["predicted_positive_rate"]:.6f}`

Output files:

- `{OUT_CSV.relative_to(ROOT)}`
- `{OOF_CSV.relative_to(ROOT)}`

"""
    OUT_MD.write_text(md, encoding="utf-8")
    with EXPERIMENT_LOG.open("a", encoding="utf-8") as fh:
        fh.write(
            "\n## Historical F1* Metric Alignment\n\n"
            f"- Selected config: `{SELECTED_CONFIG.config_id}`\n"
            f"- macro F1: {summary_lookup['macro_f1']:.6f}\n"
            f"- label 0 F1: {summary_lookup['label_0_f1']:.6f}\n"
            f"- label 1 F1: {summary_lookup['label_1_f1']:.6f}\n"
            f"- historical F1*: {summary_lookup['historical_f1_star']:.6f}\n"
            f"- finals-adjusted F1: {summary_lookup['finals_adjusted_f1']:.6f}\n"
            f"- selected threshold mean: {summary_lookup['selected_threshold']:.6f}\n"
            f"- precision: {summary_lookup['precision']:.6f}\n"
            f"- recall: {summary_lookup['recall']:.6f}\n"
            f"- predicted positive rate: {summary_lookup['predicted_positive_rate']:.6f}\n"
            f"- Report: `{OUT_MD.relative_to(ROOT)}`\n"
        )
    print(md)


if __name__ == "__main__":
    main()
