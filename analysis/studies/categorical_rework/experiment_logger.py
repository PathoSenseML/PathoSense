"""Central experiment/result logging utilities.

The logger appends one row per experiment to ``outputs/experiment_results.csv``
and keeps a Markdown copy beside it. It is intentionally dependency-free and
safe to import from lightweight preprocessing scripts.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
DEFAULT_CSV = OUTPUT_DIR / "experiment_results.csv"
DEFAULT_MD = OUTPUT_DIR / "experiment_results.md"
DEFAULT_DECISION_LOG = OUTPUT_DIR / "decision_log.md"


EXPERIMENT_COLUMNS = [
    "timestamp",
    "panel",
    "experiment_name",
    "pipeline_role",
    "preprocess_mode",
    "row_drop_enabled",
    "pca_enabled",
    "pca_variant",
    "scaling_enabled",
    "outlier_control_enabled",
    "imputation_method",
    "encoding_method",
    "feature_selection_method",
    "model_name",
    "calibration_method",
    "threshold_strategy",
    "selected_threshold",
    "blend_enabled",
    "blend_weight",
    "general_transfer_weight",
    "cancer_own_weight",
    "cv_strategy",
    "n_folds",
    "input_rows",
    "output_rows",
    "dropped_rows",
    "input_features",
    "output_features",
    "dropped_features",
    "positive_ratio_before",
    "positive_ratio_after",
    "auc",
    "f1_star",
    "finals_adjusted_f1",
    "precision",
    "recall",
    "pred_positive_mean",
    "delta_f1_vs_baseline",
    "notes",
    "output_file",
]


def log_experiment_result(result_dict: dict, csv_path: str | Path = DEFAULT_CSV) -> pd.DataFrame:
    """Append one experiment row and regenerate the Markdown table.

    Missing fields are written as ``n/a``. If a later result introduces new
    fields, the CSV is expanded without losing existing columns.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path = csv_path.with_suffix(".md")

    row = dict(result_dict)
    row.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))

    if csv_path.exists():
        existing = pd.read_csv(csv_path, dtype=str).fillna("n/a")
    else:
        existing = pd.DataFrame(columns=EXPERIMENT_COLUMNS)

    columns = list(EXPERIMENT_COLUMNS)
    for col in existing.columns:
        if col not in columns:
            columns.append(col)
    for col in row:
        if col not in columns:
            columns.append(col)

    clean_row = {col: _stringify(row.get(col, "n/a")) for col in columns}
    existing = existing.reindex(columns=columns, fill_value="n/a")
    updated = pd.concat([existing, pd.DataFrame([clean_row], columns=columns)], ignore_index=True)
    updated.to_csv(csv_path, index=False)
    md_path.write_text(_markdown_table(updated), encoding="utf-8")
    return updated


def write_decision_log(
    decisions: Iterable[str],
    path: str | Path = DEFAULT_DECISION_LOG,
    *,
    title: str = "Decision Log",
    append: bool = False,
) -> Path:
    """Write or append a readable decision log."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().isoformat(timespec="seconds")
    lines = [f"# {title}", "", f"Updated: {stamp}", ""]
    for decision in decisions:
        lines.append(f"- {decision}")
    lines.append("")
    text = "\n".join(lines)
    if append and path.exists():
        path.write_text(path.read_text(encoding="utf-8") + "\n" + text, encoding="utf-8")
    else:
        path.write_text(text, encoding="utf-8")
    return path


def _stringify(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple, set)):
        return ";".join(_stringify(v) for v in value)
    if isinstance(value, dict):
        return "; ".join(f"{k}={_stringify(v)}" for k, v in value.items())
    text = str(value)
    return text if text else "n/a"


def _markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    rows = [[str(v) for v in row] for row in df.astype(object).itertuples(index=False, name=None)]
    widths = [
        max(len(str(col)), *(len(row[i]) for row in rows)) if rows else len(str(col))
        for i, col in enumerate(cols)
    ]

    def fmt(values):
        return "| " + " | ".join(str(v).replace("\n", " ").ljust(widths[i]) for i, v in enumerate(values)) + " |"

    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    return "\n".join([fmt(cols), sep] + [fmt(row) for row in rows])
