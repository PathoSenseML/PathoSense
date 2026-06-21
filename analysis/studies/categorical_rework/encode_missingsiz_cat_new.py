"""Create static one-hot encoded CAT_new datasets for the missingsiz source.

This script does not train models and does not write experiment logs. It only
creates encoded CSV copies of the missingsiz datasets produced by
build_new_cat_datasets.py.

Encoding policy:
  * CAT_new missing/blank values are not imputed as a country/region.
  * CAT_new_is_missing is 1 for missing/blank CAT_new, else 0.
  * Region one-hot columns are 0 for missing/blank CAT_new rows.
  * The original CAT_new column is removed from the encoded output.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "outputs" / "new_cat_datasets" / "missingsiz"
OUT_DIR = SOURCE_DIR / "encoded_cat_new"
CAT_COL = "CAT_new"
MISSING_COL = "CAT_new_is_missing"

DATASET_KINDS = ("new_cat_only", "all_cats_kept")
PANELS = ("MASTER", "KANSER")


def _sanitize_category(value: str) -> str:
    text = str(value).strip().upper()
    text = re.sub(r"[^A-Z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "EMPTY"


def _load_inputs(kind: str) -> dict[str, pd.DataFrame]:
    frames = {}
    for panel in PANELS:
        path = SOURCE_DIR / kind / f"YARISMA_TRAIN_{panel}_new_cat_missingsiz_{kind}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        if CAT_COL not in df.columns:
            raise ValueError(f"{path} is missing {CAT_COL}")
        frames[panel] = df
    return frames


def _category_map(frames: dict[str, pd.DataFrame]) -> dict[str, str]:
    values: set[str] = set()
    for df in frames.values():
        s = df[CAT_COL].astype("string")
        s = s[s.notna() & s.str.strip().ne("")]
        values.update(str(v).strip() for v in s)

    mapping: dict[str, str] = {}
    used: set[str] = set()
    for value in sorted(values):
        base = _sanitize_category(value)
        name = f"{CAT_COL}_{base}"
        if name in used:
            suffix = 2
            while f"{name}_{suffix}" in used:
                suffix += 1
            name = f"{name}_{suffix}"
        mapping[value] = name
        used.add(name)
    return mapping


def _encode(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    cat = out[CAT_COL].astype("string")
    missing = cat.isna() | cat.str.strip().eq("")
    insert_at = out.columns.get_loc(CAT_COL)

    encoded = pd.DataFrame(index=out.index)
    encoded[MISSING_COL] = missing.astype(int)
    normalized = cat.fillna("").str.strip()
    for raw_value, col in mapping.items():
        encoded[col] = normalized.eq(raw_value).astype(int)

    out = out.drop(columns=[CAT_COL])
    for offset, col in enumerate(encoded.columns):
        out.insert(insert_at + offset, col, encoded[col].to_numpy())
    return out


def _summary_for(panel: str, kind: str, input_df: pd.DataFrame, output_df: pd.DataFrame, output_path: Path, mapping: dict[str, str]) -> dict[str, Any]:
    return {
        "panel": panel,
        "dataset_kind": kind,
        "output_path": str(output_path),
        "input_rows": int(input_df.shape[0]),
        "input_columns": int(input_df.shape[1]),
        "output_rows": int(output_df.shape[0]),
        "output_columns": int(output_df.shape[1]),
        "cat_new_missing_rows": int(output_df[MISSING_COL].sum()),
        "cat_new_onehot_columns": list(mapping.values()),
        "cat_new_original_removed": CAT_COL not in output_df.columns,
        "old_cat_columns_still_present": [c for c in output_df.columns if re.match(r"^CAT_\d+$", c)],
    }


def build() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "source": "missingsiz",
        "encoding_policy": {
            "cat_new_missing": "CAT_new_is_missing indicator; no region one-hot is activated for missing rows",
            "cat_new_non_missing": "one-hot encoded from observed CAT_new categories",
            "training_or_experiment_run": False,
        },
        "datasets": {},
    }

    for kind in DATASET_KINDS:
        frames = _load_inputs(kind)
        mapping = _category_map(frames)
        kind_dir = OUT_DIR / kind
        kind_dir.mkdir(parents=True, exist_ok=True)
        summary["datasets"][kind] = {
            "category_to_column": mapping,
            "panels": {},
        }
        for panel, df in frames.items():
            encoded = _encode(df, mapping)
            output_path = kind_dir / f"YARISMA_TRAIN_{panel}_new_cat_missingsiz_{kind}_catnew_onehot.csv"
            encoded.to_csv(output_path, index=False)
            summary["datasets"][kind]["panels"][panel] = _summary_for(panel, kind, df, encoded, output_path, mapping)
            print(f"[{kind}][{panel}] saved {output_path} ({encoded.shape[0]} rows, {encoded.shape[1]} cols)")

    summary_path = OUT_DIR / "cat_new_onehot_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {summary_path}")
    return summary


if __name__ == "__main__":
    build()
