"""Build raw panel datasets with the new CAT_new assignments.

Stage 1 only:
  * read raw MASTER/KANSER panel CSVs from data/
  * read the two new all-panel CAT_new CSVs from data/
  * merge by panel + Variant_ID while preserving raw panel row order
  * write separate missingli_var and missingsiz dataset variants

No model training, preprocessing, thresholding, or experiment logging happens here.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs" / "new_cat_datasets"

PANEL_FILES = {
    "MASTER": "YARISMA_TRAIN_MASTER.csv",
    "KANSER": "YARISMA_TRAIN_KANSER.csv",
}

CAT_SOURCES = {
    "missingli_var": "ALL_PANELS_CAT_new_*VAR.csv",
    "missingsiz": "ALL_PANELS_CAT_new_*Z.csv",
}

TARGET_COL = "Label"
ID_COL = "Variant_ID"
NEW_CAT_COL = "CAT_new"
OLD_CAT_PATTERN = re.compile(r"^CAT_\d+$")


def _json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _find_one(pattern: str) -> Path:
    matches = sorted(DATA_DIR.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one match for {pattern!r}, found {len(matches)}: {matches}")
    return matches[0]


def _panel_column(df: pd.DataFrame) -> str:
    matches = [c for c in df.columns if c.lower().startswith("panel")]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one panel column, found {matches}")
    return matches[0]


def _normalize_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def _normalize_panel(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def _load_cat_source(path: Path) -> pd.DataFrame:
    cat = _read_csv(path)
    panel_col = _panel_column(cat)
    missing = {ID_COL, NEW_CAT_COL} - set(cat.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    cat = cat[[panel_col, ID_COL, NEW_CAT_COL]].copy()
    cat = cat.rename(columns={panel_col: "panel"})
    cat["panel"] = _normalize_panel(cat["panel"])
    cat[ID_COL] = _normalize_id(cat[ID_COL])

    dup = cat.duplicated(["panel", ID_COL], keep=False)
    if dup.any():
        examples = cat.loc[dup, ["panel", ID_COL]].head(10).to_dict("records")
        raise ValueError(f"{path} has duplicate panel+Variant_ID keys. Examples: {examples}")

    return cat


def _merge_panel(raw_path: Path, cat: pd.DataFrame, panel: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = _read_csv(raw_path)
    if ID_COL not in raw.columns:
        raise ValueError(f"{raw_path} is missing {ID_COL}")
    if TARGET_COL not in raw.columns:
        raise ValueError(f"{raw_path} is missing {TARGET_COL}")

    raw = raw.copy()
    raw["_raw_order"] = range(len(raw))
    raw["_merge_variant_id"] = _normalize_id(raw[ID_COL])

    raw_dup = raw.duplicated("_merge_variant_id", keep=False)
    if raw_dup.any():
        examples = raw.loc[raw_dup, [ID_COL]].head(10).to_dict("records")
        raise ValueError(f"{raw_path} has duplicate Variant_ID values. Examples: {examples}")

    cat_panel = cat[cat["panel"].eq(panel)].copy()
    cat_panel["_cat_key_present"] = True
    merged = raw.merge(
        cat_panel[[ID_COL, NEW_CAT_COL, "_cat_key_present"]].rename(columns={ID_COL: "_merge_variant_id"}),
        on="_merge_variant_id",
        how="left",
        validate="one_to_one",
    )
    merged = merged.sort_values("_raw_order").reset_index(drop=True)

    matched = int(merged["_cat_key_present"].eq(True).sum())
    unmatched = int(merged["_cat_key_present"].isna().sum())
    raw_id_set = set(raw["_merge_variant_id"])
    cat_id_set = set(cat_panel[ID_COL])
    extra_cat_ids = sorted(cat_id_set - raw_id_set)
    missing_raw_ids = sorted(raw_id_set - cat_id_set)

    # Put CAT_new next to the existing categorical features, just before Label.
    helper_cols = {"_raw_order", "_merge_variant_id", "_cat_key_present"}
    base_cols = [c for c in raw.columns if c not in helper_cols]
    cols_without_label = [c for c in base_cols if c != TARGET_COL]
    ordered_cols = cols_without_label + [NEW_CAT_COL, TARGET_COL]
    merged = merged[ordered_cols]

    label_counts = merged[TARGET_COL].astype(str).value_counts(dropna=False).sort_index().to_dict()
    cat_counts = merged[NEW_CAT_COL].astype("string").fillna("__MISSING__").value_counts(dropna=False).sort_index().to_dict()
    metadata = {
        "panel": panel,
        "raw_path": raw_path,
        "raw_rows": int(len(raw)),
        "raw_columns": int(len(base_cols)),
        "output_rows": int(len(merged)),
        "output_columns": int(merged.shape[1]),
        "matched_rows": matched,
        "unmatched_rows": unmatched,
        "extra_cat_ids_not_in_raw_count": int(len(extra_cat_ids)),
        "missing_raw_ids_not_in_cat_count": int(len(missing_raw_ids)),
        "extra_cat_ids_not_in_raw_examples": extra_cat_ids[:10],
        "missing_raw_ids_not_in_cat_examples": missing_raw_ids[:10],
        "label_counts": label_counts,
        "cat_new_missing_rows": int(merged[NEW_CAT_COL].isna().sum()),
        "cat_new_value_counts": cat_counts,
        "row_order_preserved": True,
        "merge_key": ["panel", ID_COL],
    }
    return merged, metadata


def _drop_old_cat_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    old_cat_cols = [c for c in df.columns if OLD_CAT_PATTERN.match(c)]
    out = df.drop(columns=old_cat_cols)
    return out, old_cat_cols


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# New CAT Dataset Build Report",
        "",
        "Stage 1 only: raw panel rows were preserved and `CAT_new` was merged by `panel + Variant_ID`.",
        "No preprocessing, model training, threshold selection, or experiment logging was run.",
        "",
    ]
    for source_name, source_info in summary["sources"].items():
        lines.extend([
            f"## {source_name}",
            "",
            f"- CAT source: `{source_info['cat_path']}`",
            f"- CAT rows: `{source_info['cat_rows']}`",
            f"- CAT panel counts: `{source_info['cat_panel_counts']}`",
            "",
        ])
        for panel, panel_info in source_info["panels"].items():
            lines.extend([
                f"### {panel}",
                "",
                f"- All CATs kept output: `{panel_info['all_cats_kept_output_path']}`",
                f"- New CAT only output: `{panel_info['new_cat_only_output_path']}`",
                f"- Rows: `{panel_info['output_rows']}`",
                f"- All CATs kept columns: `{panel_info['output_columns']}`",
                f"- New CAT only columns: `{panel_info['new_cat_only_columns']}`",
                f"- Old CAT columns dropped in new-cat-only: `{panel_info['old_cat_columns_dropped_for_new_cat_only']}`",
                f"- Matched rows: `{panel_info['matched_rows']}`",
                f"- Unmatched rows: `{panel_info['unmatched_rows']}`",
                f"- CAT_new missing rows: `{panel_info['cat_new_missing_rows']}`",
                f"- Label counts: `{panel_info['label_counts']}`",
                "",
            ])
    path.write_text("\n".join(lines), encoding="utf-8")


def build() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "data_dir": DATA_DIR,
        "output_dir": OUT_DIR,
        "method": "left merge raw panels with CAT_new by panel + Variant_ID; preserve raw row order",
        "sources": {},
    }

    for source_name, pattern in CAT_SOURCES.items():
        cat_path = _find_one(pattern)
        cat = _load_cat_source(cat_path)
        source_dir = OUT_DIR / source_name
        source_dir.mkdir(parents=True, exist_ok=True)
        source_info: dict[str, Any] = {
            "cat_path": cat_path,
            "cat_rows": int(len(cat)),
            "cat_panel_counts": cat["panel"].value_counts().sort_index().to_dict(),
            "panels": {},
        }

        for panel, raw_name in PANEL_FILES.items():
            raw_path = DATA_DIR / raw_name
            merged, metadata = _merge_panel(raw_path, cat, panel)

            all_kept_dir = source_dir / "all_cats_kept"
            new_only_dir = source_dir / "new_cat_only"
            all_kept_dir.mkdir(parents=True, exist_ok=True)
            new_only_dir.mkdir(parents=True, exist_ok=True)

            all_kept_name = raw_path.stem + f"_new_cat_{source_name}_all_cats_kept.csv"
            all_kept_path = all_kept_dir / all_kept_name
            merged.to_csv(all_kept_path, index=False)

            new_only, dropped_old_cat_cols = _drop_old_cat_columns(merged)
            new_only_name = raw_path.stem + f"_new_cat_{source_name}_new_cat_only.csv"
            new_only_path = new_only_dir / new_only_name
            new_only.to_csv(new_only_path, index=False)

            metadata["all_cats_kept_output_path"] = all_kept_path
            metadata["new_cat_only_output_path"] = new_only_path
            metadata["new_cat_only_columns"] = int(new_only.shape[1])
            metadata["old_cat_columns_dropped_for_new_cat_only"] = dropped_old_cat_cols
            source_info["panels"][panel] = metadata
            print(f"[{source_name}][{panel}] saved all-cats-kept {all_kept_path} ({merged.shape[0]} rows, {merged.shape[1]} cols)")
            print(f"[{source_name}][{panel}] saved new-cat-only {new_only_path} ({new_only.shape[0]} rows, {new_only.shape[1]} cols)")

        summary["sources"][source_name] = source_info

    summary_path = OUT_DIR / "new_cat_dataset_build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_safe), encoding="utf-8")
    _write_markdown(summary, OUT_DIR / "new_cat_dataset_build_report.md")
    print(f"Saved {summary_path}")
    print(f"Saved {OUT_DIR / 'new_cat_dataset_build_report.md'}")
    return summary


if __name__ == "__main__":
    build()
