"""Updated categorical preprocessing for MASTER and CANCER panels.

This module is intentionally small and reusable:
  * split the new combined CAT1/CAT2 decision file by panel
  * optionally merge those updated categorical features onto the existing panel CSVs
  * fit/transform a leakage-safe numeric preprocessing object

For CV/model experiments, instantiate UpdatedCategoricalPreprocessor inside each
train fold and call fit on train only. The CLI export fits on the full panel only
to create preprocessing-only artifacts for inspection and downstream wiring.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from experiment_logger import log_experiment_result


ROOT = Path(__file__).resolve().parent
COMBINED_CAT_FILE = ROOT / "ALL_PANELS_row_by_row_CAT12_selection.csv"
PROJECT_ROOT = ROOT.parent / "denemeler" / "feature_selection"
PROJECT_PANEL_DIR = PROJECT_ROOT / "data" / "panels"
OUTPUT_DIR = ROOT / "outputs"

PANEL_MAP = {
    "MASTER": {
        "project_panel": "general",
        "source": PROJECT_PANEL_DIR / "YARISMA_TRAIN_MASTER.csv",
        "raw_out": "MASTER_categorical_raw.csv",
        "updated_out": "YARISMA_TRAIN_MASTER_updated_cat.csv",
        "processed_out": "MASTER_processed_numeric_strict.csv",
        "metadata_out": "MASTER_preprocessing_metadata_strict.json",
        "processed_no_rowdrop_out": "MASTER_processed_numeric_no_rowdrop.csv",
        "metadata_no_rowdrop_out": "MASTER_preprocessing_metadata_no_rowdrop.json",
    },
    "KANSER": {
        "project_panel": "cancer",
        "source": PROJECT_PANEL_DIR / "YARISMA_TRAIN_KANSER.csv",
        "raw_out": "CANCER_categorical_raw.csv",
        "updated_out": "YARISMA_TRAIN_CANCER_updated_cat.csv",
        "processed_out": "CANCER_processed_numeric_strict.csv",
        "metadata_out": "CANCER_preprocessing_metadata_strict.json",
        "processed_no_rowdrop_out": "CANCER_processed_numeric_no_rowdrop.csv",
        "metadata_no_rowdrop_out": "CANCER_preprocessing_metadata_no_rowdrop.json",
    },
}

KEY_COLS = ["panel", "row_index", "Variant_ID", "Label"]
ID_COLS = {"Variant_ID", "panel"}
TARGET_COL = "Label"


def _json_safe(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, Path):
        return str(value)
    return value


def load_combined_categorical(path: Path | str = COMBINED_CAT_FILE) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    if "panel" not in df.columns:
        raise ValueError(f"Expected a 'panel' column in {path}")
    print(f"[combined] original shape: {df.shape}")
    panels = df["panel"].astype(str).value_counts(dropna=False).to_dict()
    print(f"[combined] panel identifiers found: {panels}")
    return df


def split_master_cancer(
    combined: pd.DataFrame,
    output_dir: Path | str = OUTPUT_DIR / "intermediate",
    panels: Iterable[str] = ("MASTER", "KANSER"),
) -> dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out = {}
    for panel in panels:
        part = combined[combined["panel"].astype(str).str.upper().eq(panel)].copy()
        if part.empty:
            raise ValueError(f"No rows found for panel {panel!r}")
        info = PANEL_MAP[panel]
        path = output_dir / info["raw_out"]
        part.to_csv(path, index=False)
        out[panel] = part.reset_index(drop=True)
        print(f"[{panel}] shape after filtering: {part.shape}; saved {path}")
    return out


def merge_updated_categorical_features(
    panel: str,
    cat_df: pd.DataFrame,
    source_panel_path: Path | str | None = None,
) -> pd.DataFrame:
    """Overlay updated CAT_1/CAT_2 and add CAT12 decision features to a full panel.

    The new file is row-aligned with the existing panel data, but we still validate
    Variant_ID and Label before merging. Derived columns are prefixed to avoid
    colliding with existing AL/EK/CAT feature names.
    """
    panel = panel.upper()
    if panel not in PANEL_MAP:
        raise ValueError(f"Unsupported panel {panel!r}; expected one of {sorted(PANEL_MAP)}")

    source = Path(source_panel_path) if source_panel_path else PANEL_MAP[panel]["source"]
    full = pd.read_csv(source)
    cat = cat_df.reset_index(drop=True).copy()

    if len(full) != len(cat):
        raise ValueError(f"{panel}: source rows {len(full)} != categorical rows {len(cat)}")
    if "Variant_ID" in full.columns and not full["Variant_ID"].reset_index(drop=True).equals(cat["Variant_ID"]):
        raise ValueError(f"{panel}: Variant_ID order mismatch between source panel and updated categorical file")
    if TARGET_COL in full.columns and not full[TARGET_COL].reset_index(drop=True).equals(cat[TARGET_COL]):
        raise ValueError(f"{panel}: Label order mismatch between source panel and updated categorical file")

    merged = full.copy()
    for col in ("CAT_1", "CAT_2"):
        if col in cat.columns:
            merged[col] = cat[col].values

    skip = set(KEY_COLS) | {"CAT_1", "CAT_2"}
    derived_cols = [c for c in cat.columns if c not in skip]
    for col in derived_cols:
        merged[f"CAT12_{col}"] = cat[col].values

    print(
        f"[{panel}] merged updated categorical features: "
        f"{full.shape} -> {merged.shape} (+{len(derived_cols)} CAT12 columns)"
    )
    return merged


@dataclass
class UpdatedCategoricalPreprocessor:
    """Leakage-safe preprocessing for updated categorical panel data.

    Fit this object only on train data. Transform validation/test data with the
    fitted object. If drop_rows is enabled, transform returns rows with <=80%
    missing feature values; use ``last_row_keep_mask_`` to align labels.
    """

    missing_threshold: float = 0.80
    impute_numeric: bool = True
    encode_categorical: bool = True
    exclude_columns: set[str] = field(default_factory=lambda: set(ID_COLS) | {TARGET_COL})

    feature_columns_: list[str] = field(default_factory=list, init=False)
    dropped_columns_: list[str] = field(default_factory=list, init=False)
    categorical_columns_: list[str] = field(default_factory=list, init=False)
    numeric_columns_: list[str] = field(default_factory=list, init=False)
    numeric_medians_: dict[str, float] = field(default_factory=dict, init=False)
    categorical_fill_values_: dict[str, str] = field(default_factory=dict, init=False)
    encoder_: OrdinalEncoder | None = field(default=None, init=False)
    last_row_keep_mask_: pd.Series | None = field(default=None, init=False)
    fitted_shape_: tuple[int, int] | None = field(default=None, init=False)

    def fit(self, df: pd.DataFrame) -> "UpdatedCategoricalPreprocessor":
        X = self._feature_frame(df)
        self.fitted_shape_ = X.shape

        missing_rate = X.isna().mean()
        self.dropped_columns_ = missing_rate[missing_rate > self.missing_threshold].index.tolist()
        kept = X.drop(columns=self.dropped_columns_, errors="ignore")
        self.feature_columns_ = list(kept.columns)

        cat_cols = kept.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
        self.categorical_columns_ = cat_cols
        self.numeric_columns_ = [c for c in self.feature_columns_ if c not in cat_cols]

        num = kept[self.numeric_columns_].apply(pd.to_numeric, errors="coerce")
        med = num.median()
        self.numeric_medians_ = {
            c: (float(v) if pd.notna(v) and np.isfinite(v) else 0.0)
            for c, v in med.items()
        }

        self.categorical_fill_values_ = {c: "__MISSING__" for c in self.categorical_columns_}
        if self.encode_categorical and self.categorical_columns_:
            cats = self._categorical_frame(kept, fit=True)
            self.encoder_ = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-2,
            )
            self.encoder_.fit(cats)
        else:
            self.encoder_ = None
        return self

    def transform(self, df: pd.DataFrame, *, drop_rows: bool = True) -> pd.DataFrame:
        if not self.feature_columns_:
            raise RuntimeError("UpdatedCategoricalPreprocessor must be fitted before transform")

        raw = self._feature_frame(df)
        raw = raw.drop(columns=[c for c in self.dropped_columns_ if c in raw.columns], errors="ignore")
        for col in self.feature_columns_:
            if col not in raw.columns:
                raw[col] = np.nan
        raw = raw[self.feature_columns_]

        if drop_rows and self.feature_columns_:
            row_missing_rate = raw.isna().mean(axis=1)
            keep_mask = row_missing_rate <= self.missing_threshold
        else:
            keep_mask = pd.Series(True, index=raw.index)
        self.last_row_keep_mask_ = keep_mask
        raw = raw.loc[keep_mask].reset_index(drop=True)

        pieces = []
        if self.numeric_columns_:
            num = raw[self.numeric_columns_].apply(pd.to_numeric, errors="coerce")
            if self.impute_numeric:
                num = num.fillna(pd.Series(self.numeric_medians_))
            pieces.append(num.astype(float))

        if self.categorical_columns_:
            cats = self._categorical_frame(raw, fit=False)
            if self.encode_categorical:
                if self.encoder_ is None:
                    raise RuntimeError("Categorical encoder is not fitted")
                enc = pd.DataFrame(
                    self.encoder_.transform(cats),
                    columns=self.categorical_columns_,
                    index=raw.index,
                )
                pieces.append(enc.astype(float))
            else:
                pieces.append(cats)

        if not pieces:
            return pd.DataFrame(index=raw.index)

        out = pd.concat(pieces, axis=1)
        return out[self.numeric_columns_ + self.categorical_columns_]

    def fit_transform(self, df: pd.DataFrame, *, drop_rows: bool = True) -> pd.DataFrame:
        return self.fit(df).transform(df, drop_rows=drop_rows)

    def metadata(self, *, original_shape=None, final_features=None, dropped_rows_count=0) -> dict:
        encoder_categories = {}
        if self.encoder_ is not None:
            encoder_categories = {
                col: [str(v) for v in cats]
                for col, cats in zip(self.categorical_columns_, self.encoder_.categories_)
            }
        return {
            "original_shape": original_shape,
            "fitted_feature_shape": self.fitted_shape_,
            "missing_threshold": self.missing_threshold,
            "dropped_columns": self.dropped_columns_,
            "dropped_rows_count": int(dropped_rows_count),
            "categorical_columns": self.categorical_columns_,
            "numeric_columns": self.numeric_columns_,
            "encoders": {
                "type": "sklearn.preprocessing.OrdinalEncoder",
                "handle_unknown": "use_encoded_value",
                "unknown_value": -1,
                "encoded_missing_value": -2,
                "categories": encoder_categories,
            },
            "imputation_values": self.numeric_medians_,
            "final_feature_list": final_features or (self.numeric_columns_ + self.categorical_columns_),
            "impute_numeric": self.impute_numeric,
            "encode_categorical": self.encode_categorical,
            "cv_usage_note": (
                "Inspection CSV only. For CV/model evaluation, fit this preprocessing "
                "inside each train fold and transform validation/test folds with the "
                "objects fitted on that train fold."
            ),
        }

    def _feature_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        excluded = set(self.exclude_columns)
        # row_index is numeric but is an alignment key, not a biological feature.
        excluded.add("row_index")
        return df[[c for c in df.columns if c not in excluded]].copy()

    def _categorical_frame(self, df: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        cats = df[self.categorical_columns_].copy()
        for col in self.categorical_columns_:
            fill = self.categorical_fill_values_.get(col, "__MISSING__")
            cats[col] = cats[col].astype(object).where(cats[col].notna(), fill).astype(str)
        return cats


def preprocess_panel_export(
    panel: str,
    df: pd.DataFrame,
    output_dir: Path | str = OUTPUT_DIR,
    *,
    mode: str = "strict",
    drop_rows: bool = True,
):
    panel = panel.upper()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    info = PANEL_MAP[panel]
    if mode not in {"strict", "no_rowdrop"}:
        raise ValueError("mode must be 'strict' or 'no_rowdrop'")

    print(f"[{panel}][{mode}] original/full merged shape: {df.shape}")
    pre = UpdatedCategoricalPreprocessor(missing_threshold=0.80, impute_numeric=True)
    X_num = pre.fit_transform(df, drop_rows=drop_rows)
    keep = pre.last_row_keep_mask_.to_numpy()
    y = df.loc[keep, TARGET_COL].reset_index(drop=True) if TARGET_COL in df.columns else None

    dropped_rows = int((~pre.last_row_keep_mask_).sum())
    print(f"[{panel}][{mode}] dropped columns (>80% missing): {len(pre.dropped_columns_)}")
    if pre.dropped_columns_:
        print(f"[{panel}][{mode}] dropped column names: {pre.dropped_columns_}")
    print(f"[{panel}][{mode}] row-drop enabled: {drop_rows}")
    print(f"[{panel}][{mode}] dropped rows (>80% missing): {dropped_rows}")
    print(f"[{panel}][{mode}] shape after missing-value filtering: {X_num.shape}")
    print(f"[{panel}][{mode}] shape after encoding: {X_num.shape}")
    print(f"[{panel}][{mode}] shape after imputation: {X_num.shape}")

    out = X_num.copy()
    if y is not None:
        out.insert(0, TARGET_COL, y.values)
    processed_name = info["processed_out"] if mode == "strict" else info["processed_no_rowdrop_out"]
    processed_path = output_dir / processed_name
    out.to_csv(processed_path, index=False)

    metadata = pre.metadata(
        original_shape=df.shape,
        final_features=list(X_num.columns),
        dropped_rows_count=dropped_rows,
    )
    metadata["mode"] = mode
    metadata["row_drop_enabled"] = drop_rows
    metadata_path = output_dir / (info["metadata_out"] if mode == "strict" else info["metadata_no_rowdrop_out"])
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=_json_safe)

    print(f"[{panel}][{mode}] saved processed numeric dataset: {processed_path}")
    print(f"[{panel}][{mode}] saved preprocessing metadata: {metadata_path}")
    before_counts = df[TARGET_COL].astype(int).value_counts().to_dict() if TARGET_COL in df.columns else {}
    after_counts = y.astype(int).value_counts().to_dict() if y is not None else {}
    before_total = sum(before_counts.values())
    after_total = sum(after_counts.values())
    log_experiment_result({
        "panel": "CANCER" if panel == "KANSER" else panel,
        "experiment_name": f"preprocessing_{panel.lower()}_{mode}",
        "pipeline_role": "ablation" if panel == "KANSER" and mode == "strict" else "primary",
        "preprocess_mode": mode,
        "row_drop_enabled": drop_rows,
        "pca_enabled": False,
        "pca_variant": "none",
        "scaling_enabled": False,
        "outlier_control_enabled": False,
        "imputation_method": "median",
        "encoding_method": "safe ordinal / label encoding",
        "feature_selection_method": "none",
        "model_name": "n/a",
        "calibration_method": "n/a",
        "threshold_strategy": "n/a",
        "selected_threshold": "n/a",
        "blend_enabled": False,
        "blend_weight": "n/a",
        "general_transfer_weight": "n/a",
        "cancer_own_weight": "n/a",
        "cv_strategy": "n/a",
        "n_folds": "n/a",
        "input_rows": df.shape[0],
        "output_rows": out.shape[0],
        "dropped_rows": dropped_rows,
        "input_features": df.shape[1] - (1 if TARGET_COL in df.columns else 0),
        "output_features": len(X_num.columns),
        "dropped_features": len(pre.dropped_columns_),
        "positive_ratio_before": before_counts.get(1, 0) / before_total if before_total else "n/a",
        "positive_ratio_after": after_counts.get(1, 0) / after_total if after_total else "n/a",
        "auc": "n/a",
        "f1_star": "n/a",
        "finals_adjusted_f1": "n/a",
        "precision": "n/a",
        "recall": "n/a",
        "pred_positive_mean": "n/a",
        "delta_f1_vs_baseline": "n/a",
        "notes": "Preprocessing artifact only; fit preprocessing inside CV folds for model evaluation.",
        "output_file": processed_path,
    })
    return out, metadata


def run_preprocessing(
    combined_path: Path | str = COMBINED_CAT_FILE,
    output_dir: Path | str = OUTPUT_DIR,
    merge_full_panels: bool = True,
) -> None:
    output_dir = Path(output_dir)
    combined = load_combined_categorical(combined_path)
    panel_parts = split_master_cancer(combined, output_dir / "intermediate")
    print(f"[combined] shape after MASTER/CANCER filtering: {pd.concat(panel_parts.values()).shape}")

    updated_dir = output_dir / "updated_panels"
    updated_dir.mkdir(parents=True, exist_ok=True)
    for panel, cat_part in panel_parts.items():
        if merge_full_panels:
            panel_df = merge_updated_categorical_features(panel, cat_part)
            updated_path = updated_dir / PANEL_MAP[panel]["updated_out"]
            panel_df.to_csv(updated_path, index=False)
            print(f"[{panel}] saved merged updated panel: {updated_path}")
        else:
            panel_df = cat_part
        preprocess_panel_export(panel, panel_df, output_dir, mode="strict", drop_rows=True)
        preprocess_panel_export(panel, panel_df, output_dir, mode="no_rowdrop", drop_rows=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess updated MASTER/CANCER categorical features")
    parser.add_argument("--combined", default=str(COMBINED_CAT_FILE), help="Combined all-panel CAT12 CSV")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output directory")
    parser.add_argument(
        "--categorical-only",
        action="store_true",
        help="Process only the combined categorical file instead of merging onto existing full panels",
    )
    args = parser.parse_args()
    run_preprocessing(args.combined, args.output_dir, merge_full_panels=not args.categorical_only)


if __name__ == "__main__":
    main()
