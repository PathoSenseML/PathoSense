"""Compare strict vs no-row-drop preprocessing outputs.

This is a verification report only. It reads already-exported preprocessing
artifacts and does not fit models, PCA, scalers, or any CV pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
UPDATED_DIR = OUTPUT_DIR / "updated_panels"
TARGET_COL = "Label"

PANELS = {
    "MASTER": {
        "input": UPDATED_DIR / "YARISMA_TRAIN_MASTER_updated_cat.csv",
        "strict_csv": OUTPUT_DIR / "MASTER_processed_numeric_strict.csv",
        "strict_meta": OUTPUT_DIR / "MASTER_preprocessing_metadata_strict.json",
        "no_rowdrop_csv": OUTPUT_DIR / "MASTER_processed_numeric_no_rowdrop.csv",
        "no_rowdrop_meta": OUTPUT_DIR / "MASTER_preprocessing_metadata_no_rowdrop.json",
    },
    "CANCER": {
        "input": UPDATED_DIR / "YARISMA_TRAIN_CANCER_updated_cat.csv",
        "strict_csv": OUTPUT_DIR / "CANCER_processed_numeric_strict.csv",
        "strict_meta": OUTPUT_DIR / "CANCER_preprocessing_metadata_strict.json",
        "no_rowdrop_csv": OUTPUT_DIR / "CANCER_processed_numeric_no_rowdrop.csv",
        "no_rowdrop_meta": OUTPUT_DIR / "CANCER_preprocessing_metadata_no_rowdrop.json",
    },
}


def _target_stats(df: pd.DataFrame) -> tuple[dict[int, int], float]:
    if TARGET_COL not in df.columns:
        return {}, 0.0
    counts = df[TARGET_COL].astype(int).value_counts().to_dict()
    counts = {0: int(counts.get(0, 0)), 1: int(counts.get(1, 0))}
    total = counts[0] + counts[1]
    pct_pos = counts[1] / total * 100 if total else 0.0
    return counts, pct_pos


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_report(output_dir: Path = OUTPUT_DIR) -> pd.DataFrame:
    rows = []
    for panel, paths in PANELS.items():
        input_df = pd.read_csv(paths["input"])
        before_counts, before_pct_pos = _target_stats(input_df)

        for mode in ("strict", "no_rowdrop"):
            processed = pd.read_csv(paths[f"{mode}_csv"])
            meta = _read_json(paths[f"{mode}_meta"])
            after_counts, after_pct_pos = _target_stats(processed)
            features = processed.drop(columns=[TARGET_COL], errors="ignore")

            rows.append(
                {
                    "panel": panel,
                    "mode": mode,
                    "input_shape": tuple(input_df.shape),
                    "output_shape": tuple(processed.shape),
                    "dropped_rows": int(meta["dropped_rows_count"]),
                    "dropped_columns": len(meta["dropped_columns"]),
                    "final_features": len(meta["final_feature_list"]),
                    "remaining_nan_count": int(features.isna().sum().sum()),
                    "non_numeric_feature_count": int(len(features.select_dtypes(exclude="number").columns)),
                    "target_0_before": before_counts.get(0, 0),
                    "target_1_before": before_counts.get(1, 0),
                    "target_0_after": after_counts.get(0, 0),
                    "target_1_after": after_counts.get(1, 0),
                    "positive_pct_before": round(before_pct_pos, 4),
                    "positive_pct_after": round(after_pct_pos, 4),
                    "positive_pct_delta": round(after_pct_pos - before_pct_pos, 4),
                    "positive_rows_removed": before_counts.get(1, 0) - after_counts.get(1, 0),
                    "negative_rows_removed": before_counts.get(0, 0) - after_counts.get(0, 0),
                }
            )

    report = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_dir / "preprocessing_mode_comparison.csv", index=False)
    _write_markdown(report, output_dir / "preprocessing_mode_comparison.md")
    return report


def _write_markdown(report: pd.DataFrame, path: Path) -> None:
    cancer = report[report["panel"].eq("CANCER")].set_index("mode")
    lines = [
        "# Preprocessing Mode Comparison",
        "",
        "Generated from exported preprocessing artifacts. These CSVs are for inspection; CV/model evaluation must fit preprocessing inside each train fold.",
        "",
        _markdown_table(report),
        "",
    ]

    if {"strict", "no_rowdrop"}.issubset(cancer.index):
        strict = cancer.loc["strict"]
        no_drop = cancer.loc["no_rowdrop"]
        lines.extend(
            [
                "## CANCER Class-Prior Check",
                "",
                f"- Strict keeps {int(strict['output_shape'][0])} of {int(strict['input_shape'][0])} rows and drops {int(strict['dropped_rows'])} rows.",
                f"- No-row-drop keeps {int(no_drop['output_shape'][0])} of {int(no_drop['input_shape'][0])} rows.",
                f"- Strict removes {int(strict['positive_rows_removed'])} positive rows and {int(strict['negative_rows_removed'])} negative rows.",
                f"- Positive class changes from {strict['positive_pct_before']:.2f}% before preprocessing to {strict['positive_pct_after']:.2f}% after strict preprocessing.",
                f"- Compared with no-row-drop, strict shifts the CANCER positive percentage by {(strict['positive_pct_after'] - no_drop['positive_pct_after']):+.2f} percentage points.",
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    rows = [[str(v) for v in row] for row in df.astype(object).itertuples(index=False, name=None)]
    widths = [
        max(len(str(col)), *(len(row[i]) for row in rows)) if rows else len(str(col))
        for i, col in enumerate(cols)
    ]

    def fmt(values):
        return "| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(values)) + " |"

    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    return "\n".join([fmt(cols), sep] + [fmt(row) for row in rows])


def main() -> None:
    report = build_report()
    print(report.to_string(index=False))
    cancer = report[report["panel"].eq("CANCER")].set_index("mode")
    if {"strict", "no_rowdrop"}.issubset(cancer.index):
        strict = cancer.loc["strict"]
        no_drop = cancer.loc["no_rowdrop"]
        print("\nCANCER highlight:")
        print(
            f"  strict removes {int(strict['positive_rows_removed'])} positives and "
            f"{int(strict['negative_rows_removed'])} negatives; positive prior "
            f"{strict['positive_pct_before']:.2f}% -> {strict['positive_pct_after']:.2f}% "
            f"({strict['positive_pct_after'] - no_drop['positive_pct_after']:+.2f} pp vs no_rowdrop)."
        )
    print(f"\nSaved: {OUTPUT_DIR / 'preprocessing_mode_comparison.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'preprocessing_mode_comparison.md'}")


if __name__ == "__main__":
    main()
