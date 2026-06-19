"""
Reusable exploratory data analysis for the TEKNOFEST tabular datasets.

Set TARGET_COLUMN near the top, check DATASETS, then run this file or copy the
cells/functions into a notebook. The original datasets are never modified.
"""

from __future__ import annotations

import json
from io import StringIO
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
    PLOTTING_IMPORT_ERROR = None
except ImportError as exc:
    plt = None
    sns = None
    PLOTTING_AVAILABLE = False
    PLOTTING_IMPORT_ERROR = exc

try:
    from sklearn.impute import SimpleImputer

    SKLEARN_AVAILABLE = True
except ImportError:
    SimpleImputer = None
    SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1) USER SETTINGS
# ---------------------------------------------------------------------------

# Change this if your result/target column has a different name.
TARGET_COLUMN = "Label"

# Identifier columns are not used as features.
ID_COLUMNS = ["Variant_ID"]

# Dataset files. Update these paths if you move the script/notebook.
DATA_DIR = Path(__file__).resolve().parent
DATASETS = {
    "CFTR_panel": DATA_DIR / "YARISMA_TRAIN_CFTR.csv",
    "KANSER_panel": DATA_DIR / "YARISMA_TRAIN_KANSER.csv",
    "MASTER_main": DATA_DIR / "YARISMA_TRAIN_MASTER.csv",
    "PAH_panel": DATA_DIR / "YARISMA_TRAIN_PAH.csv",
}

OUTPUT_DIR = DATA_DIR / "eda_outputs"
ENCODED_OUTPUT_DIR = DATA_DIR / "encoded_datasets"

MISSING_THRESHOLD_PCT = 50.0
HIGH_OUTLIER_THRESHOLD_PCT = 5.0
HIGH_CORRELATION_THRESHOLD = 0.85
TOP_N_COLUMNS_FOR_3_COLUMN_GROUPS = 25
MAX_BOXPLOTS_PER_FIGURE = 12

TARGET_CANDIDATES = [
    "Label",
    "target",
    "Target",
    "target_column",
    "class",
    "Class",
    "y",
]

AMINO_ACID_TO_NUMBER = {
    "UNKNOWN": 0,
    "A": 1,
    "R": 2,
    "N": 3,
    "D": 4,
    "C": 5,
    "Q": 6,
    "E": 7,
    "G": 8,
    "H": 9,
    "I": 10,
    "L": 11,
    "K": 12,
    "M": 13,
    "F": 14,
    "P": 15,
    "S": 16,
    "T": 17,
    "W": 18,
    "Y": 19,
    "V": 20,
    "B": 21,
    "Z": 22,
    "X": 23,
    "U": 24,
    "O": 25,
}


# ---------------------------------------------------------------------------
# 2) LOADING AND COLUMN HELPERS
# ---------------------------------------------------------------------------


def load_dataset(path: Path) -> pd.DataFrame:
    """Load CSV or Excel files with a small amount of extension-based handling."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise ValueError(f"Unsupported file type: {path.suffix}")


def get_feature_columns(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    id_columns: Iterable[str] = ID_COLUMNS,
) -> List[str]:
    """Return feature columns, excluding identifier and target columns."""
    excluded = set(id_columns) | {target_column}
    return [col for col in df.columns if col not in excluded]


def separate_numeric_categorical_columns(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    id_columns: Iterable[str] = ID_COLUMNS,
) -> Tuple[List[str], List[str]]:
    """Separate feature columns into numeric and categorical lists."""
    feature_columns = get_feature_columns(df, target_column, id_columns)
    categorical_columns = [
        col
        for col in feature_columns
        if col.startswith("CAT_")
        or is_amino_acid_column(col, df[col])
        or not pd.api.types.is_numeric_dtype(df[col])
    ]
    numeric_columns = [
        col
        for col in feature_columns
        if pd.api.types.is_numeric_dtype(df[col]) and col not in categorical_columns
    ]
    return numeric_columns, categorical_columns


def get_prefix_groups(columns: Iterable[str]) -> Dict[str, List[str]]:
    """Group columns by expected biological prefixes."""
    prefixes = ("AL_", "CAT_", "EK_", "AA_")
    groups = {prefix.rstrip("_"): [] for prefix in prefixes}
    groups["OTHER"] = []

    for col in columns:
        matched = False
        for prefix in prefixes:
            if col.startswith(prefix):
                groups[prefix.rstrip("_")].append(col)
                matched = True
                break
        if not matched:
            groups["OTHER"].append(col)

    return groups


def show_basic_information(
    df: pd.DataFrame,
    dataset_name: str,
    target_column: str = TARGET_COLUMN,
    id_columns: Iterable[str] = ID_COLUMNS,
) -> Dict[str, object]:
    """Print basic shape, columns, dtypes, and column type separation."""
    numeric_columns, categorical_columns = separate_numeric_categorical_columns(
        df, target_column, id_columns
    )
    feature_columns = get_feature_columns(df, target_column, id_columns)
    prefix_groups = get_prefix_groups(feature_columns)

    print(f"\n{'=' * 80}")
    print(f"DATASET: {dataset_name}")
    print(f"{'=' * 80}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nColumn names:")
    print(list(df.columns))
    print("\nData types:")
    print(df.dtypes)
    print("\nNumeric feature columns:")
    print(numeric_columns)
    print("\nCategorical feature columns:")
    print(categorical_columns)
    print("\nColumn prefix groups:")
    for group_name, group_columns in prefix_groups.items():
        print(f"  {group_name}: {len(group_columns)} columns")

    return {
        "dataset_name": dataset_name,
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "prefix_groups": prefix_groups,
    }


# ---------------------------------------------------------------------------
# 3) MISSING VALUE ANALYSIS
# ---------------------------------------------------------------------------


def analyze_missing_values(
    df: pd.DataFrame,
    dataset_name: str,
    threshold_pct: float = MISSING_THRESHOLD_PCT,
) -> Dict[str, pd.DataFrame]:
    """Analyze missing values by column and row."""
    missing_count = df.isna().sum()
    missing_pct = (missing_count / len(df) * 100).round(2)

    missing_table = (
        pd.DataFrame(
            {
                "dataset_name": dataset_name,
                "column_name": df.columns,
                "missing_count": missing_count.values,
                "missing_pct": missing_pct.values,
                "data_type": df.dtypes.astype(str).values,
            }
        )
        .sort_values("missing_pct", ascending=False)
        .reset_index(drop=True)
    )

    high_missing_columns = missing_table[
        missing_table["missing_pct"] > threshold_pct
    ].copy()

    row_missing_pct = (df.isna().sum(axis=1) / df.shape[1] * 100).round(2)
    high_missing_rows = (
        pd.DataFrame(
            {
                "dataset_name": dataset_name,
                "row_index": df.index,
                "missing_count": df.isna().sum(axis=1).values,
                "missing_pct": row_missing_pct.values,
            }
        )
        .query("missing_pct > @threshold_pct")
        .sort_values("missing_pct", ascending=False)
        .reset_index(drop=True)
    )

    print("\nMissing value summary:")
    print(f"Columns with > {threshold_pct}% missing: {len(high_missing_columns)}")
    print(f"Rows with > {threshold_pct}% missing: {len(high_missing_rows)}")

    return {
        "missing_table": missing_table,
        "high_missing_columns": high_missing_columns,
        "high_missing_rows": high_missing_rows,
    }


# ---------------------------------------------------------------------------
# 4) OUTLIER ANALYSIS
# ---------------------------------------------------------------------------


def analyze_numeric_outliers(
    df: pd.DataFrame,
    dataset_name: str,
    numeric_columns: List[str],
) -> pd.DataFrame:
    """Calculate numeric statistics and IQR-based outlier information."""
    records = []

    for col in numeric_columns:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        non_missing_count = len(values)

        if non_missing_count == 0:
            records.append(
                {
                    "dataset_name": dataset_name,
                    "column_name": col,
                    "min": np.nan,
                    "max": np.nan,
                    "mean": np.nan,
                    "median": np.nan,
                    "q1": np.nan,
                    "q3": np.nan,
                    "iqr": np.nan,
                    "lower_bound": np.nan,
                    "upper_bound": np.nan,
                    "outlier_count": 0,
                    "outlier_pct": 0.0,
                    "cleaned_min": np.nan,
                    "cleaned_max": np.nan,
                    "raw_range": "",
                    "cleaned_range": "",
                }
            )
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_mask = (values < lower_bound) | (values > upper_bound)
        outlier_count = int(outlier_mask.sum())
        outlier_pct = round(outlier_count / non_missing_count * 100, 2)
        cleaned_values = values.loc[~outlier_mask]

        raw_min = values.min()
        raw_max = values.max()
        cleaned_min = cleaned_values.min() if len(cleaned_values) else np.nan
        cleaned_max = cleaned_values.max() if len(cleaned_values) else np.nan

        records.append(
            {
                "dataset_name": dataset_name,
                "column_name": col,
                "min": raw_min,
                "max": raw_max,
                "mean": values.mean(),
                "median": values.median(),
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_count": outlier_count,
                "outlier_pct": outlier_pct,
                "cleaned_min": cleaned_min,
                "cleaned_max": cleaned_max,
                "raw_range": f"{raw_min} to {raw_max}",
                "cleaned_range": f"{cleaned_min} to {cleaned_max}",
            }
        )

    outlier_table = pd.DataFrame(records).sort_values(
        "outlier_pct", ascending=False
    )

    print("\nOutlier summary:")
    print(outlier_table[["column_name", "outlier_count", "outlier_pct"]].head(20))

    return outlier_table.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5) VISUALIZATIONS
# ---------------------------------------------------------------------------


def plot_high_outlier_boxplots(
    df: pd.DataFrame,
    dataset_name: str,
    outlier_table: pd.DataFrame,
    output_dir: Path,
    threshold_pct: float = HIGH_OUTLIER_THRESHOLD_PCT,
    max_plots_per_figure: int = MAX_BOXPLOTS_PER_FIGURE,
) -> List[Path]:
    """Create box plots for columns with high outlier percentages."""
    if not PLOTTING_AVAILABLE:
        print(
            "Skipping box plots because matplotlib/seaborn is not installed. "
            f"Import error: {PLOTTING_IMPORT_ERROR}"
        )
        return []

    high_outlier_columns = outlier_table.loc[
        outlier_table["outlier_pct"] >= threshold_pct, "column_name"
    ].tolist()

    if not high_outlier_columns:
        print(f"No columns above {threshold_pct}% outliers for {dataset_name}.")
        return []

    plot_paths = []
    plot_dir = output_dir / "plots" / dataset_name
    plot_dir.mkdir(parents=True, exist_ok=True)

    for start in range(0, len(high_outlier_columns), max_plots_per_figure):
        columns_chunk = high_outlier_columns[start : start + max_plots_per_figure]
        n_cols = min(3, len(columns_chunk))
        n_rows = int(np.ceil(len(columns_chunk) / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = np.array(axes).reshape(-1)

        for ax, col in zip(axes, columns_chunk):
            sns.boxplot(y=df[col], ax=ax, color="#6BAED6")
            ax.set_title(col, fontsize=10)
            ax.set_xlabel("")
            ax.set_ylabel("")

        for ax in axes[len(columns_chunk) :]:
            ax.set_visible(False)

        fig.suptitle(
            f"{dataset_name}: numeric columns with >= {threshold_pct}% outliers",
            fontsize=14,
        )
        fig.tight_layout()

        plot_path = plot_dir / f"boxplots_high_outliers_{start // max_plots_per_figure + 1}.png"
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        plot_paths.append(plot_path)

    return plot_paths


def plot_correlation_heatmap(
    df: pd.DataFrame,
    dataset_name: str,
    numeric_columns: List[str],
    output_dir: Path,
    max_columns: int = 60,
) -> Optional[Path]:
    """Create a correlation heatmap for numeric columns without removing outliers."""
    if not PLOTTING_AVAILABLE:
        print(
            "Skipping correlation heatmap because matplotlib/seaborn is not installed. "
            f"Import error: {PLOTTING_IMPORT_ERROR}"
        )
        return None

    if len(numeric_columns) < 2:
        print(f"Not enough numeric columns for a heatmap in {dataset_name}.")
        return None

    selected_columns = numeric_columns[:max_columns]
    if len(numeric_columns) > max_columns:
        print(
            f"{dataset_name}: heatmap limited to first {max_columns} numeric columns "
            f"for readability."
        )

    corr = df[selected_columns].corr()
    plot_dir = output_dir / "plots" / dataset_name
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig_width = max(10, min(24, len(selected_columns) * 0.45))
    fig_height = max(8, min(22, len(selected_columns) * 0.45))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(
        corr,
        cmap="vlag",
        center=0,
        square=False,
        linewidths=0.1,
        cbar_kws={"shrink": 0.7},
        ax=ax,
    )
    ax.set_title(f"{dataset_name}: numeric correlation heatmap")
    fig.tight_layout()

    heatmap_path = plot_dir / "numeric_correlation_heatmap.png"
    fig.savefig(heatmap_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return heatmap_path


# ---------------------------------------------------------------------------
# 6) TARGET RELATIONSHIPS
# ---------------------------------------------------------------------------


def calculate_target_correlations(
    df: pd.DataFrame,
    dataset_name: str,
    numeric_columns: List[str],
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Rank numeric columns by absolute correlation with the target column."""
    if target_column not in df.columns:
        print(f"Target column '{target_column}' not found in {dataset_name}.")
        return pd.DataFrame(
            columns=[
                "dataset_name",
                "column_name",
                "correlation_with_target",
                "abs_correlation_with_target",
            ]
        )

    target_values = pd.to_numeric(df[target_column], errors="coerce")
    if target_values.notna().sum() < 2:
        print(
            f"Target column '{target_column}' in {dataset_name} is not numeric enough "
            "for Pearson correlation."
        )
        return pd.DataFrame(
            columns=[
                "dataset_name",
                "column_name",
                "correlation_with_target",
                "abs_correlation_with_target",
            ]
        )

    records = []
    for col in numeric_columns:
        feature_values = pd.to_numeric(df[col], errors="coerce")
        corr = feature_values.corr(target_values)
        records.append(
            {
                "dataset_name": dataset_name,
                "column_name": col,
                "correlation_with_target": corr,
                "abs_correlation_with_target": abs(corr) if pd.notna(corr) else np.nan,
            }
        )

    target_corr = pd.DataFrame(records).sort_values(
        "abs_correlation_with_target", ascending=False
    )

    print("\nStrongest numeric relationships with target:")
    print(target_corr.head(20))

    return target_corr.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 7) COLUMN-COLUMN RELATIONSHIPS
# ---------------------------------------------------------------------------


def find_highly_correlated_pairs(
    df: pd.DataFrame,
    dataset_name: str,
    numeric_columns: List[str],
    threshold: float = HIGH_CORRELATION_THRESHOLD,
) -> pd.DataFrame:
    """Find highly correlated numeric column pairs."""
    if len(numeric_columns) < 2:
        return pd.DataFrame(
            columns=[
                "dataset_name",
                "column_1",
                "column_2",
                "correlation",
                "abs_correlation",
            ]
        )

    corr = df[numeric_columns].corr()
    records = []

    for col_1, col_2 in combinations(corr.columns, 2):
        value = corr.loc[col_1, col_2]
        if pd.notna(value) and abs(value) >= threshold:
            records.append(
                {
                    "dataset_name": dataset_name,
                    "column_1": col_1,
                    "column_2": col_2,
                    "correlation": value,
                    "abs_correlation": abs(value),
                }
            )

    pairs = pd.DataFrame(records)
    if not pairs.empty:
        pairs = pairs.sort_values("abs_correlation", ascending=False).reset_index(
            drop=True
        )

    print(f"\nHighly correlated pairs, abs(corr) >= {threshold}: {len(pairs)}")
    print(pairs.head(20))

    return pairs


def select_top_columns_for_group_analysis(
    numeric_columns: List[str],
    target_corr: pd.DataFrame,
    outlier_table: pd.DataFrame,
    top_n: int = TOP_N_COLUMNS_FOR_3_COLUMN_GROUPS,
) -> List[str]:
    """
    Select a limited numeric column set for 3-column group analysis.

    Priority is target relationship first, then high outlier columns as extra
    review candidates. This avoids evaluating all 330C3 combinations.
    """
    selected = []

    if not target_corr.empty:
        selected.extend(
            target_corr["column_name"]
            .dropna()
            .head(top_n)
            .astype(str)
            .tolist()
        )

    if not outlier_table.empty and len(selected) < top_n:
        selected.extend(
            outlier_table["column_name"]
            .dropna()
            .head(top_n)
            .astype(str)
            .tolist()
        )

    selected = [col for col in dict.fromkeys(selected) if col in numeric_columns]

    if len(selected) < top_n:
        selected.extend([col for col in numeric_columns if col not in selected])
        selected = selected[:top_n]

    return selected[:top_n]


def analyze_three_column_groups(
    df: pd.DataFrame,
    dataset_name: str,
    selected_columns: List[str],
) -> pd.DataFrame:
    """
    Analyze only selected 3-column groups.

    For each group, summarize the mean absolute pairwise correlation. Groups with
    high values may indicate redundant biological measurements.
    """
    if len(selected_columns) < 3:
        return pd.DataFrame(
            columns=[
                "dataset_name",
                "column_1",
                "column_2",
                "column_3",
                "mean_abs_pairwise_corr",
                "max_abs_pairwise_corr",
            ]
        )

    corr = df[selected_columns].corr()
    records = []

    for col_1, col_2, col_3 in combinations(selected_columns, 3):
        pair_values = [
            corr.loc[col_1, col_2],
            corr.loc[col_1, col_3],
            corr.loc[col_2, col_3],
        ]
        pair_values = [abs(value) for value in pair_values if pd.notna(value)]

        if not pair_values:
            continue

        records.append(
            {
                "dataset_name": dataset_name,
                "column_1": col_1,
                "column_2": col_2,
                "column_3": col_3,
                "mean_abs_pairwise_corr": float(np.mean(pair_values)),
                "max_abs_pairwise_corr": float(np.max(pair_values)),
            }
        )

    groups = pd.DataFrame(records).sort_values(
        ["mean_abs_pairwise_corr", "max_abs_pairwise_corr"],
        ascending=False,
    )

    print(
        "\nTop 3-column groups from selected columns "
        f"({len(selected_columns)} columns analyzed):"
    )
    print(groups.head(20))

    return groups.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 8) BIOLOGIST-FRIENDLY SUMMARY
# ---------------------------------------------------------------------------


def suggest_action(
    missing_pct: float,
    outlier_pct: float,
    abs_target_corr: Optional[float],
    is_identifier: bool,
    is_target: bool,
) -> str:
    """Assign a simple review action for each column."""
    if is_identifier:
        return "DROP_CANDIDATE"
    if is_target:
        return "KEEP"
    if missing_pct > MISSING_THRESHOLD_PCT:
        return "DROP_CANDIDATE"
    if outlier_pct >= HIGH_OUTLIER_THRESHOLD_PCT:
        return "REVIEW_WITH_BIOLOGIST"
    if pd.notna(abs_target_corr) and abs_target_corr >= 0.20:
        return "KEEP"
    return "REVIEW_WITH_BIOLOGIST"


def build_biologist_summary(
    df: pd.DataFrame,
    dataset_name: str,
    missing_table: pd.DataFrame,
    outlier_table: pd.DataFrame,
    target_corr: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    id_columns: Iterable[str] = ID_COLUMNS,
) -> pd.DataFrame:
    """Build one summary row per column for review/export."""
    id_columns = set(id_columns)

    missing_map = missing_table.set_index("column_name")["missing_pct"].to_dict()
    outlier_pct_map = outlier_table.set_index("column_name")["outlier_pct"].to_dict()
    raw_range_map = outlier_table.set_index("column_name")["raw_range"].to_dict()
    cleaned_range_map = outlier_table.set_index("column_name")["cleaned_range"].to_dict()
    target_corr_map = (
        target_corr.set_index("column_name")["correlation_with_target"].to_dict()
        if not target_corr.empty
        else {}
    )
    abs_target_corr_map = (
        target_corr.set_index("column_name")["abs_correlation_with_target"].to_dict()
        if not target_corr.empty
        else {}
    )

    records = []
    for col in df.columns:
        missing_pct = float(missing_map.get(col, 0.0))
        outlier_pct = float(outlier_pct_map.get(col, 0.0))
        target_corr_value = target_corr_map.get(col, np.nan)
        abs_target_corr_value = abs_target_corr_map.get(col, np.nan)

        records.append(
            {
                "dataset_name": dataset_name,
                "column_name": col,
                "data_type": str(df[col].dtype),
                "missing_pct": missing_pct,
                "outlier_pct": outlier_pct,
                "raw_range": raw_range_map.get(col, ""),
                "cleaned_range": cleaned_range_map.get(col, ""),
                "correlation_with_target": target_corr_value,
                "suggested_action": suggest_action(
                    missing_pct=missing_pct,
                    outlier_pct=outlier_pct,
                    abs_target_corr=abs_target_corr_value,
                    is_identifier=col in id_columns,
                    is_target=col == target_column,
                ),
            }
        )

    return pd.DataFrame(records)


def export_dataset_results(
    dataset_name: str,
    output_dir: Path,
    missing_table: pd.DataFrame,
    high_missing_columns: pd.DataFrame,
    high_missing_rows: pd.DataFrame,
    outlier_table: pd.DataFrame,
    target_corr: pd.DataFrame,
    high_corr_pairs: pd.DataFrame,
    three_col_groups: pd.DataFrame,
    biologist_summary: pd.DataFrame,
) -> Dict[str, Path]:
    """Export all tables to CSV and one multi-sheet Excel workbook."""
    dataset_output_dir = output_dir / "tables" / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "missing_table": missing_table,
        "high_missing_columns": high_missing_columns,
        "high_missing_rows": high_missing_rows,
        "outlier_table": outlier_table,
        "target_correlations": target_corr,
        "highly_correlated_pairs": high_corr_pairs,
        "three_column_groups": three_col_groups,
        "biologist_summary": biologist_summary,
    }

    paths = {}
    for table_name, table in tables.items():
        csv_path = dataset_output_dir / f"{table_name}.csv"
        table.to_csv(csv_path, index=False)
        paths[table_name] = csv_path

    excel_path = dataset_output_dir / f"{dataset_name}_eda_summary.xlsx"
    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            for sheet_name, table in tables.items():
                # Excel sheet names are limited to 31 characters.
                table.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        paths["excel_workbook"] = excel_path
    except ImportError:
        print("openpyxl is not installed; CSV files were exported instead of Excel.")

    return paths


# ---------------------------------------------------------------------------
# 9) REQUIRED BASIC CHECKS AND NUMERIC CONVERSION
# ---------------------------------------------------------------------------


def detect_target_column(
    df: pd.DataFrame,
    preferred_target: str = TARGET_COLUMN,
    target_candidates: Iterable[str] = TARGET_CANDIDATES,
) -> Optional[str]:
    """Detect the dependent variable/target column."""
    if preferred_target in df.columns:
        return preferred_target

    for candidate in target_candidates:
        if candidate in df.columns:
            return candidate

    return df.columns[-1] if len(df.columns) else None


def infer_problem_type(df: pd.DataFrame, target_column: Optional[str]) -> str:
    """Infer whether the target points to classification or regression."""
    if target_column is None or target_column not in df.columns:
        return "unknown"

    target = df[target_column].dropna()
    unique_count = target.nunique()

    if unique_count <= 20:
        return "classification"

    if not pd.api.types.is_numeric_dtype(target):
        return "classification"

    return "regression"


def capture_df_info(df: pd.DataFrame) -> str:
    """Return df.info() as text so it can be written to a report file."""
    buffer = StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()


def is_amino_acid_column(column_name: str, series: pd.Series) -> bool:
    """Detect amino acid/residue columns from name and one-letter AA values."""
    lowered_name = column_name.lower()
    if column_name.startswith("AA_") or any(
        token in lowered_name for token in ("amino", "residue")
    ):
        return True

    values = series.dropna().astype(str).str.strip().str.upper()
    if values.empty:
        return False

    allowed = set(AMINO_ACID_TO_NUMBER)
    one_letter_ratio = values.map(lambda value: len(value) == 1 and value in allowed).mean()
    return bool(one_letter_ratio >= 0.95)


def get_required_column_summary(
    df: pd.DataFrame,
    target_column: Optional[str],
    id_columns: Iterable[str] = ID_COLUMNS,
) -> Dict[str, object]:
    """Summarize categorical, numeric, missing, target, and task type findings."""
    id_columns = set(id_columns)
    feature_columns = [
        col for col in df.columns if col not in id_columns and col != target_column
    ]
    categorical_columns = [
        col
        for col in feature_columns
        if col.startswith("CAT_")
        or is_amino_acid_column(col, df[col])
        or not pd.api.types.is_numeric_dtype(df[col])
    ]
    numeric_columns = [
        col
        for col in feature_columns
        if pd.api.types.is_numeric_dtype(df[col]) and col not in categorical_columns
    ]
    missing_columns = df.columns[df.isna().sum() > 0].tolist()
    amino_acid_columns = [
        col for col in categorical_columns if is_amino_acid_column(col, df[col])
    ]

    return {
        "target_column": target_column,
        "problem_type": infer_problem_type(df, target_column),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "amino_acid_columns": amino_acid_columns,
        "missing_columns": missing_columns,
    }


def build_basic_eda_report(
    df: pd.DataFrame,
    dataset_name: str,
    target_column: Optional[str],
    id_columns: Iterable[str] = ID_COLUMNS,
) -> Dict[str, object]:
    """
    Run the exact first checks requested:
    head, info, isnull().sum, describe, and target value_counts.
    """
    summary = get_required_column_summary(df, target_column, id_columns)
    target_value_counts = (
        df[target_column].value_counts(dropna=False)
        if target_column is not None and target_column in df.columns
        else pd.Series(dtype="int64")
    )

    missing_series = df.isnull().sum()
    missing_table = (
        pd.DataFrame(
            {
                "dataset_name": dataset_name,
                "column_name": missing_series.index,
                "missing_count": missing_series.values,
                "missing_pct": (missing_series.values / len(df) * 100).round(2),
                "data_type": df.dtypes.astype(str).values,
            }
        )
        .sort_values(["missing_count", "column_name"], ascending=[False, True])
        .reset_index(drop=True)
    )

    return {
        "dataset_name": dataset_name,
        "shape": df.shape,
        "head": df.head(),
        "info": capture_df_info(df),
        "isnull_sum": missing_series,
        "describe": df.describe(include="all").transpose(),
        "target_value_counts": target_value_counts,
        "summary": summary,
        "missing_table": missing_table,
    }


def encode_amino_acid_series(series: pd.Series) -> pd.Series:
    """Map amino acid letters and UNKNOWN to stable numeric IDs."""
    return series.astype("string").str.strip().str.upper().map(AMINO_ACID_TO_NUMBER)


def encode_categorical_series(series: pd.Series) -> Tuple[pd.Series, Dict[str, int]]:
    """Encode a categorical/string column with a deterministic value mapping."""
    values = series.dropna().astype(str).str.strip()
    categories = sorted(values.unique().tolist())
    mapping = {category: index for index, category in enumerate(categories)}
    encoded = series.astype("string").str.strip().map(mapping)
    return encoded, mapping


def impute_categorical_columns_before_encoding(
    df: pd.DataFrame,
    categorical_columns: List[str],
    amino_acid_columns: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Impute categorical columns before encoding."""
    imputed_df = df.copy()
    amino_acid_columns = set(amino_acid_columns)
    records = []

    for col in categorical_columns:
        missing_before = int(imputed_df[col].isna().sum())
        if missing_before == 0:
            records.append(
                {
                    "column_name": col,
                    "feature_type": "amino_acid" if col in amino_acid_columns else "categorical",
                    "imputation_stage": "before_encoding",
                    "strategy": "none_needed",
                    "fill_value": "",
                    "missing_before": 0,
                    "missing_after": 0,
                }
            )
            continue

        if col in amino_acid_columns:
            fill_value = "Unknown"
            strategy = "unknown_category"
            imputed_df[col] = imputed_df[col].fillna(fill_value)
        else:
            non_missing = imputed_df[col].dropna()
            if non_missing.empty:
                fill_value = "Unknown"
                strategy = "unknown_category_all_missing"
                imputed_df[col] = imputed_df[col].fillna(fill_value)
            else:
                strategy = "most_frequent"
                if SKLEARN_AVAILABLE:
                    cat_imputer = SimpleImputer(strategy="most_frequent")
                    imputed_df[[col]] = cat_imputer.fit_transform(imputed_df[[col]])
                    fill_value = str(cat_imputer.statistics_[0])
                else:
                    fill_value = str(non_missing.mode(dropna=True).iloc[0])
                    imputed_df[col] = imputed_df[col].fillna(fill_value)

        records.append(
            {
                "column_name": col,
                "feature_type": "amino_acid" if col in amino_acid_columns else "categorical",
                "imputation_stage": "before_encoding",
                "strategy": strategy,
                "fill_value": fill_value,
                "missing_before": missing_before,
                "missing_after": int(imputed_df[col].isna().sum()),
            }
        )

    return imputed_df, pd.DataFrame(records)


def impute_numeric_columns_after_encoding(
    df: pd.DataFrame,
    numeric_columns: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Impute numeric feature columns with median after encoding."""
    imputed_df = df.copy()
    records = []

    usable_numeric_columns = [
        col for col in numeric_columns if not imputed_df[col].isna().all()
    ]
    all_missing_numeric_columns = [
        col for col in numeric_columns if imputed_df[col].isna().all()
    ]

    if usable_numeric_columns:
        missing_before = imputed_df[usable_numeric_columns].isna().sum()
        if SKLEARN_AVAILABLE:
            num_imputer = SimpleImputer(strategy="median")
            imputed_df[usable_numeric_columns] = num_imputer.fit_transform(
                imputed_df[usable_numeric_columns]
            )
            fill_values = dict(zip(usable_numeric_columns, num_imputer.statistics_))
        else:
            fill_values = imputed_df[usable_numeric_columns].median(numeric_only=True)
            imputed_df[usable_numeric_columns] = imputed_df[
                usable_numeric_columns
            ].fillna(fill_values)

        missing_after = imputed_df[usable_numeric_columns].isna().sum()
        for col in usable_numeric_columns:
            records.append(
                {
                    "column_name": col,
                    "feature_type": "numerical",
                    "imputation_stage": "after_encoding",
                    "strategy": "median",
                    "fill_value": float(fill_values[col]),
                    "missing_before": int(missing_before[col]),
                    "missing_after": int(missing_after[col]),
                }
            )

    for col in all_missing_numeric_columns:
        missing_before = int(imputed_df[col].isna().sum())
        imputed_df[col] = 0.0
        records.append(
            {
                "column_name": col,
                "feature_type": "numerical",
                "imputation_stage": "after_encoding",
                "strategy": "constant_all_missing",
                "fill_value": 0.0,
                "missing_before": missing_before,
                "missing_after": int(imputed_df[col].isna().sum()),
            }
        )

    return imputed_df, pd.DataFrame(records)


def convert_categorical_and_amino_acid_columns_to_numeric(
    df: pd.DataFrame,
    target_column: Optional[str],
    id_columns: Iterable[str] = ID_COLUMNS,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Impute missing values, then convert CAT/string and amino acid columns.

    Variant_ID is kept unchanged as an identifier. Target is not imputed.
    """
    summary = get_required_column_summary(df, target_column, id_columns)
    categorical_columns = summary["categorical_columns"]
    amino_acid_columns = summary["amino_acid_columns"]
    numeric_columns = summary["numeric_columns"]

    converted_df, categorical_imputation = impute_categorical_columns_before_encoding(
        df,
        categorical_columns,
        amino_acid_columns,
    )
    mapping_records = []

    for col in categorical_columns:
        if col in amino_acid_columns:
            converted_df[col] = encode_amino_acid_series(converted_df[col])
            for value, code in sorted(AMINO_ACID_TO_NUMBER.items(), key=lambda item: item[1]):
                mapping_records.append(
                    {
                        "column_name": col,
                        "encoding_type": "amino_acid",
                        "original_value": "Unknown" if value == "UNKNOWN" else value,
                        "encoded_value": code,
                    }
                )
        else:
            encoded, mapping = encode_categorical_series(converted_df[col])
            converted_df[col] = encoded
            for value, code in mapping.items():
                mapping_records.append(
                    {
                        "column_name": col,
                        "encoding_type": "categorical",
                        "original_value": value,
                        "encoded_value": code,
                    }
                )

    mapping_table = pd.DataFrame(
        mapping_records,
        columns=["column_name", "encoding_type", "original_value", "encoded_value"],
    )

    all_numeric_feature_columns = numeric_columns + categorical_columns
    converted_df, numeric_imputation = impute_numeric_columns_after_encoding(
        converted_df,
        all_numeric_feature_columns,
    )

    imputation_table = pd.concat(
        [categorical_imputation, numeric_imputation],
        ignore_index=True,
    )
    return converted_df, mapping_table, imputation_table


def export_basic_eda_report(
    report: Dict[str, object],
    dataset_name: str,
    output_dir: Path,
) -> Dict[str, Path]:
    """Write requested checks to files for one dataset."""
    report_dir = output_dir / "basic_checks" / dataset_name
    report_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "head": report_dir / "head.csv",
        "isnull_sum": report_dir / "isnull_sum.csv",
        "describe": report_dir / "describe.csv",
        "target_value_counts": report_dir / "target_value_counts.csv",
        "missing_table": report_dir / "missing_table.csv",
        "summary_json": report_dir / "summary.json",
        "info_txt": report_dir / "info.txt",
    }

    report["head"].to_csv(paths["head"], index=False)
    report["isnull_sum"].rename("missing_count").to_csv(paths["isnull_sum"])
    report["describe"].to_csv(paths["describe"])
    report["target_value_counts"].rename("count").to_csv(
        paths["target_value_counts"]
    )
    report["missing_table"].to_csv(paths["missing_table"], index=False)
    paths["info_txt"].write_text(str(report["info"]), encoding="utf-8")

    json_ready_summary = {
        key: value
        for key, value in report["summary"].items()
        if key
        in {
            "target_column",
            "problem_type",
            "numeric_columns",
            "categorical_columns",
            "amino_acid_columns",
            "missing_columns",
        }
    }
    json_ready_summary["shape"] = list(report["shape"])
    paths["summary_json"].write_text(
        json.dumps(json_ready_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return paths


def analyze_and_save_numeric_dataset(
    dataset_name: str,
    dataset_path: Path,
    output_dir: Path = OUTPUT_DIR,
    encoded_output_dir: Path = ENCODED_OUTPUT_DIR,
    preferred_target: str = TARGET_COLUMN,
    id_columns: Iterable[str] = ID_COLUMNS,
) -> Dict[str, object]:
    """Run requested checks, convert string features to numeric, and save outputs."""
    df = load_dataset(dataset_path)
    target_column = detect_target_column(df, preferred_target)
    report = build_basic_eda_report(df, dataset_name, target_column, id_columns)
    report_paths = export_basic_eda_report(report, dataset_name, output_dir)

    encoded_output_dir.mkdir(parents=True, exist_ok=True)
    converted_df, mapping_table, imputation_table = convert_categorical_and_amino_acid_columns_to_numeric(
        df,
        target_column,
        id_columns,
    )

    encoded_path = encoded_output_dir / f"{dataset_path.stem}_numeric_encoded.csv"
    mapping_path = encoded_output_dir / f"{dataset_path.stem}_encoding_map.csv"
    imputation_path = encoded_output_dir / f"{dataset_path.stem}_imputation_summary.csv"
    converted_df.to_csv(encoded_path, index=False)
    mapping_table.to_csv(mapping_path, index=False)
    imputation_table.to_csv(imputation_path, index=False)

    feature_columns = [
        col
        for col in converted_df.columns
        if col not in set(id_columns) and col != target_column
    ]
    missing_after_imputation = int(converted_df[feature_columns].isnull().sum().sum())

    print(f"\n{dataset_name}")
    print(f"  Target/dependent variable: {target_column}")
    print(f"  Problem type: {report['summary']['problem_type']}")
    print(f"  Numeric columns: {len(report['summary']['numeric_columns'])}")
    print(f"  Categorical columns: {report['summary']['categorical_columns']}")
    print(f"  Amino acid columns: {report['summary']['amino_acid_columns']}")
    print(f"  Columns with missing values: {len(report['summary']['missing_columns'])}")
    print(f"  Missing feature values after imputation: {missing_after_imputation}")
    print(f"  Encoded + imputed dataset saved: {encoded_path}")

    return {
        "report": report,
        "report_paths": report_paths,
        "encoded_dataset_path": encoded_path,
        "encoding_map_path": mapping_path,
        "imputation_summary_path": imputation_path,
        "converted_dataframe": converted_df,
        "encoding_map": mapping_table,
        "imputation_summary": imputation_table,
    }


def analyze_and_save_all_numeric_datasets(
    datasets: Dict[str, Path] = DATASETS,
    output_dir: Path = OUTPUT_DIR,
    encoded_output_dir: Path = ENCODED_OUTPUT_DIR,
    preferred_target: str = TARGET_COLUMN,
    id_columns: Iterable[str] = ID_COLUMNS,
) -> Dict[str, Dict[str, object]]:
    """Run the requested EDA checks and numeric conversion for every dataset."""
    results = {}
    combined_records = []

    for dataset_name, dataset_path in datasets.items():
        results[dataset_name] = analyze_and_save_numeric_dataset(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            output_dir=output_dir,
            encoded_output_dir=encoded_output_dir,
            preferred_target=preferred_target,
            id_columns=id_columns,
        )

        summary = results[dataset_name]["report"]["summary"]
        combined_records.append(
            {
                "dataset_name": dataset_name,
                "rows": results[dataset_name]["report"]["shape"][0],
                "columns": results[dataset_name]["report"]["shape"][1],
                "target_column": summary["target_column"],
                "problem_type": summary["problem_type"],
                "numeric_column_count": len(summary["numeric_columns"]),
                "categorical_column_count": len(summary["categorical_columns"]),
                "amino_acid_column_count": len(summary["amino_acid_columns"]),
                "missing_column_count": len(summary["missing_columns"]),
                "encoded_dataset_path": str(results[dataset_name]["encoded_dataset_path"]),
                "encoding_map_path": str(results[dataset_name]["encoding_map_path"]),
                "imputation_summary_path": str(
                    results[dataset_name]["imputation_summary_path"]
                ),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    combined_summary = pd.DataFrame(combined_records)
    combined_summary.to_csv(output_dir / "basic_checks_summary.csv", index=False)

    return results


# ---------------------------------------------------------------------------
# 10) COMPLETE DATASET PIPELINE
# ---------------------------------------------------------------------------


def analyze_dataset(
    dataset_name: str,
    dataset_path: Path,
    target_column: str = TARGET_COLUMN,
    id_columns: Iterable[str] = ID_COLUMNS,
    output_dir: Path = OUTPUT_DIR,
) -> Dict[str, object]:
    """Run the full analysis for one dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(dataset_path)

    basic_info = show_basic_information(df, dataset_name, target_column, id_columns)
    numeric_columns = basic_info["numeric_columns"]

    missing_results = analyze_missing_values(df, dataset_name)
    outlier_table = analyze_numeric_outliers(df, dataset_name, numeric_columns)

    boxplot_paths = plot_high_outlier_boxplots(
        df,
        dataset_name,
        outlier_table,
        output_dir,
    )
    heatmap_path = plot_correlation_heatmap(
        df,
        dataset_name,
        numeric_columns,
        output_dir,
    )

    target_corr = calculate_target_correlations(
        df,
        dataset_name,
        numeric_columns,
        target_column,
    )
    high_corr_pairs = find_highly_correlated_pairs(df, dataset_name, numeric_columns)

    selected_columns = select_top_columns_for_group_analysis(
        numeric_columns,
        target_corr,
        outlier_table,
    )
    three_col_groups = analyze_three_column_groups(
        df,
        dataset_name,
        selected_columns,
    )

    biologist_summary = build_biologist_summary(
        df,
        dataset_name,
        missing_results["missing_table"],
        outlier_table,
        target_corr,
        target_column,
        id_columns,
    )

    exported_paths = export_dataset_results(
        dataset_name=dataset_name,
        output_dir=output_dir,
        missing_table=missing_results["missing_table"],
        high_missing_columns=missing_results["high_missing_columns"],
        high_missing_rows=missing_results["high_missing_rows"],
        outlier_table=outlier_table,
        target_corr=target_corr,
        high_corr_pairs=high_corr_pairs,
        three_col_groups=three_col_groups,
        biologist_summary=biologist_summary,
    )

    return {
        "dataframe": df,
        "basic_info": basic_info,
        "missing_results": missing_results,
        "outlier_table": outlier_table,
        "boxplot_paths": boxplot_paths,
        "heatmap_path": heatmap_path,
        "target_correlations": target_corr,
        "highly_correlated_pairs": high_corr_pairs,
        "selected_columns_for_3_column_groups": selected_columns,
        "three_column_groups": three_col_groups,
        "biologist_summary": biologist_summary,
        "exported_paths": exported_paths,
    }


def analyze_all_datasets(
    datasets: Dict[str, Path] = DATASETS,
    target_column: str = TARGET_COLUMN,
    id_columns: Iterable[str] = ID_COLUMNS,
    output_dir: Path = OUTPUT_DIR,
) -> Dict[str, Dict[str, object]]:
    """Run the full analysis for every dataset separately."""
    all_results = {}

    for dataset_name, dataset_path in datasets.items():
        all_results[dataset_name] = analyze_dataset(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            target_column=target_column,
            id_columns=id_columns,
            output_dir=output_dir,
        )

    combined_summary = pd.concat(
        [result["biologist_summary"] for result in all_results.values()],
        ignore_index=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_summary.to_csv(output_dir / "combined_biologist_summary.csv", index=False)

    combined_excel_path = output_dir / "combined_biologist_summary.xlsx"
    try:
        combined_summary.to_excel(combined_excel_path, index=False)
    except ImportError:
        print("openpyxl is not installed; combined Excel summary was not created.")

    return all_results


# ---------------------------------------------------------------------------
# 11) RUN
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    if PLOTTING_AVAILABLE:
        sns.set_theme(style="whitegrid")

    numeric_results = analyze_and_save_all_numeric_datasets()

    # Set this to True if you also want the richer plot/correlation/outlier EDA.
    RUN_FULL_EDA = False
    if RUN_FULL_EDA:
        results = analyze_all_datasets()

    print("\nDone. Basic check reports were written to:")
    print(OUTPUT_DIR / "basic_checks")
    print("\nNumeric encoded datasets were written to:")
    print(ENCODED_OUTPUT_DIR)
