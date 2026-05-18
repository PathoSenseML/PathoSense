"""Schema utilities for the YARISMA_TRAIN_* competition data.

Single source of truth for column partition + sentinel handling. Imported by
both run_pipeline.py and predict.py so the partition can't drift between
training and inference paths.
"""
from typing import Dict, List, Tuple
import pandas as pd

ID_COLS: List[str] = ["Variant_ID"]
TARGET_COL: str = "Label"
CATEGORICAL_COLS: List[str] = [
    "CAT_1", "CAT_2", "CAT_3", "CAT_4", "CAT_5", "CAT_6",
    "AA_1", "AA_2",
]

# Domain sentinels: CAT_3/4/5 use VCF's "./." for no-call; treat as NaN.
_SENTINEL_MAP: Dict[str, List[str]] = {
    "CAT_3": ["./."],
    "CAT_4": ["./."],
    "CAT_5": ["./."],
}


def normalize_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """Replace domain sentinels with NaN, in place. Idempotent."""
    for col, sentinels in _SENTINEL_MAP.items():
        if col in df.columns:
            df[col] = df[col].replace(sentinels, pd.NA)
    return df


def drop_all_nan_rows(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Drop rows whose every feature column is NaN.

    `analysis/row_audit.py` found 124 such rows across the 4 panels
    (general 116, cancer 3, pah 5, cftr 0). They have zero learnable signal —
    the model is shown identical (all-NaN) inputs with mixed labels and forced
    to guess. Both within-panel conflicted duplicate groups in row_audit were
    entirely these rows.

    Should be called at LOAD TIME (after sentinel normalisation, before any
    other transform). Skip during inference: at inference we still want a
    prediction for every input row, even all-NaN ones.
    """
    feat_cols = [c for c in df.columns if c not in ID_COLS and c != TARGET_COL]
    if not feat_cols:
        return df
    all_nan = df[feat_cols].isna().all(axis=1)
    n_drop = int(all_nan.sum())
    if n_drop > 0 and verbose:
        labels = df.loc[all_nan, TARGET_COL].value_counts().to_dict() if TARGET_COL in df.columns else {}
        print(f"  drop_all_nan_rows: removed {n_drop} all-NaN rows (labels={labels})")
    return df[~all_nan].reset_index(drop=True)


def to_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Cast the known categorical columns to pandas Categorical, in place.

    LightGBM consumes pd.Categorical natively (with native NaN handling), so
    no further encoding is needed for v1.
    """
    for col in CATEGORICAL_COLS:
        if col in df.columns and not isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype("category")
    return df


def partition_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str], str]:
    """Partition columns into (id, numeric, categorical, target).

    - id_cols: present-in-df subset of ID_COLS (currently just Variant_ID)
    - numeric_cols: every non-id, non-categorical, non-target column
    - categorical_cols: present-in-df subset of CATEGORICAL_COLS
    - target: TARGET_COL if present, else empty string

    The numeric set is derived by exclusion, so any new numeric prefix added
    by the organisers gets picked up automatically.
    """
    cols = list(df.columns)
    id_cols = [c for c in ID_COLS if c in cols]
    cat_cols = [c for c in CATEGORICAL_COLS if c in cols]
    target = TARGET_COL if TARGET_COL in cols else ""
    excluded = set(id_cols) | set(cat_cols) | ({target} if target else set())
    numeric_cols = [c for c in cols if c not in excluded]
    return id_cols, numeric_cols, cat_cols, target


# ---------------------------------------------------------------------------
# Per-column category classification
#
# Used by preprocess_data_v2 to decide which transform each column gets.
# The rule set is the team's decision tree from
#   Documents/.../Feature_lar/genel özet.xlsx (Kolon Tanıma Rehberi)
# extended with categories for engineered AA_* features.
# ---------------------------------------------------------------------------

# Numeric category labels — what each column is, semantically:
CAT_POPULATION_FREQ = "population_frequency"      # 0-1 right-skew, mean<0.01
CAT_CONSERVATION    = "conservation_signed"       # neg+pos, broad range
CAT_SCORE_CENTERED  = "score_centered"            # smaller signed-score range
CAT_IN_SILICO       = "in_silico_prob"            # 0-1 spread (bimodal or unimodal)
CAT_IN_SILICO_NEAR0 = "in_silico_near_zero"       # 0-1 with mean<0.05, std<0.05
CAT_IN_SILICO_NEAR1 = "in_silico_near_one"        # 0-1 with mean>0.95, std<0.05
CAT_CADD_PHRED      = "in_silico_cadd_phred"      # 0..50 broad range
CAT_BOUNDED_POS     = "bounded_positive_other"    # >1 max, all-positive, not CADD
CAT_AA_DERIVED      = "engineered_aa"             # the AA_* columns we engineer
CAT_CONSTANT        = "constant"                  # drop candidate
CAT_SPARSE          = "sparse"                    # <20 non-null values, drop
CAT_UNKNOWN         = "unknown_numeric"


def classify_numeric_column(col_name: str, values) -> str:
    """Apply the team's decision tree to a single numeric column.

    `values` may be a Series or ndarray; NaN handled.
    """
    import numpy as np

    if str(col_name).startswith("AA_") and col_name not in CATEGORICAL_COLS:
        # Engineered AA features (AA_blosum62, AA_grantham, ..., AA_to_stop)
        return CAT_AA_DERIVED

    v = pd.Series(values).dropna().astype(float).values
    if len(v) < 20:
        return CAT_SPARSE

    mn, mx = float(v.min()), float(v.max())
    mean_v, std_v = float(v.mean()), float(v.std())

    if std_v < 1e-8:
        return CAT_CONSTANT

    # Negative + positive numeric: conservation-like
    if mn < 0 and mx > 0:
        if mx >= 5 and mn <= -3:
            return CAT_CONSERVATION
        return CAT_SCORE_CENTERED

    # 0..50 range with max > 10 - CADD phred-like
    if mn >= 0 and 10 < mx <= 60:
        return CAT_CADD_PHRED

    # Bounded positive but max > 1 and not CADD-like
    if mn >= 0 and mx > 1.001:
        return CAT_BOUNDED_POS

    # 0-1 bounded
    if mn >= 0 and mx <= 1.001:
        frac_lt_001 = float((v < 0.01).mean())
        if mean_v < 0.01 and frac_lt_001 > 0.8:
            return CAT_POPULATION_FREQ
        if std_v < 0.05:
            return CAT_IN_SILICO_NEAR1 if mean_v > 0.95 else CAT_IN_SILICO_NEAR0
        return CAT_IN_SILICO

    return CAT_UNKNOWN


def classify_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Categorize every numeric column in df. Categoricals/IDs are skipped.

    Returns {col_name: category_label} for numeric cols only.
    """
    _, numeric_cols, _, _ = partition_columns(df)
    return {c: classify_numeric_column(c, df[c]) for c in numeric_cols}
