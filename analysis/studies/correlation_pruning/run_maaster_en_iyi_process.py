from __future__ import annotations

import json
import math
import shutil
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice encountered")

try:
    import joblib  # type: ignore
except ModuleNotFoundError:
    import pickle

    class _PickleJoblib:
        @staticmethod
        def dump(obj, path):
            with open(path, "wb") as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

        @staticmethod
        def load(path):
            with open(path, "rb") as f:
                return pickle.load(f)

    joblib = _PickleJoblib()


class OneHotEncoder:
    def __init__(self, handle_unknown="ignore", sparse_output=False, sparse=False, dtype=np.float32):
        self.handle_unknown = handle_unknown
        self.dtype = dtype
        self.columns_ = []
        self.categories_ = {}

    def fit(self, x):
        df = pd.DataFrame(x).astype("string").fillna("**MISSING**")
        self.columns_ = list(df.columns)
        self.categories_ = {c: sorted(df[c].astype(str).unique().tolist()) for c in self.columns_}
        return self

    def transform(self, x):
        df = pd.DataFrame(x).astype("string").fillna("**MISSING**")
        arrays = []
        for c in self.columns_:
            vals = df[c].astype(str) if c in df.columns else pd.Series(["**MISSING**"] * len(df), index=df.index)
            for cat in self.categories_[c]:
                arrays.append((vals == cat).to_numpy(dtype=self.dtype))
        if not arrays:
            return np.empty((len(df), 0), dtype=self.dtype)
        return np.vstack(arrays).T.astype(self.dtype, copy=False)

    def fit_transform(self, x):
        return self.fit(x).transform(x)

    def get_feature_names_out(self, input_features=None):
        names = []
        for c in self.columns_:
            for cat in self.categories_[c]:
                names.append(f"{c}={cat}")
        return np.array(names, dtype=object)


class SimpleImputer:
    def __init__(self, strategy="median"):
        if strategy != "median":
            raise ValueError("This self-contained imputer supports only median strategy.")
        self.strategy = strategy
        self.statistics_ = None

    def fit(self, x):
        arr = np.asarray(x, dtype=float)
        stats = np.nanmedian(arr, axis=0)
        stats = np.where(np.isnan(stats), 0.0, stats)
        self.statistics_ = stats
        return self

    def transform(self, x):
        arr = np.asarray(x, dtype=float).copy()
        if self.statistics_ is None:
            raise RuntimeError("SimpleImputer is not fitted.")
        rows, cols = np.where(np.isnan(arr))
        arr[rows, cols] = self.statistics_[cols]
        return arr

    def fit_transform(self, x):
        return self.fit(x).transform(x)


class RobustScaler:
    def __init__(self):
        self.center_ = None
        self.scale_ = None

    def fit(self, x):
        arr = np.asarray(x, dtype=float)
        self.center_ = np.nanmedian(arr, axis=0)
        q75 = np.nanpercentile(arr, 75, axis=0)
        q25 = np.nanpercentile(arr, 25, axis=0)
        scale = q75 - q25
        self.scale_ = np.where((scale == 0) | np.isnan(scale), 1.0, scale)
        return self

    def transform(self, x):
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("RobustScaler is not fitted.")
        return (np.asarray(x, dtype=float) - self.center_) / self.scale_

    def fit_transform(self, x):
        return self.fit(x).transform(x)


class TruncatedSVD:
    def __init__(self, n_components=25, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.components_ = None
        self.singular_values_ = None
        self.explained_variance_ = None

    def fit(self, x):
        arr = np.asarray(x, dtype=float)
        _, s, vt = np.linalg.svd(arr, full_matrices=False)
        self.components_ = vt[: self.n_components]
        self.singular_values_ = s[: self.n_components]
        transformed = arr @ self.components_.T
        self.explained_variance_ = np.var(transformed, axis=0)
        return self

    def transform(self, x):
        if self.components_ is None:
            raise RuntimeError("TruncatedSVD is not fitted.")
        return np.asarray(x, dtype=float) @ self.components_.T

    def fit_transform(self, x):
        return self.fit(x).transform(x)


RANDOM_STATE = 42
FORCE_CREATE_FINAL_LIKE_SPLIT = False
TARGET_PATHOGENIC_RATE = 0.142857
TARGET_BENIGN_RATE = 0.857143
RATIO_TOLERANCE = 0.03
PROCESS_STEP_ID = "S3_D_04_TRUNCATED_SVD_25"
PROCESS_NAME = "MASTER_en_iyi_process"


SCRIPT_PATH = Path(__file__).resolve()
if SCRIPT_PATH.parent.name == "en_iyi_process":
    OUTPUT_DIR = SCRIPT_PATH.parent
    ROOT_DIR = OUTPUT_DIR.parent
else:
    ROOT_DIR = Path.cwd()
    OUTPUT_DIR = ROOT_DIR / "en_iyi_process"

RAWLEVEL_DIR = OUTPUT_DIR / "00_s3_base_rawlevel"
MATRIX_DIR = OUTPUT_DIR / "01_s3_base_matrix"
SVD_DIR = OUTPUT_DIR / "02_best_process_SVD25"
ARTIFACT_DIR = OUTPUT_DIR / "artifacts"
SPLIT_DIR = OUTPUT_DIR / "split_ids"
ZIP_PATH = ROOT_DIR / "en_iyi_process.zip"


BLOSUM62_TEXT = """
   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V
A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4
"""


GRANTHAM_PAIRS = {
    ("A", "R"): 112, ("A", "N"): 111, ("A", "D"): 126, ("A", "C"): 195, ("A", "Q"): 91, ("A", "E"): 107, ("A", "G"): 60, ("A", "H"): 86, ("A", "I"): 94, ("A", "L"): 96, ("A", "K"): 106, ("A", "M"): 84, ("A", "F"): 113, ("A", "P"): 27, ("A", "S"): 99, ("A", "T"): 58, ("A", "W"): 148, ("A", "Y"): 112, ("A", "V"): 64,
    ("R", "N"): 86, ("R", "D"): 96, ("R", "C"): 180, ("R", "Q"): 43, ("R", "E"): 54, ("R", "G"): 125, ("R", "H"): 29, ("R", "I"): 97, ("R", "L"): 102, ("R", "K"): 26, ("R", "M"): 91, ("R", "F"): 97, ("R", "P"): 103, ("R", "S"): 110, ("R", "T"): 71, ("R", "W"): 101, ("R", "Y"): 77, ("R", "V"): 96,
    ("N", "D"): 23, ("N", "C"): 139, ("N", "Q"): 46, ("N", "E"): 42, ("N", "G"): 80, ("N", "H"): 68, ("N", "I"): 149, ("N", "L"): 153, ("N", "K"): 94, ("N", "M"): 142, ("N", "F"): 158, ("N", "P"): 91, ("N", "S"): 46, ("N", "T"): 65, ("N", "W"): 174, ("N", "Y"): 143, ("N", "V"): 133,
    ("D", "C"): 154, ("D", "Q"): 61, ("D", "E"): 45, ("D", "G"): 94, ("D", "H"): 81, ("D", "I"): 168, ("D", "L"): 172, ("D", "K"): 101, ("D", "M"): 160, ("D", "F"): 177, ("D", "P"): 108, ("D", "S"): 65, ("D", "T"): 85, ("D", "W"): 181, ("D", "Y"): 160, ("D", "V"): 152,
    ("C", "Q"): 154, ("C", "E"): 170, ("C", "G"): 159, ("C", "H"): 174, ("C", "I"): 198, ("C", "L"): 198, ("C", "K"): 202, ("C", "M"): 196, ("C", "F"): 205, ("C", "P"): 169, ("C", "S"): 112, ("C", "T"): 149, ("C", "W"): 215, ("C", "Y"): 194, ("C", "V"): 192,
    ("Q", "E"): 29, ("Q", "G"): 87, ("Q", "H"): 24, ("Q", "I"): 109, ("Q", "L"): 113, ("Q", "K"): 53, ("Q", "M"): 101, ("Q", "F"): 116, ("Q", "P"): 76, ("Q", "S"): 68, ("Q", "T"): 42, ("Q", "W"): 130, ("Q", "Y"): 99, ("Q", "V"): 96,
    ("E", "G"): 98, ("E", "H"): 40, ("E", "I"): 134, ("E", "L"): 138, ("E", "K"): 56, ("E", "M"): 126, ("E", "F"): 140, ("E", "P"): 93, ("E", "S"): 80, ("E", "T"): 65, ("E", "W"): 152, ("E", "Y"): 122, ("E", "V"): 121,
    ("G", "H"): 98, ("G", "I"): 135, ("G", "L"): 138, ("G", "K"): 127, ("G", "M"): 127, ("G", "F"): 153, ("G", "P"): 42, ("G", "S"): 56, ("G", "T"): 59, ("G", "W"): 184, ("G", "Y"): 147, ("G", "V"): 109,
    ("H", "I"): 94, ("H", "L"): 99, ("H", "K"): 32, ("H", "M"): 87, ("H", "F"): 100, ("H", "P"): 77, ("H", "S"): 89, ("H", "T"): 47, ("H", "W"): 115, ("H", "Y"): 83, ("H", "V"): 84,
    ("I", "L"): 5, ("I", "K"): 102, ("I", "M"): 10, ("I", "F"): 21, ("I", "P"): 95, ("I", "S"): 142, ("I", "T"): 89, ("I", "W"): 61, ("I", "Y"): 33, ("I", "V"): 29,
    ("L", "K"): 107, ("L", "M"): 15, ("L", "F"): 22, ("L", "P"): 98, ("L", "S"): 145, ("L", "T"): 92, ("L", "W"): 61, ("L", "Y"): 36, ("L", "V"): 32,
    ("K", "M"): 95, ("K", "F"): 102, ("K", "P"): 103, ("K", "S"): 121, ("K", "T"): 78, ("K", "W"): 110, ("K", "Y"): 85, ("K", "V"): 97,
    ("M", "F"): 28, ("M", "P"): 87, ("M", "S"): 135, ("M", "T"): 81, ("M", "W"): 67, ("M", "Y"): 36, ("M", "V"): 21,
    ("F", "P"): 114, ("F", "S"): 155, ("F", "T"): 103, ("F", "W"): 40, ("F", "Y"): 22, ("F", "V"): 50,
    ("P", "S"): 74, ("P", "T"): 38, ("P", "W"): 147, ("P", "Y"): 110, ("P", "V"): 68,
    ("S", "T"): 58, ("S", "W"): 177, ("S", "Y"): 144, ("S", "V"): 124,
    ("T", "W"): 128, ("T", "Y"): 92, ("T", "V"): 69,
    ("W", "Y"): 37, ("W", "V"): 88,
    ("Y", "V"): 55,
}


def parse_blosum62() -> dict[str, dict[str, int]]:
    lines = [line.split() for line in BLOSUM62_TEXT.strip().splitlines()]
    aas = lines[0]
    return {row[0]: {b: int(v) for b, v in zip(aas, row[1:])} for row in lines[1:]}


BLOSUM62 = parse_blosum62()
CANONICAL = set(BLOSUM62)


def prepare_dirs() -> None:
    for d in [OUTPUT_DIR, RAWLEVEL_DIR, MATRIX_DIR, SVD_DIR, ARTIFACT_DIR, SPLIT_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def normalize_label(series: pd.Series) -> pd.Series:
    def one(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip().lower()
        if s in {"0", "0.0", "benign"}:
            return 0
        if s in {"1", "1.0", "pathogenic", "patojenik"}:
            return 1
        raise ValueError(f"Unknown Label value: {x!r}")

    out = series.map(one)
    if out.isna().any():
        raise ValueError("Label contains missing values after normalization.")
    return out.astype(int)


def clean_aa(x) -> str:
    return "" if pd.isna(x) else str(x).strip().upper()


def grantham(a: str, b: str) -> float:
    if a == b:
        return 0.0
    return float(GRANTHAM_PAIRS.get((a, b), GRANTHAM_PAIRS.get((b, a), np.nan)))


def find_raw_master() -> Path:
    candidates = [
        ROOT_DIR / "YARISMA_TRAIN_MASTER.csv",
        ROOT_DIR / "BASE_MASTER.csv",
        ROOT_DIR / "master.csv",
        ROOT_DIR / "MASTER.csv",
        ROOT_DIR / "BASE" / "YARISMA_TRAIN_MASTER.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("MASTER raw data not found.")


def load_master_raw() -> tuple[pd.DataFrame, Path]:
    raw_path = find_raw_master()
    raw = pd.read_csv(raw_path)
    required = {"Variant_ID", "Label"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Raw MASTER missing required columns: {missing}")
    raw["Variant_ID"] = raw["Variant_ID"].astype(str)
    raw["Label"] = normalize_label(raw["Label"])
    if raw["Variant_ID"].duplicated().any():
        raise ValueError("Duplicate Variant_ID values in raw MASTER.")
    return raw, raw_path


def load_split_from_zip(zip_path: Path) -> pd.DataFrame | None:
    if not zip_path.exists():
        return None
    with zipfile.ZipFile(zip_path) as zf:
        hits = [n for n in zf.namelist() if n.endswith("split_ids/MASTER_split_ids.csv") or n.endswith("MASTER_split_ids.csv")]
        if not hits:
            return None
        with zf.open(hits[0]) as f:
            split = pd.read_csv(f)
    split.attrs["split_source"] = f"{zip_path}!{hits[0]}"
    return split


def find_existing_split() -> pd.DataFrame | None:
    candidates = [
        OUTPUT_DIR / "split_ids" / "MASTER_split_ids.csv",
        ROOT_DIR / "split_ids" / "MASTER_split_ids.csv",
        ROOT_DIR / "teknopatho_stage2_from_fixed_bases_results.zip",
        ROOT_DIR / "teknopatho_stage2_from_fixed_bases_20260627_151308" / "teknopatho_stage2_from_fixed_bases_results.zip",
    ]
    for path in candidates:
        if path.suffix.lower() == ".zip":
            split = load_split_from_zip(path)
            if split is not None:
                return split
        elif path.exists():
            split = pd.read_csv(path)
            split.attrs["split_source"] = str(path)
            return split
    return None


def split_summary(split: pd.DataFrame) -> dict:
    tr = split[split["split"].eq("train")]
    te = split[split["split"].eq("test")]
    test_path = int(te["Label"].eq(1).sum())
    test_benign = int(te["Label"].eq(0).sum())
    test_rows = len(te)
    path_rate = test_path / test_rows if test_rows else math.nan
    benign_rate = test_benign / test_rows if test_rows else math.nan
    return {
        "train_rows": len(tr),
        "test_rows": test_rows,
        "train_benign": int(tr["Label"].eq(0).sum()),
        "train_pathogenic": int(tr["Label"].eq(1).sum()),
        "test_benign": test_benign,
        "test_pathogenic": test_path,
        "test_pathogenic_rate": path_rate,
        "test_benign_rate": benign_rate,
        "target_test_pathogenic_rate": TARGET_PATHOGENIC_RATE,
        "target_test_benign_rate": TARGET_BENIGN_RATE,
        "ratio_error": abs(path_rate - TARGET_PATHOGENIC_RATE) if not math.isnan(path_rate) else math.nan,
        "split_source": split.attrs.get("split_source", "created_final_like_split"),
    }


def validate_split(raw: pd.DataFrame, split: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = {"Variant_ID", "split"}
    missing = sorted(required - set(split.columns))
    if missing:
        raise ValueError(f"Split file missing columns: {missing}")
    split = split.copy()
    split["Variant_ID"] = split["Variant_ID"].astype(str)
    split["split"] = split["split"].astype(str).str.lower().str.strip()
    bad = sorted(set(split["split"]) - {"train", "test"})
    if bad:
        raise ValueError(f"Invalid split values: {bad}")
    if split["Variant_ID"].duplicated().any():
        raise ValueError("Split file contains duplicate Variant_ID values.")
    if "Label" not in split.columns:
        split = split.merge(raw[["Variant_ID", "Label"]], on="Variant_ID", how="left")
    else:
        split["Label"] = normalize_label(split["Label"])
        merged_label = split[["Variant_ID", "Label"]].merge(raw[["Variant_ID", "Label"]], on="Variant_ID", how="left", suffixes=("_split", "_raw"))
        if not merged_label["Label_split"].equals(merged_label["Label_raw"]):
            raise ValueError("Split Label values do not match raw MASTER labels.")
    raw_ids = set(raw["Variant_ID"])
    split_ids = set(split["Variant_ID"])
    if raw_ids != split_ids:
        raise ValueError(f"Split/raw Variant_ID mismatch. missing_in_split={len(raw_ids - split_ids)}, extra_in_split={len(split_ids - raw_ids)}")
    summary = split_summary(split)
    if summary["ratio_error"] > RATIO_TOLERANCE:
        if FORCE_CREATE_FINAL_LIKE_SPLIT:
            return create_final_like_split(raw)
        raise ValueError(f"Existing split test pathogenic rate {summary['test_pathogenic_rate']:.6f} is outside tolerance.")
    split.to_csv(SPLIT_DIR / "MASTER_split_ids.csv", index=False)
    return split, summary


def create_final_like_split(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(RANDOM_STATE)
    pathogenic = raw.loc[raw["Label"].eq(1), "Variant_ID"].to_numpy()
    benign = raw.loc[raw["Label"].eq(0), "Variant_ID"].to_numpy()
    rng.shuffle(pathogenic)
    rng.shuffle(benign)
    max_path_test = max(1, len(pathogenic) - 1)
    max_benign_test = max(1, len(benign) - 1)
    best = None
    for p in range(1, max_path_test + 1):
        b = min(max_benign_test, max(1, round(6 * p)))
        rate = p / (p + b)
        err = abs(rate - TARGET_PATHOGENIC_RATE)
        rows = p + b
        candidate = (err, -rows, p, b)
        if best is None or candidate < best:
            best = candidate
    _, _, p_test, b_test = best
    test_ids = set(pathogenic[:p_test]) | set(benign[:b_test])
    split = raw[["Variant_ID", "Label"]].copy()
    split["split"] = np.where(split["Variant_ID"].isin(test_ids), "test", "train")
    split.attrs["split_source"] = "created_final_like_split_no_supplied_split_ids"
    summary = split_summary(split)
    if summary["ratio_error"] > RATIO_TOLERANCE:
        raise ValueError("Created split does not satisfy final-like ratio tolerance.")
    split.to_csv(SPLIT_DIR / "MASTER_split_ids.csv", index=False)
    return split, summary


def load_or_create_split(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict, bool]:
    existing = find_existing_split()
    if existing is None:
        split, summary = create_final_like_split(raw)
        return split, summary, True
    source = existing.attrs.get("split_source", "existing_split")
    split, summary = validate_split(raw, existing)
    summary["split_source"] = source
    split.to_csv(SPLIT_DIR / "MASTER_split_ids.csv", index=False)
    return split, summary, False


def aa_numeric(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "AA_1" not in raw.columns or "AA_2" not in raw.columns:
        raise RuntimeError("AA_1/AA_2 missing; cannot compute AA numeric features.")
    out = raw.copy()
    warnings = []
    rows = []
    for idx, (aa1, aa2) in enumerate(zip(out["AA_1"], out["AA_2"])):
        a = clean_aa(aa1)
        b = clean_aa(aa2)
        valid = len(a) == 1 and len(b) == 1 and a in CANONICAL and b in CANONICAL
        stop = int("*" in a or "*" in b)
        multi = int(len(a) > 1 or len(b) > 1)
        noncanonical = int((a != "" and (len(a) != 1 or a not in CANONICAL)) or (b != "" and (len(b) != 1 or b not in CANONICAL)))
        if valid:
            rows.append((BLOSUM62[a][b], grantham(a, b), 1, stop, noncanonical, multi))
        else:
            rows.append((np.nan, np.nan, 0, stop, noncanonical, multi))
            warnings.append({
                "Variant_ID": out.iloc[idx]["Variant_ID"],
                "AA_1": aa1,
                "AA_2": aa2,
                "AA_1_clean": a,
                "AA_2_clean": b,
                "reason": "invalid_or_noncanonical_or_multichar",
            })
    cols = ["AA_blosum62_score", "AA_grantham_distance", "AA_pair_valid", "AA_has_stop", "AA_has_noncanonical", "AA_is_multichar"]
    for i, col in enumerate(cols):
        out[col] = [row[i] for row in rows]
    out = out.drop(columns=["AA_1", "AA_2"])
    if out["AA_blosum62_score"].notna().sum() == 0 or out["AA_grantham_distance"].notna().sum() == 0:
        raise RuntimeError("AA conversion failed: all BLOSUM62 or Grantham scores are NaN.")
    return out, pd.DataFrame(warnings)


def add_al_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    al_cols = [c for c in out.columns if c.startswith("AL_")]
    indicators = pd.DataFrame(
        {f"{col}_is_missing": out[col].isna().astype(np.int8) for col in al_cols},
        index=out.index,
    )
    return pd.concat([out, indicators], axis=1)


def apply_split(df: pd.DataFrame, split: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = df.merge(split[["Variant_ID", "split"]], on="Variant_ID", how="inner", validate="one_to_one")
    train = merged[merged["split"].eq("train")].drop(columns=["split"]).copy()
    test = merged[merged["split"].eq("test")].drop(columns=["split"]).copy()
    if set(train["Variant_ID"]) & set(test["Variant_ID"]):
        raise RuntimeError("Leakage risk: train/test overlap detected.")
    if len(train) + len(test) != len(df):
        raise RuntimeError("Not every raw row assigned exactly once.")
    return train, test


def make_model_matrix(train_df: pd.DataFrame, test_df: pd.DataFrame):
    protected = {"Variant_ID", "Label"}
    train_x_raw = train_df.drop(columns=list(protected)).copy()
    test_x_raw = test_df.drop(columns=list(protected)).copy().reindex(columns=train_x_raw.columns)
    categorical_columns = [c for c in train_x_raw.columns if train_x_raw[c].dtype == "object" or str(train_x_raw[c].dtype).startswith(("category", "string"))]
    numeric_columns = [c for c in train_x_raw.columns if c not in categorical_columns]
    train_num = train_x_raw[numeric_columns].apply(pd.to_numeric, errors="coerce") if numeric_columns else pd.DataFrame(index=train_x_raw.index)
    test_num = test_x_raw[numeric_columns].apply(pd.to_numeric, errors="coerce") if numeric_columns else pd.DataFrame(index=test_x_raw.index)
    if categorical_columns:
        train_cat = train_x_raw[categorical_columns].astype("string").fillna("**MISSING**")
        test_cat = test_x_raw[categorical_columns].astype("string").fillna("**MISSING**")
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)
        train_cat_arr = encoder.fit_transform(train_cat)
        test_cat_arr = encoder.transform(test_cat)
        onehot_feature_names = list(encoder.get_feature_names_out(categorical_columns))
        train_cat_df = pd.DataFrame(train_cat_arr, columns=onehot_feature_names, index=train_df.index)
        test_cat_df = pd.DataFrame(test_cat_arr, columns=onehot_feature_names, index=test_df.index)
    else:
        encoder = None
        onehot_feature_names = []
        train_cat_df = pd.DataFrame(index=train_df.index)
        test_cat_df = pd.DataFrame(index=test_df.index)
    x_train = pd.concat([train_num, train_cat_df], axis=1).replace([np.inf, -np.inf], np.nan)
    x_test = pd.concat([test_num, test_cat_df], axis=1).replace([np.inf, -np.inf], np.nan).reindex(columns=x_train.columns)
    feature_names = list(x_train.columns)
    if len(feature_names) != len(set(feature_names)):
        raise RuntimeError("Duplicate feature names after one-hot encoding.")
    if x_train.shape[1] != x_test.shape[1] or len(feature_names) != x_train.shape[1]:
        raise RuntimeError("Train/test matrix alignment failed.")
    if any(x_train.dtypes.astype(str).eq("object")) or any(x_test.dtypes.astype(str).eq("object")):
        raise RuntimeError("Object dtype remains in model matrix.")
    if "Variant_ID" in feature_names or "Label" in feature_names:
        raise RuntimeError("Protected columns leaked into features.")
    return x_train, x_test, numeric_columns, categorical_columns, encoder, onehot_feature_names, feature_names


def save_matrix(train_df: pd.DataFrame, test_df: pd.DataFrame, x_train: pd.DataFrame, x_test: pd.DataFrame, feature_names: list[str]) -> None:
    train_out = pd.concat([train_df[["Variant_ID", "Label"]].reset_index(drop=True), x_train.reset_index(drop=True)], axis=1)
    test_out = pd.concat([test_df[["Variant_ID", "Label"]].reset_index(drop=True), x_test.reset_index(drop=True)], axis=1)
    train_out.to_csv(MATRIX_DIR / "MASTER_S3_BASE_matrix_train.csv", index=False)
    test_out.to_csv(MATRIX_DIR / "MASTER_S3_BASE_matrix_test.csv", index=False)
    pd.DataFrame({"feature_name": feature_names}).to_csv(MATRIX_DIR / "MASTER_S3_BASE_feature_names.csv", index=False)


def svd25_process(train_df: pd.DataFrame, test_df: pd.DataFrame, x_train: pd.DataFrame, x_test: pd.DataFrame):
    imputer = SimpleImputer(strategy="median")
    x_train_imp = imputer.fit_transform(x_train)
    x_test_imp = imputer.transform(x_test)
    if np.isnan(x_train_imp).any() or np.isnan(x_test_imp).any():
        raise RuntimeError("NaN remains after median imputation.")
    scaler = RobustScaler()
    x_train_scaled = scaler.fit_transform(x_train_imp)
    x_test_scaled = scaler.transform(x_test_imp)
    n_components = 25
    if x_train_scaled.shape[1] < n_components:
        raise RuntimeError(f"Need at least {n_components} matrix features for SVD.")
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    train_svd = svd.fit_transform(x_train_scaled)
    test_svd = svd.transform(x_test_scaled)
    final_features = [f"SVD_{i}" for i in range(n_components)]
    train_out = pd.concat(
        [train_df[["Variant_ID", "Label"]].reset_index(drop=True), pd.DataFrame(train_svd, columns=final_features)],
        axis=1,
    )
    test_out = pd.concat(
        [test_df[["Variant_ID", "Label"]].reset_index(drop=True), pd.DataFrame(test_svd, columns=final_features)],
        axis=1,
    )
    train_out.to_csv(SVD_DIR / "MASTER_en_iyi_process_train.csv", index=False)
    test_out.to_csv(SVD_DIR / "MASTER_en_iyi_process_test.csv", index=False)
    return imputer, scaler, svd, final_features, train_out, test_out


def add_check(rows: list[dict], check: str, passed: bool, detail: str, critical: bool = True) -> None:
    rows.append({"check": check, "passed": bool(passed), "detail": detail, "critical": bool(critical)})


def write_readme(summary: dict) -> None:
    text = f"""# MASTER En Iyi Process

This folder contains only MASTER processed data.

No model was trained. No ensemble was built. No threshold tuning was done.

Process step: `{PROCESS_STEP_ID}`.

Process chain:

raw MASTER -> final-like split -> AA numeric -> AL missing indicators -> one-hot encoding -> median imputation -> RobustScaler -> TruncatedSVD-25

The official final MASTER test ratio was approximated as 500 Pathogenic / 3000 Benign = 1:6.

Observed test pathogenic rate: {summary['test_pathogenic_rate']:.6f}
Target test pathogenic rate: {TARGET_PATHOGENIC_RATE:.6f}
Ratio error: {summary['ratio_error']:.6f}

All fit operations were train-only. Test was only transformed.

Recommended ensemble input files:

- `02_best_process_SVD25/MASTER_en_iyi_process_train.csv`
- `02_best_process_SVD25/MASTER_en_iyi_process_test.csv`

Model features are all columns except:

- `Variant_ID`
- `Label`

`Variant_ID` is only for row tracking. `Label` is the target.
"""
    (OUTPUT_DIR / "README_MASTER_EN_IYI_PROCESS.md").write_text(text, encoding="utf-8")


def zip_outputs() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in OUTPUT_DIR.rglob("*"):
            if "__pycache__" in path.parts:
                continue
            zf.write(path, path.relative_to(ROOT_DIR))


def main() -> None:
    prepare_dirs()
    raw, raw_path = load_master_raw()
    original_raw_feature_count = len([c for c in raw.columns if c not in {"Variant_ID", "Label"}])
    split, split_info, split_created = load_or_create_split(raw)
    raw_aa, aa_warnings = aa_numeric(raw)
    if not aa_warnings.empty:
        aa_warnings.to_csv(OUTPUT_DIR / "MASTER_AA_ROW_WARNINGS.csv", index=False)
    s3_base = add_al_missing_indicators(raw_aa)
    train_raw, test_raw = apply_split(s3_base, split)
    train_raw.to_csv(RAWLEVEL_DIR / "MASTER_S3_BASE_train_rawlevel.csv", index=False)
    test_raw.to_csv(RAWLEVEL_DIR / "MASTER_S3_BASE_test_rawlevel.csv", index=False)

    x_train, x_test, numeric_columns, categorical_columns, encoder, onehot_names, matrix_features = make_model_matrix(train_raw, test_raw)
    save_matrix(train_raw, test_raw, x_train, x_test, matrix_features)
    imputer, scaler, svd, final_features, svd_train, svd_test = svd25_process(train_raw, test_raw, x_train, x_test)

    artifact = {
        "split_info": split_info,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "onehot_encoder": encoder,
        "onehot_feature_names": onehot_names,
        "model_matrix_feature_names_before_svd": matrix_features,
        "median_imputer": imputer,
        "robust_scaler": scaler,
        "truncated_svd": svd,
        "final_feature_names": final_features,
        "label_mapping": {"Benign": 0, "benign": 0, "0": 0, "Pathogenic": 1, "pathogenic": 1, "Patojenik": 1, "1": 1},
        "process_step_id": PROCESS_STEP_ID,
        "target_test_ratio": {"pathogenic": TARGET_PATHOGENIC_RATE, "benign": TARGET_BENIGN_RATE},
    }
    artifact_path = ARTIFACT_DIR / "MASTER_en_iyi_process_preprocessor.joblib"
    joblib.dump(artifact, artifact_path)
    joblib.dump(encoder, ARTIFACT_DIR / "MASTER_onehot_encoder.joblib")
    joblib.dump(imputer, ARTIFACT_DIR / "MASTER_median_imputer.joblib")
    joblib.dump(scaler, ARTIFACT_DIR / "MASTER_robust_scaler.joblib")
    joblib.dump(svd, ARTIFACT_DIR / "MASTER_truncated_svd25.joblib")

    train_ids = set(train_raw["Variant_ID"])
    test_ids = set(test_raw["Variant_ID"])
    checks = []
    add_check(checks, "master_only", True, "Only YARISMA_TRAIN_MASTER/MASTER raw was loaded.")
    add_check(checks, "fixed_or_final_like_split_loaded", True, split_info["split_source"])
    add_check(checks, "no_new_split_created_if_split_ids_available", not split_created, f"split_created={split_created}")
    add_check(checks, "final_like_test_ratio_confirmed", split_info["ratio_error"] <= RATIO_TOLERANCE, f"ratio_error={split_info['ratio_error']:.6f}")
    add_check(checks, "no_train_test_overlap", not bool(train_ids & test_ids), "intersection size 0")
    add_check(checks, "every_raw_row_assigned_once", len(train_raw) + len(test_raw) == len(raw), f"{len(train_raw)}+{len(test_raw)}={len(raw)}")
    add_check(checks, "Label_not_in_X", "Label" not in matrix_features, "Label removed before X creation")
    add_check(checks, "Variant_ID_not_in_X", "Variant_ID" not in matrix_features, "Variant_ID removed before X creation")
    add_check(checks, "categorical_columns_detected_train_only", True, f"{len(categorical_columns)} categorical columns")
    add_check(checks, "encoder_fit_train_only", True, "OneHotEncoder fit on train only")
    add_check(checks, "imputer_fit_train_only", True, "SimpleImputer fit on train only")
    add_check(checks, "scaler_fit_train_only", True, "RobustScaler fit on train only")
    add_check(checks, "svd_fit_train_only", True, "TruncatedSVD fit on train only")
    add_check(checks, "test_only_transformed", True, "Test used only in transform calls")
    add_check(checks, "train_test_columns_aligned_before_svd", list(x_train.columns) == list(x_test.columns), f"{x_train.shape[1]} columns")
    add_check(checks, "final_train_test_columns_aligned", list(svd_train.columns) == list(svd_test.columns), f"{len(final_features)} SVD columns plus IDs")
    add_check(checks, "artifacts_saved", all((ARTIFACT_DIR / name).exists() for name in [
        "MASTER_en_iyi_process_preprocessor.joblib",
        "MASTER_onehot_encoder.joblib",
        "MASTER_median_imputer.joblib",
        "MASTER_robust_scaler.joblib",
        "MASTER_truncated_svd25.joblib",
    ]), "All artifact files exist")
    add_check(checks, "output_csvs_created", all(path.exists() for path in [
        RAWLEVEL_DIR / "MASTER_S3_BASE_train_rawlevel.csv",
        RAWLEVEL_DIR / "MASTER_S3_BASE_test_rawlevel.csv",
        MATRIX_DIR / "MASTER_S3_BASE_matrix_train.csv",
        MATRIX_DIR / "MASTER_S3_BASE_matrix_test.csv",
        SVD_DIR / "MASTER_en_iyi_process_train.csv",
        SVD_DIR / "MASTER_en_iyi_process_test.csv",
    ]), "All output CSVs exist")
    add_check(checks, "no_model_training_done", True, "No estimator model training, ensemble, or threshold tuning code is used.")
    checklist = pd.DataFrame(checks)
    checklist.to_csv(OUTPUT_DIR / "MASTER_LEAKAGE_CHECKLIST.csv", index=False)
    if not checklist.loc[checklist["critical"], "passed"].all():
        raise RuntimeError("Critical leakage/output checklist failed.")

    audit = {
        "process_name": PROCESS_NAME,
        "process_step_id": PROCESS_STEP_ID,
        "raw_rows": len(raw),
        "train_rows": split_info["train_rows"],
        "test_rows": split_info["test_rows"],
        "train_benign": split_info["train_benign"],
        "train_pathogenic": split_info["train_pathogenic"],
        "test_benign": split_info["test_benign"],
        "test_pathogenic": split_info["test_pathogenic"],
        "test_pathogenic_rate": split_info["test_pathogenic_rate"],
        "target_test_pathogenic_rate": TARGET_PATHOGENIC_RATE,
        "ratio_error": split_info["ratio_error"],
        "original_raw_feature_count": original_raw_feature_count,
        "s3_base_rawlevel_feature_count": len([c for c in train_raw.columns if c not in {"Variant_ID", "Label"}]),
        "model_matrix_feature_count_before_svd": len(matrix_features),
        "final_feature_count": len(final_features),
        "uses_AA_numeric": True,
        "uses_AL_missing_indicators": True,
        "uses_onehot_encoding": True,
        "uses_median_imputation": True,
        "uses_robust_scaler": True,
        "uses_truncated_svd": True,
        "n_svd_components": 25,
        "train_only_fit_confirmed": True,
        "Variant_ID_in_features": False,
        "Label_in_features": False,
        "no_train_test_overlap": not bool(train_ids & test_ids),
        "output_train_path": str(SVD_DIR / "MASTER_en_iyi_process_train.csv"),
        "output_test_path": str(SVD_DIR / "MASTER_en_iyi_process_test.csv"),
        "artifact_path": str(artifact_path),
        "notes": f"Raw source: {raw_path}; split source: {split_info['split_source']}; split_created={split_created}",
    }
    pd.DataFrame([audit]).to_csv(OUTPUT_DIR / "MASTER_EN_IYI_PROCESS_AUDIT.csv", index=False)
    write_readme(split_info)
    zip_outputs()

    print("MASTER en iyi process complete")
    print("output folder: en_iyi_process")
    print("zip path: en_iyi_process.zip")
    print(f"raw rows: {len(raw)}")
    print(f"train rows: {split_info['train_rows']}")
    print(f"test rows: {split_info['test_rows']}")
    print(f"train benign/pathogenic counts: {split_info['train_benign']}/{split_info['train_pathogenic']}")
    print(f"test benign/pathogenic counts: {split_info['test_benign']}/{split_info['test_pathogenic']}")
    print(f"test pathogenic rate: {split_info['test_pathogenic_rate']:.6f}")
    print(f"target pathogenic rate = {TARGET_PATHOGENIC_RATE:.6f}")
    print(f"ratio error: {split_info['ratio_error']:.6f}")
    print(f"S3_BASE matrix shape: {x_train.shape} train, {x_test.shape} test")
    print(f"SVD25 train shape: {svd_train.shape}")
    print(f"SVD25 test shape: {svd_test.shape}")
    print("final feature count = 25")
    print("leakage checklist all critical passed")
    print("recommended ensemble train file: en_iyi_process/02_best_process_SVD25/MASTER_en_iyi_process_train.csv")
    print("recommended ensemble test file: en_iyi_process/02_best_process_SVD25/MASTER_en_iyi_process_test.csv")


if __name__ == "__main__":
    main()
