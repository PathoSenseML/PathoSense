"""
Convert PHACTboost RData files to CSV.

Usage:
    python convert_rdata.py

Expects TrainingSet.RData and/or TestSet.RData in data/reference/.
Outputs TrainingSet_full.csv and TestSet_full.csv in the same directory.

Note: pyreadr fails on these files. The rdata package with a custom
constructor is required to handle 2D arrays in the R data frames.
"""

import os
import sys
import numpy as np
import pandas as pd
import rdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import BASE_DIR

REF_DIR = os.path.join(BASE_DIR, "data", "reference")

FILES = {
    "TrainingSet.RData": "TrainingSet_full.csv",
    "TestSet.RData": "TestSet_full.csv",
}


def _dataframe_constructor(obj, attrs):
    """Custom constructor that flattens 2D arrays in R data frames."""
    columns = list(attrs.get("names", [])) or None
    series = []
    for i, (k, v) in enumerate(obj.items()):
        name = columns[i] if columns and i < len(columns) else str(k)
        v = np.asarray(v)
        if v.ndim > 1:
            v = v.flatten()
        series.append(pd.Series(v, name=name))
    return pd.concat(series, axis=1)


def convert(rdata_path, csv_path):
    print(f"  Converting {os.path.basename(rdata_path)}...")
    constructor_dict = dict(rdata.conversion.DEFAULT_CLASS_MAP)
    constructor_dict["data.frame"] = _dataframe_constructor

    parsed = rdata.parser.parse_file(rdata_path)
    converter = rdata.conversion.SimpleConverter(constructor_dict=constructor_dict)
    converted = converter.convert(parsed)
    df = list(converted.values())[0]

    print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    df.to_csv(csv_path, index=False)
    print(f"  Saved to {csv_path}")
    return df


if __name__ == "__main__":
    converted = 0
    for rdata_file, csv_file in FILES.items():
        rdata_path = os.path.join(REF_DIR, rdata_file)
        csv_path = os.path.join(REF_DIR, csv_file)

        if not os.path.exists(rdata_path):
            print(f"  Skipping {rdata_file} (not found)")
            continue

        convert(rdata_path, csv_path)
        converted += 1

    if converted == 0:
        print(f"\nNo RData files found in {REF_DIR}")
        print("Download them first — see data/reference/DOWNLOAD.txt")
