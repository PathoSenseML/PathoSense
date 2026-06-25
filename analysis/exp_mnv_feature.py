"""Idea #3 — MNV combined-substitution feature. 7.1% of variants carry multi-letter AA tokens
(AG, SV, ...) = multi-nucleotide / complex changes, for which AA_grantham / AA_blosum62 fall back to
NaN. Give them a proper COMBINED score (position-wise mean Grantham / BLOSUM across the residues)
instead of NaN, so the model can use the chemistry of those 7% of variants.

Added as AA_grantham_mnv / AA_blosum_mnv (filled for ALL rows: single-letter == the usual score;
multi-letter == position-wise mean). Paired ΔF1* vs native-NaN LGBM baseline, per panel.
Run: .venv/bin/python -u analysis/exp_mnv_feature.py [panel ...]
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")

from harness import cv_evaluate, load_panel, compare
from meeting_lib import native_nan_oof
from feature_engineering import grantham, blosum62, _STD_AA_SET

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def _combined(aa1, aa2, fn):
    aa1, aa2 = str(aa1), str(aa2)
    if aa1 in ("nan", "None", "") or aa2 in ("nan", "None", ""):
        return np.nan
    # position-wise when equal length (di-residue MNV); else all-pairs mean
    if len(aa1) == len(aa2):
        pairs = list(zip(aa1, aa2))
    else:
        pairs = [(a, b) for a in aa1 for b in aa2]
    vals = [fn(a, b) for a, b in pairs if a in _STD_AA_SET and b in _STD_AA_SET]
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else np.nan


def base_fn(X_tr, y_tr, X_te):
    return native_nan_oof(X_tr, y_tr, X_te)


def mnv_fn(X_tr, y_tr, X_te):
    def add(X):
        X = X.copy()
        a1 = X["AA_1"].astype(object); a2 = X["AA_2"].astype(object)
        X["AA_grantham_mnv"] = [_combined(r, a, grantham) for r, a in zip(a1, a2)]
        X["AA_blosum_mnv"] = [_combined(r, a, blosum62) for r, a in zip(a1, a2)]
        return X
    return native_nan_oof(add(X_tr), y_tr, add(X_te))


if __name__ == "__main__":
    panels = sys.argv[1:] or ["cftr", "pah", "cancer", "general"]
    print("=== Idea #3: MNV combined-substitution feature — paired ΔF1* ===", flush=True)
    rows = []
    for pn in panels:
        df = load_panel(pn); nr = 4 if pn == "general" else 8
        b = cv_evaluate(pn, base_fn, label="base", n_repeats=nr, df=df)
        m = cv_evaluate(pn, mnv_fn, label="mnv", n_repeats=nr, df=df)
        c = compare(b, m)
        print(f"[{pn}] base F1*={c['a_f1']:.4f}  +MNV F1*={c['b_f1']:.4f}  "
              f"Δ={c['delta_f1']:+.4f} ({c['sigma']:+.2f}σ)  {c['verdict']}", flush=True)
        rows.append({"panel": pn, "base_f1": c["a_f1"], "mnv_f1": c["b_f1"],
                     "delta": c["delta_f1"], "sigma": c["sigma"]})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "mnv_feature.csv"), index=False)
    print(f"\nsaved -> mnv_feature.csv", flush=True)
