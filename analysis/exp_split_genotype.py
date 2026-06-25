"""Does splitting the CAT_3/4/5 'X/X' genotype into two single-base columns help?
They are 100% homozygous (0 het in 3678 rows), so X/X -> (X, X) are IDENTICAL halves = no new
info. Confirm empirically: baseline (native X/X categorical) vs split (add allele_1/allele_2
single-base cats). Honest paired ΔF1*.
Run: .venv/bin/python -u analysis/exp_split_genotype.py [panel ...]
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
from harness import cv_evaluate, load_panel, compare
from meeting_lib import native_nan_oof
import schema

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
GT = ["CAT_3", "CAT_4", "CAT_5"]
# register the split columns as categorical so partition_columns/fit_keep_nan treat them right
for g in GT:
    for suf in ("_a1", "_a2"):
        if g + suf not in schema.CATEGORICAL_COLS:
            schema.CATEGORICAL_COLS.append(g + suf)


def base_fn(X_tr, y_tr, X_te):
    return native_nan_oof(X_tr, y_tr, X_te)


def split_fn(X_tr, y_tr, X_te):
    def add(X):
        X = X.copy()
        for c in [g for g in GT if g in X.columns]:
            s = X[c].astype(object)
            a1 = s.str.split("/").str[0]
            a2 = s.str.split("/").str[1]
            X[f"{c}_a1"] = pd.Categorical(a1.where(a1 != ".", np.nan))
            X[f"{c}_a2"] = pd.Categorical(a2.where(a2 != ".", np.nan))
        return X
    return native_nan_oof(add(X_tr), y_tr, add(X_te))


if __name__ == "__main__":
    panels = sys.argv[1:] or ["cftr", "pah", "cancer"]
    print("=== Split X/X genotype into single-base alleles — paired ΔF1* ===", flush=True)
    for pn in panels:
        df = load_panel(pn); nr = 8
        b = cv_evaluate(pn, base_fn, label="base", n_repeats=nr, df=df)
        s = cv_evaluate(pn, split_fn, label="split", n_repeats=nr, df=df)
        c = compare(b, s)
        print(f"[{pn}] base F1*={c['a_f1']:.4f}  split F1*={c['b_f1']:.4f}  "
              f"Δ={c['delta_f1']:+.4f} ({c['sigma']:+.2f}σ)  {c['verdict']}", flush=True)
