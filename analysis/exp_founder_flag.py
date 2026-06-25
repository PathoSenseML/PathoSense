"""Tier 3 — founder-population flag (motivated by the AMI/ASJ/FIN signal in the AL->population mapping).

Founder/bottleneck populations (AMI=Amish, ASJ=Ashkenazi, FIN=Finnish) carry classic pathogenic founder
mutations (e.g. BRCA in Ashkenazi) at extreme frequencies. A "popmax is a founder population" flag — and
a "founder-private" flag (founder popmax + low population breadth) — is genetically meaningful and NOT
explicitly engineered. Features derived label-free from CAT_1/CAT_2 (+ AL presence count).

Added to native-NaN LGBM, paired ΔF1* vs baseline, general + cancer (deployed-relevant own-models).
Run: .venv/bin/python -u analysis/exp_founder_flag.py [panel ...]
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
import schema
from harness import cv_evaluate, load_panel, compare
from meeting_lib import native_nan_oof

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
FOUNDER = {"AMI", "ASJ", "FIN"}
for f in ("founder_popmax", "founder_private"):     # register engineered flags... they're numeric, fine
    pass


def _suf(s):
    return s.astype(str).str.split("_").str[-1]


def base_fn(X_tr, y_tr, X_te):
    return native_nan_oof(X_tr, y_tr, X_te)


def make_fn(kind):
    def _fn(X_tr, y_tr, X_te):
        al = [c for c in X_tr.columns if str(c).startswith("AL_")]
        def add(X):
            X = X.copy()
            s1 = _suf(X["CAT_1"]) if "CAT_1" in X.columns else pd.Series("", index=X.index)
            s2 = _suf(X["CAT_2"]) if "CAT_2" in X.columns else pd.Series("", index=X.index)
            founder = (s1.isin(FOUNDER) | s2.isin(FOUNDER)).astype(int)
            X["founder_popmax"] = founder
            if kind in ("private", "both"):
                breadth = X[al].notna().sum(axis=1) / max(len(al), 1)
                X["founder_private"] = (founder * (1 - breadth)).astype(float)
            if kind == "flag":
                X = X.drop(columns=[c for c in ("founder_private",) if c in X.columns], errors="ignore")
            return X
        return native_nan_oof(add(X_tr), y_tr, add(X_te))
    return _fn


def run(panel, nr):
    df = load_panel(panel)
    base = cv_evaluate(panel, base_fn, label="base", n_repeats=nr, df=df)
    print(f"\n[{panel}] BASE F1*={base.summary()['f1_mean']:.4f}", flush=True)
    rows = [{"panel": panel, "variant": "base", "f1": base.summary()["f1_mean"], "delta": 0.0, "sigma": 0.0}]
    for kind in ("flag", "private", "both"):
        r = cv_evaluate(panel, make_fn(kind), label=kind, n_repeats=nr, df=df)
        c = compare(base, r)
        print(f"[{panel}] founder_{kind:8s} F1*={c['b_f1']:.4f}  Δ={c['delta_f1']:+.4f} ({c['sigma']:+.2f}σ)  {c['verdict']}", flush=True)
        rows.append({"panel": panel, "variant": f"founder_{kind}", "f1": c["b_f1"], "delta": c["delta_f1"], "sigma": c["sigma"]})
    return rows


if __name__ == "__main__":
    panels = sys.argv[1:] or ["cancer", "general"]
    print("=== Tier 3: founder-population flag — paired ΔF1* ===", flush=True)
    allrows = []
    for pn in panels:
        allrows += run(pn, 6 if pn == "general" else 12)
    pd.DataFrame(allrows).to_csv(os.path.join(OUT, "founder_flag.csv"), index=False)
    print(f"\nsaved -> founder_flag.csv", flush=True)
