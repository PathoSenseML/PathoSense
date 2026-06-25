"""#12 conservation-spread + #14 missingness-aggregates as engineered features (general blend).

#12: variance/range/count across the 9 EK_ conservation tracks (track DISAGREEMENT may signal
     something the individual tracks miss).
#14: explicit AL_ missingness aggregates (count present / count missing) — the cancer win showed
     missingness is signal; general uses raw AL, maybe explicit rarity aggregates add.

Honest CV, general, F1* at prior, paired vs deployed blend. Features added BEFORE leakage-drop so
they go through the normal pipeline.

Run:  .venv/bin/python -u analysis/exp_cheap_features.py
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")

from harness import cv_evaluate, load_panel, compare
from exp_general_blend import blend_fn


def add_features(X, which):
    X = X.copy()
    ek = [c for c in X.columns if str(c).startswith("EK_")]
    al = [c for c in X.columns if str(c).startswith("AL_")]
    if "ekstd" in which and ek:
        E = X[ek].apply(pd.to_numeric, errors="coerce")
        X["FX_ek_std"] = E.std(axis=1)
        X["FX_ek_range"] = E.max(axis=1) - E.min(axis=1)
        X["FX_ek_npresent"] = E.notna().sum(axis=1).astype(float)
        X["FX_ek_min"] = E.min(axis=1)
    if "almiss" in which and al:
        A = X[al]
        X["FX_al_npresent"] = A.notna().sum(axis=1).astype(float)
        X["FX_al_frac_present"] = A.notna().mean(axis=1)
    return X


def make_fn(which):
    def _fn(X_tr, y_tr, X_te):
        return blend_fn(add_features(X_tr, which), y_tr, add_features(X_te, which))
    return _fn


if __name__ == "__main__":
    df = load_panel("general")
    print("=== #12/#14 cheap features (general) — honest CV, F1* at prior 0.143 ===", flush=True)
    ref = cv_evaluate("general", make_fn([]), label="baseline", n_repeats=5, df=df)
    print(f"  baseline F1*={ref.summary()['f1_mean']:.4f}  AUC={ref.summary()['auc_mean']:.4f}", flush=True)
    for which, tag in [(["ekstd"], "ek_spread"), (["almiss"], "al_miss"), (["ekstd", "almiss"], "both")]:
        r = cv_evaluate("general", make_fn(which), label=tag, n_repeats=5, df=df)
        c = compare(ref, r)
        print(f"  {tag:10s} F1*={c['b_f1']:.4f}  AUC={r.summary()['auc_mean']:.4f}  "
              f"Δ={c['delta_f1']:+.4f} ({c['sigma']:+.2f}σ)  {c['verdict']}", flush=True)
