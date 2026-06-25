"""Validate the ONE promising candidate from the data reanalysis: the conservation x
rarity interaction (trees can't synthesize a product of two columns as a split).

Reanalysis single-feature AUC (cancer): EK_mean x (1-maxAF) = 0.831 vs best raw 0.814,
and |corr| to best raw = 0.19 (orthogonal). Two forms:
  ek_x_af    EK_mean * (1 - maxAF)            — uses AL freq VALUES (in-panel rarity; does NOT
                                                transfer across panels, so own-model only)
  ek_x_miss  EK_mean * (1 - present_fraction) — missingness-based rarity (panel-independent,
                                                so it COULD ride the transfer scorer)
Both fit their AF-column set + present-fraction on the TRAIN fold only (batch-invariant),
added as one extra column to the native-NaN LGBM. Honest paired ΔF1* vs baseline.
Run: .venv/bin/python -u analysis/exp_reanalysis_validate.py [panel ...]
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")

from harness import cv_evaluate, load_panel, compare
from meeting_lib import native_nan_oof

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def base_fn(X_tr, y_tr, X_te):
    return native_nan_oof(X_tr, y_tr, X_te)


def make_fn(kind):
    def _fn(X_tr, y_tr, X_te):
        ek = [c for c in X_tr.columns if str(c).startswith("EK_")]
        al = [c for c in X_tr.columns if str(c).startswith("AL_")]
        af = [c for c in al if ((X_tr[c] > 0) & (X_tr[c] <= 1)).any()
              and X_tr[c].mean(skipna=True) < 0.1]                 # AF cols fixed on TRAIN

        def add(X):
            X = X.copy()
            ekmean = X[ek].astype(float).mean(axis=1)
            if kind == "ek_x_af":
                rare = 1.0 - (X[af].max(axis=1).fillna(0.0) if af else 0.0)
            else:  # ek_x_miss — panel-independent (transfer-safe) rarity
                rare = 1.0 - (X[al].notna().sum(axis=1) / max(len(al), 1))
            X["EK_x_rare"] = ekmean * rare
            return X
        return native_nan_oof(add(X_tr), y_tr, add(X_te))
    return _fn


VARIANTS = {"ek_x_af": make_fn("ek_x_af"), "ek_x_miss": make_fn("ek_x_miss")}


def run(panel, n_repeats):
    df = load_panel(panel)
    base = cv_evaluate(panel, base_fn, label="base", n_repeats=n_repeats, df=df)
    print(f"\n[{panel}] BASE F1*={base.summary()['f1_mean']:.4f}", flush=True)
    rows = [{"panel": panel, "variant": "(base)", "f1": base.summary()["f1_mean"],
             "delta_f1": 0.0, "sigma": 0.0, "verdict": "reference"}]
    for name, fn in VARIANTS.items():
        r = cv_evaluate(panel, fn, label=name, n_repeats=n_repeats, df=df)
        c = compare(base, r)
        print(f"[{panel}] {name:10s} F1*={c['b_f1']:.4f}  Δ={c['delta_f1']:+.4f} "
              f"({c['sigma']:+.2f}σ)  {c['verdict']}", flush=True)
        rows.append({"panel": panel, "variant": name, "f1": c["b_f1"],
                     "delta_f1": c["delta_f1"], "sigma": c["sigma"], "verdict": c["verdict"]})
    return rows


if __name__ == "__main__":
    panels = sys.argv[1:] or ["cancer", "cftr", "pah", "general"]
    print("=== Validate conservation×rarity interaction — paired ΔF1* (vs native-NaN LGBM) ===", flush=True)
    allrows = []
    for pn in panels:
        allrows.extend(run(pn, 4 if pn == "general" else 8))
    pd.DataFrame(allrows).to_csv(os.path.join(OUT, "reanalysis_validate.csv"), index=False)
    print(f"\nsaved -> {os.path.join(OUT, 'reanalysis_validate.csv')}", flush=True)
