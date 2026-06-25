"""Untested global knob: the leakage-drop threshold (deployed: drop single-feature AUC>0.99). Columns at
0.95-0.99 AUC are PARTIAL leaks that may overfit training artifacts → hurt generalization. Test more-
aggressive dropping (0.95/0.97/0.98) on the general LGBM (spw1_bag), cv_evaluate AUC+F1*, 2 seeds, paired
vs the deployed 0.99. If a tighter threshold robustly helps, confirm in the blend.
Run: .venv/bin/python -u analysis/exp_general_leak.py
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
from config import SEED
from harness import cv_evaluate, load_panel, compare
from schema import partition_columns
from utils import detect_leaky_columns, KNOWN_LEAKY
from run_pipeline import get_model_params
from meeting_lib import native_nan_oof

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
df = load_panel("general"); _, _num, _cat, _tgt = partition_columns(df)
SPW = (df[_tgt].astype(int) == 0).sum() / max((df[_tgt].astype(int) == 1).sum(), 1)
P = {**get_model_params(2000, SPW), "scale_pos_weight": 1.0, "bagging_freq": 1, "bagging_fraction": 0.8}


def mk(thr):
    def fn(Xtr, ytr, Xte):
        leaky = detect_leaky_columns(Xtr, ytr, auc_threshold=thr, verbose=False)
        leaky = sorted(set(leaky) | (KNOWN_LEAKY & set(Xtr.columns)))
        Xtr2 = Xtr.drop(columns=[c for c in leaky if c in Xtr.columns])
        Xte2 = Xte.drop(columns=[c for c in leaky if c in Xte.columns])
        return native_nan_oof(Xtr2, ytr, Xte2, params=P)
    return fn


if __name__ == "__main__":
    print("=== general leakage-drop threshold sweep (LGBM-only, spw1_bag) ===", flush=True)
    for seed in (SEED, SEED + 1000):
        base = cv_evaluate("general", mk(0.99), label="0.99", n_repeats=6, df=df, seed=seed)
        print(f"  seed{seed}: base(0.99) F1*={base.f1.mean():.4f} AUC={base.auc.mean():.4f}", flush=True)
        for thr in (0.98, 0.97, 0.95):
            v = cv_evaluate("general", mk(thr), label=str(thr), n_repeats=6, df=df, seed=seed)
            c = compare(base, v); da = v.auc - base.auc
            sa = da.mean()/(da.std(ddof=1)/np.sqrt(len(da))) if da.std() > 0 else 0
            print(f"    thr={thr}: F1* {v.f1.mean():.4f} Δ={c['delta_f1']:+.4f} ({c['sigma']:+.2f}σ) | "
                  f"AUC Δ={da.mean():+.4f} ({sa:+.2f}σ)", flush=True)
    print("done", flush=True)
