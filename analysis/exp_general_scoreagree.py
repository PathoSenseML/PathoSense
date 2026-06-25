"""Quick screen: score-agreement feature for general (cancer/cftr use it; general has it OFF — untested
on general). And vary the leakage-drop AUC threshold (deployed 0.99). LGBM-only native-NaN screen
(spw1_bag), cv_evaluate AUC+F1*, 2 seeds. If anything screens positive, confirm in the blend.
Run: .venv/bin/python -u analysis/exp_general_scoreagree.py
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
from config import SEED
from harness import cv_evaluate, load_panel, compare
from schema import partition_columns
from run_pipeline import get_model_params
from feature_engineering import fit_score_agreement, apply_score_agreement
from meeting_lib import native_nan_oof

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
df = load_panel("general"); _, _num, _cat, _tgt = partition_columns(df)
SPW = (df[_tgt].astype(int) == 0).sum() / max((df[_tgt].astype(int) == 1).sum(), 1)
P = {**get_model_params(2000, SPW), "scale_pos_weight": 1.0, "bagging_freq": 1, "bagging_fraction": 0.8}


def base_fn(Xtr, ytr, Xte):
    return native_nan_oof(Xtr, ytr, Xte, params=P)


def sa_fn(Xtr, ytr, Xte):
    def post(Xtr_pp, y, Xte_pp, fit_idx):
        sa = fit_score_agreement(Xtr_pp.iloc[fit_idx].reset_index(drop=True),
                                 y.iloc[fit_idx].reset_index(drop=True), top_k=7)
        return apply_score_agreement(Xtr_pp, sa), apply_score_agreement(Xte_pp, sa)
    return native_nan_oof(Xtr, ytr, Xte, params=P, post=post)


if __name__ == "__main__":
    print("=== general +score_agreement (LGBM-only screen, spw1_bag) ===", flush=True)
    for seed in (SEED, SEED + 1000):
        b = cv_evaluate("general", base_fn, label="base", n_repeats=6, df=df, seed=seed)
        s = cv_evaluate("general", sa_fn, label="sa", n_repeats=6, df=df, seed=seed)
        c = compare(b, s); da = s.auc - b.auc
        sa_sig = da.mean()/(da.std(ddof=1)/np.sqrt(len(da))) if da.std() > 0 else 0
        print(f"  seed{seed}: F1* {b.f1.mean():.4f}->{s.f1.mean():.4f} Δ={c['delta_f1']:+.4f} ({c['sigma']:+.2f}σ) | "
              f"AUC {b.auc.mean():.4f}->{s.auc.mean():.4f} Δ={da.mean():+.4f} ({sa_sig:+.2f}σ)", flush=True)
    print("done", flush=True)
