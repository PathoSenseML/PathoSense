"""#1 — Is the leakage detector discarding real signal? (overnight run)

detect_leaky_columns drops 58-106 cols/panel. Most are flagged by the MISSINGNESS-PURITY path
(all rows where the col is missing are one class) with value-AUC=0.500 — i.e. their VALUES are
useless but their MISSINGNESS perfectly splits the training labels. Today's cancer fam_flags win
showed AL missingness IS real rarity signal. So this may be over-dropping. BUT missingness-purity
can also be a dataset-construction artifact that breaks on the judges' novel benign-dominant set.

Test (general, honest CV, F1* at prior, paired): deployed drop (auc>0.99 + missingness-purity +
cat-purity) vs RELAXED drop (only true value-leaks auc>0.99 + constants). INTERPRET BY MAGNITUDE:
huge gain => suspect leakage (don't ship); modest consistent gain => likely safe rarity signal.

Run:  .venv/bin/python -u analysis/exp_leakage_reexam.py
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score
from harness import cv_evaluate, load_panel, compare, lgb_oof_and_test
from exp_general_blend import _rf_oof_test, _cat_oof_test, blend_fn  # blend_fn = deployed leakage drop
from preprocess_v2 import fit_keep_nan, apply_keep_nan
from calibration import fit_platt, apply_platt
from run_pipeline import get_model_params


def relaxed_leaky(X, y):
    """Only TRUE value-leaks: numeric single-feature AUC>0.99 or constant. NO missingness/cat purity."""
    y = np.asarray(y).astype(int); leaky = []
    for c in X.columns:
        v = X[c]
        if isinstance(v.dtype, pd.CategoricalDtype):
            continue
        vn = pd.to_numeric(v, errors="coerce"); m = vn.notna().values
        if m.sum() < 20:
            continue
        sub = vn.values[m]
        if np.nanstd(sub) < 1e-8:
            leaky.append(c); continue
        if len(np.unique(y[m])) >= 2:
            try:
                a = roc_auc_score(y[m], sub);
                if max(a, 1 - a) > 0.99:
                    leaky.append(c)
            except Exception:
                pass
    return list(set(leaky))


def _drop_relaxed(X_tr, y_tr, X_te):
    lk = relaxed_leaky(X_tr, y_tr)
    return (X_tr.drop(columns=lk, errors="ignore"),
            X_te.drop(columns=[c for c in lk if c in X_te.columns], errors="ignore"))


def blend_relaxed_fn(X_tr, y_tr, X_te):
    Xtr, Xte = _drop_relaxed(X_tr, y_tr, X_te)
    Xtr_pp, ppf = fit_keep_nan(Xtr); Xte_pp = apply_keep_nan(Xte, ppf)
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    params = get_model_params(len(X_tr), spw)
    L_t, L_o = lgb_oof_and_test(Xtr_pp, y_tr, Xte_pp, params, n_seeds=2, n_folds=5)
    R_o, R_t = _rf_oof_test(Xtr_pp, y_tr, Xte_pp, 5, spw)
    C_o, C_t = _cat_oof_test(Xtr_pp, y_tr, Xte_pp, 5, spw)
    def cal(o, t): pf = fit_platt(o, y_tr.values); return apply_platt(o, pf), apply_platt(t, pf)
    Lo, Lt = cal(L_o, L_t); Ro, Rt = cal(R_o, R_t); Co, Ct = cal(C_o, C_t)
    return (Lt + Rt + Ct) / 3, (Lo + Ro + Co) / 3


if __name__ == "__main__":
    df = load_panel("general")
    print("=== #1 leakage re-exam (general) — honest CV, F1* at prior 0.143 ===", flush=True)
    dep = cv_evaluate("general", blend_fn, label="deployed_drop", n_repeats=5, df=df)
    rel = cv_evaluate("general", blend_relaxed_fn, label="relaxed_drop", n_repeats=5, df=df)
    c = compare(dep, rel)
    ds, rs = dep.summary(), rel.summary()
    print(f"  deployed_drop F1*={ds['f1_mean']:.4f}  AUC={ds['auc_mean']:.4f}", flush=True)
    print(f"  relaxed_drop  F1*={rs['f1_mean']:.4f}  AUC={rs['auc_mean']:.4f}", flush=True)
    print(f"  Δ={c['delta_f1']:+.4f} ({c['sigma']:+.2f}σ)  ΔAUC={rs['auc_mean']-ds['auc_mean']:+.4f}  {c['verdict']}", flush=True)
    print("  INTERPRET: ΔAUC large (>0.03) => suspect leakage; modest (~0.01) + >=2σ => likely safe rarity signal", flush=True)
