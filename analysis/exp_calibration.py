"""Idea #6 — isotonic vs Platt calibration. NOTE: under threshold-at-prior, calibration is a MONOTONIC
map, so it cannot change a SINGLE model's F1* (ranking preserved). It only matters when COMBINING models
(the blend), where the calibrated scale changes the average. So we test it on the GENERAL deployed blend
(LGBM+RF+CatBoost): Platt vs isotonic calibration of the 3 components, paired ΔF1*.
Run: .venv/bin/python -u analysis/exp_calibration.py [n_repeats]
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
from sklearn.isotonic import IsotonicRegression
from harness import cv_evaluate, load_panel, compare, lgb_oof_and_test, _drop_leaky
from preprocess_v2 import fit_keep_nan, apply_keep_nan
from calibration import fit_platt, apply_platt
from run_pipeline import get_model_params
from exp_general_blend import _rf_oof_test, _cat_oof_test

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def _iso(oof, y, test):
    ir = IsotonicRegression(out_of_bounds="clip").fit(oof, y)
    return ir.predict(test), ir.predict(oof)


def blend_fn(calib):
    def _fn(X_tr, y_tr, X_te):
        y_tr = y_tr.reset_index(drop=True)
        Xtr, Xte = _drop_leaky(X_tr.reset_index(drop=True), y_tr, X_te.reset_index(drop=True))
        Xtr_pp, ppf = fit_keep_nan(Xtr); Xte_pp = apply_keep_nan(Xte, ppf)
        spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        lt, lo = lgb_oof_and_test(Xtr_pp, y_tr, Xte_pp, get_model_params(len(X_tr), spw), n_seeds=2, n_folds=5)
        ro, rt = _rf_oof_test(Xtr_pp, y_tr, Xte_pp, 5, spw)
        co, ct = _cat_oof_test(Xtr_pp, y_tr, Xte_pp, 5, spw)
        parts = [(lt, lo), (rt, ro), (ct, co)]
        if calib == "platt":
            cal = [(apply_platt(t, fit_platt(o, y_tr.values)), apply_platt(o, fit_platt(o, y_tr.values))) for t, o in parts]
        else:
            cal = [_iso(o, y_tr.values, t) for t, o in parts]
        tp = np.mean([c[0] for c in cal], axis=0); oof = np.mean([c[1] for c in cal], axis=0)
        return tp, oof
    return _fn


if __name__ == "__main__":
    nr = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    print("=== Idea #6: general blend Platt vs isotonic calibration — paired ΔF1* ===", flush=True)
    df = load_panel("general")
    p = cv_evaluate("general", blend_fn("platt"), label="platt", n_repeats=nr, df=df)
    iso = cv_evaluate("general", blend_fn("iso"), label="iso", n_repeats=nr, df=df)
    c = compare(p, iso)
    print(f"general blend  Platt F1*={c['a_f1']:.4f}  isotonic F1*={c['b_f1']:.4f}  "
          f"Δ={c['delta_f1']:+.4f} ({c['sigma']:+.2f}σ)  {c['verdict']}", flush=True)
    print("NOTE: single-model panels (pah single-LGBM) are calibration-invariant under threshold-at-prior.", flush=True)
