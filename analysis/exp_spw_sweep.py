"""#13 — Prior/importance weighting: train toward the TEST (benign-dominant) prior instead of
balanced. The deployed scale_pos_weight balances classes (~50/50); the test is benign-dominant, so
weighting the model toward the test prior may shape a better decision boundary for the operating
point (where F1 actually lives at extreme priors). Sweep the spw multiplier on general's blend.

spw_balance = (n_benign/n_pathogenic) [deployed].  Multipliers <1 downweight pathogenic further
(toward the benign-dominant test regime). 'test' = matched to the panel test prior.

Run:  .venv/bin/python -u analysis/exp_spw_sweep.py
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")

from harness import cv_evaluate, load_panel, compare, lgb_oof_and_test, _drop_leaky
from exp_general_blend import _rf_oof_test, _cat_oof_test
from preprocess_v2 import fit_keep_nan, apply_keep_nan
from calibration import fit_platt, apply_platt
from run_pipeline import get_model_params
from utils import TEST_PRIORS


def make_fn(mult):
    """mult multiplies the balanced scale_pos_weight. mult=1 == deployed."""
    def _fn(X_tr, y_tr, X_te):
        Xtr, Xte = _drop_leaky(X_tr, y_tr, X_te)
        Xtr_pp, ppf = fit_keep_nan(Xtr); Xte_pp = apply_keep_nan(Xte, ppf)
        spw0 = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        spw = spw0 * mult
        params = get_model_params(len(X_tr), spw0); params["scale_pos_weight"] = spw
        L_t, L_o = lgb_oof_and_test(Xtr_pp, y_tr, Xte_pp, params, n_seeds=2, n_folds=5)
        R_o, R_t = _rf_oof_test(Xtr_pp, y_tr, Xte_pp, 5, spw)
        C_o, C_t = _cat_oof_test(Xtr_pp, y_tr, Xte_pp, 5, spw)
        def cal(o, t): pf = fit_platt(o, y_tr.values); return apply_platt(o, pf), apply_platt(t, pf)
        Lo, Lt = cal(L_o, L_t); Ro, Rt = cal(R_o, R_t); Co, Ct = cal(C_o, C_t)
        return (Lt+Rt+Ct)/3, (Lo+Ro+Co)/3
    return _fn


if __name__ == "__main__":
    panel = "general"; prior = TEST_PRIORS[panel]
    df = load_panel(panel)
    # test-matched multiplier: spw so effective pathogenic frac == test prior
    p = df["Label"].mean()  # train pathogenic frac
    test_mult = (prior / (1 - prior)) / (p / (1 - p))  # = spw_test / spw_balance
    print(f"=== #13 spw sweep (general) — honest CV, F1* at prior {prior} ===", flush=True)
    print(f"  train P-frac={p:.3f}, test-matched spw multiplier={test_mult:.3f}", flush=True)
    ref = cv_evaluate(panel, make_fn(1.0), label="deployed", n_repeats=5, df=df)
    print(f"  deployed (balance) F1*={ref.summary()['f1_mean']:.4f}", flush=True)
    for mult, tag in [(0.5, "x0.5"), (0.25, "x0.25"), (round(test_mult, 3), "test_matched"), (2.0, "x2")]:
        r = cv_evaluate(panel, make_fn(mult), label=tag, n_repeats=5, df=df)
        c = compare(ref, r)
        print(f"  {tag:13s}(spw x{mult}) F1*={c['b_f1']:.4f}  Δ={c['delta_f1']:+.4f} "
              f"({c['sigma']:+.2f}σ)  {c['verdict']}", flush=True)
