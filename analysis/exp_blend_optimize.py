"""Are the ENSEMBLE choices (members, blend weights) optimal? (user: have all models/params
been tested?) The deployed general blend is a FIXED equal-1/3 average of LGBM+RF+CatBoost with
RF/CatBoost at hardcoded defaults, and XGBoost was never an ensemble member. Test:

  lgbm        : LightGBM only                                        (low ref)
  equal3      : (L+R+C)/3   = DEPLOYED                               <- ref
  equal4      : (L+R+C+X)/4  (+XGBoost member)
  wopt3       : weight-optimized L,R,C   (weights chosen on TRAIN OOF AUC)
  wopt4       : weight-optimized L,R,C,X

All honest per-fold CV, F1* at the general test prior, paired vs equal3. Weights are fit on the
train OOF only (no leakage). Members use default params here; member HPO is exp_blend_hpo.py.

Run:  .venv/bin/python -u analysis/exp_blend_optimize.py
"""
import os, sys, itertools
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from config import SEED
from harness import cv_evaluate, load_panel, compare, lgb_oof_and_test, _drop_leaky
from exp_general_blend import _rf_oof_test, _cat_oof_test, _cats
from preprocess_v2 import fit_keep_nan, apply_keep_nan
from calibration import fit_platt, apply_platt
from run_pipeline import get_model_params

PRIOR = 0.143


def _xgb_oof_test(Xtr, ytr, Xte, n_splits, spw):
    cats = _cats(Xtr)
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-2)
    Xtr2, Xte2 = Xtr.copy(), Xte.copy()
    if cats:
        Xtr2[cats] = enc.fit_transform(Xtr[cats].astype(object).where(Xtr[cats].notna(), "NA"))
        Xte2[cats] = enc.transform(Xte[cats].astype(object).where(Xte[cats].notna(), "NA"))
    oof = np.zeros(len(ytr))
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=SEED)
    def mk():
        return XGBClassifier(n_estimators=600, max_depth=5, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, reg_lambda=1.0, scale_pos_weight=spw,
                             tree_method="hist", eval_metric="logloss", n_jobs=4, random_state=SEED)
    for a, b in skf.split(Xtr2, ytr):
        m = mk(); m.fit(Xtr2.iloc[a], ytr.iloc[a]); oof[b] = m.predict_proba(Xtr2.iloc[b])[:, 1]
    m = mk(); m.fit(Xtr2, ytr)
    return oof, m.predict_proba(Xte2)[:, 1]


def _members(X_tr, y_tr, X_te):
    Xtr, Xte = _drop_leaky(X_tr, y_tr, X_te)
    Xtr_pp, ppf = fit_keep_nan(Xtr); Xte_pp = apply_keep_nan(Xte, ppf)
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    params = get_model_params(len(X_tr), spw)
    L_t, L_o = lgb_oof_and_test(Xtr_pp, y_tr, Xte_pp, params, n_seeds=2, n_folds=5)
    R_o, R_t = _rf_oof_test(Xtr_pp, y_tr, Xte_pp, 5, spw)
    C_o, C_t = _cat_oof_test(Xtr_pp, y_tr, Xte_pp, 5, spw)
    X_o, X_t = _xgb_oof_test(Xtr_pp, y_tr, Xte_pp, 5, spw)
    def cal(o, t):
        pf = fit_platt(o, y_tr.values); return apply_platt(o, pf), apply_platt(t, pf)
    Lo, Lt = cal(L_o, L_t); Ro, Rt = cal(R_o, R_t); Co, Ct = cal(C_o, C_t); Xo, Xt = cal(X_o, X_t)
    return ({"L": Lo, "R": Ro, "C": Co, "X": Xo}, {"L": Lt, "R": Rt, "C": Ct, "X": Xt}, y_tr.values)


def _best_weights(oofs, y, members):
    """Grid over the simplex (step .25) maximizing train-OOF AUC."""
    best_w, best = None, -1
    steps = [0, .25, .5, .75, 1.0]
    for combo in itertools.product(steps, repeat=len(members)):
        s = sum(combo)
        if abs(s - 1.0) > 1e-9:
            continue
        blend = sum(w * oofs[m] for w, m in zip(combo, members))
        a = roc_auc_score(y, blend)
        if a > best:
            best, best_w = a, combo
    return dict(zip(members, best_w))


def make_fn(kind):
    def _fn(X_tr, y_tr, X_te):
        oofs, tests, y = _members(X_tr, y_tr, X_te)
        if kind == "lgbm":
            return tests["L"], oofs["L"]
        if kind == "equal3":
            ms = ["L", "R", "C"]; w = {m: 1/3 for m in ms}
        elif kind == "equal4":
            ms = ["L", "R", "C", "X"]; w = {m: 1/4 for m in ms}
        elif kind == "wopt3":
            ms = ["L", "R", "C"]; w = _best_weights(oofs, y, ms)
        elif kind == "wopt4":
            ms = ["L", "R", "C", "X"]; w = _best_weights(oofs, y, ms)
        oof = sum(w[m] * oofs[m] for m in ms); test = sum(w[m] * tests[m] for m in ms)
        return test, oof
    return _fn


if __name__ == "__main__":
    df = load_panel("general")
    print("=== general blend optimization — honest CV, F1* at prior 0.143 ===", flush=True)
    ref = cv_evaluate("general", make_fn("equal3"), label="equal3", n_repeats=5, df=df)
    print(f"  equal3 (DEPLOYED) F1*={ref.summary()['f1_mean']:.4f}  AUC={ref.summary()['auc_mean']:.4f}", flush=True)
    for kind in ["lgbm", "equal4", "wopt3", "wopt4"]:
        r = cv_evaluate("general", make_fn(kind), label=kind, n_repeats=5, df=df)
        c = compare(ref, r)
        print(f"  {kind:7s} F1*={c['b_f1']:.4f}  AUC={r.summary()['auc_mean']:.4f}  "
              f"Δ={c['delta_f1']:+.4f} ({c['sigma']:+.2f}σ)  {c['verdict']}", flush=True)
