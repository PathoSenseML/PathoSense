"""#4 — Add a NON-tree member to the blend for diversity. (overnight)

The deployed blend is LGBM+RF+CatBoost — all tree ensembles. Today's member-HPO result showed
DIVERSITY matters more than member accuracy. A linear model (logistic on imputed+standardized+
one-hot features) decorrelates far more from the GBDTs than RF does, even if weaker alone.

Test on general (honest CV, F1* at prior, paired vs deployed equal-3):
  equal3       : (L+R+C)/3                 = DEPLOYED  <- ref
  +logit (e4)  : (L+R+C+Logit)/4
  L+C+logit    : drop RF, (L+C+Logit)/3    (logit replaces RF as the diversity member)
  +logit w.20  : 0.8*(L+R+C)/3 + 0.2*Logit (small logit weight)

Run:  .venv/bin/python -u analysis/exp_blend_nontree.py
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from config import SEED
from harness import cv_evaluate, load_panel, compare, lgb_oof_and_test, _drop_leaky
from exp_general_blend import _rf_oof_test, _cat_oof_test, _cats
from preprocess_v2 import fit_keep_nan, apply_keep_nan
from calibration import fit_platt, apply_platt
from run_pipeline import get_model_params


def _logit_pipe(Xtr):
    cats = _cats(Xtr); nums = [c for c in Xtr.columns if c not in cats]
    ct = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), nums),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=10), cats)],
        remainder="drop")
    return Pipeline([("ct", ct), ("lr", LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5))])


def _logit_oof_test(Xtr, ytr, Xte, nf=5):
    oof = np.zeros(len(ytr)); skf = StratifiedKFold(nf, shuffle=True, random_state=SEED)
    for a, b in skf.split(Xtr, ytr):
        m = _logit_pipe(Xtr); m.fit(Xtr.iloc[a], ytr.iloc[a]); oof[b] = m.predict_proba(Xtr.iloc[b])[:, 1]
    m = _logit_pipe(Xtr); m.fit(Xtr, ytr)
    return oof, m.predict_proba(Xte)[:, 1]


def make_fn(kind):
    def _fn(X_tr, y_tr, X_te):
        Xtr, Xte = _drop_leaky(X_tr, y_tr, X_te)
        Xtr_pp, ppf = fit_keep_nan(Xtr); Xte_pp = apply_keep_nan(Xte, ppf)
        spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        params = get_model_params(len(X_tr), spw)
        L_t, L_o = lgb_oof_and_test(Xtr_pp, y_tr, Xte_pp, params, n_seeds=2, n_folds=5)
        R_o, R_t = _rf_oof_test(Xtr_pp, y_tr, Xte_pp, 5, spw)
        C_o, C_t = _cat_oof_test(Xtr_pp, y_tr, Xte_pp, 5, spw)
        G_o, G_t = _logit_oof_test(Xtr_pp, y_tr, Xte_pp)
        def cal(o, t): pf = fit_platt(o, y_tr.values); return apply_platt(o, pf), apply_platt(t, pf)
        Lo, Lt = cal(L_o, L_t); Ro, Rt = cal(R_o, R_t); Co, Ct = cal(C_o, C_t); Go, Gt = cal(G_o, G_t)
        if kind == "equal3":
            return (Lt+Rt+Ct)/3, (Lo+Ro+Co)/3
        if kind == "e4_logit":
            return (Lt+Rt+Ct+Gt)/4, (Lo+Ro+Co+Go)/4
        if kind == "LCG":
            return (Lt+Ct+Gt)/3, (Lo+Co+Go)/3
        if kind == "logit_w20":
            return 0.8*(Lt+Rt+Ct)/3 + 0.2*Gt, 0.8*(Lo+Ro+Co)/3 + 0.2*Go
    return _fn


if __name__ == "__main__":
    df = load_panel("general")
    print("=== #4 non-tree blend member (general) — honest CV, F1* at prior 0.143 ===", flush=True)
    ref = cv_evaluate("general", make_fn("equal3"), label="equal3", n_repeats=5, df=df)
    print(f"  equal3 (DEPLOYED) F1*={ref.summary()['f1_mean']:.4f}  AUC={ref.summary()['auc_mean']:.4f}", flush=True)
    for kind in ["e4_logit", "LCG", "logit_w20"]:
        r = cv_evaluate("general", make_fn(kind), label=kind, n_repeats=5, df=df)
        c = compare(ref, r)
        print(f"  {kind:10s} F1*={c['b_f1']:.4f}  AUC={r.summary()['auc_mean']:.4f}  "
              f"Δ={c['delta_f1']:+.4f} ({c['sigma']:+.2f}σ)  {c['verdict']}", flush=True)
