"""Idea #3 — feature-augmented stacking + recipe-diversity ensembling, on the GENERAL deployed blend.

(a) feat_stack: a meta-learner (LogReg) over [3 calibrated base-model OOF probs + a few raw context
    features (EK_7, EK_9, popmax-AF rank, AL_n_present)] so it can weight the bases context-dependently.
    PROPER: the train meta-predictions come from inner cross_val_predict (the stacker is never evaluated
    in-sample), and every base/feature is fit on the train fold only.
(b) recipe_div: ensemble LGBM(native-NaN) + LGBM(median-impute) — same family, decorrelated by INPUT
    recipe rather than model family.
Paired ΔF1* vs the deployed equal-average blend (LGBM+RF+CatBoost), honest CV on general.
Run: .venv/bin/python -u analysis/exp_feataug_stack.py [n_repeats]
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from harness import cv_evaluate, load_panel, compare, _drop_leaky, lgb_oof_and_test
from preprocess_v2 import fit_keep_nan, apply_keep_nan
from calibration import fit_platt, apply_platt
from run_pipeline import get_model_params, preprocess_data
from exp_general_blend import _rf_oof_test, _cat_oof_test

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def _base3(X_tr, y_tr, X_te):
    Xtr, Xte = _drop_leaky(X_tr.reset_index(drop=True), y_tr.reset_index(drop=True), X_te.reset_index(drop=True))
    Xtr_pp, ppf = fit_keep_nan(Xtr); Xte_pp = apply_keep_nan(Xte, ppf)
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    lt, lo = lgb_oof_and_test(Xtr_pp, y_tr, Xte_pp, get_model_params(len(X_tr), spw), n_seeds=2, n_folds=5)
    ro, rt = _rf_oof_test(Xtr_pp, y_tr, Xte_pp, 5, spw)
    co, ct = _cat_oof_test(Xtr_pp, y_tr, Xte_pp, 5, spw)
    cal = []
    for t, o in [(lt, lo), (rt, ro), (ct, co)]:
        pf = fit_platt(o, y_tr.values); cal.append((apply_platt(t, pf), apply_platt(o, pf)))
    return cal


def _ctx(X):
    al = [c for c in X.columns if str(c).startswith("AL_")]
    af = [c for c in al if ((X[c] > 0) & (X[c] <= 1)).any() and X[c].mean(skipna=True) < 0.1]
    df = pd.DataFrame(index=X.index)
    for c in ("EK_7", "EK_9", "EK_2"):
        if c in X.columns:
            df[c] = X[c]
    df["popmax"] = X[af].max(axis=1) if af else 0.0
    df["n_present"] = X[al].notna().sum(axis=1)
    return df.astype(float)


def base_blend(X_tr, y_tr, X_te):
    cal = _base3(X_tr, y_tr, X_te)
    return np.mean([c[0] for c in cal], axis=0), np.mean([c[1] for c in cal], axis=0)


def feat_stack(X_tr, y_tr, X_te):
    y_tr = y_tr.reset_index(drop=True)
    cal = _base3(X_tr, y_tr, X_te)
    ctr = _ctx(X_tr.reset_index(drop=True)); cte = _ctx(X_te.reset_index(drop=True))
    med = ctr.median()
    Ctr = pd.concat([pd.DataFrame({f"b{i}": cal[i][1] for i in range(3)}), ctr.fillna(med).fillna(0)], axis=1)
    Cte = pd.concat([pd.DataFrame({f"b{i}": cal[i][0] for i in range(3)}), cte.fillna(med).fillna(0)], axis=1)
    sc = StandardScaler().fit(Ctr.values)
    Ctr2 = sc.transform(Ctr.values); Cte2 = sc.transform(Cte.values)
    lr = LogisticRegression(max_iter=2000, C=0.5)
    meta_tr = cross_val_predict(lr, Ctr2, y_tr, cv=5, method="predict_proba")[:, 1]  # inner OOF, no leakage
    lr.fit(Ctr2, y_tr); meta_te = lr.predict_proba(Cte2)[:, 1]
    return meta_te, meta_tr


def recipe_div(X_tr, y_tr, X_te):
    y_tr = y_tr.reset_index(drop=True)
    Xtr, Xte = _drop_leaky(X_tr.reset_index(drop=True), y_tr, X_te.reset_index(drop=True))
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1); params = get_model_params(len(X_tr), spw)
    # recipe A: native-NaN
    Atr, ppf = fit_keep_nan(Xtr); Ate = apply_keep_nan(Xte, ppf)
    la, loa = lgb_oof_and_test(Atr, y_tr, Ate, params, n_seeds=2, n_folds=5)
    # recipe B: median-impute (deployed preprocess_data)
    Btr, _, Bte, _ = preprocess_data(Xtr.copy(), Xtr.copy(), Xte.copy(), 0.95)
    lb, lob = lgb_oof_and_test(Btr, y_tr, Bte, params, n_seeds=2, n_folds=5)
    pa = fit_platt(loa, y_tr.values); pb = fit_platt(lob, y_tr.values)
    tp = (apply_platt(la, pa) + apply_platt(lb, pb)) / 2
    oof = (apply_platt(loa, pa) + apply_platt(lob, pb)) / 2
    return tp, oof


if __name__ == "__main__":
    nr = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    print("=== Idea #3: feature-augmented stacking + recipe-diversity (general) — paired ΔF1* ===", flush=True)
    df = load_panel("general")
    base = cv_evaluate("general", base_blend, label="blend", n_repeats=nr, df=df)
    for name, fn in [("feat_stack", feat_stack), ("recipe_div", recipe_div)]:
        r = cv_evaluate("general", fn, label=name, n_repeats=nr, df=df)
        c = compare(base, r)
        print(f"general  blend F1*={c['a_f1']:.4f}  {name} F1*={c['b_f1']:.4f}  "
              f"Δ={c['delta_f1']:+.4f} ({c['sigma']:+.2f}σ)  {c['verdict']}", flush=True)
