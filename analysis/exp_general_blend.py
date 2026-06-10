"""Does a heterogeneous blend (LGBM + RandomForest + CatBoost) beat the LGBM
ensemble on GENERAL's new all-features recipe? (The old blend gave general +0.017
on the OLD mRMR recipe — re-test on the new recipe, which already uses all features.)

General has DOUBLE value (panel model + transfer source), so a win here is worth most.
Honest per-fold CV, analytic F1* at the test prior, paired vs LGBM-alone.
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from catboost import CatBoostClassifier
import lightgbm as lgb

from config import SEED
from harness import cv_evaluate, load_panel, compare, lgb_oof_and_test, _drop_leaky
from preprocess_v2 import fit_keep_nan, apply_keep_nan
from calibration import fit_platt, apply_platt
from run_pipeline import get_model_params


def _cats(X):
    return [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]


def _rf_oof_test(Xtr, ytr, Xte, n_splits, spw):
    cats = _cats(Xtr)
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                         encoded_missing_value=-2)
    Xtr2, Xte2 = Xtr.copy(), Xte.copy()
    if cats:
        Xtr2[cats] = enc.fit_transform(Xtr[cats].astype(object).where(Xtr[cats].notna(), "NA"))
        Xte2[cats] = enc.transform(Xte[cats].astype(object).where(Xte[cats].notna(), "NA"))
    Xtr2 = Xtr2.fillna(-999); Xte2 = Xte2.fillna(-999)
    oof = np.zeros(len(ytr))
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=SEED)
    for a, b in skf.split(Xtr2, ytr):
        m = RandomForestClassifier(n_estimators=600, max_features="sqrt", n_jobs=4,
                                   class_weight="balanced", random_state=SEED)
        m.fit(Xtr2.iloc[a], ytr.iloc[a]); oof[b] = m.predict_proba(Xtr2.iloc[b])[:, 1]
    m = RandomForestClassifier(n_estimators=600, max_features="sqrt", n_jobs=4,
                               class_weight="balanced", random_state=SEED).fit(Xtr2, ytr)
    return oof, m.predict_proba(Xte2)[:, 1]


def _cat_oof_test(Xtr, ytr, Xte, n_splits, spw):
    cats = _cats(Xtr)
    Xtr2, Xte2 = Xtr.copy(), Xte.copy()
    for c in cats:
        Xtr2[c] = Xtr[c].astype(object).where(Xtr[c].notna(), "NA").astype(str)
        Xte2[c] = Xte[c].astype(object).where(Xte[c].notna(), "NA").astype(str)
    cat_idx = [Xtr2.columns.get_loc(c) for c in cats]
    oof = np.zeros(len(ytr))
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=SEED)
    for k, (a, b) in enumerate(skf.split(Xtr2, ytr)):
        i2, ies = train_test_split(a, test_size=0.15, random_state=SEED + k, stratify=ytr.iloc[a])
        m = CatBoostClassifier(iterations=1000, depth=6, learning_rate=0.05, scale_pos_weight=spw,
                               cat_features=cat_idx, random_seed=SEED, verbose=0,
                               allow_writing_files=False, early_stopping_rounds=40, thread_count=4)
        m.fit(Xtr2.iloc[i2], ytr.iloc[i2], eval_set=(Xtr2.iloc[ies], ytr.iloc[ies]))
        oof[b] = m.predict_proba(Xtr2.iloc[b])[:, 1]
    i2, ies = train_test_split(np.arange(len(ytr)), test_size=0.15, random_state=SEED + 99, stratify=ytr)
    m = CatBoostClassifier(iterations=1000, depth=6, learning_rate=0.05, scale_pos_weight=spw,
                           cat_features=cat_idx, random_seed=SEED, verbose=0,
                           allow_writing_files=False, early_stopping_rounds=40, thread_count=4)
    m.fit(Xtr2.iloc[i2], ytr.iloc[i2], eval_set=(Xtr2.iloc[ies], ytr.iloc[ies]))
    return oof, m.predict_proba(Xte2)[:, 1]


def lgbm_only_fn(X_tr, y_tr, X_te):
    Xtr, Xte = _drop_leaky(X_tr, y_tr, X_te)
    Xtr_pp, ppf = fit_keep_nan(Xtr); Xte_pp = apply_keep_nan(Xte, ppf)
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    params = get_model_params(len(X_tr), spw)
    tp, oof = lgb_oof_and_test(Xtr_pp, y_tr, Xte_pp, params, n_seeds=2, n_folds=5)
    pf = fit_platt(oof, y_tr.values)
    return apply_platt(tp, pf), apply_platt(oof, pf)


def blend_fn(X_tr, y_tr, X_te):
    Xtr, Xte = _drop_leaky(X_tr, y_tr, X_te)
    Xtr_pp, ppf = fit_keep_nan(Xtr); Xte_pp = apply_keep_nan(Xte, ppf)
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    params = get_model_params(len(X_tr), spw)
    l_tp, l_oof = lgb_oof_and_test(Xtr_pp, y_tr, Xte_pp, params, n_seeds=2, n_folds=5)
    lpf = fit_platt(l_oof, y_tr.values)
    r_oof, r_tp = _rf_oof_test(Xtr_pp, y_tr, Xte_pp, 5, spw)
    rpf = fit_platt(r_oof, y_tr.values)
    c_oof, c_tp = _cat_oof_test(Xtr_pp, y_tr, Xte_pp, 5, spw)
    cpf = fit_platt(c_oof, y_tr.values)
    oof = (apply_platt(l_oof, lpf) + apply_platt(r_oof, rpf) + apply_platt(c_oof, cpf)) / 3
    tp = (apply_platt(l_tp, lpf) + apply_platt(r_tp, rpf) + apply_platt(c_tp, cpf)) / 3
    return tp, oof


if __name__ == "__main__":
    df = load_panel("general")
    base = cv_evaluate("general", lgbm_only_fn, label="lgbm", n_repeats=5, df=df)
    bl = cv_evaluate("general", blend_fn, label="lgbm+rf+cat", n_repeats=5, df=df)
    c = compare(base, bl)
    print(f"general LGBM F1*={c['a_f1']:.4f}  BLEND F1*={c['b_f1']:.4f}  "
          f"Δ={c['delta_f1']:+.4f} ({c['sigma']:+.2f}σ)  {c['verdict']}", flush=True)
