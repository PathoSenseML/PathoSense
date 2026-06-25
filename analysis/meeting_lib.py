"""Shared building block for the panel-cleanup experiments.

native_nan_oof: a self-contained native-NaN LightGBM OOF+test pipeline (same recipe
as exp_general_blend.lgbm_only_fn) extended with extra levers:
  * fit_mask    — rows EXCLUDED from model fitting but kept in the returned OOF
                  (so the harness threshold step still sees a full-length, aligned
                  OOF). Used to drop CAT_6 bad-read TRAIN rows without dropping any
                  held-out / test row.
  * drop_cols   — feature columns removed before preprocessing (CAT dedup/merge).
  * label_encode— ordinal-encode the native categoricals (NaN->-1) so non-CatBoost
                  trees can consume them.
  * post(Xtr_pp, ytr, Xte_pp, fit_idx) -> (Xtr_pp2, Xte_pp2)
                  matrix-level transform fit on fit rows only (PCA / scaling / AL
                  transform for items 2 & 4). Must be batch-invariant.

OOF convention matches the codebase's lgb_oof_and_test (early-stopping eval_set = the
held-out fold); identical between baseline and variant, so paired deltas are fair.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split

import harness                       # for harness._LGB_THREADS (set per-call by cv_evaluate)
from harness import _drop_leaky
from config import SEED
from preprocess_v2 import fit_keep_nan, apply_keep_nan
from calibration import fit_platt, apply_platt
from run_pipeline import get_model_params


def _label_encode(Xtr_pp, Xte_pp):
    Xtr_pp, Xte_pp = Xtr_pp.copy(), Xte_pp.copy()
    for c in list(Xtr_pp.columns):
        if isinstance(Xtr_pp[c].dtype, pd.CategoricalDtype):
            cats = list(Xtr_pp[c].cat.categories)
            Xtr_pp[c] = Xtr_pp[c].cat.codes.astype("int32")
            Xte_pp[c] = pd.Categorical(Xte_pp[c].astype(object),
                                       categories=cats).codes.astype("int32")
    return Xtr_pp, Xte_pp


def native_nan_oof(Xtr, ytr, Xte, *, fit_mask=None, drop_cols=(), label_encode=False,
                   post=None, n_seeds=2, n_folds=5, params=None, calibrate=True):
    ytr = ytr.reset_index(drop=True).astype(int)
    Xtr = Xtr.reset_index(drop=True); Xte = Xte.reset_index(drop=True)
    dd = [c for c in drop_cols if c in Xtr.columns]
    if dd:
        Xtr = Xtr.drop(columns=dd)
        Xte = Xte.drop(columns=[c for c in dd if c in Xte.columns])
    Xtr, Xte = _drop_leaky(Xtr, ytr, Xte)

    n = len(ytr)
    fit_idx = (np.arange(n) if fit_mask is None
               else np.where(np.asarray(fit_mask, dtype=bool))[0])
    # preprocessing fit on the FIT rows only, applied to all train + test
    _, ppf = fit_keep_nan(Xtr.iloc[fit_idx].reset_index(drop=True))
    Xtr_pp = apply_keep_nan(Xtr, ppf)
    Xte_pp = apply_keep_nan(Xte, ppf)
    if label_encode:
        Xtr_pp, Xte_pp = _label_encode(Xtr_pp, Xte_pp)
    if post is not None:
        Xtr_pp, Xte_pp = post(Xtr_pp, ytr, Xte_pp, fit_idx)

    yfit = ytr.iloc[fit_idx]
    spw = (yfit == 0).sum() / max((yfit == 1).sum(), 1)
    if params is None:
        params = get_model_params(len(fit_idx), spw)
    cat_feats = [c for c in Xtr_pp.columns if isinstance(Xtr_pp[c].dtype, pd.CategoricalDtype)]
    threads = harness._LGB_THREADS
    patience = 50 if len(fit_idx) > 300 else 30
    nf = min(n_folds, max(2, int(yfit.sum()), int((yfit == 0).sum())))
    fitset = set(int(i) for i in fit_idx)

    oof_all = np.zeros((n_seeds, n)); test_all = np.zeros((n_seeds, len(Xte_pp)))
    for s in range(n_seeds):
        seed_i = SEED + s
        p = {**params, "random_state": seed_i, "n_jobs": threads}
        skf = StratifiedKFold(nf, shuffle=True, random_state=seed_i)
        oof = np.zeros(n)
        for a, b in skf.split(Xtr_pp, ytr):
            a_fit = np.array([i for i in a if i in fitset])
            m = lgb.LGBMClassifier(**p)
            m.fit(Xtr_pp.iloc[a_fit], ytr.iloc[a_fit],
                  eval_set=[(Xtr_pp.iloc[b], ytr.iloc[b])],
                  callbacks=[lgb.early_stopping(patience, verbose=False),
                             lgb.log_evaluation(0)],
                  categorical_feature=cat_feats or "auto")
            oof[b] = m.predict_proba(Xtr_pp.iloc[b])[:, 1]
        oof_all[s] = oof
        Xf, Xes, yf2, yes = train_test_split(Xtr_pp.iloc[fit_idx], yfit, test_size=0.1,
                                             random_state=seed_i, stratify=yfit)
        final = lgb.LGBMClassifier(**p)
        final.fit(Xf, yf2, eval_set=[(Xes, yes)],
                  callbacks=[lgb.early_stopping(patience, verbose=False),
                             lgb.log_evaluation(0)],
                  categorical_feature=cat_feats or "auto")
        test_all[s] = final.predict_proba(Xte_pp)[:, 1]

    oof_m = oof_all.mean(0); test_m = test_all.mean(0)
    if calibrate:
        pf = fit_platt(oof_m, ytr.values)
        return apply_platt(test_m, pf), apply_platt(oof_m, pf)
    return test_m, oof_m
