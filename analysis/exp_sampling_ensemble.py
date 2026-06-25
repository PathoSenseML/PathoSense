"""Upsample + downsample -> ENSEMBLE (page-3 circled idea, 14-06-2026 meeting).

Context: plain resample-to-50/50 already tested NEGATIVE in this project
(scale_pos_weight beats oversampling to 50/50 by +0.026). The
NEW, untested idea here is to train one model on UP-sampled data and one on
DOWN-sampled data, then probability-average them — a sampling-diversity ensemble,
not plain resampling. We give it an honest, paired shot vs the deployed baseline.

Comparisons (all paired on identical folds; F1* at the per-panel test prior):
  base   : deployed scale_pos_weight LightGBM ensemble (no resampling)   <- reference
  up     : oversample minority->50/50, no spw
  down   : undersample majority->50/50, no spw
  updown : (up + down) probability-average  <- the meeting idea

Preprocessing is held IDENTICAL across configs (fit on original train, isolating the
sampling effect on the LEARNER). general uses the reworked native-NaN all-features
recipe (full own model + transfer source); cancer uses the deployed median+mRMR recipe
(its own model is 50% of the deployed blend). pah/cftr are pure-transfer (own model
unused at w=1.0), so own-model sampling there has no deployed effect — omitted.

Run:  .venv/bin/python -u analysis/exp_sampling_ensemble.py
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, train_test_split
import lightgbm as lgb
from lightgbm import LGBMClassifier

from config import SEED
import harness as H
from harness import (cv_evaluate, load_panel, compare, _drop_leaky,
                     lgb_oof_and_test, fast_mrmr)
from preprocess_v2 import fit_keep_nan, apply_keep_nan
from calibration import fit_platt, apply_platt
from run_pipeline import get_model_params, preprocess_data


def _resample(idx, y, mode, rng):
    """Return resampled positional indices (subset/duplication of `idx`)."""
    idx = np.asarray(idx); yi = y.iloc[idx].values
    pos = idx[yi == 1]; neg = idx[yi == 0]
    if len(pos) == 0 or len(neg) == 0:
        return idx
    if mode == "up":                       # duplicate minority up to majority size
        if len(pos) < len(neg):  pos = rng.choice(pos, len(neg), replace=True)
        elif len(neg) < len(pos): neg = rng.choice(neg, len(pos), replace=True)
    elif mode == "down":                   # drop majority down to minority size
        k = min(len(pos), len(neg))
        pos = rng.choice(pos, k, replace=False); neg = rng.choice(neg, k, replace=False)
    out = np.concatenate([pos, neg]); rng.shuffle(out)
    return out


def _sampled_oof_test(X, y, Xte, params, mode, n_seeds=2, n_folds=5):
    """Multi-seed LightGBM trained on resampled folds; OOF predicted on ORIGINAL
    held-out rows (so OOF stays aligned to X for honest threshold/Platt). Early
    stopping uses a clean (un-resampled) eval slice."""
    nf = min(n_folds, max(2, int(y.sum()), int((y == 0).sum())))
    patience = 50 if len(y) > 300 else 30
    oof_all = np.zeros((n_seeds, len(y))); test_all = np.zeros((n_seeds, len(Xte)))
    for s in range(n_seeds):
        rng = np.random.RandomState(SEED + s)
        p = {**params, "random_state": SEED + s, "n_jobs": H._LGB_THREADS}
        skf = StratifiedKFold(nf, shuffle=True, random_state=SEED + s)
        oof = np.zeros(len(y))
        for a, b in skf.split(X, y):
            a = np.asarray(a)
            a_tr, a_es = train_test_split(a, test_size=0.15, random_state=SEED + s,
                                          stratify=y.iloc[a])
            ridx = _resample(a_tr, y, mode, rng)
            m = LGBMClassifier(**p)
            m.fit(X.iloc[ridx], y.iloc[ridx], eval_set=[(X.iloc[a_es], y.iloc[a_es])],
                  callbacks=[lgb.early_stopping(patience, verbose=False), lgb.log_evaluation(0)])
            oof[b] = m.predict_proba(X.iloc[b])[:, 1]
        oof_all[s] = oof
        f_tr, f_es = train_test_split(np.arange(len(y)), test_size=0.15,
                                      random_state=SEED + s, stratify=y)
        ridx = _resample(f_tr, y, mode, rng)
        m = LGBMClassifier(**p)
        m.fit(X.iloc[ridx], y.iloc[ridx], eval_set=[(X.iloc[f_es], y.iloc[f_es])],
              callbacks=[lgb.early_stopping(patience, verbose=False), lgb.log_evaluation(0)])
        test_all[s] = m.predict_proba(Xte)[:, 1]
    return test_all.mean(0), oof_all.mean(0)


# ── preprocessing variants (held identical across sampling configs) ───────────

def _prep_general(X_tr, y_tr, X_te):
    Xtr, Xte = _drop_leaky(X_tr, y_tr, X_te)
    Xtr_pp, ppf = fit_keep_nan(Xtr); Xte_pp = apply_keep_nan(Xte, ppf)
    return Xtr_pp, Xte_pp

def _prep_mrmr(X_tr, y_tr, X_te):
    Xtr, Xte = _drop_leaky(X_tr, y_tr, X_te)
    Xtr_pp, _, Xte_pp, _ = preprocess_data(Xtr.copy(), Xtr.copy(), Xte.copy(), corr_threshold=0.95)
    Xtr_pp, _, Xte_pp, _ = fast_mrmr(Xtr_pp, y_tr, Xte_pp, Xte_pp)
    return Xtr_pp, Xte_pp


def make_fn(prep, mode):
    """mode in {'base','up','down','updown'}; prep is a preprocessing function."""
    def _fn(X_tr, y_tr, X_te):
        Xtr_pp, Xte_pp = prep(X_tr, y_tr, X_te)
        if mode == "base":
            spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
            params = get_model_params(len(X_tr), spw)
            tp, oof = lgb_oof_and_test(Xtr_pp, y_tr, Xte_pp, params, n_seeds=2, n_folds=5)
        else:
            params = get_model_params(len(X_tr), 1.0)   # balanced via resampling -> no spw
            if mode in ("up", "down"):
                tp, oof = _sampled_oof_test(Xtr_pp, y_tr, Xte_pp, params, mode)
            else:  # updown ensemble
                u_tp, u_oof = _sampled_oof_test(Xtr_pp, y_tr, Xte_pp, params, "up")
                d_tp, d_oof = _sampled_oof_test(Xtr_pp, y_tr, Xte_pp, params, "down")
                tp, oof = (u_tp + d_tp) / 2, (u_oof + d_oof) / 2
        pf = fit_platt(oof, y_tr.values)
        return apply_platt(tp, pf), apply_platt(oof, pf)
    return _fn


def run(panel, prep, modes, n_repeats):
    df = load_panel(panel)
    base = cv_evaluate(panel, make_fn(prep, "base"), label="base", n_repeats=n_repeats, df=df)
    print(f"\n[{panel}] base scale_pos_weight  F1*={base.summary()['f1_mean']:.4f}", flush=True)
    for m in modes:
        r = cv_evaluate(panel, make_fn(prep, m), label=m, n_repeats=n_repeats, df=df)
        c = compare(base, r)
        print(f"[{panel}] {m:7s} F1*={c['b_f1']:.4f}  Δ={c['delta_f1']:+.4f} "
              f"({c['sigma']:+.2f}σ)  {c['verdict']}", flush=True)


if __name__ == "__main__":
    print("=== Sampling-ensemble experiment — paired honest CV (F1* at test prior) ===", flush=True)
    run("general", _prep_general, ["up", "down", "updown"], n_repeats=4)
    run("cancer",  _prep_mrmr,    ["updown"],               n_repeats=8)
    print("\nReference: deployed scale_pos_weight. Δ>0 & σ≥2 would beat it.", flush=True)
