"""Label-noise cleaning (confident-learning style), HONESTLY validated. The variant-
pathogenicity literature shows clinical labels are noisy (ClinVar reinterpretation
downgraded ~40% of some 'pathogenic' calls). We identify suspected-mislabeled TRAINING
variants via inner out-of-fold predictions (labeled-1-but-confidently-benign, or vice
versa), remove the top fraction, retrain, and predict the UNTOUCHED outer fold.

Crucial honesty: cleaning uses only the outer-TRAIN fold; the outer-TEST fold's labels
are never inspected. If removing the disagreers HELPS the outer fold, they were noise;
if it HURTS, they were signal (hard cases). Paired vs no-cleaning, analytic F1* at prior.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
from harness import (cv_evaluate, load_panel, compare, deployed_pipeline_fn,
                     lgb_oof_and_test, _drop_leaky)
from pipeline_v2 import make_fn as recipe_fn
from preprocess_v2 import fit_keep_nan, apply_keep_nan, fast_mrmr
from feature_engineering import fit_score_agreement, apply_score_agreement
from calibration import fit_platt, apply_platt
from run_pipeline import preprocess_data, get_model_params
import lightgbm as lgb

PANELS = ["general", "cancer", "pah", "cftr"]
RECIPE = {
    "general": dict(impute="none", feature_select="none", score_agree=False),
    "cancer":  dict(impute="median", feature_select="mrmr", score_agree=True),
    "pah":     dict(impute="median", feature_select="mrmr", score_agree=False),
    "cftr":    dict(impute="median", feature_select="mrmr", score_agree=True),
}


def _prep(panel, Xtr, y_tr, Xte):
    """Apply the panel recipe preprocessing+FS; return (Xtr_pp, Xte_pp)."""
    r = RECIPE[panel]
    if r["impute"] == "none":
        Xtr_pp, ppf = fit_keep_nan(Xtr); Xte_pp = apply_keep_nan(Xte, ppf)
    else:
        Xtr_pp, _, Xte_pp, _ = preprocess_data(Xtr.copy(), Xtr.copy(), Xte.copy(), 0.95)
    if r["score_agree"]:
        sa = fit_score_agreement(Xtr_pp, y_tr, top_k=7)
        Xtr_pp = apply_score_agreement(Xtr_pp, sa); Xte_pp = apply_score_agreement(Xte_pp, sa)
    if r["feature_select"] == "mrmr":
        Xtr_pp, _, Xte_pp, _ = fast_mrmr(Xtr_pp, y_tr, Xte_pp, Xte_pp)
    return Xtr_pp, Xte_pp


def clean_fn(panel, remove_frac):
    def _fn(X_tr, y_tr, X_te):
        Xtr, Xte = _drop_leaky(X_tr, y_tr, X_te)
        Xtr_pp, Xte_pp = _prep(panel, Xtr, y_tr, Xte)
        spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        params = get_model_params(len(X_tr), spw)
        nf = 3 if len(X_tr) < 200 else 5
        # inner OOF on TRAIN to score self-confidence of the given label
        _, oof = lgb_oof_and_test(Xtr_pp, y_tr, Xtr_pp, params, n_seeds=1, n_folds=nf)
        suspicion = np.where(y_tr.values == 1, 1 - oof, oof)     # high => label-disagreed
        k = int(remove_frac * len(y_tr))
        keep = np.ones(len(y_tr), bool)
        if k > 0:
            keep[np.argsort(-suspicion)[:k]] = False
        Xc = Xtr_pp[keep].reset_index(drop=True); yc = y_tr[keep].reset_index(drop=True)
        test_probs, oof_c = lgb_oof_and_test(Xc, yc, Xte_pp, params, n_seeds=1, n_folds=nf)
        pf = fit_platt(oof_c, yc.values)
        # threshold OOF must align to ALL train rows; re-score kept rows' oof (cleaned model)
        return apply_platt(test_probs, pf), apply_platt(oof_c, pf), yc
    # wrap: harness expects (test_probs, train_oof) with train_oof aligned to y_tr.
    def _wrapped(X_tr, y_tr, X_te):
        tp, oof_c, yc = _fn(X_tr, y_tr, X_te)
        return tp, oof_c, yc
    return _wrapped


def run(panel, n_repeats=6):
    """Paired: each cleaning fraction vs frac=0 (SAME new recipe, no removal), identical folds."""
    df = load_panel(panel); rep = n_repeats
    from harness import RepeatedStratifiedKFold, pick_threshold_at_prior, tpr_fpr, f1_at_prior
    from schema import partition_columns
    from utils import TEST_PRIORS
    prior = TEST_PRIORS[panel]
    _, n, c, t = partition_columns(df)
    X = df[n + c].reset_index(drop=True); y = df[t].reset_index(drop=True).astype(int)
    fracs = [0.0, 0.03, 0.06, 0.10]
    perf = {f: [] for f in fracs}
    fns = {f: clean_fn(panel, f) for f in fracs}
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=rep, random_state=42).split(X, y):
        Xtr, Xte = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True)
        ytr, yte = y.iloc[tr].reset_index(drop=True), y.iloc[te].reset_index(drop=True)
        for f in fracs:
            tp, oof_c, yc = fns[f](Xtr, ytr, Xte)
            thr = pick_threshold_at_prior(yc.values, oof_c, prior)
            tpr, fpr = tpr_fpr(yte.values, (tp >= thr).astype(int))
            perf[f].append(f1_at_prior(tpr, fpr, prior))
    base = np.array(perf[0.0])
    print(f"\n=== {panel} (new recipe, no cleaning) F1*={base.mean():.4f} ===", flush=True)
    for f in fracs[1:]:
        arr = np.array(perf[f]); d = arr - base
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0
        print(f"  clean {int(f*100):2d}%: F1*={arr.mean():.4f}  Δ={d.mean():+.4f} "
              f"({d.mean()/se if se else 0:+.2f}σ)", flush=True)


if __name__ == "__main__":
    for pn in (sys.argv[1:] or ["general", "pah"]):
        run(pn)
