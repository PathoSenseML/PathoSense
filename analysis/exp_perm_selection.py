"""Per-feature permutation-importance-guided selection — does pruning by permutation
importance beat all-features? (user idea, 2026-06-15)

Honest design (the ONLY valid way — selection fit per fold on TRAIN only):
  per outer fold:
    1. preprocess (native-NaN, all features)               [fit on train]
    2. train one LGBM on an inner fit-split                 [train only]
    3. per-feature permutation importance = AUC drop on the inner VAL split
    4. for each cut strategy, retrain the multi-seed LGBM on the selected columns,
       pick the threshold on its train OOF, score F1* on the held-out outer fold.
  Compare each strategy PAIRED vs all-features on identical folds.

Cut strategies:
  all        : keep everything (reference = deployed general LGBM recipe)
  drop_neg   : drop features whose permutation importance <= 0 (actively-harmful/dead)
               <- the one thing mRMR can't do; the real shot at a gain
  top200/100 : keep the top-K by importance (aggressive pruning sanity checks)

Note: importance is estimated on a 25% inner val split -> noisy per-feature; that noise
is part of the method's honest performance (it can drop good features by chance).
True iterative RFE (drop-worst-retrain x518) is intractable; this rank-once-per-fold +
cut-sweep captures the same decision far cheaper.

Run:  .venv/bin/python -u analysis/exp_perm_selection.py
"""
import os, sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")

from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from lightgbm import LGBMClassifier

from config import SEED
import harness as H
from harness import (load_panel, f1_at_prior, tpr_fpr, pick_threshold_at_prior,
                     _drop_leaky, lgb_oof_and_test)
from preprocess_v2 import fit_keep_nan, apply_keep_nan
from calibration import fit_platt, apply_platt
from run_pipeline import get_model_params
from schema import partition_columns

PRIOR = 0.143
N_SPLITS, N_REPEATS, N_JOBS = 5, 4, 6
PERM_REPEATS = 3


def perm_importance(model, X, y, seed):
    """AUC-drop permutation importance per single feature (dtype-preserving permute)."""
    base = roc_auc_score(y, model.predict_proba(X)[:, 1])
    rng = np.random.RandomState(seed)
    imp = {}
    for c in X.columns:
        col = X[c]; is_cat = isinstance(col.dtype, pd.CategoricalDtype)
        arr = col.to_numpy(); d = 0.0
        for _ in range(PERM_REPEATS):
            perm = rng.permutation(len(X)); Xp = X.copy()
            Xp[c] = (pd.Categorical(arr[perm], categories=col.cat.categories)
                     if is_cat else arr[perm])
            d += base - roc_auc_score(y, model.predict_proba(Xp)[:, 1])
        imp[c] = d / PERM_REPEATS
    return pd.Series(imp).sort_values(ascending=False)


def eval_fold(X_all, y_all, tr, te):
    Xtr = X_all.iloc[tr].reset_index(drop=True); Xte = X_all.iloc[te].reset_index(drop=True)
    ytr = y_all.iloc[tr].reset_index(drop=True); yte = y_all.iloc[te].reset_index(drop=True)
    Xtr, Xte = _drop_leaky(Xtr, ytr, Xte)
    Xtr_pp, ppf = fit_keep_nan(Xtr); Xte_pp = apply_keep_nan(Xte, ppf)
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    params = get_model_params(len(Xtr_pp), spw)

    # rank features by permutation importance on an inner val split (train only)
    i_fit, i_val = train_test_split(np.arange(len(ytr)), test_size=0.25,
                                    random_state=SEED, stratify=ytr)
    m = LGBMClassifier(**params, random_state=SEED, n_jobs=H._LGB_THREADS)
    m.fit(Xtr_pp.iloc[i_fit], ytr.iloc[i_fit],
          eval_set=[(Xtr_pp.iloc[i_val], ytr.iloc[i_val])],
          callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)])
    rank = perm_importance(m, Xtr_pp.iloc[i_val].reset_index(drop=True),
                           ytr.iloc[i_val].values, seed=SEED)
    cols = list(Xtr_pp.columns)
    strategies = {
        "all":      cols,
        "drop_neg": rank[rank > 0].index.tolist() or rank.head(10).index.tolist(),
        "top200":   rank.head(200).index.tolist(),
        "top100":   rank.head(100).index.tolist(),
    }
    out = {}
    for name, sel in strategies.items():
        sel = [c for c in sel if c in Xtr_pp.columns]
        tp, oof = lgb_oof_and_test(Xtr_pp[sel], ytr, Xte_pp[sel], params, n_seeds=2, n_folds=5)
        pf = fit_platt(oof, ytr.values)
        t = pick_threshold_at_prior(ytr.values, apply_platt(oof, pf), PRIOR)
        tpr, fpr = tpr_fpr(yte.values, (apply_platt(tp, pf) >= t).astype(int))
        out[name] = (f1_at_prior(tpr, fpr, PRIOR), len(sel))
    return out


def main():
    panel = "general"
    df = load_panel(panel)
    _, num_cols, cat_cols, tgt = partition_columns(df)
    X_all = df[num_cols + cat_cols].reset_index(drop=True)
    y_all = df[tgt].astype(int).reset_index(drop=True)
    splits = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                          random_state=SEED).split(X_all, y_all))
    print(f"=== {panel}: permutation-importance selection — honest CV, F1* @ {PRIOR}, "
          f"{len(splits)} folds ===", flush=True)
    from joblib import Parallel, delayed
    H._LGB_THREADS = 1
    t0 = time.time()
    res = Parallel(n_jobs=N_JOBS, backend="threading")(
        delayed(eval_fold)(X_all, y_all, tr, te) for tr, te in splits)
    names = ["all", "drop_neg", "top200", "top100"]
    f1 = {n: np.array([r[n][0] for r in res]) for n in names}
    nfeat = {n: np.mean([r[n][1] for r in res]) for n in names}
    ref = f1["all"]
    print(f"\n{'strategy':10s} {'F1*':>7s} {'SE':>7s} {'~nfeat':>7s}  {'Δ vs all':>9s} {'σ':>6s}  verdict")
    for n in names:
        d = f1[n] - ref; se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        sig = d.mean() / se if se > 0 else 0.0
        v = "ref" if n == "all" else ("DEPLOY" if d.mean() > 0 and sig >= 2 else
             "marginal" if d.mean() > 0 and sig >= 1 else "no gain")
        tag = "" if n == "all" else f"{d.mean():+9.4f} {sig:+6.2f}"
        print(f"{n:10s} {f1[n].mean():7.4f} {f1[n].std(ddof=1)/np.sqrt(len(f1[n])):7.4f} "
              f"{nfeat[n]:7.0f}  {tag}  {v}", flush=True)
    print(f"\n({time.time()-t0:.0f}s)  Deploy bar: Δ>0 at σ≥2.", flush=True)


if __name__ == "__main__":
    main()
