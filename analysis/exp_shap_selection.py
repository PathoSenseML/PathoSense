"""SHAP-importance-guided selection — can ranking columns by mean|SHAP| and keeping the
top tune general's performance? (user idea, 2026-06-15; companion to exp_perm_selection.py)

SHAP differs from permutation importance under correlation: TreeSHAP SPLITS credit across
correlated features (so each of the ~300 AL_ columns gets a nonzero share), whereas
permutation zeroes them (others compensate). So a SHAP top-K cut preserves the AL block far
better than permutation's drop_neg did -> a fairer shot for selection. But SHAP measures
contribution MAGNITUDE, not generalization HARM, and LightGBM already ignores dead features,
so the honest expectation is "match all-features, or mildly trim the useful tail" — not a gain.

Honest per-fold design (selection fit on TRAIN only):
  per outer fold: preprocess (native-NaN, all feats) -> train LGBM on inner fit-split ->
  rank features by mean|SHAP| (TreeSHAP pred_contrib) on the inner VAL split ->
  for each cut, retrain the multi-seed LGBM on the selected cols, threshold on its train OOF,
  score F1* on the held-out outer fold. Paired vs all-features on identical folds.

Run:  .venv/bin/python -u analysis/exp_shap_selection.py
"""
import os, sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")

from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
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


def shap_rank(model, X):
    """Rank features by mean|SHAP| via LightGBM native TreeSHAP (pred_contrib)."""
    contrib = model.booster_.predict(X, pred_contrib=True)   # last col = base value
    mean_abs = np.abs(contrib[:, :-1]).mean(axis=0)
    return pd.Series(mean_abs, index=list(model.booster_.feature_name())).sort_values(ascending=False)


def eval_fold(X_all, y_all, tr, te):
    Xtr = X_all.iloc[tr].reset_index(drop=True); Xte = X_all.iloc[te].reset_index(drop=True)
    ytr = y_all.iloc[tr].reset_index(drop=True); yte = y_all.iloc[te].reset_index(drop=True)
    Xtr, Xte = _drop_leaky(Xtr, ytr, Xte)
    Xtr_pp, ppf = fit_keep_nan(Xtr); Xte_pp = apply_keep_nan(Xte, ppf)
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    params = get_model_params(len(Xtr_pp), spw)

    i_fit, i_val = train_test_split(np.arange(len(ytr)), test_size=0.25,
                                    random_state=SEED, stratify=ytr)
    m = LGBMClassifier(**params, random_state=SEED, n_jobs=H._LGB_THREADS)
    m.fit(Xtr_pp.iloc[i_fit], ytr.iloc[i_fit],
          eval_set=[(Xtr_pp.iloc[i_val], ytr.iloc[i_val])],
          callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)])
    rank = shap_rank(m, Xtr_pp.iloc[i_val].reset_index(drop=True))
    nonzero = rank[rank > 0].index.tolist()      # features the model actually used
    cols = list(Xtr_pp.columns)
    strategies = {
        "all":        cols,
        "drop_zero":  nonzero or rank.head(10).index.tolist(),   # drop never-used feats
        "top250":     rank.head(250).index.tolist(),
        "top150":     rank.head(150).index.tolist(),
        "top80":      rank.head(80).index.tolist(),
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
    print(f"=== {panel}: SHAP-importance selection — honest CV, F1* @ {PRIOR}, "
          f"{len(splits)} folds ===", flush=True)
    from joblib import Parallel, delayed
    H._LGB_THREADS = 1
    t0 = time.time()
    res = Parallel(n_jobs=N_JOBS, backend="threading")(
        delayed(eval_fold)(X_all, y_all, tr, te) for tr, te in splits)
    names = ["all", "drop_zero", "top250", "top150", "top80"]
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
