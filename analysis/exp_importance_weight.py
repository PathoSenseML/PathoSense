"""Idea #1 — importance-weighted transfer scorer (covariate-shift domain adaptation).

The general scorer trains on all general variants equally, then scores a gene-specific target panel.
Importance weighting upweights the general variants most SIMILAR to the target distribution, focusing
the scorer on the relevant slice of variant space.

PROPER benchmarking:
  * Domain classifier D distinguishes general (0) vs target (1) on FEATURES ONLY (no target labels) ->
    P(target|x). Density-ratio weight for a general sample = p/(1-p), clipped. Uses only the target's
    unlabeled feature distribution, which is available at inference -> legitimate, no label leakage.
  * Both weighted and unweighted scorers are single-LGBM on the SAME general-minus-target matrix
    (build_matrices), so the ONLY difference is the sample weights.
  * Frozen OOS scorer (excludes all target Variant_IDs) -> per-fold CV blends with the target own model
    at the deployed w, threshold on train-OOF at the test prior, F1* on held-out fold. Paired.
Run: .venv/bin/python -u analysis/exp_importance_weight.py [panel ...]
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")

import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from config import SEED
from utils import TEST_PRIORS
from calibration import fit_platt, apply_platt
from harness import pick_threshold_at_prior, tpr_fpr, f1_at_prior
from run_pipeline import get_model_params
from exp_transfer import load_target_engineered, target_deployed_fn
from exp_transfer_blend import build_matrices

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
DEPLOY_W = {"cftr": 1.0, "pah": 1.0, "cancer": 0.5}
N_REPEATS = 10


def domain_weights(Xg, Xt, cats):
    """P(target|x) density-ratio weights for general rows. Features only, no labels."""
    X = pd.concat([Xg, Xt], ignore_index=True)
    d = np.r_[np.zeros(len(Xg)), np.ones(len(Xt))]
    i2, ies = train_test_split(np.arange(len(d)), test_size=0.2, random_state=SEED, stratify=d)
    m = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=SEED, n_jobs=4)
    m.fit(X.iloc[i2], d[i2], eval_set=[(X.iloc[ies], d[ies])],
          callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
          categorical_feature=cats or "auto")
    p = m.predict_proba(Xg)[:, 1]
    auc = roc_auc_score(d, m.predict_proba(X)[:, 1])
    w = np.clip(p / np.clip(1 - p, 1e-3, None), 0.05, 20.0)
    w = w * len(w) / w.sum()                      # normalize mean weight to 1
    return w, auc


def lgb_score(Xg, yg, Xt, cats, weight=None):
    params = get_model_params(len(Xg), (yg == 0).sum() / max((yg == 1).sum(), 1))
    i2, ies = train_test_split(np.arange(len(yg)), test_size=0.15, random_state=SEED + 7, stratify=yg)
    m = LGBMClassifier(**params, random_state=SEED)
    sw = None if weight is None else weight[i2]
    m.fit(Xg.iloc[i2], yg.iloc[i2], sample_weight=sw, eval_set=[(Xg.iloc[ies], yg.iloc[ies])],
          callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
          categorical_feature=cats or "auto")
    return m.predict_proba(Xt)[:, 1]


def run(panel):
    prior = TEST_PRIORS[panel]; w = DEPLOY_W[panel]
    Xg, yg, Xt, yt, cats = build_matrices(panel)
    wt, dauc = domain_weights(Xg, Xt, cats)
    g_un = lgb_score(Xg, yg, Xt, cats, None)
    g_w = lgb_score(Xg, yg, Xt, cats, wt)
    print(f"\n[{panel}] domain-clf AUC(general-vs-target)={dauc:.3f}  weight range=[{wt.min():.2f},{wt.max():.2f}]  "
          f"scorer AUC: unweighted={roc_auc_score(yt, g_un):.4f} weighted={roc_auc_score(yt, g_w):.4f}", flush=True)

    Xt_own, yt2 = load_target_engineered(panel); fn = target_deployed_fn(panel)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=N_REPEATS, random_state=SEED)
    per = {"unweighted": [], "weighted": []}
    for tr, te in rskf.split(Xt_own, yt2):
        tr, te = np.asarray(tr), np.asarray(te)
        ytr, yte = yt2.iloc[tr].reset_index(drop=True), yt2.iloc[te].reset_index(drop=True)
        e_te, e_tr = fn(Xt_own.iloc[tr].reset_index(drop=True), ytr, Xt_own.iloc[te].reset_index(drop=True))
        for key, g in (("unweighted", g_un), ("weighted", g_w)):
            pl = fit_platt(g[tr], ytr.values)
            gt, ge = apply_platt(g[tr], pl), apply_platt(g[te], pl)
            b_tr = w * gt + (1 - w) * e_tr; b_te = w * ge + (1 - w) * e_te
            thr = pick_threshold_at_prior(ytr.values, b_tr, prior)
            tpr, fpr = tpr_fpr(yte.values, (b_te >= thr).astype(int))
            per[key].append(f1_at_prior(tpr, fpr, prior))
    base = np.array(per["unweighted"]); v = np.array(per["weighted"]); d = v - base
    nf = len(base); se = d.std(ddof=1) / np.sqrt(nf); sig = d.mean() / se if se > 0 else 0.0
    vd = "DEPLOY" if d.mean() > 0 and sig >= 2 else "marginal" if d.mean() > 0 and sig >= 1 else "no gain"
    print(f"  [{panel}] unweighted F1*={base.mean():.4f}  weighted F1*={v.mean():.4f}  "
          f"Δ={d.mean():+.4f} ({sig:+.2f}σ)  -> {vd}", flush=True)
    return {"panel": panel, "unweighted": base.mean(), "weighted": v.mean(), "delta": d.mean(), "sigma": sig}


if __name__ == "__main__":
    panels = sys.argv[1:] or ["cftr", "pah", "cancer"]
    print("=== Idea #1: importance-weighted transfer scorer — honest CV ===", flush=True)
    rows = []
    for pn in panels:
        t0 = time.time(); rows.append(run(pn)); print(f"  ({pn} {time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "importance_weight.csv"), index=False)
    print(f"\nsaved -> importance_weight.csv", flush=True)
