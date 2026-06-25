"""Bonus from the general spw1_bag win: the transfer scorer trains on general, so spw1_bag
(scale_pos_weight=1 + bagging_freq=1/bagging_fraction=0.8) may lift the scorer's AUC → help the small
panels. pah's scorer is SINGLE-LGBM (no blend dilution) → full benefit; cftr/cancer scorers are
LGBM+RF+CatBoost blends → diluted. Honest per-fold CV in the DEPLOYED config, paired Δ vs default scorer.
pah/cftr = pure transfer (cheap); cancer = 0.5·transfer+0.5·own (own refit per fold). 2 seeds. Deploy if
non-negative (and doesn't drop any panel).
Run: .venv/bin/python -u analysis/exp_transfer_spwbag.py [panel ...]
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from config import SEED
from utils import TEST_PRIORS
from calibration import fit_platt, apply_platt
from harness import pick_threshold_at_prior, tpr_fpr, f1_at_prior
from transfer import fit_transfer_scorer, transfer_score, TRANSFER_PANELS
from exp_transfer import _load, load_target_engineered, target_deployed_fn

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
SB = {"scale_pos_weight": 1.0, "bagging_freq": 1, "bagging_fraction": 0.8}


def evaluate(panel):
    prior = TEST_PRIORS[panel]; w_t = TRANSFER_PANELS[panel]
    df = _load(panel); y = df["Label"].astype(int).values
    g_def = transfer_score(df, fit_transfer_scorer(panel))
    g_sb = transfer_score(df, fit_transfer_scorer(panel, model_params=SB))
    print(f"[{panel}] scorer AUC: default={roc_auc_score(y,g_def):.4f}  spw1_bag={roc_auc_score(y,g_sb):.4f}  "
          f"(Δ={roc_auc_score(y,g_sb)-roc_auc_score(y,g_def):+.4f})", flush=True)
    need_own = w_t < 1.0
    if need_own:
        Xt_own, yt = load_target_engineered(panel); fn = target_deployed_fn(panel)
        assert (yt.values == y).all()
    rows = []
    for seed in (SEED, SEED + 1000):
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=12 if not need_own else 8, random_state=seed)
        acc = {"def": [], "sb": []}
        for tr, te in rskf.split(np.zeros(len(y)), y):
            tr, te = np.asarray(tr), np.asarray(te)
            ytr, yte = y[tr], y[te]
            if need_own:
                e_te, e_tr = fn(Xt_own.iloc[tr].reset_index(drop=True), pd.Series(ytr),
                                Xt_own.iloc[te].reset_index(drop=True))
            for key, g in (("def", g_def), ("sb", g_sb)):
                pl = fit_platt(g[tr], ytr); gtr, gte = apply_platt(g[tr], pl), apply_platt(g[te], pl)
                if need_own:
                    b_tr = w_t*gtr + (1-w_t)*e_tr; b_te = w_t*gte + (1-w_t)*e_te
                else:
                    b_tr, b_te = gtr, gte
                t = pick_threshold_at_prior(ytr, b_tr, prior)
                acc[key].append(f1_at_prior(*tpr_fpr(yte, (b_te >= t).astype(int)), prior))
        a, b = np.array(acc["def"]), np.array(acc["sb"]); d = b - a
        se = d.std(ddof=1)/np.sqrt(len(d)); sig = d.mean()/se if se > 0 else 0
        print(f"  seed{seed} ({len(a)} folds): F1* {a.mean():.4f}->{b.mean():.4f}  Δ={d.mean():+.4f} ({sig:+.2f}σ)", flush=True)
        rows.append({"panel": panel, "seed": seed, "f1_def": a.mean(), "f1_sb": b.mean(),
                     "delta": d.mean(), "sigma": sig})
    return rows


if __name__ == "__main__":
    panels = sys.argv[1:] or ["pah", "cftr", "cancer"]
    print("=== transfer-scorer spw1_bag (general-trained scorer) in deployed config ===", flush=True)
    rows = []
    for pn in panels:
        t0 = time.time(); rows += evaluate(pn); print(f"  ({pn} {time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "transfer_spwbag.csv"), index=False)
    print(f"\nsaved -> transfer_spwbag.csv", flush=True)
