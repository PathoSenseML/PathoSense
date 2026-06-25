"""Confirm pah plateau-threshold win vs the DEPLOYED bootstrap threshold, robustly.
n_repeats=20, paired; sweep eps to ensure plateau isn't eps-tuned. Deploy if >=2sigma vs boot.

Run:  .venv/bin/python -u analysis/exp_pah_threshold_confirm.py
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")

from sklearn.model_selection import RepeatedStratifiedKFold
from config import SEED
from utils import TEST_PRIORS, optimize_threshold_f1_balanced
from calibration import fit_platt, apply_platt
from harness import f1_at_prior, tpr_fpr, pick_threshold_at_prior
from exp_transfer import _load, load_target_engineered
from transfer import fit_transfer_scorer, transfer_score, TRANSFER_PANELS

PANEL = "pah"


def thr_plateau(ytr, oof, prior, eps):
    ts = np.arange(0.02, 0.98, 0.005)
    f1s = np.array([f1_at_prior(*tpr_fpr(ytr, (oof >= t).astype(int)), prior) for t in ts])
    near = ts[f1s >= f1s.max() - eps]
    return float(np.median(near))


def main():
    prior = TEST_PRIORS[PANEL]; w = TRANSFER_PANELS[PANEL]   # pah w=1.0 (pure transfer)
    g_all = transfer_score(_load(PANEL), fit_transfer_scorer(PANEL))
    Xt, yt = load_target_engineered(PANEL)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=SEED)
    methods = ["sweep", "boot", "plat0.005", "plat0.01", "plat0.02"]
    f1 = {m: [] for m in methods}
    for tr, te in rskf.split(Xt, yt):
        tr, te = np.asarray(tr), np.asarray(te)
        ytr, yte = yt.iloc[tr].values, yt.iloc[te].values
        pl = fit_platt(g_all[tr], ytr)
        oof, test = apply_platt(g_all[tr], pl), apply_platt(g_all[te], pl)
        thr = {
            "sweep": pick_threshold_at_prior(ytr, oof, prior),
            "boot": optimize_threshold_f1_balanced(ytr, oof, target_prior=prior, n_bootstrap=300)[0],
            "plat0.005": thr_plateau(ytr, oof, prior, 0.005),
            "plat0.01": thr_plateau(ytr, oof, prior, 0.01),
            "plat0.02": thr_plateau(ytr, oof, prior, 0.02),
        }
        for m in methods:
            f1[m].append(f1_at_prior(*tpr_fpr(yte, (test >= thr[m]).astype(int)), prior))
    nf = len(f1["boot"]); boot = np.array(f1["boot"])
    print(f"=== pah threshold confirm ({nf} folds, prior={prior}) — paired vs DEPLOYED boot ===", flush=True)
    print(f"  boot (deployed) F1*={boot.mean():.4f}", flush=True)
    for m in ["sweep", "plat0.005", "plat0.01", "plat0.02"]:
        a = np.array(f1[m]); d = a - boot
        se = d.std(ddof=1) / np.sqrt(nf); sig = d.mean() / se if se > 0 else 0
        v = "DEPLOY" if d.mean() > 0 and sig >= 2 else ("marginal" if sig >= 1 else "no")
        print(f"  {m:10s} F1*={a.mean():.4f}  Δ={d.mean():+.4f} ({sig:+.2f}σ)  {v}", flush=True)


if __name__ == "__main__":
    main()
