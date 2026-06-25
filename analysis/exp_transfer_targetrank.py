"""Domain adaptation: rank AL_ on each TARGET panel's own distribution (vs general's grid).

AL_ values are covariate-shifted across panels (corr ~0.01-0.4; inspect_data_opportunities.py),
so the scorer's general-grid percentile for a target variant is mis-scaled. Ranking the target's
AL_ on the TARGET distribution (fit per fold on target-train -> batch-invariant) should align the
percentile semantics the scorer learned on general, recovering AL signal for pah/cftr (which need
raw AL) and possibly beating cancer's fam_flags.

Configs (raw-AL scorer; per-fold target grids via fit_target_pop_grids):
  baseline        : raw AL, general grid (current pah/cftr deployment)        <- ref
  targetrank      : raw AL, ranked on target-train distribution
  targetrank+ff   : + AL_ missingness fam_flags
Honest paired CV at the deployed w, F1* at the test prior. cancer also lists its DEPLOYED
fam_flags number for reference.

Run:  .venv/bin/python -u analysis/exp_transfer_targetrank.py [pah cftr cancer]
"""
import os, sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from config import SEED
from utils import TEST_PRIORS
from calibration import fit_platt, apply_platt
from harness import pick_threshold_at_prior, tpr_fpr, f1_at_prior, FINALS_N
from exp_transfer import _load, load_target_engineered, target_deployed_fn
from transfer import (fit_transfer_scorer, transfer_score, fit_target_pop_grids,
                      TRANSFER_PANELS)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def run(panel, n_repeats):
    prior = TEST_PRIORS[panel]; w = TRANSFER_PANELS[panel]
    print(f"\n{'='*64}\n  {panel.upper()}  (deployed w={w})\n{'='*64}", flush=True)
    raw = _load(panel)
    Xt, yt = load_target_engineered(panel)
    assert len(raw) == len(Xt)
    # raw-AL scorers (built once)
    sc = fit_transfer_scorer(panel, drop_al=False, al_features=())
    sc_ff = fit_transfer_scorer(panel, drop_al=False, al_features=("fam_flags",))
    g_base = transfer_score(raw, sc)        # general grid, fold-independent
    print(f"  baseline scorer AUC (general grid) = {roc_auc_score(yt, g_base):.4f}", flush=True)

    fn = target_deployed_fn(panel)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED)
    keys = ["baseline", "targetrank", "targetrank+ff"]
    f1 = {k: [] for k in keys}
    t0 = time.time()
    for tr, te in rskf.split(Xt, yt):
        tr, te = np.asarray(tr), np.asarray(te)
        ytr, yte = yt.iloc[tr].reset_index(drop=True), yt.iloc[te].reset_index(drop=True)
        if w >= 1.0:
            e_tr = np.zeros(len(tr)); e_te = np.zeros(len(te))
        else:
            e_te, e_tr = fn(Xt.iloc[tr].reset_index(drop=True), ytr,
                            Xt.iloc[te].reset_index(drop=True))
        raw_tr, raw_te = raw.iloc[tr], raw.iloc[te]
        grids = fit_target_pop_grids(raw_tr, sc)            # fit on target-train fold only
        grids_ff = fit_target_pop_grids(raw_tr, sc_ff)
        cfg_g = {
            "baseline":      (g_base[tr], g_base[te]),
            "targetrank":    (transfer_score(raw_tr, sc, grids), transfer_score(raw_te, sc, grids)),
            "targetrank+ff": (transfer_score(raw_tr, sc_ff, grids_ff),
                              transfer_score(raw_te, sc_ff, grids_ff)),
        }
        for k in keys:
            g_tr, g_te = cfg_g[k]
            pl = fit_platt(g_tr, ytr.values)
            gt, ge = apply_platt(g_tr, pl), apply_platt(g_te, pl)
            b_tr = w * gt + (1 - w) * e_tr; b_te = w * ge + (1 - w) * e_te
            thr = pick_threshold_at_prior(ytr.values, b_tr, prior)
            tpr, fpr = tpr_fpr(yte.values, (b_te >= thr).astype(int))
            f1[k].append(f1_at_prior(tpr, fpr, prior))
    base = np.array(f1["baseline"]); nf = len(base)
    print(f"\n  paired F1* vs baseline ({nf} folds, {time.time()-t0:.0f}s):", flush=True)
    rows = []
    for k in keys:
        arr = np.array(f1[k]); d = arr - base
        se = d.std(ddof=1) / np.sqrt(nf) if nf > 1 else 0.0
        sig = d.mean() / se if se > 0 else 0.0
        tag = "ref" if k == "baseline" else f"Δ={d.mean():+.4f} ({sig:+.2f}σ)"
        print(f"    {k:14s} F1*={arr.mean():.4f}  {tag}", flush=True)
        rows.append({"panel": panel, "config": k, "f1": arr.mean(),
                     "delta": 0 if k == "baseline" else d.mean(),
                     "sigma": 0 if k == "baseline" else sig})
    return rows


if __name__ == "__main__":
    panels = sys.argv[1:] or ["pah", "cftr", "cancer"]
    allr = []
    for pn in panels:
        allr.extend(run(pn, n_repeats=8 if pn == "cancer" else 12))
    pd.DataFrame(allr).to_csv(os.path.join(OUT, "exp_transfer_targetrank.csv"), index=False)
    print(f"\nsaved -> {os.path.join(OUT, 'exp_transfer_targetrank.csv')}", flush=True)
