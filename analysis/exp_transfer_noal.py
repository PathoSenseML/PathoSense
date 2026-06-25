"""Does dropping the (non-transferring) AL_ allele-freq VALUES improve the transfer scorer?

Data finding (inspect_data_opportunities.py): for the SAME variant across panels, AL_ values
are essentially uncorrelated (corr 0.43 cancer / 0.01 pah / 0.02 cftr, wildly different scale),
while EK_ conservation is stable (corr ~0.98). The scorer is trained on general's AL_ values
and scores the target's drifted ones -> AL_ is likely noise/dead-weight in transfer.

Test configs (deployed scorer via fit_transfer_scorer + transfer_score, full blend):
  baseline       : current (AL_ values kept, ranked on general)          <- ref
  no_AL          : drop raw AL_ values (EK_/AA_/CAT_ + missingness only)
  no_AL+famflags : drop AL_ values, add AL_ missingness fam_flags back
  famflags_add   : keep AL_ values AND add fam_flags

Reports (per panel): scorer AUC on the target (prior-independent — does AL help ranking?) and
paired F1* at the deployed w (honest CV). pah is single-LGBM (cleanest); cftr/cancer are blends.

Run:  .venv/bin/python -u analysis/exp_transfer_noal.py [pah cftr cancer]
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
from harness import pick_threshold_at_prior, tpr_fpr, f1_at_prior, FINALS_N, bootstrap_f1_mcc
from exp_transfer import _load, load_target_engineered, target_deployed_fn
from transfer import fit_transfer_scorer, transfer_score, TRANSFER_PANELS

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

CONFIGS = [   # (tag, fit_transfer_scorer kwargs)
    ("baseline",       dict()),
    ("no_AL",          dict(drop_al=True)),
    ("no_AL+famflags", dict(drop_al=True, al_features=("fam_flags",))),
    ("famflags_add",   dict(al_features=("fam_flags",))),
]


def run(panel, n_repeats):
    prior = TEST_PRIORS[panel]; n_target = FINALS_N[panel]; w = TRANSFER_PANELS[panel]
    print(f"\n{'='*64}\n  {panel.upper()}  (deployed w={w})\n{'='*64}", flush=True)
    Xt, yt = load_target_engineered(panel); raw = _load(panel)
    scores = {}
    for tag, kw in CONFIGS:
        t0 = time.time()
        g = transfer_score(raw, fit_transfer_scorer(panel, **kw))
        scores[tag] = g
        print(f"  [{panel}] {tag:15s} scorer AUC on target = {roc_auc_score(yt, g):.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
    fn = target_deployed_fn(panel)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED)
    f1 = {tag: [] for tag, _ in CONFIGS}
    for tr, te in rskf.split(Xt, yt):
        tr, te = np.asarray(tr), np.asarray(te)
        ytr, yte = yt.iloc[tr].reset_index(drop=True), yt.iloc[te].reset_index(drop=True)
        if w >= 1.0:
            e_tr = np.zeros(len(tr)); e_te = np.zeros(len(te))
        else:
            e_te, e_tr = fn(Xt.iloc[tr].reset_index(drop=True), ytr,
                            Xt.iloc[te].reset_index(drop=True))
        for tag, _ in CONFIGS:
            g = scores[tag]
            pl = fit_platt(g[tr], ytr.values)
            gt, ge = apply_platt(g[tr], pl), apply_platt(g[te], pl)
            b_tr = w * gt + (1 - w) * e_tr; b_te = w * ge + (1 - w) * e_te
            thr = pick_threshold_at_prior(ytr.values, b_tr, prior)
            tpr, fpr = tpr_fpr(yte.values, (b_te >= thr).astype(int))
            f1[tag].append(f1_at_prior(tpr, fpr, prior))
    base = np.array(f1["baseline"]); nf = len(base)
    print(f"\n  paired F1* vs baseline ({nf} folds):", flush=True)
    rows = []
    for tag, _ in CONFIGS:
        arr = np.array(f1[tag]); d = arr - base
        se = d.std(ddof=1) / np.sqrt(nf) if nf > 1 else 0.0
        sig = d.mean() / se if se > 0 else 0.0
        auc = roc_auc_score(yt, scores[tag])
        tagstr = "ref" if tag == "baseline" else f"Δ={d.mean():+.4f} ({sig:+.2f}σ)"
        print(f"    {tag:15s} F1*={arr.mean():.4f}  AUC={auc:.4f}  {tagstr}", flush=True)
        rows.append({"panel": panel, "config": tag, "f1": arr.mean(), "auc": auc,
                     "delta": 0 if tag == "baseline" else d.mean(),
                     "sigma": 0 if tag == "baseline" else sig})
    return rows


if __name__ == "__main__":
    panels = sys.argv[1:] or ["pah", "cftr", "cancer"]
    allr = []
    for pn in panels:
        allr.extend(run(pn, n_repeats=8 if pn == "cancer" else 12))
    pd.DataFrame(allr).to_csv(os.path.join(OUT, "exp_transfer_noal.csv"), index=False)
    print(f"\nsaved -> {os.path.join(OUT, 'exp_transfer_noal.csv')}", flush=True)
