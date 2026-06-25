"""Unified per-panel sweep: best {AL treatment} x {blend weight w} for the transfer pipeline.

AL_ treatment of the transfer scorer:
  rawAL    : raw AL, general grid (current pah/cftr)
  targrank : raw AL, ranked on target-train distribution (domain adaptation)
  famflags : drop raw AL, use AL_ missingness fam_flags (current cancer)
Blend weight w sweeps general-transfer vs the panel's OWN model (own brings raw-AL signal that can
decorrelate a famflags/targrank transfer — the trick that won cancer; never tried on pah which is
pure-transfer w=1.0).

Per fold: own model `e` computed once (shared); for each treatment the scorer g (Platt per fold);
blend w*g+(1-w)*e; threshold on train blend at the test prior; F1*. Paired vs the DEPLOYED config
per panel. Goal: lift pah (0.660, the weak spot) and confirm/refine cancer + cftr.

Run:  .venv/bin/python -u analysis/exp_transfer_best.py [pah cftr cancer]
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
W_GRID = [0.0, 0.25, 0.5, 0.6, 0.75, 0.85, 1.0]
TREATMENTS = ["rawAL", "targrank", "famflags"]
# deployed config per panel (treatment, w) — the reference cell
DEPLOYED = {"pah": ("rawAL", 1.0), "cftr": ("rawAL", 1.0), "cancer": ("famflags", 0.5)}


def run(panel, n_repeats):
    prior = TEST_PRIORS[panel]
    print(f"\n{'='*64}\n  {panel.upper()}\n{'='*64}", flush=True)
    raw = _load(panel); Xt, yt = load_target_engineered(panel)
    sc_raw = fit_transfer_scorer(panel, drop_al=False, al_features=())
    sc_ff = fit_transfer_scorer(panel, drop_al=True, al_features=("fam_flags",))
    g_raw = transfer_score(raw, sc_raw)                 # general grid (fold-independent)
    g_ff = transfer_score(raw, sc_ff)                   # famflags (fold-independent)
    print(f"  AUC: rawAL(gen grid)={roc_auc_score(yt,g_raw):.4f}  "
          f"famflags={roc_auc_score(yt,g_ff):.4f}", flush=True)
    fn = target_deployed_fn(panel)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED)
    cells = {(t, w): [] for t in TREATMENTS for w in W_GRID}
    t0 = time.time()
    for tr, te in rskf.split(Xt, yt):
        tr, te = np.asarray(tr), np.asarray(te)
        ytr, yte = yt.iloc[tr].reset_index(drop=True), yt.iloc[te].reset_index(drop=True)
        e_te, e_tr = fn(Xt.iloc[tr].reset_index(drop=True), ytr,
                        Xt.iloc[te].reset_index(drop=True))     # own model, shared
        grids = fit_target_pop_grids(raw.iloc[tr], sc_raw)
        g_targ_tr = transfer_score(raw.iloc[tr], sc_raw, grids)
        g_targ_te = transfer_score(raw.iloc[te], sc_raw, grids)
        gmap = {
            "rawAL":    (g_raw[tr], g_raw[te]),
            "targrank": (g_targ_tr, g_targ_te),
            "famflags": (g_ff[tr], g_ff[te]),
        }
        for t in TREATMENTS:
            g_tr, g_te = gmap[t]
            pl = fit_platt(g_tr, ytr.values)
            gt, ge = apply_platt(g_tr, pl), apply_platt(g_te, pl)
            for w in W_GRID:
                b_tr = w * gt + (1 - w) * e_tr; b_te = w * ge + (1 - w) * e_te
                thr = pick_threshold_at_prior(ytr.values, b_tr, prior)
                tpr, fpr = tpr_fpr(yte.values, (b_te >= thr).astype(int))
                cells[(t, w)].append(f1_at_prior(tpr, fpr, prior))

    dep = np.array(cells[DEPLOYED[panel]]); nf = len(dep)
    print(f"  deployed {DEPLOYED[panel]} F1*={dep.mean():.4f}  ({nf} folds, {time.time()-t0:.0f}s)", flush=True)
    rows = []
    best = None
    for t in TREATMENTS:
        for w in W_GRID:
            arr = np.array(cells[(t, w)]); d = arr - dep
            se = d.std(ddof=1) / np.sqrt(nf) if nf > 1 else 0.0
            sig = d.mean() / se if se > 0 else 0.0
            rows.append({"panel": panel, "treatment": t, "w": w, "f1": arr.mean(),
                         "delta_vs_deployed": d.mean(), "sigma": sig})
            if best is None or arr.mean() > best["f1"]:
                best = rows[-1]
    # print the top cells
    top = sorted(rows, key=lambda r: -r["f1"])[:6]
    print(f"  {'treatment':10s} {'w':>4s} {'F1*':>7s} {'Δdeployed':>10s} {'σ':>7s}", flush=True)
    for r in top:
        print(f"  {r['treatment']:10s} {r['w']:>4} {r['f1']:>7.4f} "
              f"{r['delta_vs_deployed']:>+10.4f} {r['sigma']:>+7.2f}", flush=True)
    print(f"  -> BEST: {best['treatment']} w={best['w']} F1*={best['f1']:.4f} "
          f"(Δ={best['delta_vs_deployed']:+.4f}, {best['sigma']:+.2f}σ vs deployed)", flush=True)
    return rows


if __name__ == "__main__":
    panels = sys.argv[1:] or ["pah", "cftr", "cancer"]
    allr = []
    for pn in panels:
        allr.extend(run(pn, n_repeats=10 if pn != "cancer" else 8))
    pd.DataFrame(allr).to_csv(os.path.join(OUT, "exp_transfer_best.csv"), index=False)
    print(f"\nsaved -> {os.path.join(OUT, 'exp_transfer_best.csv')}", flush=True)
