"""High-fidelity confirmation of the two transfer candidates from exp_transfer_best.py:
  pah : target-rank AL @ w=1.0   (vs deployed rawAL @ w=1.0)
  cftr: target-rank AL @ w=0.85 + rawAL @ w=0.85 + target-rank @ w=1.0 (vs deployed rawAL @ w=1.0)
n_repeats=20, paired vs deployed. Deploy cells clearing >=2sigma.

Run:  .venv/bin/python -u analysis/exp_confirm_pah_cftr.py
"""
import os, sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")

from sklearn.model_selection import RepeatedStratifiedKFold
from config import SEED
from utils import TEST_PRIORS
from calibration import fit_platt, apply_platt
from harness import pick_threshold_at_prior, tpr_fpr, f1_at_prior
from exp_transfer import _load, load_target_engineered, target_deployed_fn
from transfer import fit_transfer_scorer, transfer_score, fit_target_pop_grids

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def run(panel, candidates, n_repeats=20):
    """candidates: list of (tag, treatment in {rawAL,targrank}, w). 'deployed' = (rawAL, 1.0)."""
    prior = TEST_PRIORS[panel]
    print(f"\n{'='*60}\n  {panel.upper()}  (n_repeats={n_repeats})\n{'='*60}", flush=True)
    raw = _load(panel); Xt, yt = load_target_engineered(panel)
    sc = fit_transfer_scorer(panel, drop_al=False, al_features=())
    g_raw = transfer_score(raw, sc)
    fn = target_deployed_fn(panel)
    need_own = any(w < 1.0 for _, _, w in candidates + [("dep", "rawAL", 1.0)])
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED)
    keys = ["deployed"] + [c[0] for c in candidates]
    f1 = {k: [] for k in keys}
    t0 = time.time()
    for tr, te in rskf.split(Xt, yt):
        tr, te = np.asarray(tr), np.asarray(te)
        ytr, yte = yt.iloc[tr].reset_index(drop=True), yt.iloc[te].reset_index(drop=True)
        e_tr = e_te = None
        if need_own:
            e_te, e_tr = fn(Xt.iloc[tr].reset_index(drop=True), ytr,
                            Xt.iloc[te].reset_index(drop=True))
        grids = fit_target_pop_grids(raw.iloc[tr], sc)
        g_targ_tr = transfer_score(raw.iloc[tr], sc, grids)
        g_targ_te = transfer_score(raw.iloc[te], sc, grids)
        gmap = {"rawAL": (g_raw[tr], g_raw[te]), "targrank": (g_targ_tr, g_targ_te)}

        def score(treat, w):
            g_tr, g_te = gmap[treat]
            pl = fit_platt(g_tr, ytr.values)
            gt, ge = apply_platt(g_tr, pl), apply_platt(g_te, pl)
            etr = e_tr if e_tr is not None else 0.0
            ete = e_te if e_te is not None else 0.0
            b_tr = w * gt + (1 - w) * etr; b_te = w * ge + (1 - w) * ete
            thr = pick_threshold_at_prior(ytr.values, b_tr, prior)
            tpr, fpr = tpr_fpr(yte.values, (b_te >= thr).astype(int))
            return f1_at_prior(tpr, fpr, prior)

        f1["deployed"].append(score("rawAL", 1.0))
        for tag, treat, w in candidates:
            f1[tag].append(score(treat, w))

    dep = np.array(f1["deployed"]); nf = len(dep)
    print(f"  deployed (rawAL,w=1.0) F1*={dep.mean():.4f}  ({nf} folds, {time.time()-t0:.0f}s)", flush=True)
    rows = [{"panel": panel, "config": "deployed", "f1": dep.mean(), "delta": 0.0, "sigma": 0.0}]
    for tag, _, _ in candidates:
        arr = np.array(f1[tag]); d = arr - dep
        se = d.std(ddof=1) / np.sqrt(nf) if nf > 1 else 0.0
        sig = d.mean() / se if se > 0 else 0.0
        verdict = "DEPLOY" if d.mean() > 0 and sig >= 2 else ("marginal" if sig >= 1 else "no")
        print(f"  {tag:18s} F1*={arr.mean():.4f}  Δ={d.mean():+.4f} ({sig:+.2f}σ)  {verdict}", flush=True)
        rows.append({"panel": panel, "config": tag, "f1": arr.mean(), "delta": d.mean(), "sigma": sig})
    return rows


if __name__ == "__main__":
    allr = []
    allr += run("pah",  [("targrank_w1.0", "targrank", 1.0)], n_repeats=20)
    allr += run("cftr", [("targrank_w0.85", "targrank", 0.85),
                         ("rawAL_w0.85",    "rawAL",    0.85),
                         ("targrank_w1.0",  "targrank", 1.0)], n_repeats=20)
    pd.DataFrame(allr).to_csv(os.path.join(OUT, "exp_confirm_pah_cftr.csv"), index=False)
    print(f"\nsaved -> {os.path.join(OUT, 'exp_confirm_pah_cftr.csv')}", flush=True)
