"""Idea #1 — expand the transfer SOURCE: train the scorer on general + the OTHER small panels
(minus all target Variant_IDs) instead of general alone. EK conservation transfers cleanly across
panels (corr ~0.98), so more diverse-gene training data could lift the dominant transfer component.

Per target, compare general-only vs pooled scorer in the deployed CV (frozen OOS scorer, per-fold own
model, Platt-cal, blend at deployed w, threshold on train-OOF at the test prior, F1* held-out, paired).
Leakage-controlled: fit_transfer_scorer excludes ALL target Variant_IDs from every source.
Run: .venv/bin/python -u analysis/exp_transfer_pooled.py [panel ...]
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
from exp_transfer import _load, load_target_engineered, target_deployed_fn
from transfer import fit_transfer_scorer, transfer_score

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
DEPLOY_W = {"cftr": 1.0, "pah": 1.0, "cancer": 0.5}
POOL = {"cftr": ("general", "cancer", "pah"), "cancer": ("general", "pah", "cftr"),
        "pah": ("general", "cancer", "cftr")}
N_REPEATS = 10


def run(panel):
    prior = TEST_PRIORS[panel]; w = DEPLOY_W[panel]
    raw = _load(panel)
    g_gen = transfer_score(raw, fit_transfer_scorer(panel, sources=("general",)))
    art_pool = fit_transfer_scorer(panel, sources=POOL[panel])
    g_pool = transfer_score(raw, art_pool)
    Xt, yt = load_target_engineered(panel)
    print(f"\n[{panel}] scorer AUC: general-only={roc_auc_score(yt, g_gen):.4f}  "
          f"pooled({'+'.join(POOL[panel])})={roc_auc_score(yt, g_pool):.4f}  "
          f"(pool n_train={art_pool['n_train']})", flush=True)

    fn = target_deployed_fn(panel)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=N_REPEATS, random_state=SEED)
    per = {"general": [], "pooled": []}
    for tr, te in rskf.split(Xt, yt):
        tr, te = np.asarray(tr), np.asarray(te)
        ytr, yte = yt.iloc[tr].reset_index(drop=True), yt.iloc[te].reset_index(drop=True)
        e_te, e_tr = fn(Xt.iloc[tr].reset_index(drop=True), ytr, Xt.iloc[te].reset_index(drop=True))
        for key, g in (("general", g_gen), ("pooled", g_pool)):
            pl = fit_platt(g[tr], ytr.values)
            gt, ge = apply_platt(g[tr], pl), apply_platt(g[te], pl)
            b_tr = w * gt + (1 - w) * e_tr; b_te = w * ge + (1 - w) * e_te
            thr = pick_threshold_at_prior(ytr.values, b_tr, prior)
            tpr, fpr = tpr_fpr(yte.values, (b_te >= thr).astype(int))
            per[key].append(f1_at_prior(tpr, fpr, prior))
    base = np.array(per["general"]); v = np.array(per["pooled"]); d = v - base
    nf = len(base); se = d.std(ddof=1) / np.sqrt(nf); sig = d.mean() / se if se > 0 else 0.0
    vd = ("DEPLOY" if d.mean() > 0 and sig >= 2 else "marginal" if d.mean() > 0 and sig >= 1 else "no gain")
    print(f"  [{panel}] general F1*={base.mean():.4f}  pooled F1*={v.mean():.4f}  "
          f"Δ={d.mean():+.4f} ({sig:+.2f}σ)  -> {vd}", flush=True)
    return {"panel": panel, "general_f1": base.mean(), "pooled_f1": v.mean(), "delta": d.mean(), "sigma": sig}


if __name__ == "__main__":
    panels = sys.argv[1:] or ["cftr", "pah", "cancer"]
    print("=== Idea #1: pooled transfer source vs general-only — honest CV ===", flush=True)
    rows = []
    for pn in panels:
        t0 = time.time(); rows.append(run(pn)); print(f"  ({pn} {time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "transfer_pooled.csv"), index=False)
    print(f"\nsaved -> transfer_pooled.csv", flush=True)
