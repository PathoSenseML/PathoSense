"""Confirm + tune the cancer transfer-scorer win: drop drifted raw AL_ values, use AL_
missingness fam_flags instead (exp_transfer_noal.py showed +0.0163 F1* / +2.5σ at w=0.5).

Higher fidelity (n_repeats=15) + finer w-sweep (a better scorer may shift the optimal blend
weight). Paired Δ (new scorer − baseline) at each w. Decide deploy + best w.

Run:  .venv/bin/python -u analysis/exp_cancer_confirm.py
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
from transfer import fit_transfer_scorer, transfer_score

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
PANEL = "cancer"
W = [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]


def main():
    prior = TEST_PRIORS[PANEL]; n_target = FINALS_N[PANEL]
    raw = _load(PANEL)
    g = {
        "baseline":      transfer_score(raw, fit_transfer_scorer(PANEL)),
        "noAL_famflags": transfer_score(raw, fit_transfer_scorer(PANEL, drop_al=True,
                                                                  al_features=("fam_flags",))),
    }
    Xt, yt = load_target_engineered(PANEL)
    for k in g:
        print(f"  {k:14s} scorer AUC on cancer = {roc_auc_score(yt, g[k]):.4f}", flush=True)
    fn = target_deployed_fn(PANEL)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=15, random_state=SEED)
    per = {(k, w): [] for k in g for w in W}
    t0 = time.time()
    for tr, te in rskf.split(Xt, yt):
        tr, te = np.asarray(tr), np.asarray(te)
        ytr, yte = yt.iloc[tr].reset_index(drop=True), yt.iloc[te].reset_index(drop=True)
        e_te, e_tr = fn(Xt.iloc[tr].reset_index(drop=True), ytr,
                        Xt.iloc[te].reset_index(drop=True))     # own model, shared
        for k in g:
            pl = fit_platt(g[k][tr], ytr.values)
            gt, ge = apply_platt(g[k][tr], pl), apply_platt(g[k][te], pl)
            for w in W:
                b_tr = w * gt + (1 - w) * e_tr; b_te = w * ge + (1 - w) * e_te
                thr = pick_threshold_at_prior(ytr.values, b_tr, prior)
                tpr, fpr = tpr_fpr(yte.values, (b_te >= thr).astype(int))
                per[(k, w)].append(f1_at_prior(tpr, fpr, prior))

    nf = len(per[("baseline", 0.5)])
    print(f"\n=== cancer confirm — {nf} folds, F1* at prior {prior} ({time.time()-t0:.0f}s) ===")
    print(f"{'w':>5s} {'base F1*':>9s} {'new F1*':>9s} {'Δ(new-base@w)':>14s} {'σ':>7s}", flush=True)
    rows = []
    for w in W:
        b = np.array(per[("baseline", w)]); n = np.array(per[("noAL_famflags", w)])
        d = n - b; se = d.std(ddof=1) / np.sqrt(nf) if nf > 1 else 0.0
        sig = d.mean() / se if se > 0 else 0.0
        print(f"{w:>5} {b.mean():>9.4f} {n.mean():>9.4f} {d.mean():>+14.4f} {sig:>+7.2f}", flush=True)
        rows.append({"w": w, "base_f1": b.mean(), "new_f1": n.mean(), "delta": d.mean(), "sigma": sig})
    dep = next(r for r in rows if r["w"] == 0.5)
    best_new = max(rows, key=lambda r: r["new_f1"])
    best_base = max(rows, key=lambda r: r["base_f1"])
    # paired Δ of (new @ best_new_w) vs (baseline @ deployed 0.5) — the real deploy comparison
    nb = np.array(per[("noAL_famflags", best_new["w"])]); bb = np.array(per[("baseline", 0.5)])
    dd = nb - bb; sg = dd.mean() / (dd.std(ddof=1) / np.sqrt(nf))
    print(f"\n  deployed baseline (w=0.5): {dep['base_f1']:.4f}", flush=True)
    print(f"  best baseline: w={best_base['w']} F1*={best_base['base_f1']:.4f}", flush=True)
    print(f"  best NEW scorer: w={best_new['w']} F1*={best_new['new_f1']:.4f}", flush=True)
    print(f"  NEW@best_w vs deployed baseline@0.5: Δ={dd.mean():+.4f} ({sg:+.2f}σ)", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "exp_cancer_confirm.csv"), index=False)


if __name__ == "__main__":
    main()
