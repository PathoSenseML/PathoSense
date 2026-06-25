"""CONFIRMATION of the cancer transfer-scorer blend-weight signal (+4.13σ on the PURE scorer).
Three sources of false-positive risk to kill before believing it:
  (1) wash-out: deployed cancer = 0.5·transfer + 0.5·own, not the pure scorer. Re-test in the
      FULL deployed config.
  (2) grid-overfit: per-fold weight tuning over 66 triples on small folds may overfit the repeated
      splits. A SINGLE FIXED RF-up weight (0.25,0.50,0.25) cannot grid-overfit — if it still helps,
      the signal is real (RF is genuinely the stronger cancer component).
  (3) σ inflation from overlapping RepeatedStratifiedKFold folds → run TWO independent base seeds.

Per fold: own model refit (deployed median+mRMR via target_deployed_fn); scorer components L,R,C
fixed (scorer trained once, target IDs excluded). For each weighting form g, Platt on train, blend
{pure w=1.0, deployed w=0.5}·g_cal + (1-w)·own_cal, F1-opt-at-prior threshold on train, analytic F1*
on test. Paired Δ vs equal (1/3) within each blend mode + seed.
Run: .venv/bin/python -u analysis/exp_blendw_confirm.py
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
from transfer import fit_transfer_scorer
from exp_transfer import _load, load_target_engineered, target_deployed_fn
from exp_transfer_blendw import transfer_components, best_weights, GRID

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
PANEL = "cancer"
EQUAL = (1/3, 1/3, 1/3)
RFUP = (0.25, 0.50, 0.25)            # fixed, cannot grid-overfit


def run_seed(base_seed, nr, L, R, C, Xt_own, yt, fn, prior):
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=nr, random_state=base_seed)
    # collect F1* per fold for each (blend_mode, weighting)
    acc = {(bm, wf): [] for bm in ("pure", "deployed") for wf in ("equal", "rfup", "opt")}
    for tr, te in rskf.split(Xt_own, yt):
        tr, te = np.asarray(tr), np.asarray(te)
        ytr, yte = yt.iloc[tr].reset_index(drop=True), yt.iloc[te].reset_index(drop=True)
        e_te, e_tr = fn(Xt_own.iloc[tr].reset_index(drop=True), ytr,
                        Xt_own.iloc[te].reset_index(drop=True))
        w_opt = best_weights((L[tr], R[tr], C[tr]), ytr.values)
        for wf, wv in (("equal", EQUAL), ("rfup", RFUP), ("opt", w_opt)):
            g = wv[0]*L + wv[1]*R + wv[2]*C
            pl = fit_platt(g[tr], ytr.values)
            g_tr, g_te = apply_platt(g[tr], pl), apply_platt(g[te], pl)
            for bm, wt in (("pure", 1.0), ("deployed", 0.5)):
                b_tr = wt*g_tr + (1-wt)*e_tr
                b_te = wt*g_te + (1-wt)*e_te
                t = pick_threshold_at_prior(ytr.values, b_tr, prior)
                acc[(bm, wf)].append(f1_at_prior(*tpr_fpr(yte.values, (b_te >= t).astype(int)), prior))
    rows = []
    for bm in ("pure", "deployed"):
        base = np.array(acc[(bm, "equal")]); nf = len(base)
        for wf in ("equal", "rfup", "opt"):
            arr = np.array(acc[(bm, wf)]); d = arr - base
            se = d.std(ddof=1)/np.sqrt(nf) if nf > 1 else 0.0
            sig = d.mean()/se if se > 0 else 0.0
            print(f"    seed{base_seed} [{bm:8s}] {wf:5s} F1*={arr.mean():.4f}  Δ={d.mean():+.4f} ({sig:+.2f}σ)",
                  flush=True)
            rows.append({"seed": base_seed, "mode": bm, "weighting": wf, "f1": arr.mean(),
                         "delta": d.mean(), "sigma": sig, "folds": nf})
    return rows


if __name__ == "__main__":
    print(f"=== CONFIRM cancer scorer blend-weight (equal vs fixed RF-up vs per-fold opt) ===", flush=True)
    prior = TEST_PRIORS[PANEL]
    art = fit_transfer_scorer(PANEL)
    df = _load(PANEL)
    L, R, C = transfer_components(df, art)
    y_full = df["Label"].astype(int).values
    print(f"scorer component AUC: lgb={roc_auc_score(y_full,L):.4f} rf={roc_auc_score(y_full,R):.4f} "
          f"cat={roc_auc_score(y_full,C):.4f} equal={roc_auc_score(y_full,(L+R+C)/3):.4f} "
          f"rfup={roc_auc_score(y_full, RFUP[0]*L+RFUP[1]*R+RFUP[2]*C):.4f}", flush=True)
    Xt_own, yt = load_target_engineered(PANEL)
    assert (yt.values == y_full).all(), "row order mismatch"
    fn = target_deployed_fn(PANEL)
    allrows = []
    for bs in (SEED, SEED + 1000):
        t0 = time.time()
        allrows += run_seed(bs, 20, L, R, C, Xt_own, yt, fn, prior)
        print(f"  (seed {bs}: {time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(allrows).to_csv(os.path.join(OUT, "blendw_confirm.csv"), index=False)
    print(f"\nsaved -> blendw_confirm.csv", flush=True)
