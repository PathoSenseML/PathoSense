"""Last shot at the cancer RF-up scorer: does JOINTLY re-optimizing the transfer-vs-own weight w
recover the pure-scorer gain in deployment? Baseline = DEPLOYED (equal-weight scorer, w=0.5).
Candidates = RF-up scorer (0.25,0.5,0.25) at w in {0.4,0.5,0.6,0.7,0.75,1.0}. Paired Δ vs baseline,
analytic F1* + finals-size bootstrap (simulates the actual finals N — a gain that survives the
bootstrap is more credible). 2 seeds. Deploy bar: >=2σ AND positive AND survives bootstrap.
Run: .venv/bin/python -u analysis/exp_blendw_wsweep.py
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import RepeatedStratifiedKFold
from config import SEED
from utils import TEST_PRIORS
from calibration import fit_platt, apply_platt
from harness import pick_threshold_at_prior, tpr_fpr, f1_at_prior, bootstrap_f1_mcc, FINALS_N
from transfer import fit_transfer_scorer
from exp_transfer import _load, load_target_engineered, target_deployed_fn
from exp_transfer_blendw import transfer_components

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
PANEL = "cancer"
EQUAL = (1/3, 1/3, 1/3)
RFUP = (0.25, 0.50, 0.25)
WGRID = [0.4, 0.5, 0.6, 0.7, 0.75, 1.0]


def run_seed(bs, nr, L, R, C, Xt_own, yt, fn, prior, ntar):
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=nr, random_state=bs)
    base_f1, base_fb = [], []
    cand = {w: {"f1": [], "fb": []} for w in WGRID}
    g_eq_all = EQUAL[0]*L + EQUAL[1]*R + EQUAL[2]*C
    g_rf_all = RFUP[0]*L + RFUP[1]*R + RFUP[2]*C
    for tr, te in rskf.split(Xt_own, yt):
        tr, te = np.asarray(tr), np.asarray(te)
        ytr, yte = yt.iloc[tr].reset_index(drop=True), yt.iloc[te].reset_index(drop=True)
        e_te, e_tr = fn(Xt_own.iloc[tr].reset_index(drop=True), ytr,
                        Xt_own.iloc[te].reset_index(drop=True))
        # baseline: equal scorer @ w=0.5 (deployed)
        pe = fit_platt(g_eq_all[tr], ytr.values)
        ge_tr, ge_te = apply_platt(g_eq_all[tr], pe), apply_platt(g_eq_all[te], pe)
        b_tr, b_te = 0.5*ge_tr + 0.5*e_tr, 0.5*ge_te + 0.5*e_te
        t = pick_threshold_at_prior(ytr.values, b_tr, prior); pr = (b_te >= t).astype(int)
        base_f1.append(f1_at_prior(*tpr_fpr(yte.values, pr), prior))
        base_fb.append(bootstrap_f1_mcc(yte.values, pr, prior, ntar)[0])
        # candidates: rfup scorer @ various w
        prf = fit_platt(g_rf_all[tr], ytr.values)
        grf_tr, grf_te = apply_platt(g_rf_all[tr], prf), apply_platt(g_rf_all[te], prf)
        for w in WGRID:
            c_tr, c_te = w*grf_tr + (1-w)*e_tr, w*grf_te + (1-w)*e_te
            tc = pick_threshold_at_prior(ytr.values, c_tr, prior); pc = (c_te >= tc).astype(int)
            cand[w]["f1"].append(f1_at_prior(*tpr_fpr(yte.values, pc), prior))
            cand[w]["fb"].append(bootstrap_f1_mcc(yte.values, pc, prior, ntar)[0])
    base_f1 = np.array(base_f1); base_fb = np.array(base_fb); nf = len(base_f1)
    print(f"  seed{bs}: deployed baseline (equal,w=0.5) F1*={base_f1.mean():.4f} finals~{base_fb.mean():.4f}", flush=True)
    rows = []
    for w in WGRID:
        a = np.array(cand[w]["f1"]); d = a - base_f1
        se = d.std(ddof=1)/np.sqrt(nf); sig = d.mean()/se if se > 0 else 0.0
        db = np.array(cand[w]["fb"]) - base_fb
        seb = db.std(ddof=1)/np.sqrt(nf); sigb = db.mean()/seb if seb > 0 else 0.0
        print(f"    rfup w={w:<4} F1*={a.mean():.4f}  Δ={d.mean():+.4f} ({sig:+.2f}σ)  | "
              f"finals Δ={db.mean():+.4f} ({sigb:+.2f}σ)", flush=True)
        rows.append({"seed": bs, "w": w, "f1": a.mean(), "delta": d.mean(), "sigma": sig,
                     "finals_delta": db.mean(), "finals_sigma": sigb})
    return rows


if __name__ == "__main__":
    print("=== cancer RF-up scorer: w-sweep vs deployed (equal,w=0.5), analytic + finals-bootstrap ===", flush=True)
    prior = TEST_PRIORS[PANEL]; ntar = FINALS_N[PANEL]
    art = fit_transfer_scorer(PANEL)
    df = _load(PANEL)
    L, R, C = transfer_components(df, art)
    Xt_own, yt = load_target_engineered(PANEL)
    fn = target_deployed_fn(PANEL)
    allrows = []
    for bs in (SEED, SEED + 1000):
        t0 = time.time(); allrows += run_seed(bs, 20, L, R, C, Xt_own, yt, fn, prior, ntar)
        print(f"  (seed {bs}: {time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(allrows).to_csv(os.path.join(OUT, "blendw_wsweep.csv"), index=False)
    print(f"\nsaved -> blendw_wsweep.csv", flush=True)
