"""DEFINITIVE high-power confirmation of target-rank domain adaptation for PAH.

Target-rank = re-rank pah's AL_ allele-frequency features on pah's OWN training distribution (per fold)
before scoring, instead of using general's percentile grid. Fixes the documented cross-panel AF
covariate shift (same variant: AL corr ~0.017 general<->pah). It is the ONLY pah lever that is
consistently POSITIVE but sub-2sigma across prior tests:
  exp_transfer_targetrank +0.0081 (+1.64s) | exp_confirm_pah_cftr +0.0048 (+1.40s) | exp_pah_focus +0.49s
Three independent positives at ~1.4-1.6s = real-effect-shaped but never settled. This runs it at HIGH
POWER (4 FRESH disjoint seeds x 125 folds = 500 folds) to tighten the SE: a real +0.005-0.008 effect
sharpens to >=2-3sigma; noise regresses to 0. Pure transfer (w=1.0) so it's cheap and DIRECTLY deployable
via predict.py (fit_target_pop_grids on pah-train -> threading target_pop_grids into transfer_score).

Deploy bar: Δ>0 on ALL 4 seeds AND pooled (500-fold) sigma>=2. Run: .venv/bin/python -u analysis/exp_pah_targetrank_confirm.py
"""
import os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score

from config import SEED
from utils import TEST_PRIORS
from calibration import fit_platt, apply_platt
from harness import pick_threshold_at_prior, tpr_fpr, f1_at_prior, bootstrap_f1_mcc, FINALS_N
from exp_transfer import _load, load_target_engineered
from transfer import fit_transfer_scorer, transfer_score, fit_target_pop_grids

PANEL = "pah"; PRIOR = TEST_PRIORS[PANEL]; NT = FINALS_N["pah"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
SEEDS = [SEED + 51051, SEED + 54054, SEED + 57057, SEED + 60060]   # fresh, disjoint from prior tests


def main():
    t0 = time.time()
    raw = _load(PANEL); Xt, y = load_target_engineered(PANEL)
    sc = fit_transfer_scorer(PANEL, drop_al=False, al_features=())
    g_raw = np.asarray(transfer_score(raw, sc))
    print(f"PAH n={len(raw)} ({int(y.sum())}P/{int((y==0).sum())}B)  baseline transfer AUC={roc_auc_score(y,g_raw):.4f}")
    all_dep, all_tr = [], []          # pooled per-fold F1*
    per_seed = []
    for sd in SEEDS:
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=25, random_state=sd)
        dep, trk, depb, trkb = [], [], [], []
        for tr, te in rskf.split(Xt, y):
            tr, te = np.asarray(tr), np.asarray(te)
            ytr, yte = y.iloc[tr].reset_index(drop=True), y.iloc[te].reset_index(drop=True)
            # deployed: raw AL, general grid
            pl = fit_platt(g_raw[tr], ytr.values)
            gd_tr, gd_te = apply_platt(g_raw[tr], pl), apply_platt(g_raw[te], pl)
            td = pick_threshold_at_prior(ytr.values, gd_tr, PRIOR)
            pd_ = (gd_te >= td).astype(int)
            dep.append(f1_at_prior(*tpr_fpr(yte.values, pd_), PRIOR))
            depb.append(bootstrap_f1_mcc(yte.values, pd_, PRIOR, NT)[0])
            # target-rank: grids fit on TRAIN fold only (leakage-safe, batch-invariant at inference)
            grids = fit_target_pop_grids(raw.iloc[tr], sc)
            gr_tr = np.asarray(transfer_score(raw.iloc[tr], sc, grids))
            gr_te = np.asarray(transfer_score(raw.iloc[te], sc, grids))
            pl2 = fit_platt(gr_tr, ytr.values)
            grt, gre = apply_platt(gr_tr, pl2), apply_platt(gr_te, pl2)
            tt = pick_threshold_at_prior(ytr.values, grt, PRIOR)
            pr_ = (gre >= tt).astype(int)
            trk.append(f1_at_prior(*tpr_fpr(yte.values, pr_), PRIOR))
            trkb.append(bootstrap_f1_mcc(yte.values, pr_, PRIOR, NT)[0])
        dep, trk = np.array(dep), np.array(trk); d = trk - dep; nf = len(dep)
        se = d.std(ddof=1) / np.sqrt(nf); sg = d.mean() / se if se > 0 else 0
        per_seed.append((sd, dep.mean(), trk.mean(), d.mean(), sg, np.mean(depb), np.mean(trkb)))
        all_dep += list(dep); all_tr += list(trk)
        print(f"  seed {sd}: dep={dep.mean():.4f}  targrank={trk.mean():.4f}  Δ={d.mean():+.4f} ({sg:+.2f}σ)  "
              f"[finals: dep={np.mean(depb):.4f} trk={np.mean(trkb):.4f}]", flush=True)
    # pooled
    a, b = np.array(all_dep), np.array(all_tr); d = b - a; nf = len(d)
    se = d.std(ddof=1) / np.sqrt(nf); sg = d.mean() / se if se > 0 else 0
    allpos = all(x[3] > 0 for x in per_seed)
    print(f"\n===== POOLED ({nf} folds) =====")
    print(f"  deployed   F1*={a.mean():.4f}")
    print(f"  targetrank F1*={b.mean():.4f}   Δ={d.mean():+.4f} ({sg:+.2f}σ)   all_seeds_pos={allpos}")
    verdict = "DEPLOY (principled + Δ>0 all seeds + pooled >=2σ)" if (allpos and sg >= 2) else \
              ("MARGINAL (positive but <2σ — judgment call)" if (allpos and sg >= 1) else "REJECT")
    print(f"  VERDICT: {verdict}")
    pd.DataFrame(per_seed, columns=["seed","dep","targrank","delta","sigma","dep_finals","trk_finals"]).to_csv(
        os.path.join(OUT, "pah_targetrank_confirm.csv"), index=False)
    print(f"\n[done] {time.time()-t0:.0f}s  saved pah_targetrank_confirm.csv")


if __name__ == "__main__":
    main()
