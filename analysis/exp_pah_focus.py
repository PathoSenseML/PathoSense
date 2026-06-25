"""PAH-specific deep dive: stack the principled levers (each marginal alone) + own-model blend.
Diagnostic showed own model (AUC 0.747) is complementary to transfer (0.800): 50/50 blend AUC 0.803,
pred corr 0.49. So blend has headroom the deployed w=1.0 ignores. Combine with target-rank scorer
(AL covariate-shift fix) + plateau threshold (pah's high prior -> broad F1 plateau).

Configs (n_repeats=20, paired vs DEPLOYED = raw scorer, w=1.0, bootstrap threshold):
  deployed         raw scorer,        w=1.0, boot
  tr_plat          target-rank scorer, w=1.0, plateau         (stack scorer+threshold wins)
  tr_w85_plat      target-rank scorer, w=0.85, plateau        (+ small own-model blend)
  tr_w75_plat      target-rank scorer, w=0.75, plateau        (+ more own-model blend)
  pooled_tr_plat   pooled(gen+can+cftr) target-rank, w=1.0, plateau   (more transfer data)
Confirm any >=2sigma winner at higher fidelity before deploying.

Run:  .venv/bin/python -u analysis/exp_pah_focus.py
"""
import os, sys, time
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
from exp_transfer import _load, load_target_engineered, target_deployed_fn
from transfer import fit_transfer_scorer, transfer_score, fit_target_pop_grids

PANEL, PRIOR = "pah", TEST_PRIORS["pah"]


def thr_boot(ytr, oof):
    return optimize_threshold_f1_balanced(ytr, oof, target_prior=PRIOR, n_bootstrap=300)[0]

def thr_plateau(ytr, oof, eps=0.005):
    ts = np.arange(0.02, 0.98, 0.005)
    f1s = np.array([f1_at_prior(*tpr_fpr(ytr, (oof >= t).astype(int)), PRIOR) for t in ts])
    return float(np.median(ts[f1s >= f1s.max() - eps]))


def main():
    raw = _load(PANEL); Xt, y = load_target_engineered(PANEL)
    art = fit_transfer_scorer(PANEL)                                  # general-only (deployed)
    art_pool = fit_transfer_scorer(PANEL, sources=("general", "cancer", "cftr"))
    g_raw = transfer_score(raw, art)
    fn = target_deployed_fn(PANEL)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=SEED)
    keys = ["deployed", "tr_plat", "tr_w85_plat", "tr_w75_plat", "pooled_tr_plat"]
    f1 = {k: [] for k in keys}
    t0 = time.time()
    for tr, te in rskf.split(Xt, y):
        tr, te = np.asarray(tr), np.asarray(te)
        ytr, yte = y.iloc[tr].values, y.iloc[te].values
        # own model (for w<1 configs)
        e_te, e_tr = fn(Xt.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True),
                        Xt.iloc[te].reset_index(drop=True))
        # raw scorer (Platt on train)
        plr = fit_platt(g_raw[tr], ytr); gr_tr, gr_te = apply_platt(g_raw[tr], plr), apply_platt(g_raw[te], plr)
        # target-rank scorer (grids fit on train fold)
        grids = fit_target_pop_grids(raw.iloc[tr], art)
        gt = transfer_score(raw.iloc[tr], art, grids); gt2 = transfer_score(raw.iloc[te], art, grids)
        plt_ = fit_platt(gt, ytr); gt_tr, gt_te = apply_platt(gt, plt_), apply_platt(gt2, plt_)
        # pooled target-rank scorer
        gpr = fit_target_pop_grids(raw.iloc[tr], art_pool)
        gp = transfer_score(raw.iloc[tr], art_pool, gpr); gp2 = transfer_score(raw.iloc[te], art_pool, gpr)
        plp = fit_platt(gp, ytr); gp_tr, gp_te = apply_platt(gp, plp), apply_platt(gp2, plp)

        def score(g_tr, g_te, w, thr_fn):
            b_tr = w * g_tr + (1 - w) * e_tr; b_te = w * g_te + (1 - w) * e_te
            t = thr_fn(ytr, b_tr)
            return f1_at_prior(*tpr_fpr(yte, (b_te >= t).astype(int)), PRIOR)

        f1["deployed"].append(score(gr_tr, gr_te, 1.0, thr_boot))
        f1["tr_plat"].append(score(gt_tr, gt_te, 1.0, thr_plateau))
        f1["tr_w85_plat"].append(score(gt_tr, gt_te, 0.85, thr_plateau))
        f1["tr_w75_plat"].append(score(gt_tr, gt_te, 0.75, thr_plateau))
        f1["pooled_tr_plat"].append(score(gp_tr, gp_te, 1.0, thr_plateau))
    dep = np.array(f1["deployed"]); nf = len(dep)
    print(f"=== PAH focus ({nf} folds, prior={PRIOR}, {time.time()-t0:.0f}s) — paired vs deployed ===", flush=True)
    print(f"  deployed (raw,w1,boot) F1*={dep.mean():.4f}", flush=True)
    rows = [{"config": "deployed", "f1": dep.mean(), "delta": 0, "sigma": 0}]
    for k in keys[1:]:
        a = np.array(f1[k]); d = a - dep
        se = d.std(ddof=1) / np.sqrt(nf); sig = d.mean() / se if se > 0 else 0
        v = "DEPLOY" if d.mean() > 0 and sig >= 2 else ("marginal" if sig >= 1 else "no")
        print(f"  {k:16s} F1*={a.mean():.4f}  Δ={d.mean():+.4f} ({sig:+.2f}σ)  {v}", flush=True)
        rows.append({"config": k, "f1": a.mean(), "delta": d.mean(), "sigma": sig})
    pd.DataFrame(rows).to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "output",
                                            "exp_pah_focus.csv"), index=False)


if __name__ == "__main__":
    main()
