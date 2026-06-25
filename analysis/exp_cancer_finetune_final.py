"""Deploy gate for cancer fine-tune: does the finetuned scorer (ft_up50) at its OWN best fixed threshold
beat the CURRENT DEPLOYED cancer (rfup scorer + fixed 0.81 = ~0.735)? My v3 gain used per-fold thresholds;
deploy uses a fixed threshold, and the finetune shifts the score distribution (selector now picks ~0.73),
so the fixed threshold must be re-derived. Per fold (2 seeds): own refit; A = rfup base @ fixed 0.81;
B = finetuned (0.5/0.3/0.2) @ fixed t in a grid + @ selector. Paired Δ(B−A). Deploy B's best train-justified
threshold only if robustly >= A.
Run: .venv/bin/python -u analysis/exp_cancer_finetune_final.py
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import RepeatedStratifiedKFold
from config import SEED
from utils import TEST_PRIORS, optimize_threshold_f1_balanced
from calibration import fit_platt, apply_platt
from harness import tpr_fpr, f1_at_prior
from transfer import fit_transfer_scorer
from exp_transfer import _load, load_target_engineered, target_deployed_fn
from exp_transfer_finetune import project_matrix
from exp_transfer_blendw import transfer_components
from exp_cancer_finetune_v2 import ft_lgbm

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
RFUP = (0.25, 0.50, 0.25); FTW = (0.50, 0.30, 0.20)
A_THR = 0.81; GRID = [0.74, 0.76, 0.78, 0.80, 0.82, 0.84]


def main():
    prior = TEST_PRIORS["cancer"]
    art = fit_transfer_scorer("cancer"); base_boost = art["models"][0].booster_
    cat_feats = [c for c in art["feats"] if c in art["cat_cats"]]
    df = _load("cancer"); y = df["Label"].astype(int)
    Xs = project_matrix(df, art); L, R, C = transfer_components(df, art)
    Xt, yt = load_target_engineered("cancer"); fn = target_deployed_fn("cancer")
    rows = []
    for seed in (SEED, SEED + 1000):
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=8, random_state=seed)
        A = []; Bgrid = {t: [] for t in GRID}; Bsel = []
        for tr, te in rskf.split(Xs, y):
            tr, te = np.asarray(tr), np.asarray(te); ytr, yte = y.iloc[tr], y.iloc[te]
            e_te, e_tr = fn(Xt.iloc[tr].reset_index(drop=True), ytr.reset_index(drop=True),
                            Xt.iloc[te].reset_index(drop=True))
            # A: current deployed (rfup base @ 0.81)
            ga = RFUP[0]*L[tr]+RFUP[1]*R[tr]+RFUP[2]*C[tr]; gae = RFUP[0]*L[te]+RFUP[1]*R[te]+RFUP[2]*C[te]
            pla = fit_platt(ga, ytr.values); ba_te = 0.5*apply_platt(gae, pla)+0.5*e_te
            A.append(f1_at_prior(*tpr_fpr(yte.values, (ba_te >= A_THR).astype(int)), prior))
            # B: finetuned upweighted
            lft_te, lft_oof = ft_lgbm(Xs, y, tr, te, base_boost, cat_feats)
            gb_tr = FTW[0]*lft_oof+FTW[1]*R[tr]+FTW[2]*C[tr]; gb_te = FTW[0]*lft_te+FTW[1]*R[te]+FTW[2]*C[te]
            plb = fit_platt(gb_tr, ytr.values)
            bb_tr = 0.5*apply_platt(gb_tr, plb)+0.5*e_tr; bb_te = 0.5*apply_platt(gb_te, plb)+0.5*e_te
            for t in GRID:
                Bgrid[t].append(f1_at_prior(*tpr_fpr(yte.values, (bb_te >= t).astype(int)), prior))
            tsel, _, _ = optimize_threshold_f1_balanced(ytr.values, bb_tr, target_prior=prior, n_bootstrap=200)
            Bsel.append(f1_at_prior(*tpr_fpr(yte.values, (bb_te >= tsel).astype(int)), prior))
        A = np.array(A); nf = len(A)
        print(f"\n  seed{seed} ({nf} folds): A=deployed(rfup@0.81) F1*={A.mean():.4f}", flush=True)
        for label, arr in [("ft@selector", np.array(Bsel))] + [(f"ft@{t}", np.array(Bgrid[t])) for t in GRID]:
            d = arr - A; sig = d.mean()/(d.std(ddof=1)/np.sqrt(nf)) if d.std() > 0 else 0
            print(f"    {label:12s} F1*={arr.mean():.4f}  Δ={d.mean():+.4f} ({sig:+.2f}σ)", flush=True)
            rows.append({"seed": seed, "config": label, "f1": arr.mean(), "delta": d.mean(), "sigma": sig})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "cancer_finetune_final.csv"), index=False)
    print(f"\nsaved -> cancer_finetune_final.csv", flush=True)


if __name__ == "__main__":
    main()
