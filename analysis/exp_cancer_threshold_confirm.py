"""Confirm (2 seeds) that lowering cancer's deployed threshold from 0.85 toward ~0.82 is a robust gain.
Decision is between the deployed-method threshold (bootstrap-balanced on full train → 0.85) and a fixed
0.82 (= the median the same method picks per-fold + Öykü's independent stability optimum; NOT the peeked
test-peak 0.83). Honest per-fold CV, current deployed cancer config (rfup+fam_flags), paired Δ.
Run: .venv/bin/python -u analysis/exp_cancer_threshold_confirm.py
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
from config import SEED
from utils import TEST_PRIORS, optimize_threshold_f1_balanced
from harness import tpr_fpr, f1_at_prior
from exp_conditional_threshold import fold_preds

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
FIXED = [0.80, 0.81, 0.82, 0.83]


def main():
    prior = TEST_PRIORS["cancer"]
    rows = []
    for seed in (SEED, SEED + 1000):
        folds, _ = fold_preds("cancer", nr=10, seed=seed)
        n = len(folds)
        f1_dep = []
        fixed = {t: [] for t in FIXED}
        tdep = []
        for ytr, oof, yte, test, Xtr, Xte in folds:
            td, _, _ = optimize_threshold_f1_balanced(ytr, oof, target_prior=prior, n_bootstrap=200)
            tdep.append(td)
            f1_dep.append(f1_at_prior(*tpr_fpr(yte, (test >= td).astype(int)), prior))
            for t in FIXED:
                fixed[t].append(f1_at_prior(*tpr_fpr(yte, (test >= t).astype(int)), prior))
        f1_dep = np.array(f1_dep)
        print(f"seed{seed}: deployed-method F1*={f1_dep.mean():.4f} (mean t picked={np.mean(tdep):.3f}), {n} folds", flush=True)
        for t in FIXED:
            a = np.array(fixed[t]); d = a - f1_dep
            se = d.std(ddof=1)/np.sqrt(n); sig = d.mean()/se if se > 0 else 0.0
            print(f"   fixed t={t:.2f}  F1*={a.mean():.4f}  Δ_vs_deployed_method={d.mean():+.4f} ({sig:+.2f}σ)", flush=True)
            rows.append({"seed": seed, "threshold": t, "f1": a.mean(), "delta": d.mean(), "sigma": sig,
                         "deployed_method_f1": f1_dep.mean()})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "cancer_threshold_confirm.csv"), index=False)
    print(f"\nsaved -> cancer_threshold_confirm.csv", flush=True)


if __name__ == "__main__":
    main()
