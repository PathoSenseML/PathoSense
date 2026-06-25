"""Generalize the cancer threshold finding: does the bootstrap threshold SELECTOR overshoot on the
other panels too? Deployed thresholds: general 0.83, pah 0.82, cftr 0.90. For each panel, honest
per-fold CV (current deployed config), 2 seeds: deployed-method threshold + test F1* vs a grid of FIXED
thresholds. If a fixed threshold robustly beats the selector AND is train-justified (near the selector's
per-fold central pick), deploy it.
Run: .venv/bin/python -u analysis/exp_threshold_check_all.py [panel ...]
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
GRIDS = {"general": [0.78,0.80,0.81,0.82,0.83,0.84,0.85],
         "pah":     [0.74,0.76,0.78,0.80,0.82,0.84],
         "cftr":    [0.82,0.84,0.86,0.88,0.90,0.92]}
NR = {"general": 6, "pah": 12, "cftr": 16}     # general own-refit heavier → fewer repeats


def main(panels):
    rows = []
    for panel in panels:
        grid = GRIDS[panel]; prior = TEST_PRIORS[panel]
        print(f"\n==== {panel} (deployed config) ====", flush=True)
        for seed in (SEED, SEED + 1000):
            folds, _ = fold_preds(panel, nr=NR[panel], seed=seed)
            n = len(folds); f1_dep, tdep = [], []
            fixed = {t: [] for t in grid}
            for ytr, oof, yte, test, Xtr, Xte in folds:
                td, _, _ = optimize_threshold_f1_balanced(ytr, oof, target_prior=prior, n_bootstrap=200)
                tdep.append(td)
                f1_dep.append(f1_at_prior(*tpr_fpr(yte, (test >= td).astype(int)), prior))
                for t in grid:
                    fixed[t].append(f1_at_prior(*tpr_fpr(yte, (test >= t).astype(int)), prior))
            f1_dep = np.array(f1_dep)
            best_t = max(grid, key=lambda t: np.mean(fixed[t]))
            print(f"  seed{seed}: selector F1*={f1_dep.mean():.4f} (picks ~{np.mean(tdep):.3f}); "
                  f"best fixed t={best_t} F1*={np.mean(fixed[best_t]):.4f}", flush=True)
            for t in grid:
                a = np.array(fixed[t]); d = a - f1_dep
                se = d.std(ddof=1)/np.sqrt(n); sig = d.mean()/se if se > 0 else 0.0
                rows.append({"panel": panel, "seed": seed, "threshold": t, "f1": a.mean(),
                             "delta_vs_selector": d.mean(), "sigma": sig, "selector_pick": np.mean(tdep)})
                mark = " *" if t == best_t else ""
                print(f"      t={t:.2f}  F1*={a.mean():.4f}  Δ={d.mean():+.4f} ({sig:+.2f}σ){mark}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "threshold_check_all.csv"), index=False)
    print(f"\nsaved -> threshold_check_all.csv", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["cftr", "pah", "general"])
