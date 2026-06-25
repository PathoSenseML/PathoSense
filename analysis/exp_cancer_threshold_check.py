"""Öykü's threshold lead: she finds cancer's stable optimum ~0.81-0.82, but our deployed predict.py
picks 0.85 (optimize_threshold_f1_balanced on full train). Is 0.85 past the peak (too conservative)?

Honest per-fold CV on the CURRENT deployed cancer config (0.5·transfer[rfup+fam_flags] + 0.5·own).
Per fold: (1) deployed method = optimize_threshold_f1_balanced on train-OOF → its threshold + test F1*;
(2) analytic pick_threshold_at_prior on train-OOF → its threshold + test F1*; (3) a grid of FIXED
thresholds applied to test (to MAP the curve — the per-fold-on-train methods are the honest deployables,
the fixed grid is diagnostic of where the true optimum sits and how flat it is). Report mean test F1*
for each + the threshold the deployed method actually picks. If a fixed threshold is robustly >= the
deployed method's F1* AND the deployed method systematically overshoots, lower the deployed threshold.
Run: .venv/bin/python -u analysis/exp_cancer_threshold_check.py
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
from utils import TEST_PRIORS, optimize_threshold_f1_balanced
from harness import pick_threshold_at_prior, tpr_fpr, f1_at_prior
from exp_conditional_threshold import fold_preds

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
GRID = [0.78, 0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87]


def main():
    prior = TEST_PRIORS["cancer"]
    folds, _ = fold_preds("cancer", nr=12)          # 60 folds, current deployed cancer preds
    n = len(folds)
    f1_deployed, f1_analytic = [], []
    t_deployed, t_analytic = [], []
    fixed = {t: [] for t in GRID}
    t_test_opt = []
    for ytr, oof, yte, test, Xtr, Xte in folds:
        td, _, _ = optimize_threshold_f1_balanced(ytr, oof, target_prior=prior, n_bootstrap=200)
        ta = pick_threshold_at_prior(ytr, oof, prior)
        t_deployed.append(td); t_analytic.append(ta)
        f1_deployed.append(f1_at_prior(*tpr_fpr(yte, (test >= td).astype(int)), prior))
        f1_analytic.append(f1_at_prior(*tpr_fpr(yte, (test >= ta).astype(int)), prior))
        best_t, best = GRID[0], -1
        for t in GRID:
            f = f1_at_prior(*tpr_fpr(yte, (test >= t).astype(int)), prior)
            fixed[t].append(f)
            if f > best:
                best, best_t = f, t
        t_test_opt.append(best_t)
    f1_deployed, f1_analytic = np.array(f1_deployed), np.array(f1_analytic)
    print(f"=== cancer threshold check ({n} folds, deployed config rfup+fam_flags) ===", flush=True)
    print(f"  deployed method (bootstrap-balanced on train-OOF): F1*={f1_deployed.mean():.4f}  "
          f"mean threshold picked={np.mean(t_deployed):.3f} (median {np.median(t_deployed):.2f})", flush=True)
    print(f"  analytic pick_threshold_at_prior on train-OOF:     F1*={f1_analytic.mean():.4f}  "
          f"mean threshold picked={np.mean(t_analytic):.3f}", flush=True)
    print(f"  per-fold test-optimal threshold (DIAGNOSTIC, selection-biased): median={np.median(t_test_opt):.2f}  "
          f"mean={np.mean(t_test_opt):.3f}", flush=True)
    print(f"  --- fixed-threshold curve (test F1*; paired Δ vs deployed method) ---", flush=True)
    rows = []
    for t in GRID:
        a = np.array(fixed[t]); d = a - f1_deployed
        se = d.std(ddof=1)/np.sqrt(n); sig = d.mean()/se if se > 0 else 0.0
        flag = "  <-- deployed picks ~here" if abs(t - np.mean(t_deployed)) < 0.015 else ""
        print(f"    t={t:.2f}  F1*={a.mean():.4f}  Δ_vs_deployed={d.mean():+.4f} ({sig:+.2f}σ){flag}", flush=True)
        rows.append({"threshold": t, "f1": a.mean(), "delta_vs_deployed": d.mean(), "sigma": sig})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "cancer_threshold_check.csv"), index=False)
    print(f"\nsaved -> cancer_threshold_check.csv", flush=True)


if __name__ == "__main__":
    main()
