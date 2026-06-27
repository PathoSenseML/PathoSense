"""STEP 1 — per-panel headroom gate (cancer/pah/cftr), at each panel's OWN deployment prior.
General is already gated (STEP 0-3, closed) — cited separately, not re-run here.

For each panel, deployed config via fold_preds (transfer scorer + own blend), multi-seed paired CV:
  - realized   F1 @ the DEPLOYED FIXED threshold (general 0.81 / cancer 0.80 / pah 0.82 / cftr 0.90)
  - honest     F1 @ per-fold bootstrap-selector threshold (train-OOF, no peeking)
  - pr_max     F1 @ the test fold's OWN best threshold (fold-PEEKING ceiling = optimistic upper bound)
  - AUC (per fold)
  - paired gaps realized->honest and realized->pr_max with paired SE + sigma + per-seed signs

Classification:
  THRESHOLD-recoverable : honest (or a fixed retune) robustly > realized  -> free honest F1 sitting there
  RANKING-limited       : honest ~ realized but pr_max >> realized         -> must lift the PR curve (features/data)
  AT-CEILING            : realized ~ honest ~ pr_max                        -> leave it (like general)

Run: .venv/bin/python -u analysis/exp_step1_headroom_panels.py [panel ...]
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from config import SEED
from utils import TEST_PRIORS, optimize_threshold_f1_balanced
from harness import tpr_fpr, f1_at_prior
from exp_conditional_threshold import fold_preds

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
DEPLOYED_THR = {"general": 0.81, "cancer": 0.80, "pah": 0.82, "cftr": 0.90}
NR = {"cancer": 10, "pah": 12, "cftr": 16}
SEEDS = [SEED, SEED + 1000, SEED + 2000]


def pr_max_at_prior(yte, score, prior):
    """Best achievable f1_at_prior over all thresholds on THIS fold (fold-peeking ceiling)."""
    thr = np.unique(score)
    best = 0.0
    for t in thr:
        f = f1_at_prior(*tpr_fpr(yte, (score >= t).astype(int)), prior)
        if f > best:
            best = f
    return best


def paired(a, b):
    """mean delta (a-b), paired SE, sigma."""
    d = np.asarray(a) - np.asarray(b)
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    sig = d.mean() / se if se and se > 0 else 0.0
    return d.mean(), se, sig


def main(panels):
    rows = []
    for panel in panels:
        prior = TEST_PRIORS[panel]; tfix = DEPLOYED_THR[panel]
        print(f"\n==== {panel}  (prior={prior:.3f}, deployed fixed thr={tfix}) ====", flush=True)
        realized_all, honest_all, prmax_all, auc_all = [], [], [], []
        per_seed_real_minus_honest, per_seed_real_minus_prmax = [], []
        for seed in SEEDS:
            folds, _ = fold_preds(panel, nr=NR[panel], seed=seed)
            realized, honest, prmax, aucs = [], [], [], []
            for ytr, oof, yte, test, Xtr, Xte in folds:
                # realized @ deployed fixed threshold
                realized.append(f1_at_prior(*tpr_fpr(yte, (test >= tfix).astype(int)), prior))
                # honest recal: per-fold selector on train-OOF
                td, _, _ = optimize_threshold_f1_balanced(ytr, oof, target_prior=prior, n_bootstrap=200)
                honest.append(f1_at_prior(*tpr_fpr(yte, (test >= td).astype(int)), prior))
                # fold-peeking ceiling
                prmax.append(pr_max_at_prior(yte, test, prior))
                aucs.append(roc_auc_score(yte, test) if len(np.unique(yte)) > 1 else np.nan)
            realized, honest, prmax = map(np.array, (realized, honest, prmax))
            realized_all += realized.tolist(); honest_all += honest.tolist()
            prmax_all += prmax.tolist(); auc_all += [a for a in aucs if a == a]
            per_seed_real_minus_honest.append(np.mean(honest - realized))
            per_seed_real_minus_prmax.append(np.mean(prmax - realized))
            print(f"  seed{seed}: realized={realized.mean():.4f}  honest={honest.mean():.4f}  "
                  f"pr_max={prmax.mean():.4f}  AUC={np.nanmean(aucs):.4f}", flush=True)
        realized_all, honest_all, prmax_all = map(np.array, (realized_all, honest_all, prmax_all))
        d_h, se_h, sig_h = paired(honest_all, realized_all)     # honest gain over realized
        d_p, se_p, sig_p = paired(prmax_all, realized_all)      # peeking ceiling over realized
        auc = float(np.mean(auc_all))
        # classify
        honest_signs = [("+" if v > 0 else "-") for v in per_seed_real_minus_honest]
        if sig_h >= 2.0 and all(v > 0 for v in per_seed_real_minus_honest):
            cls = "THRESHOLD-recoverable"
        elif d_p > 0.015 and sig_p >= 2.0:
            cls = "RANKING-limited"
        else:
            cls = "AT-CEILING"
        print(f"  --> realized={realized_all.mean():.4f}  honest={honest_all.mean():.4f} "
              f"(Δ{d_h:+.4f}, {sig_h:+.2f}σ, seeds {honest_signs})  "
              f"pr_max={prmax_all.mean():.4f} (Δ{d_p:+.4f}, {sig_p:+.2f}σ)  AUC={auc:.4f}", flush=True)
        print(f"  ==> CLASS: {cls}", flush=True)
        rows.append({"panel": panel, "prior": prior, "deployed_thr": tfix,
                     "realized_F1": realized_all.mean(), "honest_F1": honest_all.mean(),
                     "pr_max_F1": prmax_all.mean(), "AUC": auc,
                     "d_honest": d_h, "sigma_honest": sig_h, "honest_seed_signs": "".join(honest_signs),
                     "d_prmax": d_p, "sigma_prmax": sig_p, "n_folds": len(realized_all), "class": cls})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "step1_headroom_panels.csv"), index=False)
    print(f"\nsaved -> step1_headroom_panels.csv\n", flush=True)
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["cancer", "pah", "cftr"])
