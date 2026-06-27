"""STEP 0 — Headroom gate. Off the deployment-faithful 4-tree cache (/tmp/general_faithful_cache.npz).
Reports: base rate; realized F1 (4-tree @0.81); honest-recal F1 (threshold on train-OOF); PR-curve max-F1
(best fold-peeking threshold @ prior); the headroom 0.537->PR-max with PAIRED SE; baseline AUC + AP.
Decides whether the threshold is the bottleneck (it isn't if realized ~ honest-recal ~ near PR-max honestly).
"""
import sys, os, warnings
sys.path.insert(0, "analysis"); sys.path.insert(0, "pipelines")
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from utils import TEST_PRIORS
from harness import f1_at_prior, tpr_fpr, pick_threshold_at_prior

PRIOR = TEST_PRIORS["general"]; CACHE = "/tmp/general_faithful_cache.npz"
IDX4 = [0, 1, 2, 3]


def f1_at(y, p, t): return f1_at_prior(*tpr_fpr(y, (p >= t).astype(int)), PRIOR)
def prmax(y, p):     # best F1 at the test prior over the whole curve (fold-peeking ceiling)
    return max(f1_at(y, p, q) for q in np.unique(np.quantile(p, np.arange(0.001, 0.999, 0.002))))


z = np.load(CACHE); n = int(z["n"])
folds = [(z[f"O{i}"], z[f"E{i}"], z[f"ytr{i}"], z[f"yte{i}"]) for i in range(n)]
realized, honest, ceil, aucs, aps = [], [], [], [], []
tr_pos = []
for O, E, ytr, yte in folds:
    otr = O[IDX4].mean(0); ote = E[IDX4].mean(0)
    realized.append(f1_at(yte, ote, 0.81))
    honest.append(f1_at(yte, ote, pick_threshold_at_prior(ytr, otr, PRIOR)))
    ceil.append(prmax(yte, ote))
    aucs.append(roc_auc_score(yte, ote)); aps.append(average_precision_score(yte, ote))
    tr_pos.append(ytr.mean())
realized, honest, ceil = map(np.array, (realized, honest, ceil))


def ms(a): return a.mean(), a.std(ddof=1)/np.sqrt(len(a))
print("=== STEP 0 — HEADROOM GATE (general, 4-tree, deployment-faithful, 15 folds) ===")
print(f"  base rate: train {np.mean(tr_pos):.3f} P  | TEST prior (F1 evaluated at) = {PRIOR:.3f}")
print(f"  AUC = {np.mean(aucs):.4f}   AUC-PR/AP = {np.mean(aps):.4f}")
print(f"  realized F1 @0.81 (DEPLOYED)     = {ms(realized)[0]:.4f} ± {ms(realized)[1]:.4f}")
print(f"  honest-recal F1 (thr on train)   = {ms(honest)[0]:.4f} ± {ms(honest)[1]:.4f}")
print(f"  PR-curve max-F1 (fold-peeking)   = {ms(ceil)[0]:.4f} ± {ms(ceil)[1]:.4f}")
# PAIRED gaps
g_honest = ceil - realized      # peeking vs deployed
g_paired_se = (ceil - honest).std(ddof=1)/np.sqrt(len(ceil))
print(f"\n  headroom (PR-max − realized)     = {g_honest.mean():+.4f}  (PEEKING gap; honest thr can't take all of it)")
print(f"  gap (PR-max − honest-recal)      = {(ceil-honest).mean():+.4f} ± {g_paired_se:.4f}  "
      f"({(ceil-honest).mean()/g_paired_se:+.1f}σ paired)")
print(f"  honest-recal vs realized@0.81    = {(honest-realized).mean():+.4f}  (≈0 → 0.81 already ≈ honest-optimal)")
within = abs((honest - realized).mean()) < (realized.std(ddof=1)/np.sqrt(len(realized)))
print(f"\n  VERDICT: deployed 0.81 is {'≈ the honest threshold optimum' if within else 'NOT the honest optimum'}; "
      f"the {g_honest.mean():.4f} to the peeking PR-max is the train→test threshold-generalization gap,")
print(f"  NOT honestly closeable -> THRESHOLD IS NOT THE LEVER. To raise F1, lift the PR curve (AUC-PR) via ranking.")
print("DONE")
