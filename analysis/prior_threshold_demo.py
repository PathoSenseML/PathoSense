"""Illustrative (numpy-only) demo: why the F1 threshold must be tuned for the
TEST prior when training is Pathogenic-dominant but the test set is
Benign-dominant (şartname v2.0 "Klinik Stres Testi").

Expected F1 at threshold t under pathogenic prior pi (large-sample limit):
    F1 = 2*pi*TPR / (2*pi*TPR + (1-pi)*FPR + pi*(1-TPR))
with TPR = P(score>=t | pathogenic), FPR = P(score>=t | benign).

This is a SIMULATION to show direction/magnitude. Real numbers come from
running pipelines/run_pipeline.py in an env with the ML stack installed.
"""
import numpy as np

rng = np.random.RandomState(0)
N = 200_000
benign = rng.beta(2.0, 3.3, N)   # benign: scores skew low
path   = rng.beta(3.3, 2.0, N)   # pathogenic: scores skew high


def auc(pos, neg):
    allv = np.concatenate([pos, neg])
    order = allv.argsort(); ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


print(f"simulated score AUC = {auc(path, benign):.3f}  (illustrative)")

ts = np.arange(0.05, 0.96, 0.005)
TPR = np.array([(path >= t).mean() for t in ts])
FPR = np.array([(benign >= t).mean() for t in ts])


def f1_at(i, pi):
    tpr, fpr = TPR[i], FPR[i]
    den = 2 * pi * tpr + (1 - pi) * fpr + pi * (1 - tpr)
    return (2 * pi * tpr / den) if den > 0 else 0.0


def best_t(pi):
    f1s = [f1_at(i, pi) for i in range(len(ts))]
    i = int(np.argmax(f1s)); return ts[i], f1s[i], i


t05, _, i05 = best_t(0.5)  # OLD: threshold tuned for the stale 50/50 assumption
for name, pi in [("general", 0.143), ("cancer", 0.167), ("pah", 0.286), ("cftr", 0.167)]:
    tpi, f1_new, ipi = best_t(pi)
    f1_old = f1_at(i05, pi)   # OLD threshold, but scored at the REAL test prior
    print(f"\n[{name}]  test prior = {pi:.3f}")
    print(f"  threshold tuned @0.5 (OLD): t={t05:.3f}   ->  F1 on test prior = {f1_old:.4f}")
    print(f"  threshold tuned @{pi:.3f} (NEW): t={tpi:.3f}   ->  F1 on test prior = {f1_new:.4f}")
    print(f"  >>> expected F1 gain from prior-aware threshold: {f1_new - f1_old:+.4f}")
