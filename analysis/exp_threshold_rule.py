"""Decision-rule lever (orthogonal to all model work): how we pick the operating point.

Deployed rule = F1-optimal-at-prior threshold swept on train-OOF (calibration-anchored: a fixed
SCORE cutoff). Alternative = RATE-MATCHING: pick the threshold on train-OOF so the predicted-
positive FRACTION equals the test prior π (rate-anchored). Mechanism for a possible win: at finals
the test set is tiny + benign-dominant and the train→test distribution shifts; a rate-anchored cutoff
is invariant to monotone score drift / miscalibration, while a score cutoff is not. Also test a
50/50 blend of the two thresholds (shrinkage toward agreement).

All thresholds are chosen on TRAIN-OOF only and applied to the held-out fold (honest). Scored by the
same analytic F1* at the per-panel prior. Uses the DEPLOYED per-fold predictions (own / transfer /
blend) via exp_conditional_threshold.fold_preds, so a win here is directly deployable in predict.py.
Run: .venv/bin/python -u analysis/exp_threshold_rule.py [panel ...]
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
from harness import pick_threshold_at_prior, tpr_fpr, f1_at_prior
from exp_conditional_threshold import fold_preds

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def rate_threshold(oof, prior):
    """Threshold so the predicted-positive fraction on OOF equals the prior."""
    q = np.quantile(oof, 1.0 - prior)
    return q


def evaluate(panel, nr):
    folds, prior = fold_preds(panel, nr)
    f1_opt, f1_rate, f1_blend = [], [], []
    for ytr, oof, yte, test, Xtr, Xte in folds:
        t_opt = pick_threshold_at_prior(ytr, oof, prior)         # deployed rule
        t_rate = rate_threshold(oof, prior)                       # rate-matching rule
        t_bl = 0.5 * t_opt + 0.5 * t_rate                         # shrinkage blend
        for t, acc in ((t_opt, f1_opt), (t_rate, f1_rate), (t_bl, f1_blend)):
            acc.append(f1_at_prior(*tpr_fpr(yte, (test >= t).astype(int)), prior))
    f1_opt, f1_rate, f1_blend = map(np.array, (f1_opt, f1_rate, f1_blend))

    def paired(other, name):
        d = other - f1_opt
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        sig = d.mean() / se if se > 0 else 0.0
        vd = "DEPLOY" if d.mean() > 0 and sig >= 2 else "marginal" if d.mean() > 0 and sig >= 1 else "no gain"
        print(f"  [{panel}] {name:11s} F1*={other.mean():.4f}  Δ={d.mean():+.4f} ({sig:+.2f}σ)  {vd}", flush=True)
        return {"panel": panel, "rule": name, "f1": other.mean(), "delta": d.mean(), "sigma": sig}

    print(f"\n[{panel}] deployed F1-opt-at-prior F1*={f1_opt.mean():.4f} ({len(folds)} folds)", flush=True)
    rows = [{"panel": panel, "rule": "f1_opt", "f1": f1_opt.mean(), "delta": 0.0, "sigma": 0.0}]
    rows.append(paired(f1_rate, "rate_match"))
    rows.append(paired(f1_blend, "blend_thr"))
    return rows


if __name__ == "__main__":
    panels = sys.argv[1:] or ["general", "cancer", "pah", "cftr"]
    print("=== Threshold-rule lever: rate-matching & blended vs deployed F1-optimal-at-prior ===", flush=True)
    allrows = []
    for pn in panels:
        t0 = time.time(); allrows += evaluate(pn, 6 if pn == "general" else 12)
        print(f"  ({pn} {time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(allrows).to_csv(os.path.join(OUT, "threshold_rule.csv"), index=False)
    print(f"\nsaved -> threshold_rule.csv", flush=True)
