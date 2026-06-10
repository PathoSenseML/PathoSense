"""Prior-robust thresholding — does tuning the decision threshold to a BAND of
plausible test priors protect F1 better than tuning to a single point?

Motivation: the organizers confirmed the test set is ~80/20 benign-dominant but
said "~20%" and flagged that the exact per-panel counts in the şartname need
re-checking (Q&A meeting, 2026-06-07). Our deployed thresholds are tuned to a
SINGLE assumed per-panel prior (utils.TEST_PRIORS). If the true test prior drifts,
a point-tuned threshold can sit off its optimum. A threshold tuned over a band
sacrifices a little at the assumed point for a flatter, higher worst-case across
the band — cheap insurance on the metric we're graded on.

This evaluates, per panel: point-tuned threshold vs band-tuned threshold, scored
(honest bootstrap F1 at prior) across a grid of TRUE priors spanning the band.
Reports the worst-case F1 over the band for each, and the cost at the assumed
prior. Deploy the band threshold only where worst-case gain > point cost.

OOF probabilities come from a single deployed-params LightGBM 5-fold CV (light;
the threshold behaviour depends on the probability distribution shape, which a
single model captures well enough for this robustness question).

Run:  .venv/bin/python -u analysis/prior_robust_threshold.py --panel all
"""
import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipelines"))

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score
import lightgbm as lgb
from lightgbm import LGBMClassifier

from config import SEED
from calibration import fit_platt, apply_platt
from utils import optimize_threshold_f1_balanced, TEST_PRIORS
from run_pipeline import get_model_params
from fair_tuneoff import prepare_panel, metrics_at_prior, OUT_DIR

PANELS_ALL = ["general", "cancer", "pah", "cftr"]
BAND_HALFWIDTH = 0.07   # ± around the assumed per-panel prior


def oof_probs(X, y, cat_feats, spw, seed=SEED):
    params = get_model_params(len(X), spw)
    n_splits = 5 if len(X) >= 200 else 3
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for itr, iva in skf.split(X, y):
        i2, ies = train_test_split(itr, test_size=0.15, random_state=seed,
                                   stratify=y.iloc[itr])
        m = LGBMClassifier(**params, random_state=seed)
        m.fit(X.iloc[i2], y.iloc[i2], eval_set=[(X.iloc[ies], y.iloc[ies])],
              callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
              categorical_feature=cat_feats or "auto")
        oof[iva] = m.predict_proba(X.iloc[iva])[:, 1]
    return oof


def optimize_threshold_f1_prior_robust(y_true, probs, band, n_priors=7,
                                       n_bootstrap=150, step=0.01, seed=SEED):
    """Threshold maximizing F1 AVERAGED over a band of priors (not a single point).
    Returns (threshold, band_mean_f1, band_worst_f1)."""
    y = np.asarray(y_true)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5, 0.0, 0.0
    n = len(y)
    rng = np.random.RandomState(seed)
    priors = np.linspace(band[0], band[1], n_priors)
    resamples = {}
    for p in priors:
        n_pos = max(1, int(round(n * p)))
        n_neg = max(1, n - n_pos)
        resamples[p] = [np.concatenate([rng.choice(pos, n_pos, replace=True),
                                        rng.choice(neg, n_neg, replace=True)])
                        for _ in range(n_bootstrap)]
    best_t, best_mean, best_worst = 0.5, 0.0, 0.0
    for t in np.arange(0.05, 0.96, step):
        preds = (probs >= t).astype(int)
        per_prior = [np.mean([f1_score(y[idx], preds[idx], zero_division=0)
                              for idx in resamples[p]]) for p in priors]
        band_mean = float(np.mean(per_prior))
        if band_mean > best_mean:
            best_mean, best_t, best_worst = band_mean, float(t), float(np.min(per_prior))
    return round(best_t, 4), round(best_mean, 4), round(best_worst, 4)


def run_panel(panel):
    X, y, cat_feats, p0 = prepare_panel(panel)
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    oof = oof_probs(X, y, cat_feats, spw)
    platt = fit_platt(oof, y.values)
    oof_cal = apply_platt(oof, platt)
    band = (round(max(0.05, p0 - BAND_HALFWIDTH), 3), round(p0 + BAND_HALFWIDTH, 3))

    t_pt, _, _ = optimize_threshold_f1_balanced(y.values, oof_cal, target_prior=p0,
                                                n_bootstrap=200)
    t_rb, band_mean_rb, worst_rb = optimize_threshold_f1_prior_robust(y.values, oof_cal, band)

    grid = np.linspace(band[0], band[1], 7)
    f1_pt = [metrics_at_prior(y.values, oof_cal, t_pt, p)[0] for p in grid]
    f1_rb = [metrics_at_prior(y.values, oof_cal, t_rb, p)[0] for p in grid]
    f1p0_pt = metrics_at_prior(y.values, oof_cal, t_pt, p0)[0]
    f1p0_rb = metrics_at_prior(y.values, oof_cal, t_rb, p0)[0]

    worst_pt, worst_rbb = min(f1_pt), min(f1_rb)
    worstcase_gain = worst_rbb - worst_pt
    p0_cost = f1p0_pt - f1p0_rb
    verdict = ("WORTH IT" if worstcase_gain > p0_cost + 0.003 else
               "marginal" if abs(worstcase_gain - p0_cost) <= 0.003 else "not worth it")

    print(f"\n{'#'*72}\n  {panel.upper()}  assumed prior p0={p0:.3f}  band={band}\n{'#'*72}")
    print(f"  point threshold  t={t_pt:.3f}   band threshold  t={t_rb:.3f}")
    print(f"  {'true prior':>11s} | " + " ".join(f"{p:>6.3f}" for p in grid))
    print(f"  {'F1 @ point':>11s} | " + " ".join(f"{v:>6.3f}" for v in f1_pt))
    print(f"  {'F1 @ band ':>11s} | " + " ".join(f"{v:>6.3f}" for v in f1_rb))
    print(f"  worst-case F1 over band:  point={worst_pt:.4f}  band={worst_rbb:.4f}  "
          f"(gain {worstcase_gain:+.4f})")
    print(f"  F1 at assumed p0={p0:.3f}:  point={f1p0_pt:.4f}  band={f1p0_rb:.4f}  "
          f"(cost {p0_cost:+.4f})")
    print(f"  VERDICT: {verdict}  (worst-case gain {worstcase_gain:+.4f} vs p0 cost {p0_cost:+.4f})")
    return {"panel": panel, "p0": p0, "band_lo": band[0], "band_hi": band[1],
            "t_point": t_pt, "t_band": t_rb, "worst_point": round(worst_pt, 4),
            "worst_band": round(worst_rbb, 4), "worstcase_gain": round(worstcase_gain, 4),
            "f1_p0_point": round(f1p0_pt, 4), "f1_p0_band": round(f1p0_rb, 4),
            "p0_cost": round(p0_cost, 4), "verdict": verdict}


def main():
    ap = argparse.ArgumentParser(description="Prior-robust threshold evaluation")
    ap.add_argument("--panel", default="all", choices=PANELS_ALL + ["all"])
    args = ap.parse_args()
    panels = PANELS_ALL if args.panel == "all" else [args.panel]
    rows = [run_panel(p) for p in panels]
    out = os.path.join(OUT_DIR, "prior_robust_threshold.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n  Saved {out}")


if __name__ == "__main__":
    main()
