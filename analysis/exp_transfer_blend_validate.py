"""Validate the BLENDED transfer scorer (LGBM+RF+CatBoost) vs single-LGBM, on the
honest harness (repeated-CV F1* at the test prior), per panel. Same folds for both;
report best-w F1* for each scorer.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import numpy as np
from scipy.stats import rankdata
from utils import TEST_PRIORS
from harness import FINALS_N
from exp_transfer import _sweep, load_target_engineered, target_deployed_fn
from exp_transfer_blend import build_matrices, lgb_score, rf_score, cat_score

W_BEST = {"pah": [0.5, 0.75, 1.0], "cftr": [0.5, 0.75, 1.0], "cancer": [0.0, 0.25, 0.5]}


def run(panel, n_repeats=6):
    prior = TEST_PRIORS[panel]; nt = FINALS_N[panel]
    Xg, yg, Xt, yt_g, cats = build_matrices(panel)
    g_lgb = lgb_score(Xg, yg, Xt, cats)
    g_rf = rf_score(Xg, yg, Xt, cats)
    g_cat = cat_score(Xg, yg, Xt, cats)
    n = len(yt_g)
    mode = os.environ.get("BLEND_MODE", "rank")
    if mode == "prob":
        g_blend = (g_lgb + g_rf + g_cat) / 3.0           # probability-average (shape-preserving)
    else:
        g_blend = (rankdata(g_lgb) + rankdata(g_rf) + rankdata(g_cat)) / (3 * n)
    Xt_own, yt = load_target_engineered(panel)
    assert (yt.values == yt_g.values).all()
    fn = target_deployed_fn(panel)
    print(f"\n=== {panel} (prior {prior}) — single-LGBM vs BLEND transfer scorer ===", flush=True)
    rows_lgb = _sweep(panel, g_lgb, Xt_own, yt, fn, prior, nt, n_repeats, "LGBM-scorer")
    rows_bl = _sweep(panel, g_blend, Xt_own, yt, fn, prior, nt, n_repeats, "BLEND-scorer")
    best_lgb = max(r["f1"] for r in rows_lgb)
    best_bl = max(r["f1"] for r in rows_bl)
    print(f"  >>> {panel}: best-w F1*  LGBM={best_lgb:.4f}  BLEND={best_bl:.4f}  "
          f"Δ={best_bl-best_lgb:+.4f}", flush=True)


if __name__ == "__main__":
    for pn in (sys.argv[1:] or ["pah", "cftr", "cancer"]):
        run(pn)
