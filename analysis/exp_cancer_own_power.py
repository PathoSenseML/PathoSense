"""Decisive higher-power test for the borderline cancer own-model upgrade.

exp_cancer_own_confirm (25 folds, 1 seed) showed blend4@.5 +1.72σ, rf@.5 +1.63σ vs the
deployed lgbm@.5 — consistently positive but sub-2σ. Resolve it with the session's robustness
bar: MULTIPLE fresh seeds, more folds, require positive-in-every-seed AND ≥2σ pooled.

3 seeds x (5x3 RepeatedStratifiedKFold) = 45 paired folds. Same deployment-faithful protocol
(per-fold threshold at prior on train-OOF, applied held-out). Configs: lgbm@.5 (deployed base),
blend4@.5, rf@.5, cat@.5. Reports per-seed Δ and pooled Δ/σ.

Run: .venv/bin/python -u analysis/exp_cancer_own_power.py
"""
import sys, warnings
sys.path.insert(0, "analysis"); sys.path.insert(0, "pipelines")
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score

from config import SEED
from utils import TEST_PRIORS
from predict import _fit_own_preprocess, _apply_own_preprocess
from general_blend import _cats, _rf_repr
from calibration import fit_platt, apply_platt
from harness import f1_at_prior, tpr_fpr
from exp_own_model_bakeoff import _load
from exp_pah_own_confirm import own_cal, _eval

PANEL = "cancer"
SEEDS = [777, 1234, 2025]


def run_seed(X, y, cfg, g, prior, spw, seed_off):
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED + seed_off)
    out = {c: [] for c in ["lgbm@.5", "blend4@.5", "rf@.5", "cat@.5"]}
    for k, (tr, te) in enumerate(outer.split(X, y)):
        Xtr0, ytr = X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True)
        Xte0, yte = X.iloc[te].reset_index(drop=True), y.iloc[te].reset_index(drop=True)
        Xtr, prep = _fit_own_preprocess(Xtr0, ytr, cfg)
        Xte = _apply_own_preprocess(Xte0, prep).reindex(columns=Xtr.columns)
        cf = _cats(Xtr); ci = [Xtr.columns.get_loc(c) for c in cf]
        _, enc = _rf_repr(Xtr, cf, fit=True) if cf else (None, None)
        g_tr, g_te = g[tr], g[te]
        pg = fit_platt(g_tr, ytr.values); g_tr_c, g_te_c = apply_platt(g_tr, pg), apply_platt(g_te, pg)
        ct, ce = {}, {}
        for kind in ["lgbm", "catboost", "rf", "xgb"]:
            t, e = own_cal(kind, Xtr, ytr, Xte, cf, enc, ci, spw, SEED + seed_off + k)
            ct[kind], ce[kind] = t, e
        b4t = np.mean([ct[c] for c in ["lgbm", "catboost", "rf", "xgb"]], axis=0)
        b4e = np.mean([ce[c] for c in ["lgbm", "catboost", "rf", "xgb"]], axis=0)
        yv_tr, yv_te = ytr.values, yte.values
        Ev = lambda o_tr, o_te: _eval(g_tr_c, o_tr, g_te_c, o_te, yv_tr, yv_te, 0.5, prior)
        out["lgbm@.5"].append(Ev(ct["lgbm"], ce["lgbm"]))
        out["blend4@.5"].append(Ev(b4t, b4e))
        out["rf@.5"].append(Ev(ct["rf"], ce["rf"]))
        out["cat@.5"].append(Ev(ct["catboost"], ce["catboost"]))
    return {k: np.array(v) for k, v in out.items()}


def main():
    df, X, y, cfg, g = _load(PANEL)
    g = np.asarray(g); prior = TEST_PRIORS[PANEL]
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    pooled = {c: [] for c in ["lgbm@.5", "blend4@.5", "rf@.5", "cat@.5"]}
    per_seed = {c: [] for c in ["blend4@.5", "rf@.5", "cat@.5"]}
    print(f"=== CANCER own-model power test — {len(SEEDS)} seeds x 15 folds, deployed lgbm@.5 base ===")
    for s in SEEDS:
        r = run_seed(X, y, cfg, g, prior, spw, s)
        base = r["lgbm@.5"]
        line = f"  seed+{s}: base={base.mean():.4f}"
        for c in ["blend4@.5", "rf@.5", "cat@.5"]:
            d = (r[c] - base).mean(); per_seed[c].append(d)
            line += f" | {c} Δ={d:+.4f}"
        print(line, flush=True)
        for c in pooled:
            pooled[c].append(r[c])
    print("  --- pooled across all seeds ---")
    base = np.concatenate(pooled["lgbm@.5"])
    for c in ["blend4@.5", "rf@.5", "cat@.5"]:
        arr = np.concatenate(pooled[c]); d = arr - base
        se = d.std(ddof=1) / np.sqrt(len(d)); sig = d.mean() / se if se > 0 else 0
        allpos = all(x > 0 for x in per_seed[c])
        verdict = "DEPLOY" if (sig >= 2 and allpos) else "reject (sub-bar)"
        print(f"  {c:11s} F1*={arr.mean():.4f}  Δ={d.mean():+.4f} ({sig:+.2f}σ)  "
              f"every-seed+={allpos}  -> {verdict}")
    print("DONE")


if __name__ == "__main__":
    main()
