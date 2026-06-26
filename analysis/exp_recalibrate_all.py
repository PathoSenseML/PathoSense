"""Comprehensive RECALIBRATED-THRESHOLD re-audit of every competitive candidate from the session.

Principle: threshold recalibration can only rescue a method whose AUC (ranking) >= baseline. The
only methods that beat their baseline on AUC were the small-panel own-model swaps (CatBoost / RF /
blend4 > own-LGBM); general preprocessing all had AUC <= the deployed blend, so no threshold helps
them. So re-audit the own-model swaps in the DEPLOYED context with an HONEST per-fold recalibrated
threshold (pick_threshold_at_prior on the blend train-OOF, applied to held-out), 3 seeds, vs the
deployed baseline. Each candidate gets its OWN optimal threshold every fold — so if a higher/lower
threshold rescues it, this finds it.

cancer: baseline = lgbm-own @ w=0.5 ; candidates {cat,rf,blend4}-own @ 0.5.
pah/cftr: baseline = pure transfer (w=1.0) ; candidates {rf,cat,blend4}-own @ 0.5 (does adding own help?).

Run: .venv/bin/python -u analysis/exp_recalibrate_all.py
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
from harness import roc_auc_score as _ra  # noqa
from exp_own_model_bakeoff import _load
from exp_pah_own_confirm import own_cal, _eval

SEEDS = [777, 1234, 2025]
# per panel: deployed weight, baseline kind (None=pure transfer), candidate kinds
PANELS = {
    "cancer": dict(w=0.5, base="lgbm", cands=["catboost", "rf", "blend4"]),
    "pah":    dict(w=0.5, base=None,   cands=["rf", "catboost", "blend4"]),
    "cftr":   dict(w=0.5, base=None,   cands=["rf", "catboost", "blend4"]),
}


def run(panel):
    cfgp = PANELS[panel]
    df, X, y, cfg, g = _load(panel); g = np.asarray(g)
    prior = TEST_PRIORS[panel]; spw = (y == 0).sum() / max((y == 1).sum(), 1)
    w = cfgp["w"]
    # accumulate held-out recalibrated-threshold F1 per (config) across seeds
    keys = (["transfer@1.0"] if cfgp["base"] is None else ["lgbm@.5"]) + [f"{k}@.5" for k in cfgp["cands"]]
    pooled = {k: [] for k in keys}; per_seed = {k: [] for k in keys}
    auc = {k: [] for k in (["lgbm", "catboost", "rf", "xgb", "blend4", "transfer"])}
    for s in SEEDS:
        outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=SEED + s)
        seed_vals = {k: [] for k in keys}
        for k_, (tr, te) in enumerate(outer.split(X, y)):
            Xtr0, ytr = X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True)
            Xte0, yte = X.iloc[te].reset_index(drop=True), y.iloc[te].reset_index(drop=True)
            Xtr, prep = _fit_own_preprocess(Xtr0, ytr, cfg)
            Xte = _apply_own_preprocess(Xte0, prep).reindex(columns=Xtr.columns)
            cf = _cats(Xtr); ci = [Xtr.columns.get_loc(c) for c in cf]
            _, enc = _rf_repr(Xtr, cf, fit=True) if cf else (None, None)
            gtr, gte = g[tr], g[te]
            pg = fit_platt(gtr, ytr.values); gtr_c, gte_c = apply_platt(gtr, pg), apply_platt(gte, pg)
            ct, ce = {}, {}
            for kind in ["lgbm", "catboost", "rf", "xgb"]:
                t, e = own_cal(kind, Xtr, ytr, Xte, cf, enc, ci, spw, SEED + s + k_)
                ct[kind], ce[kind] = t, e; auc[kind].append(roc_auc_score(yte.values, e))
            ct["blend4"] = np.mean([ct[k] for k in ["lgbm", "catboost", "rf", "xgb"]], axis=0)
            ce["blend4"] = np.mean([ce[k] for k in ["lgbm", "catboost", "rf", "xgb"]], axis=0)
            auc["blend4"].append(roc_auc_score(yte.values, ce["blend4"]))
            auc["transfer"].append(roc_auc_score(yte.values, gte))
            yv_tr, yv_te = ytr.values, yte.values
            # baseline
            if cfgp["base"] is None:
                seed_vals["transfer@1.0"].append(_eval(gtr_c, gtr_c, gte_c, gte_c, yv_tr, yv_te, 1.0, prior))
            else:
                seed_vals["lgbm@.5"].append(_eval(gtr_c, ct["lgbm"], gte_c, ce["lgbm"], yv_tr, yv_te, 0.5, prior))
            for kind in cfgp["cands"]:
                seed_vals[f"{kind}@.5"].append(_eval(gtr_c, ct[kind], gte_c, ce[kind], yv_tr, yv_te, 0.5, prior))
        for k in keys:
            arr = np.array(seed_vals[k]); per_seed[k].append(arr.mean()); pooled[k].extend(seed_vals[k])

    base_key = keys[0]; base = np.array(pooled[base_key])
    print(f"\n=== {panel}  (prior {prior:.3f}, RECALIBRATED threshold per fold, {len(base)} folds/3 seeds) ===")
    print("  own AUC: " + " ".join(f"{k}={np.mean(auc[k]):.4f}" for k in ["lgbm", "catboost", "rf", "blend4", "transfer"]))
    print(f"  baseline {base_key:14s} F1*={base.mean():.4f}")
    for k in keys[1:]:
        arr = np.array(pooled[k]); d = arr - base
        se = d.std(ddof=1)/np.sqrt(len(d)) if len(d) > 1 else 0
        sig = d.mean()/se if se > 0 else 0
        allpos = all((np.array(per_seed[k]) - np.array(per_seed[base_key]))[i] > 0 for i in range(len(SEEDS)))
        flag = "  <<< WIN" if sig >= 2 and allpos else ""
        print(f"  {k:14s} F1*={arr.mean():.4f}  Δ={d.mean():+.4f} ({sig:+.2f}σ)  every-seed+={allpos}{flag}")


if __name__ == "__main__":
    print("=== Recalibrated-threshold re-audit of competitive candidates (own-model swaps) ===")
    for p in ("cancer", "pah", "cftr"):
        run(p)
    print("\nNOTE: general preprocessing variants all had AUC <= the deployed blend, so no threshold")
    print("recalibration can make them win (worse ranking caps best-F1). Not re-run.")
    print("DONE")
