"""Clean confirmation of the ONE bake-off signal: pah own RF/CatBoost blended with the
transfer scorer beats pah's deployed pure-transfer (w=1.0).

Deployment-faithful + de-biased vs the bake-off:
  * FRESH seeds (SEED+777, different from the bake-off) → bake-off's per-seed noise won't repeat.
  * Per-fold threshold picked at the prior on the TRAIN-OOF (deployed selector), applied to the
    held-out test fold → no per-fold threshold optimism.
  * Pre-registered configs (no best-of-25): A pure transfer; B rf@.5; C cat@.5; D (rf+cat)@.5;
    plus controls E lgbm@.5 (is it RF specifically, or just adding any own model?) and
    F rf@nested-w (weight chosen per fold on train → no weight pre-selection).
  * w=.4/.6 stability for rf.
5x5 = 25 folds, paired vs A.

Run: .venv/bin/python -u analysis/exp_pah_own_confirm.py
"""
import sys, warnings
sys.path.insert(0, "analysis"); sys.path.insert(0, "pipelines")
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score

from config import SEED
from utils import TEST_PRIORS
from predict import _fit_own_preprocess, _apply_own_preprocess
from general_blend import _cats, _rf_repr
from calibration import fit_platt, apply_platt
from harness import f1_at_prior, tpr_fpr, pick_threshold_at_prior
from exp_own_model_bakeoff import _load, _fit_pred

PANEL = "pah"
WGRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def own_cal(kind, Xtr, ytr, Xte, cat_feats, enc, cat_idx, spw, seed):
    """Return (calibrated train-OOF, calibrated test) for one own-model family."""
    oof = np.zeros(len(ytr))
    skf = StratifiedKFold(3, shuffle=True, random_state=seed)
    for a, b in skf.split(Xtr, ytr):
        oof[b] = _fit_pred(kind, Xtr.iloc[a], ytr.iloc[a], Xtr.iloc[b],
                           cat_feats, enc, cat_idx, spw, seed)
    platt = fit_platt(oof, ytr.values)
    raw_te = _fit_pred(kind, Xtr, ytr, Xte, cat_feats, enc, cat_idx, spw, seed)
    return apply_platt(oof, platt), apply_platt(raw_te, platt)


def _f1(y, p, thr, prior):
    tpr, fpr = tpr_fpr(y, (p >= thr).astype(int))
    return f1_at_prior(tpr, fpr, prior)


def _eval(g_tr, o_tr, g_te, o_te, ytr, yte, w, prior, nested=False):
    """Pick threshold at prior on the blended TRAIN-OOF, apply to held-out test. F1 at prior."""
    if nested:
        w = max(WGRID, key=lambda ww: f1_at_prior(
            *tpr_fpr(ytr, ((ww * g_tr + (1 - ww) * o_tr) >= pick_threshold_at_prior(
                ytr, ww * g_tr + (1 - ww) * o_tr, prior)).astype(int)), prior))
    btr = w * g_tr + (1 - w) * o_tr
    bte = w * g_te + (1 - w) * o_te
    thr = pick_threshold_at_prior(ytr, btr, prior)
    return _f1(yte, bte, thr, prior)


def main():
    df, X, y, cfg, g = _load(PANEL)
    g = np.asarray(g); prior = TEST_PRIORS[PANEL]
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED + 777)
    configs = ["A:transfer", "B:rf@.5", "C:cat@.5", "D:rf+cat@.5",
               "E:lgbm@.5", "F:rf@nestw", "rf@.4", "rf@.6"]
    res = {c: [] for c in configs}
    own_auc = {"rf": [], "cat": [], "lgbm": []}
    for k, (tr, te) in enumerate(outer.split(X, y)):
        Xtr0, ytr = X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True)
        Xte0, yte = X.iloc[te].reset_index(drop=True), y.iloc[te].reset_index(drop=True)
        Xtr, prep = _fit_own_preprocess(Xtr0, ytr, cfg)
        Xte = _apply_own_preprocess(Xte0, prep).reindex(columns=Xtr.columns)
        cf = _cats(Xtr); ci = [Xtr.columns.get_loc(c) for c in cf]
        _, enc = _rf_repr(Xtr, cf, fit=True) if cf else (None, None)
        g_tr, g_te = g[tr], g[te]
        pg = fit_platt(g_tr, ytr.values); g_tr_c, g_te_c = apply_platt(g_tr, pg), apply_platt(g_te, pg)
        rf_tr, rf_te = own_cal("rf", Xtr, ytr, Xte, cf, enc, ci, spw, SEED + 777 + k)
        ct_tr, ct_te = own_cal("catboost", Xtr, ytr, Xte, cf, enc, ci, spw, SEED + 777 + k)
        lg_tr, lg_te = own_cal("lgbm", Xtr, ytr, Xte, cf, enc, ci, spw, SEED + 777 + k)
        rc_tr, rc_te = (rf_tr + ct_tr) / 2, (rf_te + ct_te) / 2
        yv_tr, yv_te = ytr.values, yte.values
        own_auc["rf"].append(roc_auc_score(yv_te, rf_te))
        own_auc["cat"].append(roc_auc_score(yv_te, ct_te))
        own_auc["lgbm"].append(roc_auc_score(yv_te, lg_te))
        res["A:transfer"].append(_eval(g_tr_c, g_tr_c, g_te_c, g_te_c, yv_tr, yv_te, 1.0, prior))
        res["B:rf@.5"].append(_eval(g_tr_c, rf_tr, g_te_c, rf_te, yv_tr, yv_te, 0.5, prior))
        res["C:cat@.5"].append(_eval(g_tr_c, ct_tr, g_te_c, ct_te, yv_tr, yv_te, 0.5, prior))
        res["D:rf+cat@.5"].append(_eval(g_tr_c, rc_tr, g_te_c, rc_te, yv_tr, yv_te, 0.5, prior))
        res["E:lgbm@.5"].append(_eval(g_tr_c, lg_tr, g_te_c, lg_te, yv_tr, yv_te, 0.5, prior))
        res["F:rf@nestw"].append(_eval(g_tr_c, rf_tr, g_te_c, rf_te, yv_tr, yv_te, None, prior, nested=True))
        res["rf@.4"].append(_eval(g_tr_c, rf_tr, g_te_c, rf_te, yv_tr, yv_te, 0.4, prior))
        res["rf@.6"].append(_eval(g_tr_c, rf_tr, g_te_c, rf_te, yv_tr, yv_te, 0.6, prior))

    base = np.array(res["A:transfer"])
    print(f"=== PAH own-model blend — CLEAN confirmation (25 folds, fresh seed, nested threshold) ===")
    print(f"  prior={prior:.3f}  own AUC: rf={np.mean(own_auc['rf']):.4f} "
          f"cat={np.mean(own_auc['cat']):.4f} lgbm={np.mean(own_auc['lgbm']):.4f}")
    print(f"  A:transfer (deployed)  F1*={base.mean():.4f}")
    for c in configs[1:]:
        arr = np.array(res[c]); d = arr - base
        se = d.std(ddof=1) / np.sqrt(len(d)); sig = d.mean() / se if se > 0 else 0
        flag = "  <-- ≥2σ" if sig >= 2 else ""
        print(f"  {c:14s} F1*={arr.mean():.4f}  Δ={d.mean():+.4f} ({sig:+.2f}σ){flag}")
    print("DONE")


if __name__ == "__main__":
    main()
