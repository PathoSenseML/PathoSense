"""Clean confirmation for cancer: can a CatBoost / XGB / 4-way-blend OWN model beat the
deployed LGBM-own inside cancer's 0.5·transfer + 0.5·own blend?

Same de-biased protocol as exp_pah_own_confirm: FRESH seed (SEED+777), per-fold threshold
picked at the prior on the TRAIN-OOF and applied to the held-out fold, pre-registered
configs (no best-of-25). Baseline = lgbm@.5 (the deployed cancer own at the deployed weight).
25 folds, paired vs baseline. Also reports pure-transfer (w=1) and nested-weight for context.

Run: .venv/bin/python -u analysis/exp_cancer_own_confirm.py
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
from harness import f1_at_prior, tpr_fpr, pick_threshold_at_prior
from exp_own_model_bakeoff import _load
from exp_pah_own_confirm import own_cal, _eval

PANEL = "cancer"


def main():
    df, X, y, cfg, g = _load(PANEL)
    g = np.asarray(g); prior = TEST_PRIORS[PANEL]
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED + 777)
    configs = ["A:lgbm@.5(dep)", "B:cat@.5", "C:xgb@.5", "D:blend4@.5", "E:rf@.5",
               "F:blend4@nestw", "transfer@1.0", "blend4@.25", "blend4@.75"]
    res = {c: [] for c in configs}
    auc = {"lgbm": [], "catboost": [], "rf": [], "xgb": [], "blend4": [], "transfer": []}
    for k, (tr, te) in enumerate(outer.split(X, y)):
        Xtr0, ytr = X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True)
        Xte0, yte = X.iloc[te].reset_index(drop=True), y.iloc[te].reset_index(drop=True)
        Xtr, prep = _fit_own_preprocess(Xtr0, ytr, cfg)
        Xte = _apply_own_preprocess(Xte0, prep).reindex(columns=Xtr.columns)
        cf = _cats(Xtr); ci = [Xtr.columns.get_loc(c) for c in cf]
        _, enc = _rf_repr(Xtr, cf, fit=True) if cf else (None, None)
        g_tr, g_te = g[tr], g[te]
        pg = fit_platt(g_tr, ytr.values); g_tr_c, g_te_c = apply_platt(g_tr, pg), apply_platt(g_te, pg)
        cals_tr, cals_te = {}, {}
        for kind in ["lgbm", "catboost", "rf", "xgb"]:
            t, e = own_cal(kind, Xtr, ytr, Xte, cf, enc, ci, spw, SEED + 777 + k)
            cals_tr[kind], cals_te[kind] = t, e
            auc[kind].append(roc_auc_score(yte.values, e))
        b4_tr = np.mean([cals_tr[c] for c in ["lgbm", "catboost", "rf", "xgb"]], axis=0)
        b4_te = np.mean([cals_te[c] for c in ["lgbm", "catboost", "rf", "xgb"]], axis=0)
        auc["blend4"].append(roc_auc_score(yte.values, b4_te))
        auc["transfer"].append(roc_auc_score(yte.values, g_te))
        yv_tr, yv_te = ytr.values, yte.values
        E = lambda o_tr, o_te, w, nest=False: _eval(g_tr_c, o_tr, g_te_c, o_te, yv_tr, yv_te, w, prior, nested=nest)
        res["A:lgbm@.5(dep)"].append(E(cals_tr["lgbm"], cals_te["lgbm"], 0.5))
        res["B:cat@.5"].append(E(cals_tr["catboost"], cals_te["catboost"], 0.5))
        res["C:xgb@.5"].append(E(cals_tr["xgb"], cals_te["xgb"], 0.5))
        res["D:blend4@.5"].append(E(b4_tr, b4_te, 0.5))
        res["E:rf@.5"].append(E(cals_tr["rf"], cals_te["rf"], 0.5))
        res["F:blend4@nestw"].append(E(b4_tr, b4_te, None, nest=True))
        res["transfer@1.0"].append(E(g_tr_c, g_te_c, 1.0))
        res["blend4@.25"].append(E(b4_tr, b4_te, 0.25))
        res["blend4@.75"].append(E(b4_tr, b4_te, 0.75))

    base = np.array(res["A:lgbm@.5(dep)"])
    print("=== CANCER own-model blend — CLEAN confirmation (25 folds, fresh seed, nested threshold) ===")
    print(f"  prior={prior:.3f}  transfer-AUC={np.mean(auc['transfer']):.4f}  own AUC: " +
          " ".join(f"{k}={np.mean(auc[k]):.4f}" for k in ["lgbm", "catboost", "rf", "xgb", "blend4"]))
    print(f"  A:lgbm@.5 (deployed)  F1*={base.mean():.4f}")
    for c in configs[1:]:
        arr = np.array(res[c]); d = arr - base
        se = d.std(ddof=1) / np.sqrt(len(d)); sig = d.mean() / se if se > 0 else 0
        flag = "  <-- ≥2σ" if sig >= 2 else ""
        print(f"  {c:16s} F1*={arr.mean():.4f}  Δ={d.mean():+.4f} ({sig:+.2f}σ){flag}")
    print("DONE")


if __name__ == "__main__":
    main()
