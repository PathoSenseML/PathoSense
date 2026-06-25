"""Does a 5-model transfer-scorer blend (LGBM+RF+CatBoost+XGBoost+ExtraTrees) beat the
DEPLOYED transfer scorer? This is the propagating version of the session's ens4 win: the
deployed small-panel score IS the transfer scorer, so a better scorer lifts the deployed
number directly (unlike own-model gains, which wash out in the blend).

We already deploy a 3-model blend (LGBM+RF+CatBoost) because it beat single-LGBM — so the
open question is the 2 extra models. Honest harness (exp_cancer_confirm style): each scorer
is frozen + OOS (trained on general MINUS all target Variant_IDs), Platt-cal per fold, blended
with the target own-model at the DEPLOYED weight, threshold on train-OOF at the test prior,
F1* on the held-out fold. Paired per-fold deltas.

Deployed baselines: cftr = 3-blend @ w=1.0 ; pah = single-LGBM @ w=1.0 ; cancer = 3-blend @ w=0.5
(NOTE: deployed cancer scorer also drops AL+uses fam_flags; build_matrices keeps all features, so
the cancer row here is indicative of the all-features recipe, not the exact deployed scorer).
Run: .venv/bin/python -u analysis/exp_transfer_5blend.py [panel ...]
"""
import os, sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")

import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score

from config import SEED
from utils import TEST_PRIORS
from calibration import fit_platt, apply_platt
from harness import pick_threshold_at_prior, tpr_fpr, f1_at_prior
from exp_transfer import load_target_engineered, target_deployed_fn
from exp_transfer_blend import build_matrices, lgb_score, rf_score, cat_score

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
DEPLOY_W = {"cftr": 1.0, "pah": 1.0, "cancer": 0.5}
N_REPEATS = 10


def _ord(Xg, Xt, cats, fillna=None):
    Xg2, Xt2 = Xg.copy(), Xt.copy()
    if cats:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-2)
        Xg2[cats] = enc.fit_transform(Xg[cats].astype(object).where(Xg[cats].notna(), "NA"))
        Xt2[cats] = enc.transform(Xt[cats].astype(object).where(Xt[cats].notna(), "NA"))
    if fillna is not None:
        Xg2 = Xg2.fillna(fillna); Xt2 = Xt2.fillna(fillna)
    return Xg2, Xt2


def xgb_score(Xg, yg, Xt, cats):
    Xg2, Xt2 = _ord(Xg, Xt, cats, fillna=None)          # XGB handles NaN natively
    spw = (yg == 0).sum() / max((yg == 1).sum(), 1)
    m = xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=4, subsample=0.8,
                          colsample_bytree=0.8, reg_lambda=1.0, scale_pos_weight=spw,
                          tree_method="hist", eval_metric="logloss", n_jobs=6,
                          random_state=SEED).fit(Xg2, yg)
    return m.predict_proba(Xt2)[:, 1]


def et_score(Xg, yg, Xt, cats):
    Xg2, Xt2 = _ord(Xg, Xt, cats, fillna=-999.0)        # ExtraTrees can't take NaN
    m = ExtraTreesClassifier(n_estimators=600, max_features="sqrt", class_weight="balanced",
                             n_jobs=6, random_state=SEED).fit(Xg2, yg)
    return m.predict_proba(Xt2)[:, 1]


def run(panel):
    prior = TEST_PRIORS[panel]; w = DEPLOY_W[panel]
    Xg, yg, Xt, yt, cats = build_matrices(panel)
    print(f"\n[{panel}] building scorers (n_train={len(yg)})...", flush=True)
    sc = {"lgb": lgb_score(Xg, yg, Xt, cats), "rf": rf_score(Xg, yg, Xt, cats),
          "cat": cat_score(Xg, yg, Xt, cats), "xgb": xgb_score(Xg, yg, Xt, cats),
          "et": et_score(Xg, yg, Xt, cats)}
    g3 = (sc["lgb"] + sc["rf"] + sc["cat"]) / 3.0                          # deployed cftr/cancer blend
    g5 = (sc["lgb"] + sc["rf"] + sc["cat"] + sc["xgb"] + sc["et"]) / 5.0   # +XGB +ExtraTrees
    deployed = sc["lgb"] if panel == "pah" else g3                        # pah deploys single-LGBM
    print("  scorer AUC: " + "  ".join(f"{k}={roc_auc_score(yt, v):.4f}" for k, v in sc.items())
          + f"  | 3blend={roc_auc_score(yt, g3):.4f}  5blend={roc_auc_score(yt, g5):.4f}", flush=True)

    Xt_own, yt2 = load_target_engineered(panel)
    fn = target_deployed_fn(panel)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=N_REPEATS, random_state=SEED)
    cfg = {"deployed": deployed, "g5": g5}
    per = {k: [] for k in cfg}
    for tr, te in rskf.split(Xt_own, yt2):
        tr, te = np.asarray(tr), np.asarray(te)
        ytr, yte = yt2.iloc[tr].reset_index(drop=True), yt2.iloc[te].reset_index(drop=True)
        e_te, e_tr = fn(Xt_own.iloc[tr].reset_index(drop=True), ytr, Xt_own.iloc[te].reset_index(drop=True))
        for k, g in cfg.items():
            pl = fit_platt(g[tr], ytr.values)
            gt, ge = apply_platt(g[tr], pl), apply_platt(g[te], pl)
            b_tr = w * gt + (1 - w) * e_tr; b_te = w * ge + (1 - w) * e_te
            thr = pick_threshold_at_prior(ytr.values, b_tr, prior)
            tpr, fpr = tpr_fpr(yte.values, (b_te >= thr).astype(int))
            per[k].append(f1_at_prior(tpr, fpr, prior))
    base = np.array(per["deployed"]); v = np.array(per["g5"]); d = v - base
    nf = len(base); se = d.std(ddof=1) / np.sqrt(nf); sig = d.mean() / se if se > 0 else 0.0
    verdict = ("DEPLOY" if d.mean() > 0 and sig >= 2 else "marginal" if d.mean() > 0 and sig >= 1 else "no gain")
    print(f"  [{panel}] deployed F1*={base.mean():.4f}  5blend F1*={v.mean():.4f}  "
          f"Δ={d.mean():+.4f} ({sig:+.2f}σ)  -> {verdict}", flush=True)
    return {"panel": panel, "deployed_f1": base.mean(), "g5_f1": v.mean(),
            "delta": d.mean(), "sigma": sig, "verdict": verdict}


if __name__ == "__main__":
    panels = sys.argv[1:] or ["cftr", "pah", "cancer"]
    print("=== 5-model transfer blend (+XGB +ExtraTrees) vs deployed scorer — honest CV ===", flush=True)
    rows = []
    for pn in panels:
        t0 = time.time(); rows.append(run(pn)); print(f"  ({pn} {time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "transfer_5blend.csv"), index=False)
    print(f"\nsaved -> {os.path.join(OUT, 'transfer_5blend.csv')}", flush=True)
