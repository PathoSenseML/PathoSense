"""Check the ENSEMBLE WEIGHTS between the transfer-scorer component models.

The deployed transfer blend averages LGBM+RF+CatBoost with FIXED equal weights (1/3 each).
Question: are equal weights optimal, or does a learned weighting help — and can a learned
stacker rescue XGB/ExtraTrees (which the EQUAL 5-blend wasted by averaging redundant trees)?

Per panel we compare, at the deployed transfer weight w:
  equal3   (lgb+rf+cat)/3                       — DEPLOYED baseline
  stack3   logistic-regression stacker on the 3 deployed components
  stack5   logistic-regression stacker on all 5 (lgb,rf,cat,xgb,et) — can downweight redundant ones
The stacker's TRAIN meta-preds are produced by internal OOF (cross_val_predict) so the threshold
+ Platt are not fit on in-sample stacker outputs (honest). Frozen OOS component scorers (general
minus all target IDs). Paired per-fold ΔF1* at the test prior. We also print the AVERAGE learned
weights so you can see what the optimal mix actually is.
Run: .venv/bin/python -u analysis/exp_blend_weights.py [panel ...]
"""
import os, sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

from config import SEED
from utils import TEST_PRIORS
from calibration import fit_platt, apply_platt
from harness import pick_threshold_at_prior, tpr_fpr, f1_at_prior
from exp_transfer import load_target_engineered, target_deployed_fn
from exp_transfer_blend import build_matrices, lgb_score, rf_score, cat_score
from exp_transfer_5blend import xgb_score, et_score

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
DEPLOY_W = {"cftr": 1.0, "pah": 1.0, "cancer": 0.5}
N_REPEATS = 10
COMP5 = ["lgb", "rf", "cat", "xgb", "et"]


def _stack(C_tr, ytr, C_te):
    """Return (train OOF meta, test meta, coef) for an LR stacker on component probs."""
    lr = LogisticRegression(max_iter=2000, C=1.0)
    meta_tr = cross_val_predict(lr, C_tr, ytr, cv=5, method="predict_proba")[:, 1]
    lr.fit(C_tr, ytr)
    return meta_tr, lr.predict_proba(C_te)[:, 1], lr.coef_[0]


def run(panel):
    prior = TEST_PRIORS[panel]; w = DEPLOY_W[panel]
    Xg, yg, Xt, yt, cats = build_matrices(panel)
    print(f"\n[{panel}] building components (n_train={len(yg)})...", flush=True)
    sc = {"lgb": lgb_score(Xg, yg, Xt, cats), "rf": rf_score(Xg, yg, Xt, cats),
          "cat": cat_score(Xg, yg, Xt, cats), "xgb": xgb_score(Xg, yg, Xt, cats),
          "et": et_score(Xg, yg, Xt, cats)}
    print("  component AUC: " + "  ".join(f"{k}={roc_auc_score(yt, v):.3f}" for k in COMP5 for v in [sc[k]]),
          flush=True)
    C3 = np.column_stack([sc[k] for k in ["lgb", "rf", "cat"]])
    C5 = np.column_stack([sc[k] for k in COMP5])
    eq3 = C3.mean(axis=1)

    Xt_own, yt2 = load_target_engineered(panel)
    fn = target_deployed_fn(panel)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=N_REPEATS, random_state=SEED)
    per = {"equal3": [], "stack3": [], "stack5": []}
    coef3, coef5 = [], []
    for tr, te in rskf.split(Xt_own, yt2):
        tr, te = np.asarray(tr), np.asarray(te)
        ytr, yte = yt2.iloc[tr].reset_index(drop=True), yt2.iloc[te].reset_index(drop=True)
        e_te, e_tr = fn(Xt_own.iloc[tr].reset_index(drop=True), ytr, Xt_own.iloc[te].reset_index(drop=True))
        s3_tr, s3_te, w3 = _stack(C3[tr], ytr.values, C3[te]); coef3.append(w3)
        s5_tr, s5_te, w5 = _stack(C5[tr], ytr.values, C5[te]); coef5.append(w5)
        variants = {"equal3": (eq3[tr], eq3[te]), "stack3": (s3_tr, s3_te), "stack5": (s5_tr, s5_te)}
        for k, (g_tr, g_te) in variants.items():
            pl = fit_platt(g_tr, ytr.values)
            gt, ge = apply_platt(g_tr, pl), apply_platt(g_te, pl)
            b_tr = w * gt + (1 - w) * e_tr; b_te = w * ge + (1 - w) * e_te
            thr = pick_threshold_at_prior(ytr.values, b_tr, prior)
            tpr, fpr = tpr_fpr(yte.values, (b_te >= thr).astype(int))
            per[k].append(f1_at_prior(tpr, fpr, prior))

    base = np.array(per["equal3"]); nf = len(base)
    cm3 = np.mean(coef3, axis=0); cm5 = np.mean(coef5, axis=0)
    print(f"  avg stacker weights  3-model [lgb,rf,cat]={np.round(cm3,2)}", flush=True)
    print(f"                       5-model [lgb,rf,cat,xgb,et]={np.round(cm5,2)}", flush=True)
    print(f"  [{panel}] equal3 F1*={base.mean():.4f}", flush=True)
    rows = [{"panel": panel, "method": "equal3", "f1": base.mean(), "delta": 0.0, "sigma": 0.0}]
    for k in ("stack3", "stack5"):
        v = np.array(per[k]); d = v - base
        se = d.std(ddof=1) / np.sqrt(nf); sig = d.mean() / se if se > 0 else 0.0
        vd = ("DEPLOY" if d.mean() > 0 and sig >= 2 else "marginal" if d.mean() > 0 and sig >= 1 else "no gain")
        print(f"  {k:8s} F1*={v.mean():.4f}  Δ={d.mean():+.4f} ({sig:+.2f}σ)  -> {vd}", flush=True)
        rows.append({"panel": panel, "method": k, "f1": v.mean(), "delta": d.mean(), "sigma": sig})
    return rows


if __name__ == "__main__":
    panels = sys.argv[1:] or ["cftr", "pah", "cancer"]
    print("=== Transfer-blend component weights: equal vs learned stacker — honest CV ===", flush=True)
    allrows = []
    for pn in panels:
        t0 = time.time(); allrows += run(pn); print(f"  ({pn} {time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(allrows).to_csv(os.path.join(OUT, "blend_weights.csv"), index=False)
    print(f"\nsaved -> {os.path.join(OUT, 'blend_weights.csv')}", flush=True)
