"""Test A — does adding a ONE-HOT LINEAR component to the GENERAL deployed blend help?

The deployed general model = avg of calibrated LGBM + RF + CatBoost (all trees, native-NaN matrix).
The 5-blend (adding XGB/ExtraTrees) failed because more trees are redundant. A one-hot LogReg/SVM is
maximally DECORRELATED (different family + different encoding), so it could add what trees can't.

Per outer fold: each component produces train-OOF + test probs (inner CV), Platt-calibrated; variants
blend subsets (prob-average); threshold on the blend train-OOF at the test prior; F1* on the held-out
fold. Paired vs the deployed trees-only blend. general has DOUBLE value (own model + transfer source).
Run: .venv/bin/python -u analysis/exp_tree_linear_blend.py [n_repeats]
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score

from config import SEED
from schema import partition_columns
from utils import TEST_PRIORS
from harness import (load_panel, lgb_oof_and_test, _drop_leaky, pick_threshold_at_prior,
                     tpr_fpr, f1_at_prior)
from preprocess_v2 import fit_keep_nan, apply_keep_nan
from calibration import fit_platt, apply_platt
from run_pipeline import get_model_params
from exp_general_blend import _rf_oof_test, _cat_oof_test
from exp_encoding_bakeoff import _onehot_scaled

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def _sk_oof(Xtr, ytr, Xte, make, nf):
    oof = np.zeros(len(ytr)); skf = StratifiedKFold(nf, shuffle=True, random_state=SEED)
    for a, b in skf.split(Xtr, ytr):
        m = make(); m.fit(Xtr.iloc[a], ytr.iloc[a]); oof[b] = m.predict_proba(Xtr.iloc[b])[:, 1]
    m = make(); m.fit(Xtr, ytr); return oof, m.predict_proba(Xte)[:, 1]


def components(X_tr, y_tr, X_te):
    """Calibrated (test, oof) per component model. Trees on native-NaN matrix; LogReg/SVM on one-hot."""
    Xtr, Xte = _drop_leaky(X_tr.reset_index(drop=True), y_tr.reset_index(drop=True), X_te.reset_index(drop=True))
    Xtr_pp, ppf = fit_keep_nan(Xtr); Xte_pp = apply_keep_nan(Xte, ppf)
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1); nf = 5
    out = {}
    lt, lo = lgb_oof_and_test(Xtr_pp, y_tr, Xte_pp, get_model_params(len(X_tr), spw), n_seeds=2, n_folds=nf)
    out["lgb"] = (lt, lo)
    ro, rt = _rf_oof_test(Xtr_pp, y_tr, Xte_pp, nf, spw); out["rf"] = (rt, ro)
    co, ct = _cat_oof_test(Xtr_pp, y_tr, Xte_pp, nf, spw); out["cat"] = (ct, co)
    Xtr_oh, Xte_oh = _onehot_scaled(Xtr_pp, Xte_pp)
    lro, lrt = _sk_oof(Xtr_oh, y_tr, Xte_oh, lambda: LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced"), nf)
    out["lr"] = (lrt, lro)
    so, st = _sk_oof(Xtr_oh, y_tr, Xte_oh, lambda: SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced", random_state=SEED), nf)
    out["svm"] = (st, so)
    # calibrate each (platt on its own oof)
    cal = {}
    for k, (t, o) in out.items():
        pf = fit_platt(o, y_tr.values); cal[k] = (apply_platt(t, pf), apply_platt(o, pf))
    return cal


VARIANTS = {
    "trees(deployed)": ["lgb", "rf", "cat"],
    "trees+lr":        ["lgb", "rf", "cat", "lr"],
    "trees+svm":       ["lgb", "rf", "cat", "svm"],
    "trees+lr+svm":    ["lgb", "rf", "cat", "lr", "svm"],
    "linear_only":     ["lr", "svm"],
}


def main(n_repeats):
    panel = "general"; prior = TEST_PRIORS[panel]
    df = load_panel(panel); _, num, cat, tgt = partition_columns(df)
    X = df[num + cat].reset_index(drop=True); y = df[tgt].astype(int).reset_index(drop=True)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED)
    per = {v: [] for v in VARIANTS}; aucs = {k: [] for k in ["lgb", "rf", "cat", "lr", "svm"]}
    t0 = time.time()
    for fi, (tr, te) in enumerate(rskf.split(X, y)):
        ytr, yte = y.iloc[tr].reset_index(drop=True), y.iloc[te].reset_index(drop=True)
        cal = components(X.iloc[tr], ytr, X.iloc[te])
        for k in aucs:
            aucs[k].append(roc_auc_score(yte, cal[k][0]))
        for v, comps in VARIANTS.items():
            b_te = np.mean([cal[k][0] for k in comps], axis=0)
            b_tr = np.mean([cal[k][1] for k in comps], axis=0)
            thr = pick_threshold_at_prior(ytr.values, b_tr, prior)
            tpr, fpr = tpr_fpr(yte.values, (b_te >= thr).astype(int))
            per[v].append(f1_at_prior(tpr, fpr, prior))
        if fi == 0:
            print("  (fold 0 done)", flush=True)
    nf = len(per["trees(deployed)"])
    print(f"\n=== general tree+linear blend — {nf} folds, F1* at prior {prior} ({time.time()-t0:.0f}s) ===")
    print("  component test-AUC: " + "  ".join(f"{k}={np.mean(v):.3f}" for k, v in aucs.items()), flush=True)
    base = np.array(per["trees(deployed)"])
    print(f"  trees(deployed) F1*={base.mean():.4f}", flush=True)
    rows = [{"variant": "trees(deployed)", "f1": base.mean(), "delta": 0.0, "sigma": 0.0}]
    for v in VARIANTS:
        if v == "trees(deployed)":
            continue
        x = np.array(per[v]); d = x - base; se = d.std(ddof=1) / np.sqrt(nf); sig = d.mean() / se if se > 0 else 0.0
        vd = ("DEPLOY" if d.mean() > 0 and sig >= 2 else "marginal" if d.mean() > 0 and sig >= 1 else "no gain")
        print(f"  {v:16s} F1*={x.mean():.4f}  Δ={d.mean():+.4f} ({sig:+.2f}σ)  -> {vd}", flush=True)
        rows.append({"variant": v, "f1": x.mean(), "delta": d.mean(), "sigma": sig})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "tree_linear_blend_general.csv"), index=False)
    print(f"\nsaved -> tree_linear_blend_general.csv", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
