"""Test B — add a ONE-HOT LINEAR component to the deployed TRANSFER scorer (propagates to
cftr/cancer). The 3-tree transfer blend (LGBM+RF+CatBoost) didn't gain from +XGB/+ExtraTrees
(redundant trees). A one-hot LogReg/SVM is maximally decorrelated — does it help where trees didn't?

Frozen OOS scorers on general-minus-target; per-fold Platt + blend with own at deployed w; threshold on
train-OOF at the test prior; F1* held-out; paired vs deployed 3-tree blend.
Run: .venv/bin/python -u analysis/exp_transfer_treelinear.py [panel ...]
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score

from config import SEED
from utils import TEST_PRIORS
from calibration import fit_platt, apply_platt
from harness import pick_threshold_at_prior, tpr_fpr, f1_at_prior
from exp_transfer import load_target_engineered, target_deployed_fn
from exp_transfer_blend import build_matrices, lgb_score, rf_score, cat_score
from exp_encoding_bakeoff import _onehot_scaled

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
DEPLOY_W = {"cftr": 1.0, "pah": 1.0, "cancer": 0.5}
N_REPEATS = int(os.environ.get("NR", 8))


def lr_score(Xg, yg, Xt):
    Xg2, Xt2 = _onehot_scaled(Xg, Xt)
    m = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced").fit(Xg2, yg)
    return m.predict_proba(Xt2)[:, 1]


def svm_score(Xg, yg, Xt):
    Xg2, Xt2 = _onehot_scaled(Xg, Xt)
    m = SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced", random_state=SEED).fit(Xg2, yg)
    return m.predict_proba(Xt2)[:, 1]


def run(panel):
    prior = TEST_PRIORS[panel]; w = DEPLOY_W[panel]
    Xg, yg, Xt, yt, cats = build_matrices(panel)
    print(f"\n[{panel}] building scorers (n_train={len(yg)})...", flush=True)
    sc = {"lgb": lgb_score(Xg, yg, Xt, cats), "rf": rf_score(Xg, yg, Xt, cats),
          "cat": cat_score(Xg, yg, Xt, cats), "lr": lr_score(Xg, yg, Xt), "svm": svm_score(Xg, yg, Xt)}
    print("  scorer AUC on target: " + "  ".join(f"{k}={roc_auc_score(yt, v):.3f}" for k, v in sc.items()), flush=True)
    G = {"trees(deployed)": ["lgb", "rf", "cat"], "trees+lr": ["lgb", "rf", "cat", "lr"],
         "trees+svm": ["lgb", "rf", "cat", "svm"], "trees+lr+svm": ["lgb", "rf", "cat", "lr", "svm"]}
    g = {v: np.mean([sc[k] for k in comps], axis=0) for v, comps in G.items()}

    Xt_own, yt2 = load_target_engineered(panel)
    fn = target_deployed_fn(panel)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=N_REPEATS, random_state=SEED)
    per = {v: [] for v in G}
    for tr, te in rskf.split(Xt_own, yt2):
        tr, te = np.asarray(tr), np.asarray(te)
        ytr, yte = yt2.iloc[tr].reset_index(drop=True), yt2.iloc[te].reset_index(drop=True)
        e_te, e_tr = fn(Xt_own.iloc[tr].reset_index(drop=True), ytr, Xt_own.iloc[te].reset_index(drop=True))
        for v in G:
            pl = fit_platt(g[v][tr], ytr.values)
            gt, ge = apply_platt(g[v][tr], pl), apply_platt(g[v][te], pl)
            b_tr = w * gt + (1 - w) * e_tr; b_te = w * ge + (1 - w) * e_te
            thr = pick_threshold_at_prior(ytr.values, b_tr, prior)
            tpr, fpr = tpr_fpr(yte.values, (b_te >= thr).astype(int))
            per[v].append(f1_at_prior(tpr, fpr, prior))
    base = np.array(per["trees(deployed)"]); nf = len(base)
    print(f"  [{panel}] trees(deployed) F1*={base.mean():.4f}", flush=True)
    rows = [{"panel": panel, "variant": "trees(deployed)", "f1": base.mean(), "delta": 0.0, "sigma": 0.0}]
    for v in G:
        if v == "trees(deployed)":
            continue
        x = np.array(per[v]); d = x - base; se = d.std(ddof=1) / np.sqrt(nf); sig = d.mean() / se if se > 0 else 0.0
        vd = ("DEPLOY" if d.mean() > 0 and sig >= 2 else "marginal" if d.mean() > 0 and sig >= 1 else "no gain")
        print(f"  {v:14s} F1*={x.mean():.4f}  Δ={d.mean():+.4f} ({sig:+.2f}σ)  -> {vd}", flush=True)
        rows.append({"panel": panel, "variant": v, "f1": x.mean(), "delta": d.mean(), "sigma": sig})
    return rows


if __name__ == "__main__":
    panels = sys.argv[1:] or ["cftr", "cancer"]
    print("=== Transfer scorer + one-hot LINEAR — honest CV ===", flush=True)
    allrows = []
    for pn in panels:
        t0 = time.time(); allrows += run(pn); print(f"  ({pn} {time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(allrows).to_csv(os.path.join(OUT, "transfer_treelinear.csv"), index=False)
    print(f"\nsaved -> transfer_treelinear.csv", flush=True)
