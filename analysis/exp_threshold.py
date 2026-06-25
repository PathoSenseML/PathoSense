"""#8 — Threshold-selection robustness (the operating point IS the F1 bottleneck at extreme priors).
Per panel/fold, get the deployed blended (train OOF, test probs), then compare threshold methods:
  sweep   : argmax analytic F1* at prior on train OOF (harness default)        <- ref
  boot    : optimize_threshold_f1_balanced (deployed predict.py, 300x bootstrap)
  plateau : median threshold among the near-optimal plateau (within 0.01 F1 of max)
Test F1* at the panel prior, paired. Deploy a >=2sigma winner (esp. tiny cftr/pah).

Run:  .venv/bin/python -u analysis/exp_threshold.py
"""
import os, sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")

from sklearn.model_selection import RepeatedStratifiedKFold
from config import SEED
from utils import TEST_PRIORS, optimize_threshold_f1_balanced
from calibration import fit_platt, apply_platt
from harness import (cv_evaluate, load_panel, f1_at_prior, tpr_fpr, pick_threshold_at_prior)
from exp_transfer import _load, load_target_engineered, target_deployed_fn
from transfer import fit_transfer_scorer, transfer_score, TRANSFER_PANELS
from exp_general_blend import blend_fn


def thr_sweep(ytr, oof, prior):
    return pick_threshold_at_prior(ytr, oof, prior)

def thr_boot(ytr, oof, prior):
    t, _, _ = optimize_threshold_f1_balanced(ytr, oof, target_prior=prior, n_bootstrap=300)
    return t

def thr_plateau(ytr, oof, prior, eps=0.01):
    ts = np.arange(0.02, 0.98, 0.005); f1s = []
    for t in ts:
        tpr, fpr = tpr_fpr(ytr, (oof >= t).astype(int)); f1s.append(f1_at_prior(tpr, fpr, prior))
    f1s = np.array(f1s); mx = f1s.max()
    near = ts[f1s >= mx - eps]
    return float(np.median(near))


def f1_test(yte, test_probs, t, prior):
    tpr, fpr = tpr_fpr(yte, (test_probs >= t).astype(int))
    return f1_at_prior(tpr, fpr, prior)


def fold_probs_small(panel, g_all, fn, tr, te, w):
    """Blended (train OOF, test) probs for a small-panel transfer pipeline."""
    Xt, yt = fold_probs_small.cache
    ytr = yt.iloc[tr].reset_index(drop=True)
    pl = fit_platt(g_all[tr], ytr.values)
    g_tr, g_te = apply_platt(g_all[tr], pl), apply_platt(g_all[te], pl)
    if w >= 1.0:
        return g_tr, g_te
    e_te, e_tr = fn(Xt.iloc[tr].reset_index(drop=True), ytr, Xt.iloc[te].reset_index(drop=True))
    return w * g_tr + (1 - w) * e_tr, w * g_te + (1 - w) * e_te


def run_small(panel, n_repeats):
    prior = TEST_PRIORS[panel]; w = TRANSFER_PANELS[panel]
    raw = _load(panel); Xt, yt = load_target_engineered(panel)
    g_all = transfer_score(raw, fit_transfer_scorer(panel))
    fn = target_deployed_fn(panel); fold_probs_small.cache = (Xt, yt)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED)
    methods = {"sweep": thr_sweep, "boot": thr_boot, "plateau": thr_plateau}
    f1 = {m: [] for m in methods}
    for tr, te in rskf.split(Xt, yt):
        tr, te = np.asarray(tr), np.asarray(te)
        ytr, yte = yt.iloc[tr].values, yt.iloc[te].values
        oof, test = fold_probs_small(panel, g_all, fn, tr, te, w)
        for m, fnc in methods.items():
            f1[m].append(f1_test(yte, test, fnc(ytr, oof, prior), prior))
    return prior, f1


def run_general(n_repeats):
    prior = TEST_PRIORS["general"]; df = load_panel("general")
    from harness import _eval_one_fold  # not used; replicate via blend_fn
    Xall = df[[c for c in df.columns if c not in ("Variant_ID", "Label")]].reset_index(drop=True)
    y = df["Label"].astype(int).reset_index(drop=True)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED)
    methods = {"sweep": thr_sweep, "boot": thr_boot, "plateau": thr_plateau}
    f1 = {m: [] for m in methods}
    for tr, te in rskf.split(Xall, y):
        tr, te = np.asarray(tr), np.asarray(te)
        ytr, yte = y.iloc[tr].reset_index(drop=True), y.iloc[te].values
        test, oof = blend_fn(Xall.iloc[tr].reset_index(drop=True), ytr,
                             Xall.iloc[te].reset_index(drop=True))
        for m, fnc in methods.items():
            f1[m].append(f1_test(yte, test, fnc(ytr.values, oof, prior), prior))
    return prior, f1


def report(panel, prior, f1):
    base = np.array(f1["sweep"]); nf = len(base)
    print(f"\n[{panel}] (prior={prior}, {nf} folds)  sweep F1*={base.mean():.4f}", flush=True)
    rows = [{"panel": panel, "method": "sweep", "f1": base.mean(), "delta": 0, "sigma": 0}]
    for m in ["boot", "plateau"]:
        a = np.array(f1[m]); d = a - base
        se = d.std(ddof=1) / np.sqrt(nf) if nf > 1 else 0.0
        sig = d.mean() / se if se > 0 else 0.0
        v = "DEPLOY" if d.mean() > 0 and sig >= 2 else ("marginal" if sig >= 1 else "no")
        print(f"  {m:8s} F1*={a.mean():.4f}  Δ={d.mean():+.4f} ({sig:+.2f}σ)  {v}", flush=True)
        rows.append({"panel": panel, "method": m, "f1": a.mean(), "delta": d.mean(), "sigma": sig})
    return rows


if __name__ == "__main__":
    allr = []
    for pn, nr in [("pah", 10), ("cftr", 10), ("cancer", 8)]:
        prior, f1 = run_small(pn, nr); allr += report(pn, prior, f1)
    prior, f1 = run_general(5); allr += report("general", prior, f1)
    pd.DataFrame(allr).to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "output",
                                            "exp_threshold.csv"), index=False)
    print("\nsaved -> analysis/output/exp_threshold.csv", flush=True)
