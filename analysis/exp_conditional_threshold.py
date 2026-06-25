"""Angle 2 — conditional/subgroup thresholds. We've only ever used ONE global threshold per panel. If the
deployed score's calibration differs by stratum, per-stratum thresholds could lift F1 without touching the
model. Strata (chosen for plausibly-different score behaviour):
  popmax_present : CAT_1 observed (variant in gnomAD) vs missing (ultra-rare/novel)
  cons_median    : EK_7 conservation above/below the train median

Deployed predictions per panel: general = own native-NaN LGBM; cancer = 0.5·transfer+0.5·own; pah/cftr =
pure transfer. Per fold: per-stratum threshold picked on train-OOF-in-stratum (≥20 samples else global
fallback), applied to test-in-stratum; pooled test F1*. Paired vs single global threshold.
Run: .venv/bin/python -u analysis/exp_conditional_threshold.py [panel ...]
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import RepeatedStratifiedKFold
from config import SEED
from utils import TEST_PRIORS
from schema import partition_columns
from calibration import fit_platt, apply_platt
from harness import load_panel, pick_threshold_at_prior, tpr_fpr, f1_at_prior
from meeting_lib import native_nan_oof
from exp_transfer import _load, load_target_engineered, target_deployed_fn
from transfer import fit_transfer_scorer, transfer_score

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
DEPLOY_W = {"cftr": 1.0, "pah": 1.0, "cancer": 0.5}


def strata(Xtr, Xte, name, tr_ref=None):
    if name == "popmax_present":
        return (Xtr["CAT_1"].notna().values.astype(int) if "CAT_1" in Xtr else np.zeros(len(Xtr), int),
                Xte["CAT_1"].notna().values.astype(int) if "CAT_1" in Xte else np.zeros(len(Xte), int))
    if name == "cons_median":
        med = Xtr["EK_7"].median()
        return ((Xtr["EK_7"] > med).fillna(False).values.astype(int),
                (Xte["EK_7"] > med).fillna(False).values.astype(int))
    raise ValueError(name)


def fold_preds(panel, nr, seed=SEED):
    prior = TEST_PRIORS[panel]; folds = []
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=nr, random_state=seed)
    if panel == "general":
        df = load_panel("general"); _, num, cat, tgt = partition_columns(df)
        X = df[num + cat].reset_index(drop=True); y = df[tgt].astype(int).reset_index(drop=True)
        for tr, te in rskf.split(X, y):
            Xtr, Xte = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True)
            ytr, yte = y.iloc[tr].reset_index(drop=True), y.iloc[te].reset_index(drop=True)
            test, oof = native_nan_oof(Xtr, ytr, Xte)
            folds.append((ytr.values, oof, yte.values, test, Xtr, Xte))
    else:
        g = transfer_score(_load(panel), fit_transfer_scorer(panel)); w = DEPLOY_W[panel]
        Xt, yt = load_target_engineered(panel); fn = target_deployed_fn(panel)
        for tr, te in rskf.split(Xt, yt):
            tr, te = np.asarray(tr), np.asarray(te)
            ytr, yte = yt.iloc[tr].reset_index(drop=True), yt.iloc[te].reset_index(drop=True)
            Xtr, Xte = Xt.iloc[tr].reset_index(drop=True), Xt.iloc[te].reset_index(drop=True)
            e_te, e_tr = fn(Xtr, ytr, Xte)
            pl = fit_platt(g[tr], ytr.values)
            oof = w * apply_platt(g[tr], pl) + (1 - w) * e_tr
            test = w * apply_platt(g[te], pl) + (1 - w) * e_te
            folds.append((ytr.values, oof, yte.values, test, Xtr, Xte))
    return folds, prior


def evaluate(panel, nr):
    folds, prior = fold_preds(panel, nr)
    glob = []
    for ytr, oof, yte, test, Xtr, Xte in folds:
        t = pick_threshold_at_prior(ytr, oof, prior)
        glob.append(f1_at_prior(*tpr_fpr(yte, (test >= t).astype(int)), prior))
    glob = np.array(glob)
    print(f"\n[{panel}] global-threshold F1*={glob.mean():.4f} ({len(folds)} folds)", flush=True)
    rows = [{"panel": panel, "strat": "global", "f1": glob.mean(), "delta": 0.0, "sigma": 0.0}]
    for name in ("popmax_present", "cons_median"):
        cond = []
        for ytr, oof, yte, test, Xtr, Xte in folds:
            s_tr, s_te = strata(Xtr, Xte, name)
            t_glob = pick_threshold_at_prior(ytr, oof, prior)
            pred = np.zeros(len(yte), int)
            for s in np.unique(s_tr):
                trm = s_tr == s; tem = s_te == s
                t_s = pick_threshold_at_prior(ytr[trm], oof[trm], prior) if trm.sum() >= 20 else t_glob
                pred[tem] = (test[tem] >= t_s).astype(int)
            cond.append(f1_at_prior(*tpr_fpr(yte, pred), prior))
        cond = np.array(cond); d = cond - glob
        se = d.std(ddof=1) / np.sqrt(len(d)); sig = d.mean() / se if se > 0 else 0.0
        vd = "DEPLOY" if d.mean() > 0 and sig >= 2 else "marginal" if d.mean() > 0 and sig >= 1 else "no gain"
        print(f"  [{panel}] cond:{name:14s} F1*={cond.mean():.4f}  Δ={d.mean():+.4f} ({sig:+.2f}σ)  {vd}", flush=True)
        rows.append({"panel": panel, "strat": name, "f1": cond.mean(), "delta": d.mean(), "sigma": sig})
    return rows


if __name__ == "__main__":
    panels = sys.argv[1:] or ["general", "cancer", "pah", "cftr"]
    print("=== Angle 2: conditional/subgroup thresholds vs single global ===", flush=True)
    allrows = []
    for pn in panels:
        t0 = time.time(); allrows += evaluate(pn, 4 if pn == "general" else 10)
        print(f"  ({pn} {time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(allrows).to_csv(os.path.join(OUT, "conditional_threshold.csv"), index=False)
    print(f"\nsaved -> conditional_threshold.csv", flush=True)
