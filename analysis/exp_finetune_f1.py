"""#3 — Honest F1* for continued-training fine-tuning (the AUC probe showed pah +2.7σ, cancer +3.9σ).
Build on each panel's CURRENT deployed scorer (fit_transfer_scorer resolves the deployed config, incl.
cancer's fam_flags). Per outer fold: fine-tune K trees on the target train fold (init_model=base),
inner 3-fold OOF on train for honest Platt+threshold, score the held-out fold. Blend at the deployed
w (pah 1.0 pure-transfer; cancer 0.5 with own model). Paired vs the base (no fine-tune).

Run:  .venv/bin/python -u analysis/exp_finetune_f1.py
"""
import os, sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from lightgbm import LGBMClassifier
from config import SEED
from utils import TEST_PRIORS
from calibration import fit_platt, apply_platt
from harness import pick_threshold_at_prior, tpr_fpr, f1_at_prior, FINALS_N
from exp_transfer import _load, load_target_engineered, target_deployed_fn
from transfer import fit_transfer_scorer, TRANSFER_PANELS
from exp_transfer_finetune import project_matrix


def _ft(Xs, y, tr, te, base_booster, cat_feats, K, lr, n_inner=3):
    """Return (test_probs, train_oof) for a K-tree fine-tune (init_model=base) — inner CV for OOF."""
    tr = np.asarray(tr)
    oof = np.zeros(len(tr))
    skf = StratifiedKFold(n_inner, shuffle=True, random_state=SEED)
    def fit(idx):
        spw = (y.iloc[idx] == 0).sum() / max((y.iloc[idx] == 1).sum(), 1)
        m = LGBMClassifier(n_estimators=K, learning_rate=lr, num_leaves=15, min_child_samples=10,
                           reg_lambda=2.0, scale_pos_weight=spw, random_state=SEED, n_jobs=4, verbose=-1)
        m.fit(Xs.iloc[idx], y.iloc[idx], init_model=base_booster, categorical_feature=cat_feats or "auto")
        return m
    for itr_rel, iva_rel in skf.split(Xs.iloc[tr], y.iloc[tr]):
        itr, iva = tr[itr_rel], tr[iva_rel]
        oof[iva_rel] = fit(itr).predict_proba(Xs.iloc[iva])[:, 1]
    test = fit(tr).predict_proba(Xs.iloc[te])[:, 1]
    return test, oof


def run(panel, Ks, n_repeats):
    prior = TEST_PRIORS[panel]; w = TRANSFER_PANELS[panel]
    art = fit_transfer_scorer(panel)
    base = art["models"][0].booster_
    cat_feats = [c for c in art["feats"] if c in art["cat_cats"]]
    Xs = project_matrix(_load(panel), art)
    Xt, y = load_target_engineered(panel)
    g_base = art["models"][0].predict_proba(Xs)[:, 1]    # out-of-sample on target (excl target IDs)
    fn = target_deployed_fn(panel)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED)
    keys = ["base"] + [f"ft{K}" for K in Ks]
    f1 = {k: [] for k in keys}
    t0 = time.time()
    for tr, te in rskf.split(Xs, y):
        tr, te = np.asarray(tr), np.asarray(te)
        ytr, yte = y.iloc[tr], y.iloc[te]
        if w >= 1.0:
            e_tr = np.zeros(len(tr)); e_te = np.zeros(len(te))
        else:
            e_te, e_tr = fn(Xt.iloc[tr].reset_index(drop=True), ytr.reset_index(drop=True),
                            Xt.iloc[te].reset_index(drop=True))
        def blend_f1(g_tr_raw, g_te_raw, oof_for_platt, y_for_platt):
            pl = fit_platt(oof_for_platt, y_for_platt)
            gt, ge = apply_platt(g_tr_raw, pl), apply_platt(g_te_raw, pl)
            b_tr = w * gt + (1 - w) * e_tr; b_te = w * ge + (1 - w) * e_te
            thr = pick_threshold_at_prior(ytr.values, b_tr, prior)
            tpr, fpr = tpr_fpr(yte.values, (b_te >= thr).astype(int))
            return f1_at_prior(tpr, fpr, prior)
        # base: out-of-sample, Platt on base g[tr] directly
        f1["base"].append(blend_f1(g_base[tr], g_base[te], g_base[tr], ytr.values))
        for K in Ks:
            ft_te, ft_oof = _ft(Xs, y, tr, te, base, cat_feats, K, 0.02)
            f1[f"ft{K}"].append(blend_f1(ft_oof, ft_te, ft_oof, ytr.values))
    b = np.array(f1["base"]); nf = len(b)
    print(f"\n[{panel}] base F1*={b.mean():.4f}  ({nf} folds, {time.time()-t0:.0f}s)", flush=True)
    rows = [{"panel": panel, "config": "base", "f1": b.mean(), "delta": 0, "sigma": 0}]
    for K in Ks:
        a = np.array(f1[f"ft{K}"]); d = a - b
        se = d.std(ddof=1) / np.sqrt(nf) if nf > 1 else 0.0
        sig = d.mean() / se if se > 0 else 0.0
        v = "DEPLOY" if d.mean() > 0 and sig >= 2 else ("marginal" if sig >= 1 else "no")
        print(f"  ft+{K:3d} F1*={a.mean():.4f}  Δ={d.mean():+.4f} ({sig:+.2f}σ)  {v}", flush=True)
        rows.append({"panel": panel, "config": f"ft{K}", "f1": a.mean(), "delta": d.mean(), "sigma": sig})
    return rows


if __name__ == "__main__":
    allr = []
    allr += run("pah", [80, 150], n_repeats=10)
    allr += run("cancer", [30, 60], n_repeats=8)
    pd.DataFrame(allr).to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "output",
                                            "exp_finetune_f1.csv"), index=False)
    print("\nsaved -> analysis/output/exp_finetune_f1.csv", flush=True)
