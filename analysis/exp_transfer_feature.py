"""Angle 1 — transfer score as an INPUT FEATURE (feature-level stacking) vs the deployed fixed-weight
probability blend. Deployed = w·transfer + (1-w)·own (linear). Here the frozen OOS transfer score is fed
INTO the own model as a column (appended post-mRMR so it survives selection), letting the GBDT learn a
non-linear, variant-CONDITIONAL combination (e.g. trust transfer more when conservation is rich).

Frozen OOS scorer (excludes all target Variant_IDs); g[tr]/g[te] both valid (no label leak). Per fold:
own-with-g trained on [own feats + g], threshold on its train-OOF at the test prior, F1* on held-out fold.
Paired vs the deployed blend (w from TRANSFER_PANELS). cancer/pah/cftr.
Run: .venv/bin/python -u analysis/exp_transfer_feature.py [panel ...]
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import RepeatedStratifiedKFold
from config import SEED
from utils import TEST_PRIORS
from calibration import fit_platt, apply_platt
from harness import (_drop_leaky, fast_mrmr, lgb_oof_and_test, pick_threshold_at_prior,
                     tpr_fpr, f1_at_prior)
from run_pipeline import preprocess_data, get_model_params
from feature_engineering import fit_score_agreement, apply_score_agreement
from exp_transfer import _load, load_target_engineered, target_deployed_fn
from transfer import fit_transfer_scorer, transfer_score

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
DEPLOY_W = {"cftr": 1.0, "pah": 1.0, "cancer": 0.5}
N_REPEATS = 10


def own_with_g(panel, Xtr, ytr, Xte, g_tr, g_te):
    sa = panel in ("cancer", "cftr")
    Xd_tr, Xd_te = _drop_leaky(Xtr, ytr, Xte)
    Xtr_pp, _, Xte_pp, _ = preprocess_data(Xd_tr.copy(), Xd_tr.copy(), Xd_te.copy(), 0.95)
    if sa:
        saf = fit_score_agreement(Xtr_pp, ytr, top_k=7)
        Xtr_pp = apply_score_agreement(Xtr_pp, saf); Xte_pp = apply_score_agreement(Xte_pp, saf)
    Xtr_pp, _, Xte_pp, _ = fast_mrmr(Xtr_pp, ytr, Xte_pp, Xte_pp)
    Xtr_pp = Xtr_pp.copy(); Xte_pp = Xte_pp.copy()
    Xtr_pp["transfer_g"] = np.asarray(g_tr); Xte_pp["transfer_g"] = np.asarray(g_te)  # the new feature
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    nf = 3 if len(Xtr) < 200 else 5
    tp, oof = lgb_oof_and_test(Xtr_pp, ytr, Xte_pp, get_model_params(len(Xtr), spw), n_seeds=5, n_folds=nf)
    pf = fit_platt(oof, ytr.values)
    return apply_platt(tp, pf), apply_platt(oof, pf)


def run(panel):
    prior = TEST_PRIORS[panel]; w = DEPLOY_W[panel]
    g = transfer_score(_load(panel), fit_transfer_scorer(panel))      # frozen OOS
    Xt, yt = load_target_engineered(panel)
    own_blend = target_deployed_fn(panel)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=N_REPEATS, random_state=SEED)
    per = {"deployed_blend": [], "g_as_feature": []}
    t0 = time.time()
    for tr, te in rskf.split(Xt, yt):
        tr, te = np.asarray(tr), np.asarray(te)
        ytr, yte = yt.iloc[tr].reset_index(drop=True), yt.iloc[te].reset_index(drop=True)
        Xtr, Xte = Xt.iloc[tr].reset_index(drop=True), Xt.iloc[te].reset_index(drop=True)
        # deployed blend
        e_te, e_tr = own_blend(Xtr, ytr, Xte)
        pl = fit_platt(g[tr], ytr.values); gt, ge = apply_platt(g[tr], pl), apply_platt(g[te], pl)
        b_tr = w * gt + (1 - w) * e_tr; b_te = w * ge + (1 - w) * e_te
        thr = pick_threshold_at_prior(ytr.values, b_tr, prior)
        per["deployed_blend"].append(f1_at_prior(*tpr_fpr(yte.values, (b_te >= thr).astype(int)), prior))
        # g as feature
        f_te, f_tr = own_with_g(panel, Xtr, ytr, Xte, g[tr], g[te])
        thr2 = pick_threshold_at_prior(ytr.values, f_tr, prior)
        per["g_as_feature"].append(f1_at_prior(*tpr_fpr(yte.values, (f_te >= thr2).astype(int)), prior))
    base = np.array(per["deployed_blend"]); v = np.array(per["g_as_feature"]); d = v - base
    nf = len(base); se = d.std(ddof=1) / np.sqrt(nf); sig = d.mean() / se if se > 0 else 0.0
    vd = "DEPLOY" if d.mean() > 0 and sig >= 2 else "marginal" if d.mean() > 0 and sig >= 1 else "no gain"
    print(f"\n[{panel}] deployed-blend F1*={base.mean():.4f}  g-as-feature F1*={v.mean():.4f}  "
          f"Δ={d.mean():+.4f} ({sig:+.2f}σ) [{nf} folds, {time.time()-t0:.0f}s] -> {vd}", flush=True)
    return {"panel": panel, "deployed_blend": base.mean(), "g_as_feature": v.mean(), "delta": d.mean(), "sigma": sig}


if __name__ == "__main__":
    panels = sys.argv[1:] or ["cancer", "pah", "cftr"]
    print("=== Angle 1: transfer score as INPUT FEATURE vs deployed prob-blend ===", flush=True)
    rows = []
    for pn in panels:
        rows.append(run(pn))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "transfer_feature.csv"), index=False)
    print(f"\nsaved -> transfer_feature.csv", flush=True)
