"""Isolate WHY cancer ft_up (+0.0085) works: is it the FINE-TUNE or just upweighting the LGBM?
  base      : rfup (0.25 lgb_base, 0.5 rf, 0.25 cat) — current deployed
  base_up   : upweight the NON-finetuned lgb (0.5 lgb_base, 0.3 rf, 0.2 cat) — reweight-only control
  ft_up40/50/60 : fine-tune lgb +30, weight it 0.4/0.5/0.6 (rest split rf>cat) — finetune + upweight
If ft_up >> base_up robustly AND holds across weights → fine-tune is the real driver, deploy. 2 seeds.
Run: .venv/bin/python -u analysis/exp_cancer_finetune_v3.py
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import RepeatedStratifiedKFold
from config import SEED
from utils import TEST_PRIORS
from calibration import fit_platt, apply_platt
from harness import pick_threshold_at_prior, tpr_fpr, f1_at_prior
from transfer import fit_transfer_scorer
from exp_transfer import _load, load_target_engineered, target_deployed_fn
from exp_transfer_finetune import project_matrix
from exp_transfer_blendw import transfer_components
from exp_cancer_finetune_v2 import ft_lgbm

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
THR = 0.81
# (w_lgb, w_rf, w_cat)
CONF = {"base": (0.25, 0.50, 0.25, False), "base_up": (0.50, 0.30, 0.20, False),
        "ft_up40": (0.40, 0.35, 0.25, True), "ft_up50": (0.50, 0.30, 0.20, True),
        "ft_up60": (0.60, 0.25, 0.15, True)}


def main():
    prior = TEST_PRIORS["cancer"]
    art = fit_transfer_scorer("cancer"); base_boost = art["models"][0].booster_
    cat_feats = [c for c in art["feats"] if c in art["cat_cats"]]
    df = _load("cancer"); y = df["Label"].astype(int)
    Xs = project_matrix(df, art); L, R, C = transfer_components(df, art)
    Xt, yt = load_target_engineered("cancer"); fn = target_deployed_fn("cancer")
    rows = []
    for seed in (SEED, SEED + 1000):
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=8, random_state=seed)
        acc = {k: [] for k in CONF}
        for tr, te in rskf.split(Xs, y):
            tr, te = np.asarray(tr), np.asarray(te); ytr, yte = y.iloc[tr], y.iloc[te]
            e_te, e_tr = fn(Xt.iloc[tr].reset_index(drop=True), ytr.reset_index(drop=True),
                            Xt.iloc[te].reset_index(drop=True))
            lft_te, lft_oof = ft_lgbm(Xs, y, tr, te, base_boost, cat_feats)
            def score(gtr, gte):
                pl = fit_platt(gtr, ytr.values)
                t = pick_threshold_at_prior(ytr.values, 0.5*apply_platt(gtr, pl)+0.5*e_tr, prior)
                b_te = 0.5*apply_platt(gte, pl)+0.5*e_te
                return f1_at_prior(*tpr_fpr(yte.values, (b_te >= t).astype(int)), prior)
            for k, (wl, wr, wc, ft) in CONF.items():
                Ltr = lft_oof if ft else L[tr]; Lte = lft_te if ft else L[te]
                acc[k].append(score(wl*Ltr+wr*R[tr]+wc*C[tr], wl*Lte+wr*R[te]+wc*C[te]))
        base = np.array(acc["base"]); nf = len(base)
        print(f"  seed{seed} ({nf} folds): base F1*={base.mean():.4f}", flush=True)
        for k in CONF:
            if k == "base": continue
            a = np.array(acc[k]); d = a - base
            sig = d.mean()/(d.std(ddof=1)/np.sqrt(nf)) if d.std() > 0 else 0
            print(f"    {k:8s} F1*={a.mean():.4f}  Δ={d.mean():+.4f} ({sig:+.2f}σ)", flush=True)
            rows.append({"seed": seed, "config": k, "f1": a.mean(), "delta": d.mean(), "sigma": sig})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "cancer_finetune_v3.csv"), index=False)
    print(f"\nsaved -> cancer_finetune_v3.csv", flush=True)


if __name__ == "__main__":
    main()
