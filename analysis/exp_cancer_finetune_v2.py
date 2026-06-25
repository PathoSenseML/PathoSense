"""Re-verify per-panel FINE-TUNING (general-pretrain → panel warm-start) on the CURRENT deployed cancer
config (rfup blend + threshold 0.81). The old test (+1.78σ) fine-tuned the LGBM-only scorer pre-rfup; now
the scorer LGBM is weighted 0.25, so test both:
  ft_rfup : fine-tune LGBM +K trees, keep rfup weights (0.25 lgb_ft, 0.5 rf, 0.25 cat)  [diluted]
  ft_up   : fine-tune LGBM +K trees, UPWEIGHT it (0.5 lgb_ft, 0.3 rf, 0.2 cat)  [leverage adaptation]
Deployed cancer = 0.5·transfer + 0.5·own, threshold 0.81. Per fold: inner-CV fine-tune OOF (honest),
own model refit, paired ΔF1* vs current (base LGBM, rfup). 2 seeds. K=30 (the AUC-best from the probe).
Run: .venv/bin/python -u analysis/exp_cancer_finetune_v2.py
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from config import SEED
from utils import TEST_PRIORS
from calibration import fit_platt, apply_platt
from harness import pick_threshold_at_prior, tpr_fpr, f1_at_prior
from transfer import fit_transfer_scorer
from exp_transfer import _load, load_target_engineered, target_deployed_fn
from exp_transfer_finetune import project_matrix
from exp_transfer_blendw import transfer_components

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
THR = 0.81; K = 30; LR = 0.02


def ft_lgbm(Xs, y, tr, te, base, cat_feats):
    tr = np.asarray(tr); oof = np.zeros(len(tr))
    skf = StratifiedKFold(3, shuffle=True, random_state=SEED)
    def fit(idx):
        spw = (y.iloc[idx] == 0).sum() / max((y.iloc[idx] == 1).sum(), 1)
        m = LGBMClassifier(n_estimators=K, learning_rate=LR, num_leaves=15, min_child_samples=10,
                           reg_lambda=2.0, scale_pos_weight=spw, random_state=SEED, n_jobs=4, verbose=-1)
        m.fit(Xs.iloc[idx], y.iloc[idx], init_model=base, categorical_feature=cat_feats or "auto")
        return m
    for itr_rel, iva_rel in skf.split(Xs.iloc[tr], y.iloc[tr]):
        itr, iva = tr[itr_rel], tr[iva_rel]
        oof[iva_rel] = fit(itr).predict_proba(Xs.iloc[iva])[:, 1]
    return fit(tr).predict_proba(Xs.iloc[te])[:, 1], oof


def main():
    prior = TEST_PRIORS["cancer"]
    art = fit_transfer_scorer("cancer")
    base_boost = art["models"][0].booster_
    cat_feats = [c for c in art["feats"] if c in art["cat_cats"]]
    df = _load("cancer"); y = df["Label"].astype(int)
    Xs = project_matrix(df, art)
    L_base, R, C = transfer_components(df, art)        # rf_p, cat_p fixed (general-minus-cancer)
    Xt, yt = load_target_engineered("cancer"); fn = target_deployed_fn("cancer")
    print(f"cancer finetune-v2 (K={K}); base scorer AUC lgb={roc_auc_score(y,L_base):.4f}", flush=True)
    RFUP = (0.25, 0.50, 0.25); UP = (0.50, 0.30, 0.20)
    rows = []
    for seed in (SEED, SEED + 1000):
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=8, random_state=seed)
        acc = {"base": [], "ft_rfup": [], "ft_up": []}
        for tr, te in rskf.split(Xs, y):
            tr, te = np.asarray(tr), np.asarray(te)
            ytr, yte = y.iloc[tr], y.iloc[te]
            e_te, e_tr = fn(Xt.iloc[tr].reset_index(drop=True), ytr.reset_index(drop=True),
                            Xt.iloc[te].reset_index(drop=True))
            lft_te, lft_oof = ft_lgbm(Xs, y, tr, te, base_boost, cat_feats)
            def score(g_tr, g_te):
                pl = fit_platt(g_tr, ytr.values)
                b_tr = 0.5 * apply_platt(g_tr, pl) + 0.5 * e_tr
                b_te = 0.5 * apply_platt(g_te, pl) + 0.5 * e_te
                t = pick_threshold_at_prior(ytr.values, b_tr, prior)
                return f1_at_prior(*tpr_fpr(yte.values, (b_te >= t).astype(int)), prior)
            acc["base"].append(score(RFUP[0]*L_base[tr]+RFUP[1]*R[tr]+RFUP[2]*C[tr],
                                     RFUP[0]*L_base[te]+RFUP[1]*R[te]+RFUP[2]*C[te]))
            acc["ft_rfup"].append(score(RFUP[0]*lft_oof+RFUP[1]*R[tr]+RFUP[2]*C[tr],
                                        RFUP[0]*lft_te+RFUP[1]*R[te]+RFUP[2]*C[te]))
            acc["ft_up"].append(score(UP[0]*lft_oof+UP[1]*R[tr]+UP[2]*C[tr],
                                      UP[0]*lft_te+UP[1]*R[te]+UP[2]*C[te]))
        base = np.array(acc["base"]); nf = len(base)
        print(f"  seed{seed} ({nf} folds): base F1*={base.mean():.4f}", flush=True)
        for k in ("ft_rfup", "ft_up"):
            a = np.array(acc[k]); d = a - base
            sig = d.mean()/(d.std(ddof=1)/np.sqrt(nf)) if d.std() > 0 else 0
            print(f"    {k:8s} F1*={a.mean():.4f}  Δ={d.mean():+.4f} ({sig:+.2f}σ)", flush=True)
            rows.append({"seed": seed, "config": k, "f1": a.mean(), "delta": d.mean(), "sigma": sig})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "cancer_finetune_v2.csv"), index=False)
    print(f"\nsaved -> cancer_finetune_v2.csv", flush=True)


if __name__ == "__main__":
    main()
