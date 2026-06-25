"""DEFINITIVE per-member masking gate against the ACTUAL deployed general blend (heavy tuned RF 1100 +
CatBoost 1300 + XGB), folds PARALLELIZED across CPU cores (the real speedup vs the broken GPU stack).
Configs: base vs lgbm_prune (LGBM on pruned, others full) vs lgbm_cmean (LGBM on cluster-mean, others
full). 3 seeds, F1*/AUC. Deploy if robustly non-negative both metrics across seeds.
Run: .venv/bin/python -u analysis/exp_mask_final.py
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
import harness
from config import SEED
from utils import TEST_PRIORS
from schema import partition_columns
from calibration import fit_platt, apply_platt
from harness import load_panel, _drop_leaky, lgb_oof_and_test, pick_threshold_at_prior, tpr_fpr, f1_at_prior
from preprocess_v2 import fit_keep_nan, apply_keep_nan
from general_blend import _rf_repr, _cat_repr, _xgb_model, _rf_general, _cat_general
from run_pipeline import get_model_params
from exp_mask_v2 import clusters_of, transform

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
CONFIGS = {"base": ("full", "full"), "lgbm_prune": ("prune", "full"), "lgbm_cmean": ("cmean", "full")}


def blend(Xtr, ytr, Xte, lgbp, lgm, otm, clusters, reps, numc):
    n = len(ytr); spw = (ytr == 0).sum()/max((ytr == 1).sum(), 1)
    Xl_tr = transform(Xtr, lgm, clusters, reps, numc); Xl_te = transform(Xte, lgm, clusters, reps, numc)
    lt, lo = lgb_oof_and_test(Xl_tr, ytr, Xl_te, lgbp, n_seeds=3, n_folds=5)
    lpl = fit_platt(lo, ytr.values); lo, lt = apply_platt(lo, lpl), apply_platt(lt, lpl)
    Xo = transform(Xtr, otm, clusters, reps, numc); Xo_te_ = transform(Xte, otm, clusters, reps, numc)
    cf = [c for c in Xo.columns if str(Xo[c].dtype) == "category"]
    Xo_tr, enc = _rf_repr(Xo, cf, fit=True); Xo_te, _ = _rf_repr(Xo_te_, cf, enc=enc, fit=False)
    Xc_tr, Xc_te = _cat_repr(Xo, cf), _cat_repr(Xo_te_, cf); ci = [Xc_tr.columns.get_loc(c) for c in cf]
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED); rf_o = np.zeros(n); xg_o = np.zeros(n); cb_o = np.zeros(n)
    for k, (a, b) in enumerate(skf.split(Xo_tr, ytr)):
        rf_o[b] = _rf_general().fit(Xo_tr.iloc[a], ytr.iloc[a]).predict_proba(Xo_tr.iloc[b])[:, 1]
        xg_o[b] = _xgb_model().fit(Xo_tr.iloc[a], ytr.iloc[a]).predict_proba(Xo_tr.iloc[b])[:, 1]
        i2, ies = train_test_split(a, test_size=0.15, random_state=SEED+k, stratify=ytr.iloc[a])
        cb = _cat_general(ci, spw); cb.fit(Xc_tr.iloc[i2], ytr.iloc[i2], eval_set=(Xc_tr.iloc[ies], ytr.iloc[ies])); cb_o[b] = cb.predict_proba(Xc_tr.iloc[b])[:, 1]
    rpl, xpl, cpl = fit_platt(rf_o, ytr.values), fit_platt(xg_o, ytr.values), fit_platt(cb_o, ytr.values)
    rf_f = _rf_general().fit(Xo_tr, ytr); xg_f = _xgb_model().fit(Xo_tr, ytr)
    i2, ies = train_test_split(np.arange(n), test_size=0.15, random_state=SEED+99, stratify=ytr)
    cb_f = _cat_general(ci, spw); cb_f.fit(Xc_tr.iloc[i2], ytr.iloc[i2], eval_set=(Xc_tr.iloc[ies], ytr.iloc[ies]))
    oof = (lo + apply_platt(rf_o, rpl) + apply_platt(cb_o, cpl) + apply_platt(xg_o, xpl)) / 4.0
    test = (lt + apply_platt(rf_f.predict_proba(Xo_te)[:, 1], rpl) + apply_platt(cb_f.predict_proba(Xc_te)[:, 1], cpl)
            + apply_platt(xg_f.predict_proba(Xo_te)[:, 1], xpl)) / 4.0
    return oof, test


def one_fold(tr, te, X, y, lgbp, prior, clusters, reps, numc):
    harness._LGB_THREADS = 2
    Xtr0, Xte0 = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True)
    ytr, yte = y.iloc[tr].reset_index(drop=True), y.iloc[te].values
    Xtr0, Xte0 = _drop_leaky(Xtr0, ytr, Xte0)
    Xtr_pp, ppf = fit_keep_nan(Xtr0); Xte_pp = apply_keep_nan(Xte0, ppf)
    for c in Xtr_pp.columns:
        if str(Xtr_pp[c].dtype) == "category":
            Xte_pp[c] = pd.Categorical(Xte_pp[c].astype(object), categories=Xtr_pp[c].cat.categories)
    out = {}
    for mode, (lgm, otm) in CONFIGS.items():
        oof, test = blend(Xtr_pp, ytr, Xte_pp, lgbp, lgm, otm, clusters, reps, numc)
        t = pick_threshold_at_prior(ytr.values, oof, prior)
        out[mode] = (f1_at_prior(*tpr_fpr(yte, (test >= t).astype(int)), prior), roc_auc_score(yte, test))
    return out


def main():
    prior = TEST_PRIORS["general"]
    df = load_panel("general"); _, num, cat, tgt = partition_columns(df)
    X = df[num + cat].reset_index(drop=True); y = df[tgt].astype(int).reset_index(drop=True)
    spw = (y == 0).sum()/max((y == 1).sum(), 1)
    lgbp = {**get_model_params(2000, spw), "scale_pos_weight": 1.0, "bagging_freq": 1, "bagging_fraction": 0.8}
    Xd0, _ = _drop_leaky(X, y, X); Xpp0, _ = fit_keep_nan(Xd0)
    clusters, reps, numc = clusters_of(Xpp0)
    print(f"=== DEFINITIVE per-member masking vs deployed heavy blend (parallel folds) ===", flush=True)
    rows = []
    for seed in (SEED + 1000, SEED + 2000, SEED + 3000):
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed)
        splits = list(rskf.split(X, y))
        res = Parallel(n_jobs=6, backend="loky")(
            delayed(one_fold)(tr, te, X, y, lgbp, prior, clusters, reps, numc) for tr, te in splits)
        agg = {m: {"f1": [], "auc": []} for m in CONFIGS}
        for r in res:
            for m in CONFIGS:
                agg[m]["f1"].append(r[m][0]); agg[m]["auc"].append(r[m][1])
        bf = np.array(agg["base"]["f1"]); ba = np.array(agg["base"]["auc"]); nf = len(bf)
        print(f"  seed{seed} ({nf}f): base F1*={bf.mean():.4f} AUC={ba.mean():.4f}", flush=True)
        for m in CONFIGS:
            if m == "base":
                continue
            f = np.array(agg[m]["f1"]); a = np.array(agg[m]["auc"]); d = f - bf; da = a - ba
            sig = d.mean()/(d.std(ddof=1)/np.sqrt(nf)) if d.std() > 0 else 0
            print(f"    {m:11s} F1*={f.mean():.4f} Δ={d.mean():+.4f} ({sig:+.2f}σ) | AUC Δ={da.mean():+.4f}", flush=True)
            rows.append({"seed": seed, "mode": m, "f1": f.mean(), "df1": d.mean(), "sigma": sig, "dauc": da.mean()})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "mask_final.csv"), index=False)
    print(f"\nsaved -> mask_final.csv", flush=True)


if __name__ == "__main__":
    main()
