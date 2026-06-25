"""Methodologically-correct redundancy test for the general blend. The AL block = 16 big >0.99 clusters
(one 79-col cluster!) over-representing a few signals. Test, in the blend:
  prune_all  : keep 1 rep per cluster for ALL members (re-confirm).
  lgbm_prune : LGBM gets pruned features (it benefits), RF/Cat/XGB keep FULL (they're redundancy-robust).
  cmean_all  : replace each cluster with its MEAN (denoise+balance, keep info) for ALL members.
  lgbm_cmean : cluster-mean for LGBM only, others FULL.
Per-member feature matrices is the key untested idea. Lighter RF/Cat for the screen (paired Δ still valid);
F1*/AUC, 2 seeds. Winner re-confirmed in the full deployed config before deploy.
Run: .venv/bin/python -u analysis/exp_mask_v2.py
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
from collections import defaultdict
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from config import SEED
from utils import TEST_PRIORS
from schema import partition_columns
from calibration import fit_platt, apply_platt
from harness import load_panel, _drop_leaky, lgb_oof_and_test, pick_threshold_at_prior, tpr_fpr, f1_at_prior
from preprocess_v2 import fit_keep_nan, apply_keep_nan
from general_blend import _rf_repr, _cat_repr, _xgb_model
from run_pipeline import get_model_params

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def clusters_of(Xpp):
    numc = [c for c in Xpp.columns if str(Xpp[c].dtype) != "category"]
    C = Xpp[numc].apply(pd.to_numeric, errors="coerce").corr().abs().fillna(0.0).values
    parent = {c: c for c in numc}
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for i in range(len(numc)):
        for j in range(i + 1, len(numc)):
            if C[i, j] > 0.99:
                parent[find(numc[i])] = find(numc[j])
    cl = defaultdict(list)
    for c in numc:
        cl[find(c)].append(c)
    miss = Xpp[numc].apply(pd.to_numeric, errors="coerce").isna().mean()
    clusters = list(cl.values())
    reps = [min(v, key=lambda c: miss[c]) for v in clusters]   # lowest-missingness rep
    return clusters, reps, numc


def transform(Xpp, mode, clusters, reps, numc):
    if mode == "full":
        return Xpp
    present = set(Xpp.columns)
    cats = [c for c in Xpp.columns if str(Xpp[c].dtype) == "category"]
    if mode == "prune":
        keep = set()
        for v, rp in zip(clusters, reps):
            vp = [c for c in v if c in present]
            if vp:
                keep.add(rp if rp in present else vp[0])
        return Xpp[[c for c in Xpp.columns if (c in keep) or (c in cats)]]
    if mode == "cmean":
        out = {}
        for v, rp in zip(clusters, reps):
            vp = [c for c in v if c in present]
            if not vp:
                continue
            name = rp if rp in present else vp[0]
            if len(vp) == 1:
                out[name] = pd.to_numeric(Xpp[vp[0]], errors="coerce")
            else:
                out[f"CM_{name}"] = Xpp[vp].apply(pd.to_numeric, errors="coerce").mean(axis=1)  # nan-skip
        return pd.concat([pd.DataFrame(out, index=Xpp.index), Xpp[cats]], axis=1)
    raise ValueError(mode)


def blend(Xtr, ytr, Xte, lgbp, lgb_mode, oth_mode, clusters, reps, numc):
    n = len(ytr); spw = (ytr == 0).sum()/max((ytr == 1).sum(), 1)
    # LGBM on its matrix (native cats)
    Xl_tr = transform(Xtr, lgb_mode, clusters, reps, numc); Xl_te = transform(Xte, lgb_mode, clusters, reps, numc)
    catf = [c for c in Xl_tr.columns if str(Xl_tr[c].dtype) == "category"]
    lt, lo = lgb_oof_and_test(Xl_tr, ytr, Xl_te, lgbp, n_seeds=3, n_folds=5)
    lpl = fit_platt(lo, ytr.values); lo, lt = apply_platt(lo, lpl), apply_platt(lt, lpl)
    # RF/Cat/XGB on the "others" matrix (ordinal/cat repr)
    Xo = transform(Xtr, oth_mode, clusters, reps, numc); Xo_te_ = transform(Xte, oth_mode, clusters, reps, numc)
    cf = [c for c in Xo.columns if str(Xo[c].dtype) == "category"]
    Xo_tr, enc = _rf_repr(Xo, cf, fit=True); Xo_te, _ = _rf_repr(Xo_te_, cf, enc=enc, fit=False)
    Xc_tr, Xc_te = _cat_repr(Xo, cf), _cat_repr(Xo_te_, cf); ci = [Xc_tr.columns.get_loc(c) for c in cf]
    def mkrf(): return RandomForestClassifier(n_estimators=400, max_features=0.3, max_depth=9, min_samples_leaf=10,
                                              n_jobs=4, class_weight="balanced", random_state=SEED)
    def mkcat(): return CatBoostClassifier(iterations=600, depth=5, learning_rate=0.03, l2_leaf_reg=2.65,
                                           scale_pos_weight=spw, cat_features=ci, random_seed=SEED, verbose=0,
                                           allow_writing_files=False, early_stopping_rounds=40, thread_count=4)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED); rf_o = np.zeros(n); xg_o = np.zeros(n); cb_o = np.zeros(n)
    for k, (a, b) in enumerate(skf.split(Xo_tr, ytr)):
        rf_o[b] = mkrf().fit(Xo_tr.iloc[a], ytr.iloc[a]).predict_proba(Xo_tr.iloc[b])[:, 1]
        xg_o[b] = _xgb_model().fit(Xo_tr.iloc[a], ytr.iloc[a]).predict_proba(Xo_tr.iloc[b])[:, 1]
        i2, ies = train_test_split(a, test_size=0.15, random_state=SEED+k, stratify=ytr.iloc[a])
        cb = mkcat(); cb.fit(Xc_tr.iloc[i2], ytr.iloc[i2], eval_set=(Xc_tr.iloc[ies], ytr.iloc[ies])); cb_o[b] = cb.predict_proba(Xc_tr.iloc[b])[:, 1]
    rpl, xpl, cpl = fit_platt(rf_o, ytr.values), fit_platt(xg_o, ytr.values), fit_platt(cb_o, ytr.values)
    rf_f = mkrf().fit(Xo_tr, ytr); xg_f = _xgb_model().fit(Xo_tr, ytr)
    i2, ies = train_test_split(np.arange(n), test_size=0.15, random_state=SEED+99, stratify=ytr)
    cb_f = mkcat(); cb_f.fit(Xc_tr.iloc[i2], ytr.iloc[i2], eval_set=(Xc_tr.iloc[ies], ytr.iloc[ies]))
    oof = (lo + apply_platt(rf_o, rpl) + apply_platt(cb_o, cpl) + apply_platt(xg_o, xpl)) / 4.0
    test = (lt + apply_platt(rf_f.predict_proba(Xo_te)[:, 1], rpl) + apply_platt(cb_f.predict_proba(Xc_te)[:, 1], cpl)
            + apply_platt(xg_f.predict_proba(Xo_te)[:, 1], xpl)) / 4.0
    return oof, test


CONFIGS = {"base": ("full", "full"), "prune_all": ("prune", "prune"), "lgbm_prune": ("prune", "full"),
           "cmean_all": ("cmean", "cmean"), "lgbm_cmean": ("cmean", "full")}


def main():
    prior = TEST_PRIORS["general"]
    df = load_panel("general"); _, num, cat, tgt = partition_columns(df)
    X = df[num + cat].reset_index(drop=True); y = df[tgt].astype(int).reset_index(drop=True)
    spw = (y == 0).sum()/max((y == 1).sum(), 1)
    lgbp = {**get_model_params(2000, spw), "scale_pos_weight": 1.0, "bagging_freq": 1, "bagging_fraction": 0.8}
    Xd0, _ = _drop_leaky(X, y, X); Xpp0, _ = fit_keep_nan(Xd0)
    clusters, reps, numc = clusters_of(Xpp0)
    print(f"=== mask v2 (per-member + cluster-mean); {len(clusters)} clusters from {len(numc)} numeric ===", flush=True)
    rows = []
    for seed in (SEED + 1000, SEED + 2000):
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
        acc = {m: {"f1": [], "auc": []} for m in CONFIGS}
        for tr, te in rskf.split(X, y):
            tr, te = np.asarray(tr), np.asarray(te)
            Xtr0, Xte0 = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True)
            ytr, yte = y.iloc[tr].reset_index(drop=True), y.iloc[te].values
            Xtr0, Xte0 = _drop_leaky(Xtr0, ytr, Xte0)
            Xtr_pp, ppf = fit_keep_nan(Xtr0); Xte_pp = apply_keep_nan(Xte0, ppf)
            for c in Xtr_pp.columns:
                if str(Xtr_pp[c].dtype) == "category":
                    Xte_pp[c] = pd.Categorical(Xte_pp[c].astype(object), categories=Xtr_pp[c].cat.categories)
            for mode, (lgm, otm) in CONFIGS.items():
                oof, test = blend(Xtr_pp, ytr, Xte_pp, lgbp, lgm, otm, clusters, reps, numc)
                t = pick_threshold_at_prior(ytr.values, oof, prior)
                acc[mode]["f1"].append(f1_at_prior(*tpr_fpr(yte, (test >= t).astype(int)), prior))
                acc[mode]["auc"].append(roc_auc_score(yte, test))
        bf = np.array(acc["base"]["f1"]); ba = np.array(acc["base"]["auc"]); nf = len(bf)
        print(f"  seed{seed} ({nf}f): base F1*={bf.mean():.4f} AUC={ba.mean():.4f}", flush=True)
        for mode in CONFIGS:
            if mode == "base":
                continue
            f = np.array(acc[mode]["f1"]); a = np.array(acc[mode]["auc"]); d = f - bf; da = a - ba
            sig = d.mean()/(d.std(ddof=1)/np.sqrt(nf)) if d.std() > 0 else 0
            print(f"    {mode:11s} F1*={f.mean():.4f} Δ={d.mean():+.4f} ({sig:+.2f}σ) | AUC Δ={da.mean():+.4f}", flush=True)
            rows.append({"seed": seed, "mode": mode, "f1": f.mean(), "df1": d.mean(), "sigma": sig, "dauc": da.mean()})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "mask_v2.csv"), index=False)
    print(f"\nsaved -> mask_v2.csv", flush=True)


if __name__ == "__main__":
    main()
