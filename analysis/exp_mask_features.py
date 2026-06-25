"""User/heatmap idea: the deployed general matrix still has ~291 cols with a >0.99 partner (the
corr>0.95 drop misses them: corr(min_periods=30) returns NaN for sparse AL pairs → they survive).
Test MASKING (dropping) the over-correlated / less-effective columns:
  redun@t : drop columns that have a |corr|>t partner already kept (keep the LOWEST-missingness rep);
            label-free, applied via native_nan_oof's post-hook on the preprocessed matrix.
  imp@q   : drop the bottom-q fraction of features by per-fold LGBM gain (honest, fit on train fold).
Gate F1*/AUC vs deployed (no extra pruning), general, 2 seeds. Deploy if robustly non-negative.
Run: .venv/bin/python -u analysis/exp_mask_features.py
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
import lightgbm as lgb
from config import SEED
from harness import cv_evaluate, load_panel, compare, _drop_leaky
from schema import partition_columns
from preprocess_v2 import fit_keep_nan
from meeting_lib import native_nan_oof
from run_pipeline import get_model_params

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
df = load_panel("general"); _, NUM, CAT, TGT = partition_columns(df)
X = df[NUM + CAT].reset_index(drop=True); y = df[TGT].astype(int).reset_index(drop=True)
SPW = (y == 0).sum() / max((y == 1).sum(), 1)
P = {**get_model_params(2000, SPW), "scale_pos_weight": 1.0, "bagging_freq": 1, "bagging_fraction": 0.8}


def redundancy_droplist(t):
    """Label-free: on the full preprocessed general matrix, drop cols with a |corr|>t partner already
    kept; keep the lower-missingness representative per cluster."""
    Xd, _ = _drop_leaky(X, y, X); Xpp, _ = fit_keep_nan(Xd)
    numc = [c for c in Xpp.columns if str(Xpp[c].dtype) != "category"]
    M = Xpp[numc].apply(pd.to_numeric, errors="coerce")
    miss = M.isna().mean()                                  # missingness per col
    C = M.corr().abs().fillna(0.0).values
    order = list(miss.sort_values().index)                  # most-complete first = preferred reps
    pos = {c: i for i, c in enumerate(numc)}
    kept, drop = [], set()
    for c in order:
        if c in drop:
            continue
        kept.append(c)
        ci = pos[c]
        for c2 in numc:
            if c2 != c and c2 not in drop and C[ci, pos[c2]] > t:
                drop.add(c2)
    return sorted(drop)


def mk_redun(t):
    dl = set(redundancy_droplist(t))
    def post(Xtr_pp, ytr, Xte_pp, fit_idx):
        d = [c for c in Xtr_pp.columns if c in dl]
        return Xtr_pp.drop(columns=d), Xte_pp.drop(columns=[c for c in d if c in Xte_pp.columns])
    def fn(Xtr, ytr, Xte):
        return native_nan_oof(Xtr, ytr, Xte, params=P, post=post)
    return fn, len(dl)


def mk_imp(q):
    """Drop bottom-q by per-fold LGBM gain (honest: importance fit on the fit rows only)."""
    def post(Xtr_pp, ytr, Xte_pp, fit_idx):
        catf = [c for c in Xtr_pp.columns if str(Xtr_pp[c].dtype) == "category"]
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, n_jobs=-1,
                               verbose=-1, random_state=SEED)
        m.fit(Xtr_pp.iloc[fit_idx], ytr.iloc[fit_idx], categorical_feature=catf or "auto")
        imp = pd.Series(m.feature_importances_, index=Xtr_pp.columns)
        ndrop = int(q * len(imp)); dl = set(imp.sort_values().index[:ndrop])
        d = [c for c in Xtr_pp.columns if c in dl]
        return Xtr_pp.drop(columns=d), Xte_pp.drop(columns=[c for c in d if c in Xte_pp.columns])
    def fn(Xtr, ytr, Xte):
        return native_nan_oof(Xtr, ytr, Xte, params=P, post=post)
    return fn


def base_fn(Xtr, ytr, Xte):
    return native_nan_oof(Xtr, ytr, Xte, params=P)


if __name__ == "__main__":
    print("=== feature masking on general (redundancy + low-importance), F1*+AUC ===", flush=True)
    variants = {}
    for t in (0.99, 0.95, 0.90):
        fn, nd = mk_redun(t); variants[f"redun>{t} (drop {nd})"] = fn
    for q in (0.25, 0.50):
        variants[f"imp-bottom{int(q*100)}%"] = mk_imp(q)
    rows = []
    for seed in (SEED, SEED + 1000):
        base = cv_evaluate("general", base_fn, label="base", n_repeats=6, df=df, seed=seed)
        print(f"\n  seed{seed}: base F1*={base.f1.mean():.4f} AUC={base.auc.mean():.4f}", flush=True)
        for name, fn in variants.items():
            v = cv_evaluate("general", fn, label=name, n_repeats=6, df=df, seed=seed)
            c = compare(base, v); da = v.auc.mean() - base.auc.mean()
            print(f"    {name:22s} F1*={v.f1.mean():.4f} Δ={c['delta_f1']:+.4f} ({c['sigma']:+.2f}σ) | AUC Δ={da:+.4f}", flush=True)
            rows.append({"seed": seed, "variant": name, "f1": v.f1.mean(), "df1": c["delta_f1"], "sigma": c["sigma"], "dauc": da})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "mask_features.csv"), index=False)
    print(f"\nsaved -> mask_features.csv", flush=True)
