"""FOUNDATIONAL TEST — is the deployed preprocessing earning its place, or undercutting everything
we layered on top of it this session?

The deployed native-NaN preprocessing (preprocess_v2.fit_keep_nan) does, in order:
  dropcs   drop constant/sparse columns
  rank     rank-transform (pct) the population-frequency AL columns
  clip     clip conservation to [p1, p99]
  missind  add missing-indicator columns (5-95% NaN)
  lowvar   drop near-zero-variance columns
  corr     drop one of each |corr|>0.95 pair
We reimplement it with per-step toggles (flex_fit/flex_apply, output-identical to fit_keep_nan when all
steps on) and test, paired vs the full deployed config on identical folds (native-cat LGBM, honest F1*):
  MINIMAL    no steps (raw numeric + native NaN + native cats)  -> does all the preprocessing even help?
  -rank / -clip / -missind / -corr / -dropcs   leave-one-out    -> is any single step harmful/inert?
If MINIMAL >= full, or a leave-one-out beats full, the preprocessing is undercutting and the session's
interventions should be re-tested on the better baseline.
Run: .venv/bin/python -u analysis/exp_preprocess_ablation.py [panel ...]
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")

from harness import cv_evaluate, load_panel, compare, _drop_leaky, lgb_oof_and_test
from schema import (partition_columns, classify_columns, CAT_POPULATION_FREQ, CAT_CONSERVATION,
                    CAT_SCORE_CENTERED, CAT_CONSTANT, CAT_SPARSE)
from calibration import fit_platt, apply_platt
from run_pipeline import get_model_params

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
ALL_STEPS = ("dropcs", "rank", "clip", "missind", "lowvar", "corr")


def flex_fit(Xtr, steps):
    _, num_cols, cat_cols, _ = partition_columns(Xtr)
    Xn = Xtr[num_cols].copy(); cats = classify_columns(Xtr)
    f = {"cat_cols": list(cat_cols), "drop_c": [], "pop_grids": {}, "cons_clip": {}, "miss_cols": []}
    if "dropcs" in steps:
        f["drop_c"] = [c for c, k in cats.items() if k in (CAT_CONSTANT, CAT_SPARSE)]
        Xn.drop(columns=f["drop_c"], inplace=True, errors="ignore")
    if "rank" in steps:
        for c in [c for c in Xn.columns if cats.get(c) == CAT_POPULATION_FREQ]:
            nn = Xn[c].dropna().sort_values().values
            if not len(nn):
                continue
            f["pop_grids"][c] = (nn, np.linspace(1 / len(nn), 1.0, len(nn)))
            Xn[c] = Xn[c].rank(pct=True, na_option="keep")
    if "clip" in steps:
        for c in [c for c in Xn.columns if cats.get(c) in (CAT_CONSERVATION, CAT_SCORE_CENTERED)]:
            s = Xn[c].dropna()
            if len(s) < 10:
                continue
            lo, hi = float(s.quantile(.01)), float(s.quantile(.99)); f["cons_clip"][c] = (lo, hi)
            Xn[c] = Xn[c].clip(lo, hi)
    if "missind" in steps:
        miss = Xn.isna().mean(); f["miss_cols"] = miss[(miss > .05) & (miss < .95)].index.tolist()
        for c in f["miss_cols"]:
            Xn[f"{c}_missing"] = Xn[c].isna().astype(int)
    if "lowvar" in steps:
        Xn.drop(columns=Xn.var()[lambda v: v < 1e-6].index.tolist(), inplace=True, errors="ignore")
    if "corr" in steps:
        raw = Xtr[[c for c in Xn.columns if c in Xtr.columns]]
        if raw.shape[1] > 50:
            corr = raw.corr(min_periods=30).abs()
            up = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))
            Xn.drop(columns=[c for c in up.columns if (up[c] > 0.95).any()], inplace=True, errors="ignore")
    f["num_kept"] = list(Xn.columns)
    if cat_cols:
        Xn = pd.concat([Xn.reset_index(drop=True), Xtr[cat_cols].reset_index(drop=True)], axis=1)
    return Xn, f


def flex_apply(X, f):
    Xn = X[[c for c in X.columns if c not in f["cat_cols"]]].copy()
    Xn.drop(columns=f["drop_c"], inplace=True, errors="ignore")
    for c, (nn, rk) in f["pop_grids"].items():
        if c not in Xn.columns:
            continue
        v = Xn[c].values.astype(float); m = ~np.isnan(v); out = np.full_like(v, np.nan)
        if m.any() and len(nn):
            out[m] = np.interp(v[m], nn, rk, left=rk[0], right=rk[-1])
        Xn[c] = out
    for c, (lo, hi) in f["cons_clip"].items():
        if c in Xn.columns:
            Xn[c] = Xn[c].clip(lo, hi)
    for c in f["miss_cols"]:
        if c in Xn.columns:
            Xn[f"{c}_missing"] = Xn[c].isna().astype(int)
    for c in f["num_kept"]:
        if c not in Xn.columns:
            Xn[c] = np.nan
    out = Xn[f["num_kept"]]
    if f["cat_cols"]:
        out = pd.concat([out.reset_index(drop=True), X[f["cat_cols"]].reset_index(drop=True)], axis=1)
    return out


def fn(steps):
    def _fn(X_tr, y_tr, X_te):
        y_tr = y_tr.reset_index(drop=True)
        Xtr, Xte = _drop_leaky(X_tr.reset_index(drop=True), y_tr, X_te.reset_index(drop=True))
        Xtr_pp, f = flex_fit(Xtr, set(steps)); Xte_pp = flex_apply(Xte, f)
        spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        nf = 3 if len(X_tr) < 200 else 5
        tp, oof = lgb_oof_and_test(Xtr_pp, y_tr, Xte_pp, get_model_params(len(X_tr), spw), n_seeds=2, n_folds=nf)
        pf = fit_platt(oof, y_tr.values); return apply_platt(tp, pf), apply_platt(oof, pf)
    return _fn


def run(panel, nr):
    df = load_panel(panel)
    full = cv_evaluate(panel, fn(ALL_STEPS), label="full", n_repeats=nr, df=df)
    print(f"\n[{panel}] FULL (deployed preprocessing) F1*={full.summary()['f1_mean']:.4f}", flush=True)
    configs = {"MINIMAL(raw)": ()}
    for s in ("dropcs", "rank", "clip", "missind", "corr"):
        configs[f"-{s}"] = tuple(x for x in ALL_STEPS if x != s)
    rows = [{"panel": panel, "config": "full", "f1": full.summary()["f1_mean"], "delta": 0.0, "sigma": 0.0}]
    for name, steps in configs.items():
        r = cv_evaluate(panel, fn(steps), label=name, n_repeats=nr, df=df)
        c = compare(full, r)   # b-a = config - full ; positive => config BEATS full (preprocessing undercut)
        flag = "  <== beats full" if c["delta_f1"] > 0 and c["sigma"] >= 1.5 else ""
        print(f"[{panel}] {name:14s} F1*={c['b_f1']:.4f}  Δ(vs full)={c['delta_f1']:+.4f} ({c['sigma']:+.2f}σ){flag}", flush=True)
        rows.append({"panel": panel, "config": name, "f1": c["b_f1"], "delta": c["delta_f1"], "sigma": c["sigma"]})
    return rows


if __name__ == "__main__":
    panels = sys.argv[1:] or ["cftr", "pah", "cancer", "general"]
    print("=== Preprocessing ablation — config vs FULL deployed (positive Δ = preprocessing undercuts) ===", flush=True)
    allrows = []
    for pn in panels:
        allrows += run(pn, 4 if pn == "general" else 8)
    pd.DataFrame(allrows).to_csv(os.path.join(OUT, "preprocess_ablation.csv"), index=False)
    print(f"\nsaved -> preprocess_ablation.csv", flush=True)
