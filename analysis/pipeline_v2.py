"""Unified, recipe-driven pipeline function — the basis for the reworked production
pipeline. One make_fn(panel, recipe) used by experiments AND (after validation) the
integrated pipelines/ code, so train/inference can't drift.

Recipe keys (all optional, defaults = deployed behaviour):
  impute:         "median" (deployed) | "none"  (native LightGBM NaN; mRMR still
                  needs complete data so it imputes a private copy for MI only)
  feature_select: "mrmr" (deployed) | "none" (use all post-preprocess features)
  al_features:    list subset of {"fam_flags","al_aggs"} force-added after preprocess
  score_agree:    bool (deployed: cancer/cftr True)
  interactions:   bool (deployed: cancer/cftr True) — applied at engineering time,
                  so handled by the caller's load; here only score_agree/al/fs/impute
  model_params:   dict overriding get_model_params
  n_seeds:        ensemble seeds (screening 1, deploy 5)

Everything stateful is fit on the train fold only and applied to the test fold.
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))

from harness import (lgb_oof_and_test, _drop_leaky, fast_mrmr)
from feature_engineering import fit_score_agreement, apply_score_agreement
from calibration import fit_platt, apply_platt
from run_pipeline import preprocess_data, get_model_params
from schema import (partition_columns, classify_columns, CAT_POPULATION_FREQ,
                    CAT_CONSERVATION, CAT_SCORE_CENTERED, CAT_CONSTANT, CAT_SPARSE)


# ── native-NaN preprocessing (deployed transforms, no median imputation) ──────

def fit_keep_nan(Xtr):
    """Fit native-NaN preprocessing on the train frame; return (Xtr_pp, fitted).
    Transforms: drop constants/sparse, population rank, conservation clip, missing-
    indicators, low-var + NaN-aware corr drop. Numeric NaN preserved (LightGBM native).
    `fitted` holds everything needed to replay on any future batch (apply_keep_nan).
    """
    _, num_cols, cat_cols, _ = partition_columns(Xtr)
    Xn = Xtr[num_cols].copy()
    cats = classify_columns(Xtr)
    drop_c = [c for c, k in cats.items() if k in (CAT_CONSTANT, CAT_SPARSE)]
    Xn.drop(columns=drop_c, inplace=True, errors="ignore")
    pop = [c for c in Xn.columns if cats.get(c) == CAT_POPULATION_FREQ]
    cons = [c for c in Xn.columns if cats.get(c) in (CAT_CONSERVATION, CAT_SCORE_CENTERED)]
    pop_grids, cons_clip = {}, {}
    for c in pop:
        nn = Xn[c].dropna().sort_values().values
        if len(nn) == 0:
            continue
        rk = np.linspace(1 / len(nn), 1.0, len(nn))
        pop_grids[c] = (nn, rk)
        Xn[c] = Xn[c].rank(pct=True, na_option="keep")
    for c in cons:
        s = Xn[c].dropna()
        if len(s) < 10:
            continue
        lo, hi = float(s.quantile(.01)), float(s.quantile(.99))
        cons_clip[c] = (lo, hi); Xn[c] = Xn[c].clip(lo, hi)
    miss = Xn.isna().mean()
    miss_cols = miss[(miss > .05) & (miss < .95)].index.tolist()
    for c in miss_cols:
        Xn[f"{c}_missing"] = Xn[c].isna().astype(int)
    lowvar = Xn.var()[lambda v: v < 1e-6].index.tolist()
    Xn.drop(columns=lowvar, inplace=True, errors="ignore")
    raw = Xtr[[c for c in Xn.columns if c in Xtr.columns]]
    dc = []
    if raw.shape[1] > 50:
        corr = raw.corr(min_periods=30).abs()
        up = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))
        dc = [c for c in up.columns if (up[c] > 0.95).any()]
        Xn.drop(columns=dc, inplace=True, errors="ignore")
    num_kept = list(Xn.columns)
    if cat_cols:
        Xn = pd.concat([Xn, Xtr[cat_cols].reset_index(drop=True)], axis=1)
    fitted = {"drop_c": drop_c, "pop_grids": pop_grids, "cons_clip": cons_clip,
              "miss_cols": miss_cols, "num_kept": num_kept, "cat_cols": list(cat_cols)}
    return Xn, fitted


def apply_keep_nan(X, fitted):
    """Replay fit_keep_nan transforms on a new batch — pure, batch-invariant."""
    Xn = X[[c for c in X.columns if c not in fitted["cat_cols"]]].copy()
    Xn.drop(columns=fitted["drop_c"], inplace=True, errors="ignore")
    for c, (nn, rk) in fitted["pop_grids"].items():
        if c not in Xn.columns:
            continue
        v = Xn[c].values.astype(float); m = ~np.isnan(v); out = np.full_like(v, np.nan)
        if m.any() and len(nn):
            out[m] = np.interp(v[m], nn, rk, left=rk[0], right=rk[-1])
        Xn[c] = out
    for c, (lo, hi) in fitted["cons_clip"].items():
        if c in Xn.columns:
            Xn[c] = Xn[c].clip(lo, hi)
    for c in fitted["miss_cols"]:
        if c in Xn.columns:
            Xn[f"{c}_missing"] = Xn[c].isna().astype(int)
    for c in fitted["num_kept"]:
        if c not in Xn.columns:
            Xn[c] = np.nan
    out = Xn[fitted["num_kept"]]
    if fitted["cat_cols"]:
        out = pd.concat([out.reset_index(drop=True),
                         X[fitted["cat_cols"]].reset_index(drop=True)], axis=1)
    return out


def preprocess_keep_nan(Xtr, Xte):
    """Convenience: fit on Xtr, apply to Xte (CV use). Returns (Xtr_pp, Xte_pp)."""
    Xtr_pp, fitted = fit_keep_nan(Xtr)
    return Xtr_pp, apply_keep_nan(Xte, fitted)


# ── AL family / aggregate features (fit on train) ────────────────────────────

def fit_al(Xtr, kinds):
    al = [c for c in Xtr.columns if c.startswith("AL_")]
    fitted = {"al_cols": al, "kinds": list(kinds)}
    if "fam_flags" in kinds:
        mc = Xtr[al].isna().sum(); fams = {}
        for c in al:
            fams.setdefault(int(mc[c]), []).append(c)
        fitted["families"] = fams
    if "al_aggs" in kinds:
        M = Xtr[al]
        fitted["af_cols"] = [c for c in al if ((M[c] > 0) & (M[c] < 1)).any()
                             and M[c].mean(skipna=True) < 0.1]
    return fitted


def apply_al(X, fitted):
    al = fitted["al_cols"]; kinds = fitted["kinds"]; out = {}
    if "fam_flags" in kinds:
        flags = {}
        for fid, cols in fitted["families"].items():
            cols = [c for c in cols if c in X.columns]
            flags[f"ALfam_{fid}"] = X[cols].notna().any(axis=1).astype(int) if cols else 0
        out.update(flags)
        out["ALn_fams"] = pd.DataFrame(flags, index=X.index).sum(axis=1)
    if "al_aggs" in kinds:
        M = X[al]
        out["ALn_present"] = M.notna().sum(axis=1)
        af = fitted.get("af_cols", [])
        if af:
            out["ALmaxaf"] = M[af].max(axis=1).fillna(0.0)
            out["ALmeanaf"] = M[af].mean(axis=1).fillna(0.0)
    return pd.DataFrame(out, index=X.index)


# ── unified pipeline_fn ───────────────────────────────────────────────────────

def make_fn(panel, recipe=None, n_seeds=1):
    r = dict(impute="median", feature_select="mrmr", al_features=(),
             score_agree=(panel in ("cancer", "cftr")), model_params=None)
    r.update(recipe or {})

    def _fn(X_tr, y_tr, X_te):
        Xtr, Xte = _drop_leaky(X_tr, y_tr, X_te)
        alf = fit_al(Xtr, r["al_features"]) if r["al_features"] else None
        ALtr = apply_al(Xtr, alf) if alf else None
        ALte = apply_al(Xte, alf) if alf else None

        if r["impute"] == "none":
            Xtr_pp, Xte_pp = preprocess_keep_nan(Xtr, Xte)
        else:
            Xtr_pp, _, Xte_pp, _ = preprocess_data(Xtr.copy(), Xtr.copy(), Xte.copy(), 0.95)

        if r["score_agree"]:
            saf = fit_score_agreement(Xtr_pp, y_tr, top_k=7)
            Xtr_pp = apply_score_agreement(Xtr_pp, saf); Xte_pp = apply_score_agreement(Xte_pp, saf)

        if r["feature_select"] == "mrmr":
            Xtr_pp, _, Xte_pp, _ = fast_mrmr(Xtr_pp, y_tr, Xte_pp, Xte_pp)

        if alf is not None:
            Xtr_pp = pd.concat([Xtr_pp.reset_index(drop=True), ALtr.reset_index(drop=True)], axis=1)
            Xte_pp = pd.concat([Xte_pp.reset_index(drop=True), ALte.reset_index(drop=True)], axis=1)
            Xtr_pp = Xtr_pp.loc[:, ~Xtr_pp.columns.duplicated()]
            Xte_pp = Xte_pp.loc[:, ~Xte_pp.columns.duplicated()]

        spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        params = get_model_params(len(X_tr), spw)
        if r["model_params"]:
            params.update(r["model_params"])
        nf = 3 if len(X_tr) < 200 else 5
        tp, oof = lgb_oof_and_test(Xtr_pp, y_tr, Xte_pp, params, n_seeds=n_seeds, n_folds=nf)
        pf = fit_platt(oof, y_tr.values)
        return apply_platt(tp, pf), apply_platt(oof, pf)
    return _fn
