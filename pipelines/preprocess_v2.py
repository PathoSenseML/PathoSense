"""Native-NaN preprocessing + AL_ family features — the validated reworked
preprocessing (2026-06-10). Single source of truth: imported by run_pipeline,
predict, and transfer so the train / inference / projection paths cannot drift.

Validated wins over the deployed median-impute + mRMR chain (honest per-fold CV,
analytic F1* at the test prior, paired vs deployed on GENERAL):
  * native NaN (no median imputation; LightGBM splits on missing natively)  +~0.02
  * all features (no mRMR feature selection; the 334-col AL_ block is informative) +~0.03-0.04
  * fam_flags (17 AL_ missingness-cohort presence flags) recover rarity signal   +~0.04
The deployed mRMR-60 over-pruned the AL_ block; native NaN preserves the
missingness signal that median-fill destroyed.

Every transform is fit on the train frame only (fit_keep_nan / fit_al) and replayed
purely on any future batch (apply_keep_nan / apply_al) — batch-invariant by
construction (no statistic is computed from the inference set).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from config import SEED
from schema import (partition_columns, classify_columns, CAT_POPULATION_FREQ,
                    CAT_CONSERVATION, CAT_SCORE_CENTERED, CAT_CONSTANT, CAT_SPARSE)


# ── fast mRMR (output-identical to run_pipeline.select_features_mrmr, 37-83x faster) ──

def fast_mrmr(X_train, y_train, X_val, X_test, max_features=0):
    """Minimum-redundancy-maximum-relevance selection. Precomputes the |corr| matrix
    once (vs the deployed version recomputing pairwise corr inside the greedy loop),
    so it is 37-83x faster while selecting the IDENTICAL feature set. NaN in the
    encoded matrix is imputed (column median) for the MI/corr math only — the returned
    columns preserve their original NaN."""
    from sklearn.feature_selection import mutual_info_classif
    n = len(X_train)
    if max_features <= 0:
        max_features = 20 if n < 200 else (40 if n < 500 else 60)
        max_features = min(max_features, len(X_train.columns))
    n_before = len(X_train.columns)
    if n_before <= max_features:
        return X_train, X_val, X_test, {"method": "none_needed",
                                        "n_before": n_before, "n_after": n_before}
    cols = list(X_train.columns)
    enc = np.empty((n, n_before), dtype=float)
    discrete = []
    for j, c in enumerate(cols):
        s = X_train[c]
        if isinstance(s.dtype, pd.CategoricalDtype):
            enc[:, j] = s.cat.codes.to_numpy(); discrete.append(True)
        else:
            enc[:, j] = s.to_numpy(dtype=float); discrete.append(False)
    if not np.isfinite(enc).all():
        col_med = np.nanmedian(np.where(np.isfinite(enc), enc, np.nan), axis=0)
        col_med = np.where(np.isfinite(col_med), col_med, 0.0)
        bad = ~np.isfinite(enc); enc[bad] = np.take(col_med, np.where(bad)[1])
    mi = mutual_info_classif(enc, y_train, discrete_features=discrete,
                             random_state=SEED, n_neighbors=max(3, min(5, n // 30)))
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.abs(np.corrcoef(enc, rowvar=False))
    C[np.isnan(C)] = 0.0
    selected = [int(np.argmax(mi))]
    cand = [j for j in range(n_before) if j != selected[0]]
    while len(selected) < max_features and cand:
        red = C[np.ix_(cand, selected)].mean(axis=1)
        best = cand[int(np.argmax(mi[cand] - red))]
        selected.append(best); cand.remove(best)
    names = [cols[j] for j in selected]
    info = {"method": "mRMR_fast", "n_before": n_before, "n_after": len(names),
            "top_5": {cols[j]: round(float(mi[j]), 4) for j in selected[:5]}}
    return X_train[names], X_val[names], X_test[names], info


# ── native-NaN per-category preprocessing ────────────────────────────────────

def fit_keep_nan(Xtr: pd.DataFrame):
    """Fit native-NaN preprocessing; return (Xtr_pp, fitted).
    Drop constants/sparse → population rank → conservation clip (p1/p99) →
    missing-indicators (5-95% NaN) → low-var drop → NaN-aware corr>0.95 drop.
    Numeric NaN preserved for LightGBM's native handling. Categoricals pass through.
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
        # reset BOTH blocks so a non-contiguous input index can't misalign the concat
        # (apply_keep_nan already resets both; production passes reset-index data, so
        # this is identical there — purely a robustness fix for arbitrary inputs).
        Xn = pd.concat([Xn.reset_index(drop=True),
                        Xtr[cat_cols].reset_index(drop=True)], axis=1)
    fitted = {"drop_c": drop_c, "pop_grids": pop_grids, "cons_clip": cons_clip,
              "miss_cols": miss_cols, "num_kept": num_kept, "cat_cols": list(cat_cols)}
    return Xn, fitted


def apply_keep_nan(X: pd.DataFrame, fitted: dict) -> pd.DataFrame:
    """Replay fit_keep_nan on a new batch — pure, batch-invariant."""
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


# ── AL_ missingness-family presence features ─────────────────────────────────

def fit_al(Xtr: pd.DataFrame, kinds=("fam_flags",)) -> dict:
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


def apply_al(X: pd.DataFrame, fitted: dict) -> pd.DataFrame:
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
