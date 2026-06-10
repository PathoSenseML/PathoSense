"""Preprocessing reconciliation — deployed pipeline vs teammate-report choices.

Each axis is tested as a PAIRED config-vs-deployed comparison on identical repeated
K-folds, scored by analytic F1 at the per-panel test prior (harness.py). We keep
whatever clears ~1.5 sigma and stack wins.

Divergent decisions under test:
  A. Imputation: median+indicator (deployed) vs NO impute (native LGBM NaN) vs
     teammate empty-cols->0.0.
  B. Feature selection: mRMR (deployed) vs none (all features) vs mRMR-80/120.
  C. Categorical encoding: native pd.Categorical (deployed) vs ordinal label-encode.
  D. Class balancing: scale_pos_weight (deployed) vs oversample-to-50/50 (teammate)
     vs is_unbalance vs none.
"""
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))

from harness import (cv_evaluate, load_panel, compare, deployed_pipeline_fn,
                     lgb_oof_and_test, _drop_leaky, fast_mrmr)
from feature_engineering import fit_score_agreement, apply_score_agreement
from calibration import fit_platt, apply_platt
from config import SEED
from utils import TEST_PRIORS
from run_pipeline import preprocess_data, get_model_params

PANELS = ["general", "cancer", "pah", "cftr"]
# Screening settings: fewer repeats + 1 seed (paired comparison cancels seed-count).
REPEATS = {"general": 4, "cancer": 6, "pah": 6, "cftr": 8}
SCREEN_SEEDS = 1
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def _sa_for(panel):
    return panel in ("cancer", "cftr")


def make_fn(panel, *, impute="median", feat_select="mrmr", mrmr_n=0,
            cat_encode="native", balance="spw", platt=True, n_seeds=SCREEN_SEEDS):
    """Build a configurable pipeline_fn. Everything stateful fits on train only."""
    sa = _sa_for(panel)

    def _fn(X_tr, y_tr, X_te):
        Xtr, Xte = _drop_leaky(X_tr, y_tr, X_te)

        # preprocessing — reuse deployed transforms, then optionally override impute
        Xtr_pp, _, Xte_pp, info = preprocess_data(Xtr.copy(), Xtr.copy(), Xte.copy(),
                                                  corr_threshold=0.95)
        if impute == "none":
            # undo median imputation: restore NaN where original was NaN.
            # preprocess_data already filled; easiest is to re-run a NaN-preserving
            # variant. Instead, we reconstruct: re-apply the same column set but keep NaN.
            Xtr_pp, Xte_pp = _reprocess_keep_nan(Xtr, Xte, y_tr, info)
        elif impute == "zero":
            # teammate-style: fill remaining NaN with 0 instead of median (already
            # median-filled, so emulate by re-deriving). Use NaN-preserving then 0-fill.
            Xtr_pp, Xte_pp = _reprocess_keep_nan(Xtr, Xte, y_tr, info)
            Xtr_pp = Xtr_pp.fillna(0.0); Xte_pp = Xte_pp.fillna(0.0)

        if sa:
            saf = fit_score_agreement(Xtr_pp, y_tr, top_k=7)
            Xtr_pp = apply_score_agreement(Xtr_pp, saf)
            Xte_pp = apply_score_agreement(Xte_pp, saf)

        if cat_encode == "ordinal":
            Xtr_pp, Xte_pp = _ordinal_encode(Xtr_pp, Xte_pp)

        if feat_select == "mrmr":
            Xtr_pp, _, Xte_pp, _ = fast_mrmr(Xtr_pp, y_tr, Xte_pp, Xte_pp,
                                             max_features=mrmr_n)
        # feat_select == "none": keep all columns

        spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        params = get_model_params(len(X_tr), spw)
        if balance == "none":
            params["scale_pos_weight"] = 1.0
        elif balance == "is_unbalance":
            params.pop("scale_pos_weight", None); params["is_unbalance"] = True
        elif balance == "oversample":
            params["scale_pos_weight"] = 1.0
            Xtr_pp, y_tr2 = _oversample_balanced(Xtr_pp, y_tr)
        if balance == "oversample":
            y_use = y_tr2
        else:
            y_use = y_tr

        nf = 3 if len(X_tr) < 200 else 5
        test_probs, oof = lgb_oof_and_test(Xtr_pp, y_use, Xte_pp, params,
                                           n_seeds=n_seeds, n_folds=nf)
        # OOF for threshold must align to the ORIGINAL y_tr; for oversample the oof
        # is on the resampled set, which biases threshold. Refit a clean OOF on the
        # original rows for thresholding when oversampling.
        if balance == "oversample":
            _, oof = lgb_oof_and_test(_reslice(Xtr_pp, len(X_tr)), y_tr, Xte_pp,
                                      {**params}, n_seeds=1, n_folds=nf)
        if platt:
            pf = fit_platt(oof, y_tr.values)
            return apply_platt(test_probs, pf), apply_platt(oof, pf)
        return test_probs, oof
    return _fn


def _reprocess_keep_nan(Xtr, Xte, y_tr, info):
    """Re-run the deployed preprocessing column ops but DO NOT median-impute.
    Mirrors preprocess_data transforms (drop constants, pop-rank, conservation clip,
    missing-indicators, low-var/corr drop) while preserving native NaN for LGBM.
    """
    from schema import (partition_columns, classify_columns, CAT_POPULATION_FREQ,
                        CAT_CONSERVATION, CAT_SCORE_CENTERED, CAT_CONSTANT, CAT_SPARSE)
    _, num_cols, cat_cols, _ = partition_columns(Xtr)
    Xn_tr = Xtr[num_cols].copy(); Xn_te = Xte[num_cols].copy()
    cats = classify_columns(Xtr)
    drop_c = [c for c, k in cats.items() if k in (CAT_CONSTANT, CAT_SPARSE)]
    Xn_tr.drop(columns=drop_c, inplace=True, errors="ignore")
    Xn_te.drop(columns=drop_c, inplace=True, errors="ignore")
    pop = [c for c in Xn_tr.columns if cats.get(c) == CAT_POPULATION_FREQ]
    cons = [c for c in Xn_tr.columns if cats.get(c) in (CAT_CONSERVATION, CAT_SCORE_CENTERED)]
    for c in pop:
        s = Xn_tr[c]; nn = s.dropna().sort_values().values
        if len(nn) == 0: continue
        rk = np.linspace(1/len(nn), 1.0, len(nn))
        Xn_tr[c] = s.rank(pct=True, na_option="keep")
        for df in (Xn_te,):
            v = df[c].values.astype(float); m = ~np.isnan(v)
            out = np.full_like(v, np.nan)
            if m.any(): out[m] = np.interp(v[m], nn, rk, left=rk[0], right=rk[-1])
            df[c] = out
    for c in cons:
        s = Xn_tr[c].dropna()
        if len(s) < 10: continue
        lo, hi = float(s.quantile(.01)), float(s.quantile(.99))
        Xn_tr[c] = Xn_tr[c].clip(lo, hi); Xn_te[c] = Xn_te[c].clip(lo, hi)
    miss = Xn_tr.isna().mean()
    for c in miss[(miss > .05) & (miss < .95)].index.tolist():
        Xn_tr[f"{c}_missing"] = Xn_tr[c].isna().astype(int)
        Xn_te[f"{c}_missing"] = Xn_te[c].isna().astype(int)
    # NO imputation. low-var drop (on non-NaN variance) + NaN-aware corr drop.
    lowvar = Xn_tr.var()[lambda v: v < 1e-6].index.tolist()
    Xn_tr.drop(columns=lowvar, inplace=True, errors="ignore")
    Xn_te.drop(columns=lowvar, inplace=True, errors="ignore")
    raw = Xtr[[c for c in Xn_tr.columns if c in Xtr.columns]]
    if raw.shape[1] > 50:
        corr = raw.corr(min_periods=30).abs()
        up = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))
        dc = [c for c in up.columns if (up[c] > 0.95).any()]
        Xn_tr.drop(columns=dc, inplace=True, errors="ignore")
        Xn_te.drop(columns=dc, inplace=True, errors="ignore")
    if cat_cols:
        Xn_tr = pd.concat([Xn_tr, Xtr[cat_cols].reset_index(drop=True)], axis=1)
        Xn_te = pd.concat([Xn_te, Xte[cat_cols].reset_index(drop=True)], axis=1)
    return Xn_tr, Xn_te


def _ordinal_encode(Xtr, Xte):
    from sklearn.preprocessing import OrdinalEncoder
    cat_cols = [c for c in Xtr.columns if isinstance(Xtr[c].dtype, pd.CategoricalDtype)]
    if not cat_cols:
        return Xtr, Xte
    Xtr = Xtr.copy(); Xte = Xte.copy()
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                         encoded_missing_value=-2)
    Xtr[cat_cols] = enc.fit_transform(Xtr[cat_cols].astype(object).where(
        Xtr[cat_cols].notna(), "NA"))
    Xte[cat_cols] = enc.transform(Xte[cat_cols].astype(object).where(
        Xte[cat_cols].notna(), "NA"))
    return Xtr, Xte


def _oversample_balanced(X, y):
    rng = np.random.RandomState(SEED)
    pos = np.where(y.values == 1)[0]; neg = np.where(y.values == 0)[0]
    n = max(len(pos), len(neg))
    pi = rng.choice(pos, n, replace=True); ni = rng.choice(neg, n, replace=True)
    idx = np.concatenate([pi, ni]); rng.shuffle(idx)
    return X.iloc[idx].reset_index(drop=True), pd.Series(y.values[idx])


def _reslice(X, n):
    return X.iloc[:n].reset_index(drop=True) if len(X) > n else X


def run():
    rows = []
    for pn in PANELS:
        df = load_panel(pn)
        rep = REPEATS[pn]
        base = cv_evaluate(pn, deployed_pipeline_fn(pn, n_seeds=SCREEN_SEEDS),
                           label="deployed", n_repeats=rep, df=df)
        print(f"\n=== {pn} (prior={TEST_PRIORS[pn]:.3f}) deployed F1*="
              f"{base.summary()['f1_mean']:.4f}±{base.summary()['f1_se']:.4f} ===")
        configs = {
            "no_impute_nativeNaN": make_fn(pn, impute="none"),
            "impute_zero(teammate)": make_fn(pn, impute="zero"),
            "no_feature_select": make_fn(pn, feat_select="none"),
            "mrmr_80": make_fn(pn, mrmr_n=80),
            "ordinal_cats(teammate)": make_fn(pn, cat_encode="ordinal"),
            "oversample_50_50(teammate)": make_fn(pn, balance="oversample"),
            "is_unbalance": make_fn(pn, balance="is_unbalance"),
            "no_balance": make_fn(pn, balance="none"),
        }
        for name, fn in configs.items():
            try:
                r = cv_evaluate(pn, fn, label=name, n_repeats=rep, df=df)
            except Exception as e:
                print(f"  {name:28s} FAILED: {type(e).__name__}: {str(e)[:80]}")
                continue
            c = compare(base, r)
            print(f"  {name:28s} F1*={c['b_f1']:.4f}  Δ={c['delta_f1']:+.4f} "
                  f"({c['sigma']:+.2f}σ)  {c['verdict']}")
            rows.append({"panel": pn, "config": name, **c})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "exp_preprocess.csv"), index=False)
    print(f"\nsaved -> {os.path.join(OUT, 'exp_preprocess.csv')}")


if __name__ == "__main__":
    run()
