"""Validate the deployed-behavior audit fixes before keeping them.

  #3  fast_mrmr constant pre-drop  -> selected set must now MATCH the reference
      select_features_mrmr (the validated-deployed intent) on cancer/pah/cftr.
  #4  base-rate-aware leak rule    -> leaky set must be UNCHANGED on general/cancer
      (the real AF leak is preserved); cftr-own may drop fewer (dormant, w=1.0).
  #6  nested early-stopping OOF     -> paired honest CV (general + cancer own model),
      OLD (ES on the OOF fold = mild optimism) vs NEW (ES nested in train fold).
      Bar: NEW must NOT regress F1* at the test prior beyond noise.

Run: .venv/bin/python -u analysis/exp_audit_fix_validate.py
"""
import sys, warnings
sys.path.insert(0, "analysis"); sys.path.insert(0, "pipelines")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

from config import SEED
from schema import normalize_sentinels, partition_columns
from utils import detect_leaky_columns, KNOWN_LEAKY, TEST_PRIORS
from recipe import PANEL_RECIPE
from predict import _engineer, _fit_own_preprocess, _lgbm_redun_keep, REDUN_PRUNE_LGBM
from preprocess_v2 import fast_mrmr
from run_pipeline import select_features_mrmr, get_model_params, preprocess_data
from harness import f1_at_prior, tpr_fpr
from schema import drop_all_nan_rows
from utils import PANELS


def _load(panel):
    df = pd.read_csv(PANELS[panel]); df = normalize_sentinels(df); df = drop_all_nan_rows(df)
    cfg = PANEL_RECIPE[panel]
    df_eng = _engineer(df.copy(), cfg)
    _, num_cols, cat_cols, tgt = partition_columns(df_eng)
    y = df_eng[tgt].reset_index(drop=True)
    X = df_eng[num_cols + cat_cols].reset_index(drop=True)
    leaky = detect_leaky_columns(X[num_cols], y, auc_threshold=0.99, verbose=False)
    X = X.drop(columns=sorted(set(leaky) | (KNOWN_LEAKY & set(X.columns))))
    return df, X, y, cfg


# ---------- #3 fast_mrmr equivalence to the reference ----------
def check_mrmr():
    print("\n=== #3  fast_mrmr selected-set vs reference select_features_mrmr ===")
    for panel in ("cancer", "pah", "cftr"):
        _, X, y, cfg = _load(panel)
        if cfg["feature_select"] != "mrmr":
            print(f"  {panel}: no mrmr in recipe, skip"); continue
        Xpp, _, _, _ = preprocess_data(X.copy(), X.copy(), X.copy(), 0.95)
        new_sel = list(fast_mrmr(Xpp, y, Xpp, Xpp)[0].columns)
        ref_sel = list(select_features_mrmr(Xpp, y, Xpp, Xpp)[0].columns)
        set_match = set(new_sel) == set(ref_sel)
        order_match = new_sel == ref_sel
        extra = set(new_sel) - set(ref_sel)
        print(f"  {panel:7s} n={len(Xpp.columns)} -> sel {len(new_sel)} | "
              f"set_match={set_match} order_match={order_match} "
              f"fast-only={sorted(extra) if extra else '∅'}")


# ---------- #4 leak-rule neutrality ----------
def _old_missingness_leaky(X, y, min_count=10):
    """The OLD base-rate-naive rule: any column whose >min_count missing rows are all one class."""
    out = []
    for col in X.columns:
        vals = X[col]; nm = int(vals.isna().sum())
        if nm > min_count and y[vals.isna()].nunique() == 1:
            out.append(col)
    return set(out)


def _new_missingness_leaky(X, y, min_count=10):
    """The NEW base-rate-aware rule (binomial chance < 1e-3)."""
    out = []
    for col in X.columns:
        vals = X[col]; nm = int(vals.isna().sum())
        if nm > min_count:
            ym = y[vals.isna()]
            if ym.nunique() == 1:
                cls = int(ym.iloc[0]); base = float((y == cls).mean())
                if base ** nm < 1e-3:
                    out.append(col)
    return set(out)


def check_leak():
    print("\n=== #4  missingness leak rule: OLD (nunique==1) vs NEW (binomial<1e-3) ===")
    for panel in ("general", "cancer", "cftr"):
        _, X, y, _ = _load(panel)
        _, num_cols, _, _ = partition_columns(pd.concat([X, y.rename("__y")], axis=1)) \
            if False else (None, list(X.columns), None, None)
        old = _old_missingness_leaky(X, y); new = _new_missingness_leaky(X, y)
        dropped_old_only = old - new
        print(f"  {panel:7s} prior={TEST_PRIORS[panel]:.3f} | OLD flags {len(old)} | "
              f"NEW flags {len(new)} | OLD-only (now kept) {len(dropped_old_only)}"
              f"{'  <-- DEPLOYED set CHANGED' if (panel in ('general','cancer') and dropped_old_only) else ''}")


# ---------- #6 nested early-stopping honesty ----------
def _oof(X, y, params, n_folds, seed_i, nested):
    patience = 50 if len(y) > 300 else 30
    p = {**params, "random_state": seed_i}
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed_i)
    oof = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        if nested:
            in_tr, in_es = train_test_split(tr, test_size=0.15, random_state=seed_i,
                                            stratify=y.iloc[tr])
        else:
            in_tr, in_es = tr, va
        m = lgb.LGBMClassifier(**p)
        m.fit(X.iloc[in_tr], y.iloc[in_tr], eval_set=[(X.iloc[in_es], y.iloc[in_es])],
              callbacks=[lgb.early_stopping(patience, verbose=False), lgb.log_evaluation(0)])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return oof


def _f1star(y, oof, prior):
    qs = np.quantile(oof, np.arange(.02, .98, .01))
    return max(f1_at_prior(*tpr_fpr(y.values, (oof >= q).astype(int)), prior) for q in qs)


def check_es():
    print("\n=== #6  LGBM OOF: OLD (ES on OOF fold) vs NEW (nested ES) — 5 seeds, paired ===")
    for panel in ("general", "cancer"):
        _, X, y, cfg = _load(panel)
        Xpp, _ = _fit_own_preprocess(X, y, cfg)
        if panel in REDUN_PRUNE_LGBM:
            Xpp = Xpp[_lgbm_redun_keep(Xpp)]
        spw = (y == 0).sum() / max((y == 1).sum(), 1)
        params = get_model_params(len(Xpp), spw)
        n_folds = 3 if len(Xpp) < 200 else 5
        prior = TEST_PRIORS[panel]
        oa, na, of, nf = [], [], [], []
        for s in range(5):
            o_old = _oof(Xpp, y, params, n_folds, SEED + s, nested=False)
            o_new = _oof(Xpp, y, params, n_folds, SEED + s, nested=True)
            oa.append(roc_auc_score(y, o_old)); na.append(roc_auc_score(y, o_new))
            of.append(_f1star(y, o_old, prior)); nf.append(_f1star(y, o_new, prior))
        oa, na, of, nf = map(np.array, (oa, na, of, nf))
        dA, dF = na - oa, nf - of
        seA = dA.std(ddof=1) / np.sqrt(5); seF = dF.std(ddof=1) / np.sqrt(5)
        print(f"  {panel:7s} OOF-AUC {oa.mean():.4f}->{na.mean():.4f} Δ={dA.mean():+.4f} "
              f"({dA.mean()/seA if seA>0 else 0:+.2f}σ)  |  "
              f"F1*@{prior:.3f} {of.mean():.4f}->{nf.mean():.4f} Δ={dF.mean():+.4f} "
              f"({dF.mean()/seF if seF>0 else 0:+.2f}σ)")
    print("  (Δ<0 expected for AUC = less optimism; bar = F1* not worse beyond ~-2σ)")


if __name__ == "__main__":
    check_mrmr()
    check_leak()
    check_es()
    print("\nDONE")
