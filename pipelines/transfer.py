"""Cross-panel transfer scorer — the rework's biggest win (2026-06-10).

A model trained on the large GENERAL panel generalizes to the small panels far
better than their own data-starved models. Honest, leakage-corrected CV (exclude
ALL target Variant_IDs from the general scorer — finals target variants are novel):
  cftr   general scorer AUC 0.91 vs own 0.77  -> pure general (w=1.0) F1* 0.52->0.72 (+0.21, 3.8σ)
  cancer general scorer AUC 0.93 vs own 0.91  -> pure general (w=1.0) F1* 0.66->0.71 (+0.046, 2.4σ)
  pah    general scorer AUC 0.74 ≈ own 0.74   -> equal blend (w=0.5) F1* 0.55->0.60 (+0.044, 2.6σ)
(63-69% of small-panel Variant_IDs are shared with general; the prior feature-vector
exclusion missed them and leaked — Variant_ID exclusion is the correct control.)

This module trains the transfer SCORER (fit_transfer_scorer) and projects target
variants into its feature space for scoring (transfer_score). predict.py fits the
Platt calibration + blend weight + threshold on the target's own training data.

Leakage / purity: the scorer excludes the target panel's training Variant_IDs, so the
Platt calibration that predict.py fits on target-train probabilities sees OUT-OF-SAMPLE
general scores — matching the finals condition (novel target variants). Projection is a
pure replay of frozen transforms (preprocess_v2), so it is batch-invariant.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from config import SEED
from schema import (normalize_sentinels, drop_all_nan_rows, to_categorical,
                    partition_columns)
from feature_engineering import engineer_aa_features
from utils import detect_leaky_columns, KNOWN_LEAKY, PANELS
from run_pipeline import get_model_params
from preprocess_v2 import fit_keep_nan, apply_keep_nan, fit_al, apply_al
from general_blend import _rf_repr, _cat_repr, _cat_model

# Per-panel general-scorer blend weight w (general share); 1-w is the panel-own share.
# From analysis/exp_transfer*.py honest leakage-corrected CV (improved scorer, 30 folds).
#   cftr  1.0  blended scorer, pure transfer -> F1* 0.529->0.836
#   cancer 0.5 blended scorer                -> F1* 0.665->0.708
#   pah   1.0  single-LGBM scorer            -> F1* 0.549->0.660
TRANSFER_PANELS = {"cftr": 1.0, "cancer": 0.5, "pah": 1.0}

# Panels whose transfer scorer is a heterogeneous BLEND (LGBM+RF+CatBoost, prob-avg).
# Validated to beat the single-LGBM transfer scorer (analysis/exp_transfer_blend_validate.py):
# cftr +0.039, cancer +0.012. pah is EXCLUDED — the blend hurts it (-0.012), so pah keeps the
# single LightGBM transfer scorer.
TRANSFER_BLEND = {"cftr", "cancer"}


def _load_raw(panel):
    df = pd.read_csv(PANELS[panel])
    df = normalize_sentinels(df)
    df = drop_all_nan_rows(df, verbose=False)
    return df.reset_index(drop=True)


def fit_transfer_scorer(target_panel, sources=("general",), al_features=()):
    """Train a general(-pooled) LightGBM scorer on `sources` MINUS all target
    Variant_IDs, using the improved recipe (native-NaN preprocessing + ALL features).
    al_features is empty by default: with all features kept, the AL_ fam_flags are
    redundant (validated — they slightly hurt vs noFS alone). Returns a frozen
    artifact for pure projection at inference."""
    tids = set(_load_raw(target_panel)["Variant_ID"])
    dg = pd.concat([_load_raw(s)[lambda d: ~d["Variant_ID"].isin(tids)]
                    for s in sources], ignore_index=True).reset_index(drop=True)
    dg = to_categorical(dg); dg = engineer_aa_features(dg)
    _, n, c, t = partition_columns(dg)
    yg = dg[t].reset_index(drop=True); Xg = dg[n + c].reset_index(drop=True)
    leaky = sorted(set(detect_leaky_columns(Xg, yg, 0.99, verbose=False))
                   | (KNOWN_LEAKY & set(Xg.columns)))
    Xg = Xg.drop(columns=leaky, errors="ignore")
    raw_cols = list(Xg.columns)

    alf = fit_al(Xg, al_features) if al_features else None
    Xg_pp, ppf = fit_keep_nan(Xg)
    Xg_sel = Xg_pp
    if alf is not None:
        Xg_sel = pd.concat([Xg_pp.reset_index(drop=True),
                            apply_al(Xg, alf).reset_index(drop=True)], axis=1)
    Xg_sel = Xg_sel.loc[:, ~Xg_sel.columns.duplicated()]
    feats = list(Xg_sel.columns)
    cat_feats = [c for c in feats if isinstance(Xg_sel[c].dtype, pd.CategoricalDtype)]
    cat_cats = {c: list(Xg_sel[c].cat.categories) for c in cat_feats}

    spw = (yg == 0).sum() / max((yg == 1).sum(), 1)
    params = get_model_params(len(Xg_sel), spw)
    i2, ies = train_test_split(np.arange(len(yg)), test_size=0.15,
                               random_state=SEED + 7, stratify=yg)
    m = LGBMClassifier(**params, random_state=SEED)
    m.fit(Xg_sel.iloc[i2], yg.iloc[i2], eval_set=[(Xg_sel.iloc[ies], yg.iloc[ies])],
          callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
          categorical_feature=cat_feats or "auto")

    # heterogeneous blend (LGBM + RF + CatBoost, prob-averaged) for flagged panels
    blend = None
    if target_panel in TRANSFER_BLEND:
        spw = (yg == 0).sum() / max((yg == 1).sum(), 1)
        X_ord, enc = _rf_repr(Xg_sel, cat_feats, fit=True)
        rf = RandomForestClassifier(n_estimators=600, max_features="sqrt", n_jobs=4,
                                    class_weight="balanced", random_state=SEED).fit(X_ord, yg)
        X_cat = _cat_repr(Xg_sel, cat_feats)
        cat_idx = [X_cat.columns.get_loc(c) for c in cat_feats]
        i2b, iesb = train_test_split(np.arange(len(yg)), test_size=0.15,
                                     random_state=SEED + 5, stratify=yg)
        cb = _cat_model(cat_idx, spw)
        cb.fit(X_cat.iloc[i2b], yg.iloc[i2b], eval_set=(X_cat.iloc[iesb], yg.iloc[iesb]))
        blend = {"rf_model": rf, "encoder": enc, "cat_model": cb, "cat_feats": cat_feats}

    return {"model": m, "raw_cols": raw_cols, "ppf": ppf, "alf": alf, "feats": feats,
            "cat_cats": cat_cats, "blend": blend, "sources": list(sources),
            "n_excluded": len(tids), "n_train": len(yg)}


def transfer_score(df_raw: pd.DataFrame, art: dict) -> np.ndarray:
    """Project raw target variants into the general scorer's space and return raw
    general probabilities. df_raw is a panel-style frame (post sentinel-normalise;
    Variant_ID/Label columns are ignored). Pure replay — batch-invariant."""
    df = to_categorical(df_raw.copy())
    df = engineer_aa_features(df)
    X = df.reindex(columns=art["raw_cols"])
    Xs = apply_keep_nan(X, art["ppf"])
    if art["alf"] is not None:
        Xs = pd.concat([Xs.reset_index(drop=True),
                        apply_al(X, art["alf"]).reset_index(drop=True)], axis=1)
        Xs = Xs.loc[:, ~Xs.columns.duplicated()]
    for col in art["feats"]:
        if col not in Xs.columns:
            Xs[col] = np.nan
    Xs = Xs[art["feats"]]
    for col, cats in art["cat_cats"].items():
        Xs[col] = pd.Categorical(Xs[col].astype(object), categories=cats)
    lgbm_p = art["model"].predict_proba(Xs)[:, 1]
    bl = art.get("blend")
    if bl is None:
        return lgbm_p
    X_ord, _ = _rf_repr(Xs, bl["cat_feats"], enc=bl["encoder"], fit=False)
    rf_p = bl["rf_model"].predict_proba(X_ord)[:, 1]
    X_cat = _cat_repr(Xs, bl["cat_feats"])
    cat_p = bl["cat_model"].predict_proba(X_cat)[:, 1]
    return (lgbm_p + rf_p + cat_p) / 3.0          # probability-average (shape-preserving)
