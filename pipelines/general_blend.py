"""Heterogeneous blend for the GENERAL panel: average the LightGBM ensemble's
calibrated probs with a RandomForest and a CatBoost (different families → decorrelated
errors). Validated on the NEW all-features recipe (honest per-fold CV, analytic F1* at
the test prior): general 0.5195 → 0.5302 (+0.0107, 1.92σ). general has DOUBLE value
(panel model + transfer source), so a win here is worth most.

All components train on the SAME native-NaN feature matrix the general own-model uses.
Every fitted object (RF, CatBoost, ordinal encoder, both Platt calibrators) is frozen
at train time and replayed at inference — pure / batch-invariant (RF & CatBoost are
deterministic given fixed inputs; no statistic is computed from the inference set).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from catboost import CatBoostClassifier
import lightgbm as lgb

from config import SEED
from calibration import fit_platt, apply_platt


def _cats(X):
    return [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]


def _rf_repr(X, cat_feats, enc=None, fit=False):
    Xr = X.copy()
    if cat_feats:
        s = X[cat_feats].astype(object).where(X[cat_feats].notna(), "NA")
        if fit:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                 encoded_missing_value=-2)
            Xr[cat_feats] = enc.fit_transform(s)
        else:
            Xr[cat_feats] = enc.transform(s)
    return Xr.fillna(-999.0), enc


def _cat_repr(X, cat_feats):
    Xc = X.copy()
    for c in cat_feats:
        Xc[c] = X[c].astype(object).where(X[c].notna(), "NA").astype(str)
    return Xc


def fit_general_blend(X_pp, y, lgb_oof_cal, spw, n_splits=5):
    """Fit RF + CatBoost on X_pp (the general own-model native-NaN matrix), produce
    inner-OOF calibrated probs, and return frozen artifacts + the blended OOF (avg of
    the LGBM-ensemble calibrated OOF and the RF/CatBoost calibrated OOF). Caller uses
    the blended OOF to pick the deployed threshold."""
    cat_feats = _cats(X_pp)
    X_ord, enc = _rf_repr(X_pp, cat_feats, fit=True)
    X_cat = _cat_repr(X_pp, cat_feats)
    cat_idx = [X_cat.columns.get_loc(c) for c in cat_feats]

    skf = StratifiedKFold(n_splits, shuffle=True, random_state=SEED)
    rf_oof = np.zeros(len(y)); cat_oof = np.zeros(len(y))
    for k, (a, b) in enumerate(skf.split(X_pp, y)):
        rf = RandomForestClassifier(n_estimators=600, max_features="sqrt", n_jobs=4,
                                    class_weight="balanced", random_state=SEED)
        rf.fit(X_ord.iloc[a], y.iloc[a]); rf_oof[b] = rf.predict_proba(X_ord.iloc[b])[:, 1]
        i2, ies = train_test_split(a, test_size=0.15, random_state=SEED + k, stratify=y.iloc[a])
        cb = _cat_model(cat_idx, spw)
        cb.fit(X_cat.iloc[i2], y.iloc[i2], eval_set=(X_cat.iloc[ies], y.iloc[ies]))
        cat_oof[b] = cb.predict_proba(X_cat.iloc[b])[:, 1]
    rf_platt = fit_platt(rf_oof, y.values); cat_platt = fit_platt(cat_oof, y.values)
    blend_oof = (np.asarray(lgb_oof_cal) + apply_platt(rf_oof, rf_platt)
                 + apply_platt(cat_oof, cat_platt)) / 3.0

    rf_final = RandomForestClassifier(n_estimators=600, max_features="sqrt", n_jobs=4,
                                      class_weight="balanced", random_state=SEED).fit(X_ord, y)
    i2, ies = train_test_split(np.arange(len(y)), test_size=0.15, random_state=SEED + 99, stratify=y)
    cat_final = _cat_model(cat_idx, spw)
    cat_final.fit(X_cat.iloc[i2], y.iloc[i2], eval_set=(X_cat.iloc[ies], y.iloc[ies]))
    art = {"rf_model": rf_final, "cat_model": cat_final, "encoder": enc,
           "cat_feats": cat_feats, "cat_idx": cat_idx,
           "rf_platt": rf_platt, "cat_platt": cat_platt}
    return art, blend_oof


def _cat_model(cat_idx, spw):
    return CatBoostClassifier(iterations=1000, depth=6, learning_rate=0.05,
                              scale_pos_weight=spw, cat_features=cat_idx, random_seed=SEED,
                              verbose=0, allow_writing_files=False, early_stopping_rounds=40,
                              thread_count=4)


def general_blend_probs(X_pp, lgb_probs_cal, art):
    """Inference: average the LGBM-ensemble calibrated probs with the frozen RF and
    CatBoost calibrated probs. X_pp is the general own-model native-NaN matrix."""
    X_ord, _ = _rf_repr(X_pp, art["cat_feats"], enc=art["encoder"], fit=False)
    rf_p = apply_platt(art["rf_model"].predict_proba(X_ord)[:, 1], art["rf_platt"])
    X_cat = _cat_repr(X_pp, art["cat_feats"])
    cat_p = apply_platt(art["cat_model"].predict_proba(X_cat)[:, 1], art["cat_platt"])
    return (np.asarray(lgb_probs_cal) + rf_p + cat_p) / 3.0
