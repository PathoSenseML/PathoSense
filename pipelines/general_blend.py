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
from xgboost import XGBClassifier
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


def _xgb_model():
    # validated 4th blend member (analysis/exp_general_blend_expand.py): adding XGBoost decorrelates
    # cleanly → general blend F1* +0.005 / AUC +0.001 (both seeds). ExtraTrees/HistGB rejected (hurt AUC).
    return XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=5, subsample=0.8,
                         colsample_bytree=0.8, reg_lambda=1.5, eval_metric="logloss",
                         n_jobs=4, random_state=SEED)


ADD_TABPFN_MEMBER = False   # reverted 2026-06-27 — TabPFN adds AUC but no robust deployed F1 (see below)


def fit_general_blend(X_pp, y, lgb_oof_cal, spw, n_splits=5):
    """Fit RF + CatBoost + XGBoost on X_pp (the general own-model native-NaN matrix), produce
    inner-OOF calibrated probs, and return frozen artifacts + the blended OOF (avg of the LGBM-
    ensemble calibrated OOF and the RF/CatBoost/XGB calibrated OOF). Caller uses the blended OOF
    to pick the deployed threshold. XGBoost uses the same ordinal repr as RF (NaN→-999)."""
    cat_feats = _cats(X_pp)
    X_ord, enc = _rf_repr(X_pp, cat_feats, fit=True)
    X_cat = _cat_repr(X_pp, cat_feats)
    cat_idx = [X_cat.columns.get_loc(c) for c in cat_feats]

    skf = StratifiedKFold(n_splits, shuffle=True, random_state=SEED)
    rf_oof = np.zeros(len(y)); cat_oof = np.zeros(len(y)); xgb_oof = np.zeros(len(y))
    for k, (a, b) in enumerate(skf.split(X_pp, y)):
        rf = _rf_general()
        rf.fit(X_ord.iloc[a], y.iloc[a]); rf_oof[b] = rf.predict_proba(X_ord.iloc[b])[:, 1]
        xg = _xgb_model(); xg.fit(X_ord.iloc[a], y.iloc[a])
        xgb_oof[b] = xg.predict_proba(X_ord.iloc[b])[:, 1]
        i2, ies = train_test_split(a, test_size=0.15, random_state=SEED + k, stratify=y.iloc[a])
        cb = _cat_general(cat_idx, spw)
        cb.fit(X_cat.iloc[i2], y.iloc[i2], eval_set=(X_cat.iloc[ies], y.iloc[ies]))
        cat_oof[b] = cb.predict_proba(X_cat.iloc[b])[:, 1]
    rf_platt = fit_platt(rf_oof, y.values); cat_platt = fit_platt(cat_oof, y.values)
    xgb_platt = fit_platt(xgb_oof, y.values)
    members_oof = [np.asarray(lgb_oof_cal), apply_platt(rf_oof, rf_platt),
                   apply_platt(cat_oof, cat_platt), apply_platt(xgb_oof, xgb_platt)]

    rf_final = _rf_general().fit(X_ord, y)
    xgb_final = _xgb_model().fit(X_ord, y)
    i2, ies = train_test_split(np.arange(len(y)), test_size=0.15, random_state=SEED + 99, stratify=y)
    cat_final = _cat_general(cat_idx, spw)
    cat_final.fit(X_cat.iloc[i2], y.iloc[i2], eval_set=(X_cat.iloc[ies], y.iloc[ies]))
    art = {"rf_model": rf_final, "cat_model": cat_final, "xgb_model": xgb_final, "encoder": enc,
           "cat_feats": cat_feats, "cat_idx": cat_idx,
           "rf_platt": rf_platt, "cat_platt": cat_platt, "xgb_platt": xgb_platt, "tabpfn": None}
    # OPTIONAL TabPFN-v2.6 5th member — REVERTED to OFF by default (2026-06-27). A deployment-faithful
    # held-out CV (analysis/exp_general_final_number.py + exp_general_capture.py) showed it adds +0.0017
    # AUC but ZERO robust F1 at the deployed operating point (paired Δ −0.0016…+0.0011, <1σ, not seed-
    # stable; F1-optimized weights collapse to equal) — not worth a gated TabPFN/GPU dependency + the
    # batch-invariance compromise. Set ADD_TABPFN_MEMBER=True to re-enable. tabpfn_member.py is kept.
    if ADD_TABPFN_MEMBER:
        from tabpfn_member import fit_tabpfn_member
        tab_art, tab_oof_cal = fit_tabpfn_member(X_pp, y, cat_feats, n_splits)
        if tab_art is not None:
            members_oof.append(tab_oof_cal); art["tabpfn"] = tab_art
            print("  + TabPFN-v2.6 member: 5-way EQUAL-weight blend (general only)")
    blend_oof = np.mean(members_oof, axis=0)
    return art, blend_oof


def _cat_model(cat_idx, spw):
    # shared with the transfer scorer (transfer.py) — DO NOT retune here (scorer is locked).
    return CatBoostClassifier(iterations=1000, depth=6, learning_rate=0.05,
                              scale_pos_weight=spw, cat_features=cat_idx, random_seed=SEED,
                              verbose=0, allow_writing_files=False, early_stopping_rounds=40,
                              thread_count=4)


# Optuna-tuned RF + CatBoost for the GENERAL OWN blend only (analysis/exp_general_rfcat_hpo.py +
# exp_general_rfcat_gate.py). RF default (unlimited depth + sqrt) over-fit the 518-col matrix; tuned
# (depth 9, max_features 0.3, min_leaf 10) is a regularization fix: OOF AUC +0.0198. In the 4-way blend
# (held-out seeds) tuned RF+CatBoost give F1* +0.0019/+0.0039 and AUC +0.0014/+0.0015 (both seeds +).
# DEPLOYED → general ~0.536→~0.539. Kept separate from _cat_model so the transfer scorer is untouched.
def _rf_general():
    return RandomForestClassifier(n_estimators=1100, max_features=0.3, max_depth=9,
                                  min_samples_leaf=10, n_jobs=4, class_weight="balanced",
                                  random_state=SEED)


def _cat_general(cat_idx, spw):
    return CatBoostClassifier(iterations=1300, depth=5, learning_rate=0.0268, l2_leaf_reg=2.65,
                              scale_pos_weight=spw, cat_features=cat_idx, random_seed=SEED,
                              verbose=0, allow_writing_files=False, early_stopping_rounds=40,
                              thread_count=4)


def general_blend_probs(X_pp, lgb_probs_cal, art):
    """Inference: average the LGBM-ensemble calibrated probs with the frozen RF, CatBoost and
    XGBoost calibrated probs. X_pp is the general own-model native-NaN matrix. Back-compatible:
    pre-XGB artifacts (no 'xgb_model') fall back to the 3-way average."""
    X_ord, _ = _rf_repr(X_pp, art["cat_feats"], enc=art["encoder"], fit=False)
    rf_p = apply_platt(art["rf_model"].predict_proba(X_ord)[:, 1], art["rf_platt"])
    X_cat = _cat_repr(X_pp, art["cat_feats"])
    cat_p = apply_platt(art["cat_model"].predict_proba(X_cat)[:, 1], art["cat_platt"])
    members = [np.asarray(lgb_probs_cal), rf_p, cat_p]
    if art.get("xgb_model") is not None:                       # pre-XGB artifacts -> 3-way
        members.append(apply_platt(art["xgb_model"].predict_proba(X_ord)[:, 1], art["xgb_platt"]))
    if art.get("tabpfn") is not None:                          # optional TabPFN-v2.6 5th member
        from tabpfn_member import tabpfn_member_probs
        members.append(tabpfn_member_probs(X_pp, art["tabpfn"]))
    return np.mean(members, axis=0)
