"""Cross-panel transfer scorer — the rework's biggest win (2026-06-10).

A model trained on the large GENERAL panel generalizes to the small panels far
better than their own data-starved models. Honest, leakage-corrected CV (exclude
ALL target Variant_IDs from the general scorer — finals target variants are novel):
  cftr   pure general transfer (w=1.0)        -> F1* 0.529->0.836  (general scorer AUC ~0.91 >> own 0.77)
  cancer 0.5 transfer + 0.5 own blend (w=0.5) -> F1* 0.665->0.708  (transfer AUC ~0.92 ≈ own; blend wins)
  pah    pure general transfer (w=1.0)        -> F1* 0.549->0.660  (transfer AUC ~0.80 > own)
(DEPLOYED weights are TRANSFER_PANELS below — the SINGLE SOURCE OF TRUTH. cancer=0.5 blend; cftr/pah=1.0 pure.
 Earlier drafts of this header had cancer/pah swapped — do NOT reconcile docstrings against memory, trust the code.)
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
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold

from config import SEED
from schema import (normalize_sentinels, drop_all_nan_rows, to_categorical,
                    partition_columns)
from feature_engineering import engineer_aa_features
from utils import detect_leaky_columns, KNOWN_LEAKY, PANELS
from models import get_model_params
from preprocess_v2 import fit_keep_nan, apply_keep_nan, fit_al, apply_al
from general_blend import _rf_repr, _cat_repr, _cat_model

# Per-panel general-scorer blend weight w (general share); 1-w is the panel-own share.
# From analysis/exp_transfer*.py honest leakage-corrected CV (improved scorer, 30 folds).
#   cftr  1.0  blended scorer, pure transfer -> F1* 0.529->0.836
#   cancer 0.5 blended scorer                -> F1* 0.665->0.708
#   pah   1.0  single-LGBM scorer            -> F1* 0.549->0.660
TRANSFER_PANELS = {"cftr": 1.0, "cancer": 0.5, "pah": 1.0}

# Panels whose transfer scorer applies TARGET-RANK domain adaptation: re-rank the target's AL_ allele-
# frequency features on the TARGET panel's OWN training distribution (fit_target_pop_grids) instead of
# general's percentile grid — fixing the cross-panel AF covariate shift (same variant: AL corr ~0.017
# general<->pah, because the organizers re-extracted AFs per panel). DEPLOYED for pah (2026-06-28):
# definitive 4-fresh-seed / 500-fold confirm = +0.0043 F1* (+3.19σ pooled; per-seed +2.32/+2.70/+1.94σ
# and one flat −0.40σ) — a principled, leakage-safe (grids fit on target-train only) and batch-invariant
# win. The lone consistently-positive pah lever after population-features (leak), all 165 architecture×
# blend cells, and TabPFN were rejected. (analysis/exp_pah_targetrank_confirm.py, pah_targetrank_confirm.csv,
# analysis/exp_pah_targetrank_confirm.py.)
TRANSFER_TARGETRANK = {"pah"}

# Panels whose transfer scorer is a heterogeneous BLEND (LGBM+RF+CatBoost, prob-avg).
# Validated to beat the single-LGBM transfer scorer (analysis/exp_transfer_blend_validate.py):
# cftr +0.039, cancer +0.012. pah is EXCLUDED — the blend hurts it (-0.012), so pah keeps the
# single LightGBM transfer scorer.
TRANSFER_BLEND = {"cftr", "cancer"}

# Per-panel weights (w_lgb, w_rf, w_cat) for the heterogeneous blend; default = equal 1/3.
# cancer: RF-up (0.25,0.50,0.25). On the PURE cancer scorer this is a robust +0.034 F1* (+10σ, both
# seeds; a FIXED weight, so not grid-overfit) — though it is AUC-NEUTRAL (an operating-point/score-
# shape effect, not better ranking). In the deployed 0.5·transfer+0.5·own blend it shrinks to a sub-
# 2σ, seed-unstable +0.003 (`exp_transfer_blendw.py` / `exp_blendw_confirm.py` / `exp_blendw_wsweep.py`).
# Below the ≥2σ deploy bar, but zero-cost and never-negative across 400 folds → shipped by user call
# (2026-06-22) as a small calculated bet. cftr is EXCLUDED — RF-up HURTS it (−2.8σ; equal blend is best).
TRANSFER_BLEND_WEIGHTS = {"cancer": (0.25, 0.50, 0.25)}

# Panels whose transfer-scorer blend adds a 4th member: XGBoost (same ordinal repr as RF). XGBoost
# decorrelates cleanly in the general OWN blend (+0.005 F1*); test whether it also lifts the general-
# trained transfer scorer for cftr/cancer. blend_weights become 4-tuples (TRANSFER_BLEND_WEIGHTS4).
TRANSFER_ADD_XGB = set()
TRANSFER_BLEND_WEIGHTS4 = {}        # {panel: (w_lgb, w_rf, w_cat, w_xgb)}; default = equal 1/4


def _xgb_scorer():
    return XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=5, subsample=0.8,
                         colsample_bytree=0.8, reg_lambda=1.5, eval_metric="logloss",
                         n_jobs=4, random_state=SEED)

# Per-panel FINE-TUNING: warm-start (continue-boost K trees, init_model=base) the scorer's LGBM on the
# TARGET panel's own training rows, then UPWEIGHT the now-panel-adapted LGBM in the blend. Validated
# CANCER-ONLY: +0.0085 F1* (both seeds, +1.18σ, weight-robust 0.4-0.6, FINETUNE-driven — reweighting the
# base LGBM is negative; AUC-supported +3.86σ). pah collapses (F1* −3.97σ), cftr flat → cancer only.
# TESTED & REJECTED (2026-06-23) — kept as inert infra (default-off) for reproducibility. Fine-tuning the
# cancer scorer LGBM + upweighting it looked like +0.0085 F1* (exp_cancer_finetune_v3.py) but that was vs a
# PER-FOLD-threshold baseline. Against the ACTUAL deployed cancer (rfup + FIXED threshold 0.81 = ~0.735),
# the finetune LOSES at every threshold, both seeds (exp_cancer_finetune_final.py: ft@selector −0.9/−1.9σ,
# all fixed −0.6 to −1.9σ). Reason: finetune improves AUC (+3.86σ) but the F1 gain is operating-point, and
# the fixed-0.81 threshold win already captured that margin — they don't stack. pah collapses (−3.97σ),
# cftr flat. So NO panel deploys finetune. Set a {panel: cfg} dict here to re-enable.
TRANSFER_FINETUNE = {}


def project_target(df_raw, art, target_pop_grids=None):
    """Project raw target rows into the scorer's frozen model matrix Xs. Pure replay of the fitted
    transforms (batch-invariant). target_pop_grids: see transfer_score."""
    df = engineer_aa_features(to_categorical(df_raw.copy()))
    X = df.reindex(columns=art["raw_cols"])
    ppf = art["ppf"] if target_pop_grids is None else {**art["ppf"], "pop_grids": target_pop_grids}
    Xs = apply_keep_nan(X, ppf)
    if art["alf"] is not None:
        Xs = pd.concat([Xs.reset_index(drop=True), apply_al(X, art["alf"]).reset_index(drop=True)], axis=1)
        Xs = Xs.loc[:, ~Xs.columns.duplicated()]
    for col in art["feats"]:
        if col not in Xs.columns:
            Xs[col] = np.nan
    Xs = Xs[art["feats"]]
    for col, cats in art["cat_cats"].items():
        Xs[col] = pd.Categorical(Xs[col].astype(object), categories=cats)
    return Xs

_project_target = project_target   # back-compat name for the finetune helpers


def _finetune_booster(base_booster, Xs, y, cfg, cat_feats):
    """Continue-boost K trees on (Xs, y) warm-started from base_booster. Returns the LGBMClassifier."""
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    m = LGBMClassifier(n_estimators=cfg["K"], learning_rate=cfg["lr"], num_leaves=cfg["num_leaves"],
                       min_child_samples=cfg["min_child_samples"], reg_lambda=cfg["reg_lambda"],
                       scale_pos_weight=spw, random_state=SEED, n_jobs=4, verbose=-1)
    m.fit(Xs, y, init_model=base_booster, categorical_feature=cat_feats or "auto")
    return m


def finetune_and_store(art, df_target_raw, y, cfg):
    """Fit the full-panel finetuned LGBM and store it in art (used at inference). Mutates+returns art."""
    Xs = _project_target(df_target_raw, art)
    cat_feats = [c for c in art["feats"] if c in art["cat_cats"]]
    art["finetuned_booster"] = _finetune_booster(art["models"][0].booster_, Xs,
                                                 y.reset_index(drop=True), cfg, cat_feats)
    art["blend_weights"] = tuple(cfg["weights"])
    return art


def crossfit_finetuned_g(art, df_target_raw, y, cfg, n_folds=5):
    """OOF blended finetuned transfer scores on the target panel — honest for Platt/threshold (the
    finetune sees only the train fold; rf/cat are the frozen general-trained components)."""
    Xs = _project_target(df_target_raw, art); yv = y.reset_index(drop=True)
    cat_feats = [c for c in art["feats"] if c in art["cat_cats"]]
    base = art["models"][0].booster_; bl = art["blend"]
    X_ord, _ = _rf_repr(Xs, bl["cat_feats"], enc=bl["encoder"], fit=False)
    rf_p = bl["rf_model"].predict_proba(X_ord)[:, 1]
    cat_p = bl["cat_model"].predict_proba(_cat_repr(Xs, bl["cat_feats"]))[:, 1]
    w = cfg["weights"]; lgb_oof = np.zeros(len(yv))
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=SEED)
    for a, b in skf.split(Xs, yv):
        m = _finetune_booster(base, Xs.iloc[a], yv.iloc[a], cfg, cat_feats)
        lgb_oof[b] = m.predict_proba(Xs.iloc[b])[:, 1]
    return w[0] * lgb_oof + w[1] * rf_p + w[2] * cat_p

# ── tunables (deploy a winner by flipping these; defaults = original behavior) ──
# All three were TESTED 2026-06-15 (analysis/exp_transfer_improve.py + exp_transfer_hpo.py,
# analysis/exp_transfer_noal.py) and REJECTED — the defaults below are the
# validated optimum. Kept as inert, default-off knobs for reproducibility + easy re-test.
# LightGBM seeds averaged in the scorer (1 = original single-seed). Scalar or per-panel dict.
#   tested: marginal at best (cancer ms10 +1.7σ, sub-2σ; neutral cftr; negative pah).
TRANSFER_N_SEEDS = 1
# Monotone-constraint subset for the scorer LGBM: "none" | "ek" | "ek_aa". Scalar or per-panel dict.
#   tested: REFUTED — hurts pah (−9σ) and cftr (−4σ); models use real non-monotone conservation signal.
MONOTONE_SUBSET = "none"
# Optional HPO-tuned LightGBM param overrides for the scorer (None = get_model_params default).
#   tested: tuned-to-general params hurt the transfer targets (pah −5.8σ, cftr −2.9σ). Scalar dict
#   (all panels) or {panel: {param: value}}.
TRANSFER_PARAMS = None
# Drop raw AL_ allele-freq VALUES from the scorer model input (they don't transfer across panels:
# same variant has corr ~0.01-0.4 / different scale — see analysis/inspect_data_opportunities.py).
# EK_ conservation transfers cleanly. Scalar bool or per-panel dict. (fam_flags still addable via
# TRANSFER_AL_FEATURES since apply_al reads full raw_cols at score time.)
#   DEPLOYED for cancer: raw AL is covariate-shifted; the AL_ missingness fam_flags are a cleaner,
#   panel-independent rarity signal that decorrelates from the own model -> blended cancer F1*
#   0.701->0.717 (+0.0162, +3.39σ, 75 folds; analysis/exp_cancer_confirm.py). pah/cftr need raw AL
#   (dropping it hurts -8σ/-12σ), so they keep it.
TRANSFER_DROP_AL = {"cancer": True}
# AL_ derived features added to the scorer (resolved per panel; () = none). cancer uses missingness
# fam_flags in place of the dropped raw AL_ values (see TRANSFER_DROP_AL).
TRANSFER_AL_FEATURES = {"cancer": ("fam_flags",)}

import re as _re
_EK_RE = _re.compile(r"^EK_\d+$")
# signed monotone directions for the well-grounded engineered AA features (ek_aa subset only)
_AA_MONO = {"AA_grantham": 1, "AA_grantham_high": 1, "AA_is_radical": 1,
            "AA_blosum62": -1, "AA_is_conservative": -1}


def _resolve(cfg, panel):
    """A tunable may be a scalar (all panels) or a {panel: value} dict."""
    return cfg.get(panel) if isinstance(cfg, dict) else cfg


def _monotone_vec(feats, cat_feats, subset):
    """Monotone-constraint vector aligned to `feats` (= the fit-matrix column order).
    +1 for EK_ conservation (higher -> more pathogenic); for 'ek_aa' also the signed AA_
    features; 0 for categoricals, AL_ (NaN-heavy, ambiguous), and everything else.
    Returns None when no constraint applies (so params stay untouched)."""
    if subset in (None, "none"):
        return None
    cats = set(cat_feats)
    vec = []
    for f in feats:
        if f in cats:
            vec.append(0)
        elif _EK_RE.match(f):
            vec.append(1)
        elif subset == "ek_aa" and f in _AA_MONO:
            vec.append(_AA_MONO[f])
        else:
            vec.append(0)
    return vec if any(v != 0 for v in vec) else None


def _load_raw(panel):
    df = pd.read_csv(PANELS[panel])
    df = normalize_sentinels(df)
    df = drop_all_nan_rows(df, verbose=False)
    return df.reset_index(drop=True)


def fit_transfer_scorer(target_panel, sources=("general",), al_features=None,
                        n_seeds=None, monotone=None, model_params=None, drop_al=None):
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
    raw_cols = list(Xg.columns)            # full (incl AL_) — apply_al reads these at score time
    drop_al = _resolve(TRANSFER_DROP_AL, target_panel) if drop_al is None else drop_al
    if al_features is None:
        al_features = _resolve(TRANSFER_AL_FEATURES, target_panel) or ()

    alf = fit_al(Xg, al_features) if al_features else None
    # AL_ allele-freq VALUES don't transfer (different scale/order per panel); drop_al trains the
    # scorer on EK_/AA_/CAT_ only. fam_flags (AL_ missingness) still addable via al_features.
    Xg_model = Xg
    if drop_al:
        Xg_model = Xg.drop(columns=[cc for cc in Xg.columns if str(cc).startswith("AL_")],
                           errors="ignore")
    Xg_pp, ppf = fit_keep_nan(Xg_model)
    Xg_sel = Xg_pp
    if alf is not None:
        Xg_sel = pd.concat([Xg_pp.reset_index(drop=True),
                            apply_al(Xg, alf).reset_index(drop=True)], axis=1)
    Xg_sel = Xg_sel.loc[:, ~Xg_sel.columns.duplicated()]
    feats = list(Xg_sel.columns)
    cat_feats = [c for c in feats if isinstance(Xg_sel[c].dtype, pd.CategoricalDtype)]
    cat_cats = {c: list(Xg_sel[c].cat.categories) for c in cat_feats}

    n_seeds = _resolve(TRANSFER_N_SEEDS, target_panel) if n_seeds is None else n_seeds
    monotone = _resolve(MONOTONE_SUBSET, target_panel) if monotone is None else monotone
    if model_params is None:
        model_params = _resolve(TRANSFER_PARAMS, target_panel)
    spw = (yg == 0).sum() / max((yg == 1).sum(), 1)
    params = get_model_params(len(Xg_sel), spw)
    if model_params:
        params = {**params, **model_params}
    mono = _monotone_vec(feats, cat_feats, monotone)
    if mono is not None:
        params = {**params, "monotone_constraints": mono}  # aligned to feats column order
    # multi-seed LightGBM: vary BOTH the model seed and the ES-split seed (the latter is the
    # real decorrelation source — subsample is inert without bagging_freq). n_seeds=1 -> identical
    # to the original single model (SEED, ES-split SEED+7). transfer_score averages art["models"].
    models = []
    for i in range(n_seeds):
        i2, ies = train_test_split(np.arange(len(yg)), test_size=0.15,
                                   random_state=SEED + 7 + i, stratify=yg)
        mi = LGBMClassifier(**params, random_state=SEED + i)
        mi.fit(Xg_sel.iloc[i2], yg.iloc[i2], eval_set=[(Xg_sel.iloc[ies], yg.iloc[ies])],
               callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
               categorical_feature=cat_feats or "auto")
        models.append(mi)
    m = models[0]

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
        if target_panel in TRANSFER_ADD_XGB:
            blend["xgb_model"] = _xgb_scorer().fit(X_ord, yg)

    if blend is not None and blend.get("xgb_model") is not None:
        blend_weights = TRANSFER_BLEND_WEIGHTS4.get(target_panel, (0.25, 0.25, 0.25, 0.25))
    else:
        blend_weights = TRANSFER_BLEND_WEIGHTS.get(target_panel, (1/3, 1/3, 1/3))
    return {"model": m, "models": models, "n_seeds": n_seeds, "monotone": monotone,
            "raw_cols": raw_cols, "ppf": ppf, "alf": alf, "feats": feats,
            "cat_cats": cat_cats, "blend": blend, "blend_weights": blend_weights,
            "sources": list(sources), "n_excluded": len(tids), "n_train": len(yg)}


def fit_target_pop_grids(df_target_train_raw: pd.DataFrame, art: dict) -> dict:
    """Fit per-column AF rank grids on the TARGET panel's own training distribution, for the
    same population columns the scorer ranked on general (art['ppf']['pop_grids']). Used for
    domain adaptation: AL_ values are covariate-shifted across panels (corr ~0.01-0.4), so
    ranking the target on ITS OWN distribution aligns the percentile semantics the scorer learned
    on general. Returns {col: (sorted_values, ranks)} to pass as transfer_score(target_pop_grids=)."""
    df = to_categorical(df_target_train_raw.copy()); df = engineer_aa_features(df)
    X = df.reindex(columns=art["raw_cols"])
    grids = {}
    for c in art["ppf"].get("pop_grids", {}):
        if c not in X.columns:
            continue
        nn = pd.to_numeric(X[c], errors="coerce").dropna().sort_values().values
        if len(nn):
            grids[c] = (nn, np.linspace(1 / len(nn), 1.0, len(nn)))
    return grids


def transfer_score(df_raw: pd.DataFrame, art: dict, target_pop_grids=None) -> np.ndarray:
    """Project raw target variants into the general scorer's space and return raw
    general probabilities. df_raw is a panel-style frame (post sentinel-normalise;
    Variant_ID/Label columns are ignored). Pure replay — batch-invariant.

    target_pop_grids (optional, from fit_target_pop_grids): override the general AF rank grids
    with target-distribution grids (domain adaptation for the AL_ covariate shift). Fit on
    target-train only -> still batch-invariant at inference."""
    return transfer_score_matrix(project_target(df_raw, art, target_pop_grids), art)

def transfer_score_matrix(Xs: pd.DataFrame, art: dict) -> np.ndarray:
    """Score an already-projected matrix (from project_target)."""
    ft = art.get("finetuned_booster")
    if ft is not None:                            # panel fine-tuned LGBM (cancer) replaces the base
        lgbm_p = ft.predict_proba(Xs)[:, 1]
    else:
        mdls = art.get("models") or [art["model"]]   # avg multi-seed; old pickles -> [model]
        lgbm_p = np.mean([mm.predict_proba(Xs)[:, 1] for mm in mdls], axis=0)
    bl = art.get("blend")
    if bl is None:
        return lgbm_p
    X_ord, _ = _rf_repr(Xs, bl["cat_feats"], enc=bl["encoder"], fit=False)
    rf_p = bl["rf_model"].predict_proba(X_ord)[:, 1]
    X_cat = _cat_repr(Xs, bl["cat_feats"])
    cat_p = bl["cat_model"].predict_proba(X_cat)[:, 1]
    w = art.get("blend_weights", (1/3, 1/3, 1/3))                  # cancer = RF-up; else equal
    if bl.get("xgb_model") is not None:                            # 4-way blend (LGBM+RF+CatBoost+XGB)
        xgb_p = bl["xgb_model"].predict_proba(X_ord)[:, 1]
        return w[0] * lgbm_p + w[1] * rf_p + w[2] * cat_p + w[3] * xgb_p
    return w[0] * lgbm_p + w[1] * rf_p + w[2] * cat_p              # weighted prob-average (shape-preserving)
