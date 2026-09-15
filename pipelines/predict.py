"""
PathoSense — Train/Save & Inference Pipeline (2026-06-10 rework).

Per-panel recipe (pipelines/recipe.py) + cross-panel transfer (pipelines/transfer.py):
  general — own model only, IMPROVED recipe: native-NaN preprocessing + ALL features
            (no mRMR) + AL_ fam_flags. (general is the transfer source.)
  cftr    — pure general-transfer scorer (w=1.0); own model unused.
  cancer  — 0.5·general-transfer + 0.5·own(median+mRMR) blend (w=0.5).
  pah     — pure general-transfer scorer (w=1.0); own model unused.
  (Deployed weights live ONLY in transfer.TRANSFER_PANELS — the single source of truth.)

All preprocessing/feature/calibration steps are fit on the panel's training data and
replayed purely at inference (batch-invariant). The transfer scorer excludes the
panel's training Variant_IDs, so its Platt calibration (fit on panel-train probs) sees
out-of-sample general scores — matching the finals condition (novel target variants).

Usage:
    python predict.py train --panel all
    python predict.py infer --panel cftr --input test.csv --output preds.csv
    python predict.py submit --input test.csv --team-name "..." --team-id TEAM_101 \
        --application-id BASVURU_101   # finals JSON (TEKNOFEST guide §3.2)
    (--input also takes several files, e.g. one CSV per panel, merged into one JSON)
"""
import os
import sys
import argparse
import json
import math
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.metrics import f1_score, roc_auc_score
from config import SEED
from schema import (normalize_sentinels, to_categorical, partition_columns,
                    drop_all_nan_rows)
from feature_engineering import (
    engineer_aa_features, engineer_acmg_features, engineer_interaction_features,
    fit_score_agreement, apply_score_agreement,
)
from calibration import fit_platt, apply_platt
from preprocess_v2 import fit_keep_nan, apply_keep_nan, fit_al, apply_al, fast_mrmr
from utils import (detect_leaky_columns, optimize_threshold_f1,
                   optimize_threshold_f1_balanced, KNOWN_LEAKY, PANELS, TEST_PRIORS)
from models import (preprocess_data, get_model_params, train_multi_seed_ensemble)
from recipe import PANEL_RECIPE
from transfer import (TRANSFER_PANELS, fit_transfer_scorer, transfer_score,
                      project_target, transfer_score_matrix,
                      TRANSFER_FINETUNE, crossfit_finetuned_g, finetune_and_store,
                      TRANSFER_TARGETRANK, fit_target_pop_grids)
from general_blend import fit_general_blend, general_blend_probs

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Per-panel deployed-threshold override (else optimize_threshold_f1_balanced on full-train OOF).
# cancer: the bootstrap selector overshot to 0.85 (a single full-train draw on the DOWNSLOPE). Honest
# per-fold CV (current rfup+fam_flags scorer) shows the method's per-fold picks CENTER at ~0.81
# (0.805/0.814 across 2 seeds = train-justified, not a peeked test-peak) and that a fixed 0.81 beats
# the adaptive selection by +0.011..+0.016 (>=1.86σ both seeds) and beats fixed-0.85 by ~+0.008-0.010
# → cancer F1* ~0.720 → ~0.735. (analysis/exp_cancer_threshold_check.py + exp_cancer_threshold_confirm.py;
# lead from Öykü's threshold-stability analysis.)
# general: selector picks 0.827 (→0.83) but honest per-fold optimum is 0.82; fixed 0.82 beats the
# selector +0.005 (1.46σ/1.95σ) and fixed-0.83 +0.003, both seeds → ~0.530→~0.533 (small, train-justified).
# pah (selector 0.822 = optimum) and cftr (selector 0.888 ≈ deployed 0.90; the test-better 0.92 is ABOVE
# the selector's center = small-sample tail-chasing, not train-justified) keep the data-driven selector.
# (analysis/exp_threshold_check_all.py)
# RECALIBRATED 2026-06-26 after the audit fixes (#6 nested-ES OOF, #3 fast_mrmr) shifted general's &
# cancer's calibration: re-running the honest per-fold grid on the POST-FIX deployed blend (faithful;
# 3 seeds; analysis/exp_threshold_recalibrate.py) moves both DOWN one notch — general 0.82→0.81
# (+0.0048, +3.51σ, per-seed [.81,.81,.80], selector center .82-.83 → overshoot), cancer 0.81→0.80
# (+0.0062, +2.87σ, at selector center ~.79). cancer 0.84 showed a bigger +0.018 but sits ABOVE the
# selector center on a bumpy landscape (same tail-chasing pattern as cftr-0.92) → NOT deployed.
THRESHOLD_OVERRIDE = {"cancer": 0.80, "general": 0.81}

# Per-panel LGBM-param overrides on top of get_model_params. general: scale_pos_weight=1 (no class
# balancing — the threshold-at-prior handles the prior; rework found it best) + ACTIVATE bagging
# (the deployed subsample=0.8 is INERT without bagging_freq). On the general LGBM this is +0.0082 F1*
# (+2.52σ) / +0.0031 AUC (+4.36σ); in the LGBM+RF+CatBoost blend it dilutes to +0.0021/+0.0030 F1*
# (both seeds +, AUC + both) → DEPLOYED. (analysis/exp_general_params.py + exp_general_spwbag_confirm.py.)
# ONLY general — the small panels need their class weighting + no row subsampling on tiny data.
LGBM_PARAM_OVERRIDE = {"general": {"scale_pos_weight": 1.0, "bagging_freq": 1, "bagging_fraction": 0.8}}

# Per-member redundancy prune: the general matrix has ~16 >0.99-correlated clusters (one 79-col!) that
# the corr-drop misses (corr(min_periods=30)=NaN for sparse AL pairs). They hurt the greedy LightGBM but
# NOT the redundancy-robust RF (max_features=0.3)/CatBoost/XGB. So prune them for the LGBM ONLY, keep full
# features for the blend members → F1* +0.0018 / AUC + (positive all 3 seeds; analysis/exp_mask_final.py).
# Pruning the WHOLE blend instead is negative (exp_mask_blend_gate.py). general only.
REDUN_PRUNE_LGBM = {"general"}


def _lgbm_redun_keep(X_pp, thr=0.99):
    """Columns to keep for the LGBM: drop numeric cols with a |corr|>thr partner already kept (keep the
    lowest-missingness representative per cluster); all categoricals kept. Label-free."""
    from collections import defaultdict
    cats = [c for c in X_pp.columns if isinstance(X_pp[c].dtype, pd.CategoricalDtype)]
    numc = [c for c in X_pp.columns if c not in cats]
    if len(numc) < 2:
        return list(X_pp.columns)
    M = X_pp[numc].apply(pd.to_numeric, errors="coerce")
    C = M.corr().abs().fillna(0.0).values
    miss = M.isna().mean()
    parent = {c: c for c in numc}
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for i in range(len(numc)):
        for j in range(i + 1, len(numc)):
            if C[i, j] > thr:
                parent[find(numc[i])] = find(numc[j])
    cl = defaultdict(list)
    for c in numc:
        cl[find(c)].append(c)
    keep_num = {min(v, key=lambda c: miss[c]) for v in cl.values()}
    return [c for c in X_pp.columns if (c in keep_num) or (c in cats)]


# ---------------------------------------------------------------------------
# Feature engineering (per-panel recipe)
# ---------------------------------------------------------------------------

def _engineer(df, cfg):
    df = to_categorical(df)
    df = engineer_aa_features(df)
    if cfg.get("acmg"):
        df = engineer_acmg_features(df)
    if cfg.get("interactions"):
        df = engineer_interaction_features(df)
    return df


# ---------------------------------------------------------------------------
# Own-model preprocessing (keep_nan or median), with pure replay
# ---------------------------------------------------------------------------

def _fit_own_preprocess(X, y, cfg):
    """Fit own-model preprocessing + feature selection. Returns (X_pp, prep_art)."""
    if cfg["prep"] == "keep_nan":
        alf = fit_al(X, cfg["al_features"]) if cfg.get("al_features") else None
        X_pp, ppf = fit_keep_nan(X)
        if alf is not None:
            X_pp = pd.concat([X_pp.reset_index(drop=True),
                              apply_al(X, alf).reset_index(drop=True)], axis=1)
            X_pp = X_pp.loc[:, ~X_pp.columns.duplicated()]
        sa = None
        if cfg.get("score_agree"):
            sa = fit_score_agreement(X_pp, y, top_k=7)
            X_pp = apply_score_agreement(X_pp, sa)
        feats = list(X_pp.columns)
        cat_cats = {c: list(X_pp[c].cat.categories) for c in feats
                    if isinstance(X_pp[c].dtype, pd.CategoricalDtype)}
        prep_art = {"kind": "keep_nan", "ppf": ppf, "alf": alf, "sa": sa,
                    "feats": feats, "cat_cats": cat_cats}
        return X_pp[feats], prep_art

    # median + (optional mRMR) — the deployed chain
    X_pp, _, _, info = preprocess_data(X.copy(), X.copy(), X.copy(), 0.95)
    sa = None
    if cfg.get("score_agree"):
        sa = fit_score_agreement(X_pp, y, top_k=7)
        X_pp = apply_score_agreement(X_pp, sa)
    if cfg["feature_select"] == "mrmr":
        X_sel, _, _, _ = fast_mrmr(X_pp, y, X_pp, X_pp)
        feats = list(X_sel.columns)
    else:
        feats = list(X_pp.columns)
    cat_cats = {c: list(X_pp[c].cat.categories) for c in feats
                if isinstance(X_pp[c].dtype, pd.CategoricalDtype)}
    prep_art = {"kind": "median", "fitted": info["fitted"], "sa": sa,
                "feats": feats, "cat_cats": cat_cats}
    return X_pp[feats], prep_art


def _apply_own_preprocess(X, prep_art):
    if prep_art["kind"] == "keep_nan":
        X_pp = apply_keep_nan(X, prep_art["ppf"])
        if prep_art["alf"] is not None:
            X_pp = pd.concat([X_pp.reset_index(drop=True),
                              apply_al(X, prep_art["alf"]).reset_index(drop=True)], axis=1)
            X_pp = X_pp.loc[:, ~X_pp.columns.duplicated()]
    else:
        X_pp = _apply_median_preprocess(X, prep_art["fitted"])
    if prep_art["sa"] is not None:
        X_pp = apply_score_agreement(X_pp, prep_art["sa"])
    for c in prep_art["feats"]:
        if c not in X_pp.columns:
            X_pp[c] = np.nan
    X_pp = X_pp[prep_art["feats"]]
    for c, cats in prep_art["cat_cats"].items():
        X_pp[c] = pd.Categorical(X_pp[c].astype(object), categories=cats)
    return X_pp


def _apply_median_preprocess(X, fitted):
    """Replay the deployed median preprocessing (mirrors run_pipeline.preprocess_data)."""
    X = X.copy()
    X.drop(columns=[c for c in fitted.get("drop_constants", []) if c in X.columns],
           inplace=True, errors="ignore")
    for col, (gv, gr) in fitted.get("pop_quantile_grids", {}).items():
        if col not in X.columns or len(gv) == 0:
            continue
        v = X[col].values.astype(float); m = ~np.isnan(v); out = np.full_like(v, np.nan)
        if m.any():
            out[m] = np.interp(v[m], gv, gr, left=gr[0], right=gr[-1])
        X[col] = out
    for col, (lo, hi) in fitted.get("conservation_clip", {}).items():
        if col in X.columns:
            X[col] = X[col].clip(lo, hi)
    for col in fitted.get("miss_indicator_cols", []) + fitted.get("cat_miss_indicator_cols", []):
        if col in X.columns:
            X[f"{col}_missing"] = X[col].isna().astype(int)
    X = X.fillna(fitted["medians"])
    for col in fitted.get("low_var_after_transform", []) + fitted.get("high_corr_cols", []):
        if col in X.columns:
            X.drop(columns=[col], inplace=True)
    num_kept = fitted.get("numeric_cols_kept", [])
    cat_kept = fitted.get("categorical_cols", [])
    for c in num_kept:
        if c not in X.columns:
            X[c] = 0
    return X[[c for c in num_kept if c in X.columns] + [c for c in cat_kept if c in X.columns]]


# ---------------------------------------------------------------------------
# Train & save
# ---------------------------------------------------------------------------

def train_and_save(panel_name):
    cfg = PANEL_RECIPE[panel_name]
    w = TRANSFER_PANELS.get(panel_name)          # general-transfer weight, None if no transfer
    prior = TEST_PRIORS.get(panel_name, 0.5)
    panel_dir = os.path.join(MODELS_DIR, panel_name)
    os.makedirs(panel_dir, exist_ok=True)
    print(f"\n  {panel_name.upper()}  recipe={cfg}  transfer_w={w}")

    df = pd.read_csv(PANELS[panel_name])
    df = normalize_sentinels(df)
    df = drop_all_nan_rows(df)
    df_eng = _engineer(df.copy(), cfg)
    _, num_cols, cat_cols, tgt = partition_columns(df_eng)
    y = df_eng[tgt].reset_index(drop=True)
    X = df_eng[num_cols + cat_cols].reset_index(drop=True)
    print(f"  {len(df)} samples ({int(y.sum())}P / {int((y==0).sum())}B)")

    leaky = detect_leaky_columns(X[num_cols], y, auc_threshold=0.99, verbose=False)
    all_leaky = sorted(set(leaky) | (KNOWN_LEAKY & set(X.columns)))
    X = X.drop(columns=all_leaky)
    feature_cols = list(X.columns)

    need_own = (w is None) or (w < 1.0)
    own = {}
    if need_own:
        X_pp, prep_art = _fit_own_preprocess(X, y, cfg)
        spw = (y == 0).sum() / max((y == 1).sum(), 1)
        params = get_model_params(len(X_pp), spw)
        if panel_name in LGBM_PARAM_OVERRIDE:
            params = {**params, **LGBM_PARAM_OVERRIDE[panel_name]}
            print(f"  LGBM param override: {LGBM_PARAM_OVERRIDE[panel_name]}")
        n_folds = 3 if len(X_pp) < 200 else 5
        lgbm_keep = None; X_pp_lgbm = X_pp
        if panel_name in REDUN_PRUNE_LGBM:
            lgbm_keep = _lgbm_redun_keep(X_pp)
            X_pp_lgbm = X_pp[lgbm_keep]
            print(f"  LGBM redundancy prune (>0.99 near-dups): {len(X_pp.columns)} -> {len(lgbm_keep)} cols")
        models, oof, _, oof_std = train_multi_seed_ensemble(
            X_pp_lgbm, y, X_pp_lgbm, params, n_seeds=5, n_folds=n_folds)
        platt_own = fit_platt(oof, y.values)
        own_oof_cal = apply_platt(oof, platt_own)
        print(f"  own model: {len(prep_art['feats'])} feats, OOF AUC={roc_auc_score(y, oof):.4f}")
        blend_art = None
        if cfg.get("blend"):
            blend_art, own_oof_cal = fit_general_blend(X_pp, y, own_oof_cal, spw, n_splits=n_folds)
            print(f"  + RF+CatBoost blend (decorrelation): blended-OOF AUC={roc_auc_score(y, own_oof_cal):.4f}")
        own = {"prep_art": prep_art, "models": models, "platt": platt_own, "blend": blend_art,
               "lgbm_keep": lgbm_keep,   # per-member: LGBM uses pruned cols; blend uses full X_pp
               "params": {k: v for k, v in params.items() if k != "verbose"}}
    else:
        own_oof_cal = None

    transfer = {}
    if w is not None:
        art = fit_transfer_scorer(panel_name)
        # target-rank domain adaptation: fit AL_ rank grids on THIS panel's own training distribution
        # (leakage-safe — train rows only; batch-invariant at inference). None for non-targrank panels.
        tr_grids = fit_target_pop_grids(df, art) if panel_name in TRANSFER_TARGETRANK else None
        ft_cfg = TRANSFER_FINETUNE.get(panel_name)
        if ft_cfg:
            # panel fine-tune: CROSS-FIT OOF finetuned scores for honest Platt/threshold, then store the
            # full-panel finetuned booster for OUT-OF-SAMPLE finals inference (novel target variants).
            g = crossfit_finetuned_g(art, df, y, ft_cfg)
            art = finetune_and_store(art, df, y, ft_cfg)
            print(f"  transfer scorer FINE-TUNED on {panel_name} (+{ft_cfg['K']} trees, w={ft_cfg['weights']}); "
                  f"OOF-finetuned AUC={roc_auc_score(y, g):.4f}")
        else:
            g = transfer_score(df, art, target_pop_grids=tr_grids)   # OOS for this panel's rows
            print(f"  transfer scorer: AUC on {panel_name}={roc_auc_score(y, g):.4f} "
                  f"(trained on {art['n_train']} rows, excl {art['n_excluded']} ids)"
                  + ("  [target-rank AL]" if tr_grids else ""))
        platt_g = fit_platt(g, y.values)
        g_oof_cal = apply_platt(g, platt_g)
        transfer = {"art": art, "platt": platt_g, "w": w, "target_grids": tr_grids}
    else:
        g_oof_cal = None

    # blend OOF -> threshold at the test prior
    if w is None:
        blend_oof = own_oof_cal
    elif w >= 1.0:
        blend_oof = g_oof_cal
    else:
        blend_oof = w * g_oof_cal + (1 - w) * own_oof_cal

    f1_t_raw, f1_val_raw = optimize_threshold_f1(y, blend_oof)
    threshold, f1_bal, f1_std = optimize_threshold_f1_balanced(
        y.values, blend_oof, target_prior=prior, n_bootstrap=300)
    if panel_name in THRESHOLD_OVERRIDE:
        sel = threshold
        threshold = THRESHOLD_OVERRIDE[panel_name]
        print(f"  threshold OVERRIDE: selector picked {sel} -> deploying fixed {threshold} "
              f"(honest CV: fixed beats selector +~0.01)")
    print(f"  deployed threshold (test prior {prior:.3f}): t={threshold} "
          f"F1*={f1_bal:.4f}±{f1_std:.4f}  (raw-prior F1={f1_val_raw:.4f})")

    artifacts = {
        "panel": panel_name, "recipe": cfg, "transfer_w": w,
        "leaky_cols": all_leaky, "feature_cols": feature_cols,
        "own": own, "transfer": transfer,
        "threshold": threshold, "f1_estimate": f1_bal, "f1_std": f1_std,
        "training_prior": float(y.mean()), "test_prior": prior,
    }
    path = os.path.join(panel_dir, "artifacts.joblib")
    joblib.dump(artifacts, path)
    print(f"  saved -> {path}")
    return artifacts


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_artifacts(panel_name):
    art_path = os.path.join(MODELS_DIR, panel_name, "artifacts.joblib")
    if not os.path.exists(art_path):
        raise FileNotFoundError(
            f"no model at {art_path}. Run: python predict.py train --panel {panel_name}")
    art = joblib.load(art_path)
    # Byte-identical replay (guide §6: the jury may re-run and must get the submitted JSON).
    # RandomForest.predict_proba with n_jobs>1 sums tree votes in a thread-dependent order,
    # which moves probabilities by ~1e-16 between runs — same classes, different bytes.
    # Serial summation is deterministic; the fitted trees are untouched.
    for blend in (art.get("own") or {}).get("blend"), ((art.get("transfer") or {}).get("art") or {}).get("blend"):
        if blend and blend.get("rf_model") is not None:
            blend["rf_model"].set_params(n_jobs=1)
    return art

def score_frame(df, art):
    """Score a raw panel-style frame with the deployed artifacts. Returns a dict:
    probs (blended, calibrated), own_probs / transfer_probs (calibrated member streams or None),
    and the model matrices the LightGBM members saw (X_lgbm for the own model, Xs for the transfer
    scorer) so callers can compute per-row feature contributions on exactly what was scored."""
    cfg = art["recipe"]; w = art["transfer_w"]
    df = normalize_sentinels(df.copy())
    if "Label" in df.columns:
        df = df.drop(columns=["Label"])

    e_cal = X_lgbm = None
    if art["own"]:
        df_eng = _engineer(df.copy(), cfg)
        avail = [c for c in art["feature_cols"] if c in df_eng.columns]
        X_pp = _apply_own_preprocess(df_eng[avail].copy(), art["own"]["prep_art"])
        keep = art["own"].get("lgbm_keep")        # per-member: LGBM on pruned cols, blend on full X_pp
        X_lgbm = X_pp[[c for c in keep if c in X_pp.columns]] if keep else X_pp
        e = np.mean([m.predict_proba(X_lgbm)[:, 1] for m in art["own"]["models"]], axis=0)
        e_cal = apply_platt(e, art["own"]["platt"])
        if art["own"].get("blend") is not None:
            e_cal = general_blend_probs(X_pp, e_cal, art["own"]["blend"])

    g_cal = Xs = None
    if art["transfer"]:
        Xs = project_target(df, art["transfer"]["art"],
                            target_pop_grids=art["transfer"].get("target_grids"))
        g = transfer_score_matrix(Xs, art["transfer"]["art"])
        g_cal = apply_platt(g, art["transfer"]["platt"])

    if w is None:
        probs = e_cal
    elif w >= 1.0:
        probs = g_cal
    else:
        probs = w * g_cal + (1 - w) * e_cal
    return {"probs": probs, "own_probs": e_cal, "transfer_probs": g_cal,
            "X_lgbm": X_lgbm, "Xs": Xs}

def predict(input_csv, panel_name, output_csv):
    try:
        art = load_artifacts(panel_name)
    except FileNotFoundError as e:
        print(f"ERROR: {e}"); sys.exit(1)
    print(f"  {panel_name.upper()}: transfer_w={art['transfer_w']}, threshold={art['threshold']}")

    df = pd.read_csv(input_csv)
    variant_ids = df["Variant_ID"].copy() if "Variant_ID" in df.columns else None
    probs = score_frame(df, art)["probs"]
    preds = (probs >= art["threshold"]).astype(int)

    out = {}
    if variant_ids is not None:
        out["Variant_ID"] = variant_ids.values
    out["prediction"] = preds
    out["probability"] = probs.round(6)
    pd.DataFrame(out).to_csv(output_csv, index=False)
    print(f"  {int(preds.sum())} Pathogenic / {len(preds)-int(preds.sum())} Benign -> {output_csv}")
    return pd.DataFrame(out)

# --- Finals submission JSON (guide §3.2, university level) ---

PANEL_KEYS = ("general", "cancer", "pah", "cftr")
ID_CANDIDATES = ("id", "Variant_ID", "variant_id", "ID")
# Organizer vocabulary (şartname §3: "MASTER, Kanser, CFTR, PAH"; training files
# YARISMA_TRAIN_{MASTER,KANSER,PAH,CFTR}.csv). Accepted on input; emitted on output
# whenever the test file carries no panel column of its own.
PANEL_ALIASES = {"master": "general", "genel": "general", "general": "general",
                 "kanser": "cancer", "cancer": "cancer", "pah": "pah", "cftr": "cftr"}
PANEL_LABELS = {"general": "MASTER", "cancer": "KANSER", "pah": "PAH", "cftr": "CFTR"}


def _panel_key(value):
    k = PANEL_ALIASES.get(str(value).strip().lower())
    if k is None:
        raise ValueError(f"unknown panel value {value!r}; expected one of "
                         f"{tuple(PANEL_ALIASES)} (case-insensitive)")
    return k


def build_submission(team_name, team_id, application_id, ids, panels, probs, preds,
                     prob_mode="p1", with_probabilities=False):
    """Assemble the guide §3.2 JSON. ids/panels must be verbatim text from the test file (ids are
    never altered — §3). probs are P(class 1).

    prob_mode: default "p1" ships P(class 1) — the guide says predicted_prob feeds AUROC-like
    metrics, which need a monotone positive-class score, and its example (class "0", prob 0.87)
    is exactly a high-threshold decision on P(1)=0.87, which the "own decision rule" note explains.
    "predicted" = confidence in the predicted class, only if the sample check demands it.
    with_probabilities (default False): the row set matches the §3.2 university template exactly
    (id, panel, predicted_class, predicted_prob). True adds the {"0","1"} dict from the §3 common
    rule ("probabilities alanı zorunludur"), whose scope is ambiguous — opt in only if the sample
    validator demands it; predicted_prob then equals probabilities["1"].
    """
    rows = []
    for vid, pnl, p, k in zip(ids, panels, probs, preds):
        p = float(p)
        cls = str(int(k))
        rec = {"id": vid, "panel": pnl, "predicted_class": cls,
               "predicted_prob": p if (prob_mode == "p1" or k == 1) else 1.0 - p}
        if with_probabilities:
            rec["probabilities"] = {"0": 1.0 - p, "1": p}
        rows.append(rec)
    return {"team_name": team_name, "team_id": team_id, "application_id": application_id,
            "competition_level": "UNIVERSITE_VE_UZERI", "predictions": rows}


def validate_submission(obj, ids):
    """Enforce the guide's hard rules before writing; raises ValueError naming the offense."""
    for key in ("team_name", "team_id", "application_id", "competition_level", "predictions"):
        if key not in obj:
            raise ValueError(f"submission missing required field: {key}")
    if obj["competition_level"] != "UNIVERSITE_VE_UZERI":
        raise ValueError(f"competition_level must be UNIVERSITE_VE_UZERI, got {obj['competition_level']!r}")
    rows = obj["predictions"]
    if len(rows) != len(ids):
        raise ValueError(f"{len(rows)} prediction rows for {len(ids)} test examples (every id exactly once)")
    seen = set()
    for r in rows:
        vid = r["id"]
        if vid in seen:
            raise ValueError(f"duplicate id in predictions: {vid!r}")
        seen.add(vid)
        q = r["predicted_prob"]
        if not (isinstance(q, (int, float)) and math.isfinite(q) and 0.0 <= q <= 1.0):
            raise ValueError(f"predicted_prob out of [0,1] for id {vid!r}: {q!r}")
        if r["predicted_class"] not in ("0", "1"):
            raise ValueError(f"predicted_class must be '0' or '1' for id {vid!r}: {r['predicted_class']!r}")
        if "probabilities" in r:
            pr = r["probabilities"]
            if set(pr) < {"0", "1"} or not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in pr.values())\
                    or abs(pr["0"] + pr["1"] - 1.0) > 0.001:
                raise ValueError(f"probabilities must give finite '0'/'1' values summing to 1 (±0.001) for id {vid!r}")
    missing = set(map(str, ids)) - seen
    if missing:
        raise ValueError(f"{len(missing)} test ids missing from predictions, e.g. {sorted(missing)[:3]}")


def write_submission(obj, path):
    # allow_nan=False makes json.dump raise on any NaN/Infinity — the guide's hard no.
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
    with open(path, encoding="utf-8") as f:   # re-read: what lands on the wire must parse
        back = json.load(f)
    if len(back["predictions"]) != len(obj["predictions"]):
        raise ValueError(f"wrote {path} but it does not re-parse to the same row count")


def _infer_panel_from_name(path):
    """Guess the panel from the file name (training-style names like
    YARISMA_TEST_KANSER.csv). Returns None when nothing matches."""
    name = os.path.basename(str(path)).lower()
    if "kanser" in name or "cancer" in name:
        return "cancer"
    if "cftr" in name:
        return "cftr"
    if "pah" in name:
        return "pah"
    if "master" in name or "general" in name or "genel" in name:
        return "general"
    return None


def _split_panel_assignment(spec):
    """'panel=path' → (panel, path); a plain path → (None, path). A '=' inside
    a real path is safe: the left side must be a valid panel name to count."""
    if "=" in spec:
        maybe_panel, _, maybe_path = spec.partition("=")
        try:
            return _panel_key(maybe_panel), maybe_path
        except ValueError:
            pass
    return None, spec


def _panel_keys(raw, fallback, source):
    """Per-row panel keys for one input file. A present panel column always wins;
    the fallback covers files without one (e.g. one CSV per panel, like training)."""
    if "panel" in raw.columns:
        if raw["panel"].isna().any():
            raise ValueError(f"{source}: missing panel value at row(s) "
                             f"{list(raw.index[raw['panel'].isna()])[:5]}")
        return raw["panel"].map(_panel_key)
    if fallback:
        return pd.Series([fallback] * len(raw), index=raw.index).map(_panel_key)
    raise ValueError(f"{source}: no panel column; pass --panel <general|cancer|pah|cftr>")


def submit(input_csv, team_name, team_id, application_id, output=None, panel=None,
           id_col=None, prob_mode="p1", with_probabilities=False):
    """input_csv: one test CSV or a list of them (e.g. one file per panel). panel:
    fallback panel name, or a list aligned with input_csv for files without a
    panel column. All files are merged into ONE guide-§3.2 JSON."""
    inputs = [input_csv] if isinstance(input_csv, (str, os.PathLike)) else list(input_csv)
    assigned, paths = zip(*[_split_panel_assignment(s) for s in inputs])
    if panel is None:
        fallbacks = [None] * len(paths)
    elif isinstance(panel, str):
        fallbacks = [panel] * len(paths)
    else:
        fallbacks = list(panel)
        if len(fallbacks) != len(paths):
            raise ValueError(f"{len(paths)} input files but {len(fallbacks)} --panel values "
                             "(pass one panel per file, in order, or a single panel)")
    for p, a in zip(fallbacks, assigned):
        if p is not None and a is not None and _panel_key(p) != a:
            raise ValueError(f"conflicting panels for one file: --panel {p} vs "
                             f"panel={a} assignment (pick one)")
    inputs = paths

    all_ids, all_panels, all_probs, all_preds = [], [], [], []
    for path, fallback, assign in zip(inputs, fallbacks, assigned):
        effective = fallback or assign or _infer_panel_from_name(path)
        raw = pd.read_csv(path, dtype=str)      # verbatim text for id/panel
        df = pd.read_csv(path)                  # real dtypes for scoring

        col = id_col or next((c for c in ID_CANDIDATES if c in raw.columns), None)
        if col is None or col not in raw.columns:
            raise ValueError(f"{path}: no id column found (tried {ID_CANDIDATES}); pass --id-col")
        if raw[col].isna().any():
            raise ValueError(f"{path}: empty id at row(s) {list(raw.index[raw[col].isna()])[:5]}")
        ids = raw[col]
        if ids.duplicated().any():
            raise ValueError(f"{path}: duplicate ids in test file: "
                             f"{ids[ids.duplicated()].iloc[:3].tolist()}")

        keys = _panel_keys(raw, effective, path)
        # A panel column is echoed verbatim (the organizers' own text wins). Otherwise we
        # must name the panel ourselves, so use THEIR vocabulary, not our internal keys.
        panels = raw["panel"] if "panel" in raw.columns \
            else pd.Series([PANEL_LABELS[_panel_key(effective)]] * len(raw), index=raw.index)

        probs = np.full(len(df), np.nan)
        thresholds = np.full(len(df), np.nan)
        for key in sorted(keys.unique()):
            art = load_artifacts(key)   # FileNotFoundError carries its own train hint
            mask = (keys == key).values
            probs[mask] = np.asarray(score_frame(df.loc[mask], art)["probs"], dtype=float)
            thresholds[mask] = art["threshold"]
            print(f"  {key.upper()}: {int(mask.sum())} rows from {path}, "
                  f"threshold={art['threshold']}")
        if np.isnan(probs).any():
            raise ValueError(f"{path}: model produced NaN probability for "
                             f"{int(np.isnan(probs).sum())} row(s)")
        preds = (probs >= thresholds).astype(int)
        all_ids.append(ids)
        all_panels.append(panels)
        all_probs.append(probs)
        all_preds.append(preds)

    ids = pd.concat(all_ids, ignore_index=True)
    panels = pd.concat(all_panels, ignore_index=True)
    probs = np.concatenate(all_probs)
    preds = np.concatenate(all_preds)

    obj = build_submission(team_name, team_id, application_id, ids, panels, probs, preds,
                           prob_mode=prob_mode, with_probabilities=with_probabilities)
    validate_submission(obj, ids)
    # guide §3 recommended name TEAM_<TAKIM_ID>_FINAL.json: team_id "TEAM_101" -> TEAM_101_FINAL.json
    takim_id = team_id[5:] if team_id.upper().startswith("TEAM_") else team_id
    out = output or f"TEAM_{takim_id}_FINAL.json"
    write_submission(obj, out)
    print(f"  {int(preds.sum())} Pathogenic / {len(preds) - int(preds.sum())} Benign "
          f"({prob_mode} prob" + (" + probabilities dict" if with_probabilities else "") + ")")
    print(f"  submission -> {out}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PathoSense Train/Predict")
    sub = parser.add_subparsers(dest="command", required=True)
    p_train = sub.add_parser("train")
    p_train.add_argument("--panel", required=True,
                         choices=["general", "cancer", "pah", "cftr", "all"])
    p_infer = sub.add_parser("infer")
    p_infer.add_argument("--panel", required=True, choices=["general", "cancer", "pah", "cftr"])
    p_infer.add_argument("--input", required=True)
    p_infer.add_argument("--output", default="predictions.csv")
    p_sub = sub.add_parser("submit", help="finals JSON per the TEKNOFEST guide §3.2")
    p_sub.add_argument("--input", required=True, nargs="+",
                       help="test CSV(s): one mixed-panel file, or one file per panel as "
                       "panel=path pairs (e.g. --input general=g.csv cancer=k.csv ...), "
                       "which needs no --panel and no ordering" )
    p_sub.add_argument("--team-name", required=True, help="exactly as in KYS")
    p_sub.add_argument("--team-id", required=True, help="e.g. TEAM_101")
    p_sub.add_argument("--application-id", required=True, help="e.g. BASVURU_101")
    p_sub.add_argument("--output", default=None,
                       help="default: TEAM_<team-id>_FINAL.json")
    p_sub.add_argument("--panel", choices=["general", "cancer", "pah", "cftr"], default=None,
                       nargs="*", help="panel for files without a panel column (overrides the "
                       "file-name guess): one value for all files, or one per file in "
                       "--input order — keep the two orders aligned")
    p_sub.add_argument("--id-col", default=None,
                       help=f"id column name; default auto-detect {ID_CANDIDATES}")
    p_sub.add_argument("--prob-mode", choices=["p1", "predicted"], default="p1",
                       help="'p1' (default) = P(class 1), the monotone score AUROC/PR-AUC need — the "
                            "guide's example (class '0', prob 0.87) is consistent with this under a "
                            "high threshold, and its 'own decision rule' note explains the disagreement; "
                            "'predicted' = confidence in the predicted class, if the sample check demands it")
    p_sub.add_argument("--probabilities", action="store_true",
                       help="add the {'0','1'} probabilities dict from the §3 common rules; off by "
                            "default so rows match the §3.2 university template exactly")
    args = parser.parse_args()

    if args.command == "train":
        panels = ["general", "cancer", "pah", "cftr"] if args.panel == "all" else [args.panel]
        for pn in panels:
            print(f"\n{'='*60}")
            train_and_save(pn)
    elif args.command == "infer":
        predict(args.input, args.panel, args.output)
    elif args.command == "submit":
        try:
            submit(args.input, args.team_name, args.team_id, args.application_id,
                   output=args.output, panel=args.panel, id_col=args.id_col,
                   prob_mode=args.prob_mode, with_probabilities=args.probabilities)
        except (ValueError, FileNotFoundError) as e:
            print(f"ERROR: {e}"); sys.exit(1)


if __name__ == "__main__":
    main()
