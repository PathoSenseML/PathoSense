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
"""
import os
import sys
import argparse
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
    return joblib.load(art_path)

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
    args = parser.parse_args()

    if args.command == "train":
        panels = ["general", "cancer", "pah", "cftr"] if args.panel == "all" else [args.panel]
        for pn in panels:
            print(f"\n{'='*60}")
            train_and_save(pn)
    elif args.command == "infer":
        predict(args.input, args.panel, args.output)


if __name__ == "__main__":
    main()
