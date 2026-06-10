"""Heterogeneous per-panel BLEND test — does averaging decorrelated model families
beat the deployed LGBM ensemble?

Rationale: the fair tune-off (fair_tuneoff.py) measured RF and CatBoost as
individually competitive with the deployed 5-seed LGBM ensemble, but they are
different families with decorrelated errors. Seed-averaging within LGBM can't
capture cross-family diversity; a blend can. This is the one lever with a real
mechanistic reason to help — but it still must clear the same honest bar.

Protocol (identical philosophy to fair_tuneoff): RepeatedStratifiedKFold outer.
Per fold, each component produces calibrated probabilities both on outer-train
(inner-OOF, for the blend's threshold) and outer-test (model trained on
outer-train). Blends are equal-weight averages of the calibrated probs (learned
weights overfit 111–370-row panels). The blend threshold is tuned on the
outer-TRAIN blend-OOF at the test prior, frozen, applied to outer-test. F1*/MCC
at the test prior on the held-out fold; PAIRED vs the LGBM ensemble alone.

Components: lgbm_ensemble (deployed, 5-seed), random_forest, catboost — each
tuned + native-categorical + early-stopped exactly as in fair_tuneoff.
Blend variants (all from the same per-fold probs, ~free): ens+rf, ens+cat,
ens+rf+cat.

Run:  .venv/bin/python -u analysis/blend_test.py --panel pah   [--quick]
"""
import os
import sys
import time
import json
import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipelines"))

from sklearn.model_selection import RepeatedStratifiedKFold

from config import SEED
from calibration import fit_platt, apply_platt
from schema import (
    normalize_sentinels, drop_all_nan_rows, to_categorical, partition_columns,
)
from feature_engineering import (
    engineer_aa_features, engineer_acmg_features, engineer_interaction_features,
    fit_score_agreement, apply_score_agreement,
)
from utils import (
    optimize_threshold_f1_balanced, TEST_PRIORS,
    detect_leaky_columns, KNOWN_LEAKY, ACMG_PANELS, PANELS,
)
from run_pipeline import (
    train_multi_seed_ensemble, get_model_params,
    preprocess_data, select_features_mrmr, BEST_CONFIG_PER_PANEL,
)
from fair_tuneoff import (
    Candidate, GRIDS, prepare_panel, build_representations, metrics_at_prior,
    DEPLOYED_F1, OUT_DIR,
)


def prepare_panel_tracks(panel):
    """Like fair_tuneoff.prepare_panel but with the panel's DEPLOYED per-panel tracks
    on (score_agreement / interactions per BEST_CONFIG_PER_PANEL), so cancer/cftr are
    tested on the same feature set the submission model actually uses. Tracks are fit
    on the full panel (mild, equal-across-configs optimism — fine for relative blend Δ)."""
    cfg = BEST_CONFIG_PER_PANEL.get(panel, {})
    df = pd.read_csv(PANELS[panel])
    df = normalize_sentinels(df)
    df = drop_all_nan_rows(df, verbose=False)
    df = to_categorical(df)
    df = engineer_aa_features(df)
    if panel in ACMG_PANELS:
        df = engineer_acmg_features(df)
    if cfg.get("use_interactions", False):
        df = engineer_interaction_features(df)
    _, num_cols, cat_cols, tgt = partition_columns(df)
    y = df[tgt].reset_index(drop=True)
    X = df[num_cols + cat_cols].reset_index(drop=True)
    leaky = detect_leaky_columns(X, y, auc_threshold=0.99, verbose=False)
    drop = sorted(set(leaky) | (KNOWN_LEAKY & set(X.columns)))
    if drop:
        X = X.drop(columns=drop)
    Xpp, _, _, _ = preprocess_data(X.copy(), X.copy(), X.copy(), corr_threshold=0.95)
    if cfg.get("use_score_agree", False):
        sa = fit_score_agreement(Xpp, y, top_k=7)
        Xpp = apply_score_agreement(Xpp, sa)
    Xsel, _, _, _ = select_features_mrmr(Xpp, y, Xpp, Xpp)
    feats = list(Xsel.columns)
    X = Xpp[feats].reset_index(drop=True)
    cat_feats = [c for c in feats if isinstance(X[c].dtype, pd.CategoricalDtype)]
    return X, y, cat_feats, TEST_PRIORS.get(panel, 0.5)

PANELS_ALL = ["general", "cancer", "pah", "cftr"]


def ensemble_fold_probs(X_native, y, spw, tr, te):
    """Deployed 5-seed LGBM ensemble: calibrated (outer-train OOF, outer-test) probs."""
    Xtr = X_native.iloc[tr].reset_index(drop=True)
    ytr = y.iloc[tr].reset_index(drop=True)
    Xte = X_native.iloc[te].reset_index(drop=True)
    n_folds = 3 if len(Xtr) < 200 else 5
    params = get_model_params(len(Xtr), spw)
    _, oof, test, _ = train_multi_seed_ensemble(
        Xtr, ytr, Xte, params, n_seeds=5, n_folds=n_folds)
    pl = fit_platt(oof, ytr.values)
    return apply_platt(oof, pl), apply_platt(test, pl)


def candidate_fold_probs(cand, tr, te, n_inner, seed, prior):
    """A tuned single-family Candidate: calibrated (outer-train OOF, outer-test) probs.
    Mirrors fair_tuneoff.Candidate.eval_fold but returns probabilities, not metrics."""
    oof = cand.inner_oof(cand.best_hp, tr, prior, n_inner, seed)
    i2, es = cand._carve_es(tr, seed + 99)
    model = cand._fit(cand.best_hp, i2, es)
    test_raw = cand._proba(model, te)
    sub_y = cand.y.iloc[tr].values
    pl = fit_platt(oof, sub_y)
    return apply_platt(oof, pl), apply_platt(test_raw, pl)


def score(name, oof_train, test_probs, y_tr, y_te, prior):
    """Threshold on outer-train probs, score outer-test at the test prior."""
    thr, _, _ = optimize_threshold_f1_balanced(y_tr, oof_train, target_prior=prior,
                                               n_bootstrap=150)
    f1, mcc = metrics_at_prior(y_te, test_probs, thr, prior)
    return f1, mcc, thr


def run_panel(panel, quick=False, prepare_fn=prepare_panel):
    tag = " +tracks" if prepare_fn is prepare_panel_tracks else ""
    print(f"\n{'#'*72}\n  BLEND TEST — {panel.upper()}{tag}\n{'#'*72}", flush=True)
    X, y, cat_feats, prior = prepare_fn(panel)
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    X_native, X_cat, X_ord = build_representations(X, cat_feats)
    n_inner, n_splits = 3, 5
    n_repeats = 1 if quick else 2
    print(f"  n={len(X)}  feats={len(X.columns)}  test_prior={prior:.3f}  "
          f"outer={n_splits}x{n_repeats}  deployed F1*={DEPLOYED_F1.get(panel, float('nan')):.4f}",
          flush=True)

    # tune the single-family components once (reuses fair_tuneoff tuning)
    rf = Candidate("random_forest", X_ord, y, spw, cat_feats, GRIDS["random_forest"], quick=quick)
    cb = Candidate("catboost", X_cat, y, spw, cat_feats, GRIDS["catboost"], quick=quick)
    for c in (rf, cb):
        hp, auc = c.tune(prior, n_inner, SEED)
        print(f"  tuned {c.name:14s} AUC={auc:.4f} hp={hp}", flush=True)

    variants = ["ensemble", "ens+rf", "ens+cat", "ens+rf+cat"]
    per_fold = {v: [] for v in variants}
    rows = []
    outer = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    t0 = time.time()
    for fold, (tr, te) in enumerate(outer.split(X, y)):
        tr, te = np.asarray(tr), np.asarray(te)
        y_tr, y_te = y.iloc[tr].values, y.iloc[te].values
        ft = time.time()
        e_oof, e_te = ensemble_fold_probs(X_native, y, spw, tr, te)
        r_oof, r_te = candidate_fold_probs(rf, tr, te, n_inner, SEED + fold, prior)
        c_oof, c_te = candidate_fold_probs(cb, tr, te, n_inner, SEED + fold, prior)

        blends = {
            "ensemble":   (e_oof, e_te),
            "ens+rf":     ((e_oof + r_oof) / 2, (e_te + r_te) / 2),
            "ens+cat":    ((e_oof + c_oof) / 2, (e_te + c_te) / 2),
            "ens+rf+cat": ((e_oof + r_oof + c_oof) / 3, (e_te + r_te + c_te) / 3),
        }
        line = []
        for v, (oof, tep) in blends.items():
            f1, mcc, thr = score(v, oof, tep, y_tr, y_te, prior)
            per_fold[v].append((f1, mcc))
            rows.append({"panel": panel, "fold": fold, "variant": v, "f1": f1, "mcc": mcc})
            line.append(f"{v}={f1:.3f}")
        print(f"  fold {fold+1:2d}/{n_splits*n_repeats}: " + "  ".join(line) +
              f"  ({round(time.time()-ft,1)}s)", flush=True)

    # aggregate, paired vs ensemble
    base = np.array([f for f, _ in per_fold["ensemble"]])
    print(f"\n  {panel.upper()} — F1* mean±std over {len(base)} folds "
          f"(paired Δ vs ensemble):", flush=True)
    summary = {}
    for v in variants:
        arr = np.array([f for f, _ in per_fold[v]])
        mcc = np.mean([m for _, m in per_fold[v]])
        d = arr - base
        se = d.std() / max(np.sqrt(len(d)), 1)
        if v == "ensemble":
            verdict = "(incumbent)"
        elif d.mean() > 2 * se:
            verdict = "HELPS"
        elif d.mean() < -2 * se:
            verdict = "HURTS"
        else:
            verdict = "no effect (noise)"
        summary[v] = (arr.mean(), arr.std(), mcc, d.mean(), se)
        dstr = "   —   " if v == "ensemble" else f"{d.mean():+.4f}±{se:.4f}"
        wins = "" if v == "ensemble" else f"  wins {int((d>0).sum())}/{len(d)}"
        print(f"    {v:12s} F1*={arr.mean():.4f}±{arr.std():.3f}  MCC*={mcc:.4f}  "
              f"Δ={dstr}{wins}  -> {verdict}", flush=True)
    return rows, summary


def main():
    ap = argparse.ArgumentParser(description="Heterogeneous per-panel blend test")
    ap.add_argument("--panel", default="all", choices=PANELS_ALL + ["all"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tracks", action="store_true",
                    help="use deployed per-panel feature tracks (settles cancer/cftr)")
    args = ap.parse_args()
    panels = PANELS_ALL if args.panel == "all" else [args.panel]
    prep = prepare_panel_tracks if args.tracks else prepare_panel
    all_rows = []
    for p in panels:
        rows, _ = run_panel(p, quick=args.quick, prepare_fn=prep)
        all_rows.extend(rows)
    suffix = "_tracks" if args.tracks else ""
    out = os.path.join(OUT_DIR, f"blend_test{suffix}.csv")
    pd.DataFrame(all_rows).to_csv(out, index=False)
    print(f"\n  Saved {out}", flush=True)


if __name__ == "__main__":
    main()
