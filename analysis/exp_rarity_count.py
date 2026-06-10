"""AL_-block rarity COUNT feature — honest test-prior A/B for GENERAL and CANCER.

Candidate (stateless, batch-invariant, leakage-safe):
    MISS_AL_count = df[[c for c in df.columns if c.startswith('AL_')]].isna().sum(axis=1)

The 334 AL_ allele-frequency cohort columns are present (non-NaN) for common
variants and missing (NaN) for rare ones. Rarer variant -> more cohorts missing
-> higher prior on pathogenicity. preprocess_data imputes those NaNs (median) and
drops the high-missing AL_ columns, DESTROYING the row-level missingness pattern,
so the deployed model never sees it. This feature snapshots that pattern as a
single count at LOAD time (before leakage-drop / preprocess) so it survives into
mRMR. Single-feature sAUC: general 0.683, cancer 0.770 (confirmed on the panels).

Leakage note: this is NOT the perfect-missingness AF leakage (those near-AUC=1.0
columns are caught by detect_leaky_columns at auc_threshold=0.99 and dropped). The
COUNT aggregate sits at 0.68-0.77 — the legitimate "rare = pathogenic" biology,
well under the 0.99 leak gate, computable per-row on the unlabeled test set with
no labels / no external lookup / no test-set stats. Missing is counted, never
imputed to 0.

A/B (paired RepeatedStratifiedKFold, 5x3 = 15 folds, scored at the per-panel TEST
prior via the deployed single-LGBM ablation chain in feature_ablation.eval_fold):
  general : deployed_base                vs deployed_base + MISS_AL_count
            (general deploys no tracks; feature_ablation.prepare_config is the
             deployed feature set)
  cancer  : deployed_tracks (score_agree+interactions) vs + MISS_AL_count
            (cancer deploys score_agree+interactions per BEST_CONFIG_PER_PANEL;
             baseline mirrors blend_test.prepare_panel_tracks)

pah (sAUC 0.536, flat) and cftr (0.609, weak) are intentionally NOT tested.

Run:  .venv/bin/python -u analysis/exp_rarity_count.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipelines"))

from sklearn.model_selection import RepeatedStratifiedKFold

from config import SEED
from schema import (
    normalize_sentinels, drop_all_nan_rows, to_categorical, partition_columns,
)
from feature_engineering import (
    engineer_aa_features, engineer_acmg_features, engineer_interaction_features,
    fit_score_agreement, apply_score_agreement,
)
from utils import (
    detect_leaky_columns, KNOWN_LEAKY, TEST_PRIORS, ACMG_PANELS, PANELS,
)
from run_pipeline import (
    preprocess_data, select_features_mrmr, BEST_CONFIG_PER_PANEL,
)

# reuse the deployed-chain prepare + single-LGBM honest fold evaluator
from feature_ablation import prepare_config, eval_fold

RARITY_COL = "MISS_AL_count"


def add_miss_al_count(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless per-row count of missing AL_ (allele-frequency cohort) columns.
    Computed on the AL_ block BEFORE any imputation, so the missingness pattern
    is preserved. No labels, no fitted state, no test-set statistics."""
    al = [c for c in df.columns if c.startswith("AL_")]
    df = df.copy()
    df[RARITY_COL] = df[al].isna().sum(axis=1).astype("int32")
    return df


def prepare_tracks_config(panel, add_rarity):
    """Deployed per-panel tracks (score_agree/interactions per BEST_CONFIG_PER_PANEL),
    optionally + MISS_AL_count injected at LOAD time. Mirrors
    blend_test.prepare_panel_tracks; returns the prepare_config-shaped tuple so
    feature_ablation.eval_fold can consume it directly."""
    cfg = BEST_CONFIG_PER_PANEL.get(panel, {})
    df = pd.read_csv(PANELS[panel])
    df = normalize_sentinels(df)
    df = drop_all_nan_rows(df, verbose=False)
    df = to_categorical(df)
    if add_rarity:                       # inject BEFORE leakage-drop / preprocess
        df = add_miss_al_count(df)
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
    return X, y, cat_feats, feats, []


def prepare_base_config(panel, add_rarity):
    """General: deployed base feature set (no tracks). Uses feature_ablation's
    prepare_config with the rarity fn as the `extra` hook (applied before
    leakage-drop / preprocess), exactly as the deployed chain would treat a
    load-time feature."""
    extra = add_miss_al_count if add_rarity else None
    return prepare_config(panel, use_aa=True, extra=extra)


PREPARE = {
    "general": prepare_base_config,   # no deployed tracks
    "cancer":  prepare_tracks_config, # score_agree + interactions
}


def run_panel(panel, n_splits=5, n_repeats=3):
    prep = PREPARE[panel]
    prior = TEST_PRIORS[panel]
    print(f"\n{'#'*72}\n  RARITY-COUNT A/B — {panel.upper()}  (test_prior={prior:.3f})\n{'#'*72}",
          flush=True)

    configs = {"base": False, "base+rarity": True}
    summ = {}
    feats_by_cfg = {}
    for name, add_rarity in configs.items():
        X, y, cat_feats, feats, _ = prep(panel, add_rarity)
        feats_by_cfg[name] = feats
        spw = (y == 0).sum() / max((y == 1).sum(), 1)
        selected = RARITY_COL in feats
        outer = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                        random_state=SEED)
        vals, t0 = [], time.time()
        for fold, (tr, te) in enumerate(outer.split(X, y)):
            f1, mcc, _ = eval_fold(X, y, spw, cat_feats, np.asarray(tr), np.asarray(te),
                                   prior, seed=SEED + fold)
            vals.append((f1, mcc))
        arr = np.array(vals)
        summ[name] = {"f1": arr[:, 0], "mcc": arr[:, 1], "n_feats": len(feats),
                      "selected": selected}
        print(f"  {name:12s} F1*={arr[:,0].mean():.4f}±{arr[:,0].std():.3f}  "
              f"MCC*={arr[:,1].mean():.4f}  feats={len(feats)}  "
              f"{RARITY_COL}_selected={selected}  ({round(time.time()-t0,1)}s)", flush=True)

    base, cand = summ["base"]["f1"], summ["base+rarity"]["f1"]
    d = cand - base
    se = d.std() / max(np.sqrt(len(d)), 1)
    clears = bool(d.mean() > 2 * se)
    mcc_d = summ["base+rarity"]["mcc"].mean() - summ["base"]["mcc"].mean()
    verdict = ("WIN (>2SE)" if clears else
               "HURTS" if d.mean() < -2 * se else "noise (within 2SE)")
    print(f"\n  paired Δ (base+rarity − base), {len(d)} folds:", flush=True)
    print(f"    ΔF1*={d.mean():+.4f} ± {se:.4f} SE   (wins {int((d>0).sum())}/{len(d)})  "
          f"clears_2SE={clears}  ->  {verdict}", flush=True)
    print(f"    ΔMCC*={mcc_d:+.4f}", flush=True)
    return {
        "panel": panel,
        "baseline_f1": float(base.mean()),
        "candidate_f1": float(cand.mean()),
        "delta_f1": float(d.mean()),
        "se": float(se),
        "n_folds": int(len(d)),
        "clears_2se": clears,
        "mcc_delta": float(mcc_d),
        "selected": summ["base+rarity"]["selected"],
        "n_feats_base": summ["base"]["n_feats"],
        "n_feats_cand": summ["base+rarity"]["n_feats"],
    }


def main():
    results = [run_panel(p) for p in ("general", "cancer")]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output",
                       "exp_rarity_count.csv")
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"\n  Saved {out}", flush=True)
    print(f"\n{'='*72}\n  FINAL\n{'='*72}", flush=True)
    for r in results:
        print(f"  {r['panel']:8s} base={r['baseline_f1']:.4f} -> "
              f"cand={r['candidate_f1']:.4f}  ΔF1*={r['delta_f1']:+.4f}±{r['se']:.4f}  "
              f"clears_2SE={r['clears_2se']}  ΔMCC*={r['mcc_delta']:+.4f}  "
              f"selected={r['selected']}", flush=True)


if __name__ == "__main__":
    main()
