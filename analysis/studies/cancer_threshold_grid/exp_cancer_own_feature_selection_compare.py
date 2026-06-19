"""Cancer-own feature-selection / recipe comparison under fixed transfer BLEND.

Controlled setup:
  * transfer scorer: existing general-minus-cancer BLEND probability scorer
  * BLEND_MODE=prob
  * blend weight w=0.5
  * Variant_ID is excluded by the existing engineered feature loader
  * feature selection is fit inside each outer training fold only
"""
import argparse
import os
import sys
import time
import warnings
from dataclasses import dataclass

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "6")
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINES = os.path.join(HERE, "..", "pipelines")
sys.path.insert(0, HERE)
sys.path.insert(0, PIPELINES)

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
import lightgbm as lgb
from lightgbm import LGBMClassifier

from calibration import apply_platt, fit_platt
from config import SEED
from exp_cancer_weight_threshold_grid import build_probability_blend_scorer
from exp_transfer import load_target_engineered
from feature_engineering import fit_score_agreement, apply_score_agreement
from harness import (
    FINALS_N,
    _drop_leaky,
    bootstrap_f1_mcc,
    f1_at_prior,
    fast_mrmr,
    lgb_oof_and_test,
    mcc_at_prior_analytic,
    tpr_fpr,
)
from preprocess_v2 import fit_keep_nan, apply_keep_nan
from run_pipeline import get_model_params, preprocess_data
from utils import TEST_PRIORS


OUT = os.path.join(HERE, "output")
PANEL = "cancer"
WEIGHT = 0.5
FIXED_THRESHOLD = 0.82
GRID_THRESHOLDS = np.round(np.arange(0.70, 0.90 + 0.001, 0.01), 2)
TOP_K_GRID = [20, 40, 60, 80, 120]
CURRENT_STABLE_F1_STAR = 0.7136
BEST_CALIBRATION_GRID_F1_STAR = 0.7162


@dataclass(frozen=True)
class Recipe:
    key: str
    label: str
    prep: str
    selector: str
    k: int | None = None


def build_recipes():
    recipes = [
        Recipe(
            key="A_median_mrmr_40",
            label="A) median + mRMR 40",
            prep="median",
            selector="mrmr",
            k=40,
        ),
        Recipe(
            key="B_native_nan_all",
            label="B) native NaN + all features",
            prep="keep_nan",
            selector="none",
        ),
    ]
    recipes += [
        Recipe(
            key=f"C_lgb_importance_top_{k}",
            label=f"C) native NaN + LGB importance top-{k}",
            prep="keep_nan",
            selector="lgb_importance",
            k=k,
        )
        for k in TOP_K_GRID
    ]
    recipes += [
        Recipe(
            key=f"D_permutation_top_{k}",
            label=f"D) native NaN + permutation top-{k}",
            prep="keep_nan",
            selector="permutation",
            k=k,
        )
        for k in TOP_K_GRID
    ]
    return recipes


def safe_auc(y_true, probs):
    try:
        return float(roc_auc_score(y_true, probs))
    except ValueError:
        return np.nan


def metric_row(y_true, probs, threshold, prior, n_target):
    preds = (np.asarray(probs, dtype=float) >= threshold).astype(int)
    tpr, fpr = tpr_fpr(y_true, preds)
    finals_f1, finals_mcc = bootstrap_f1_mcc(y_true, preds, prior, n_target)
    return {
        "f1_star": f1_at_prior(tpr, fpr, prior),
        "finals_f1": finals_f1,
        "finals_mcc": finals_mcc,
        "f1_positive": f1_score(y_true, preds, zero_division=0),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "mcc": matthews_corrcoef(y_true, preds),
        "mcc_star": mcc_at_prior_analytic(tpr, fpr, prior),
        "roc_auc": safe_auc(y_true, probs),
        "predicted_positives": int(preds.sum()),
    }


def _cat_cols(X):
    return [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]


def _fit_selector_model(X, y, random_state):
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    params = {
        **get_model_params(len(X), spw),
        "random_state": random_state,
        "n_jobs": 1,
        "importance_type": "gain",
    }
    cat_cols = _cat_cols(X)
    idx_tr, idx_va = train_test_split(
        np.arange(len(y)),
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )
    model = LGBMClassifier(**params)
    model.fit(
        X.iloc[idx_tr],
        y.iloc[idx_tr],
        eval_set=[(X.iloc[idx_va], y.iloc[idx_va])],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
        categorical_feature=cat_cols or "auto",
    )
    return model, idx_va


def _top_names(columns, scores, k):
    scores = np.asarray(scores, dtype=float)
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    order = np.lexsort((np.arange(len(columns)), -scores))
    k = min(int(k), len(columns))
    return [columns[i] for i in order[:k]]


def select_lgb_importance(Xtr, ytr, k, seed):
    model, _ = _fit_selector_model(Xtr, ytr, seed)
    return _top_names(list(Xtr.columns), model.feature_importances_, k)


def select_permutation_importance(Xtr, ytr, k, seed):
    model, idx_va = _fit_selector_model(Xtr, ytr, seed)
    Xva = Xtr.iloc[idx_va]
    yva = ytr.iloc[idx_va]
    scoring = "roc_auc" if yva.nunique() == 2 else "average_precision"
    result = permutation_importance(
        model,
        Xva,
        yva,
        scoring=scoring,
        n_repeats=5,
        random_state=seed,
        n_jobs=1,
    )
    return _top_names(list(Xtr.columns), result.importances_mean, k)


def preprocess_for_recipe(recipe, X_tr_raw, y_tr, X_te_raw, selector_seed):
    Xtr_raw, Xte_raw = _drop_leaky(X_tr_raw, y_tr, X_te_raw)

    if recipe.prep == "median":
        Xtr, _, Xte, _ = preprocess_data(
            Xtr_raw.copy(), Xtr_raw.copy(), Xte_raw.copy(), corr_threshold=0.95
        )
    elif recipe.prep == "keep_nan":
        Xtr, fitted = fit_keep_nan(Xtr_raw.copy())
        Xte = apply_keep_nan(Xte_raw.copy(), fitted)
    else:
        raise ValueError(f"Unknown prep: {recipe.prep}")

    # Cancer's deployed own model uses score-agreement features. Keep that cancer
    # feature-engineering lever fixed while changing selection / NaN handling.
    saf = fit_score_agreement(Xtr, y_tr, top_k=7)
    Xtr = apply_score_agreement(Xtr, saf)
    Xte = apply_score_agreement(Xte, saf)

    if recipe.selector == "mrmr":
        Xtr, _, Xte, _ = fast_mrmr(Xtr, y_tr, Xte, Xte, max_features=recipe.k or 40)
    elif recipe.selector == "none":
        pass
    elif recipe.selector == "lgb_importance":
        names = select_lgb_importance(Xtr, y_tr, recipe.k, selector_seed)
        Xtr, Xte = Xtr[names], Xte[names]
    elif recipe.selector == "permutation":
        names = select_permutation_importance(Xtr, y_tr, recipe.k, selector_seed)
        Xtr, Xte = Xtr[names], Xte[names]
    else:
        raise ValueError(f"Unknown selector: {recipe.selector}")

    return Xtr.reset_index(drop=True), Xte.reset_index(drop=True)


def cancer_own_recipe_probs(recipe, X_tr_raw, y_tr, X_te_raw, *, n_own_seeds, selector_seed):
    Xtr, Xte = preprocess_for_recipe(recipe, X_tr_raw, y_tr, X_te_raw, selector_seed)
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    params = get_model_params(len(Xtr), spw)
    n_folds = 3 if len(Xtr) < 200 else 5
    test_probs, oof = lgb_oof_and_test(
        Xtr,
        y_tr,
        Xte,
        params,
        n_seeds=n_own_seeds,
        n_folds=n_folds,
    )
    calibrator = fit_platt(oof, y_tr.values)
    return apply_platt(test_probs, calibrator), apply_platt(oof, calibrator), len(Xtr.columns)


def evaluate_seed(g_all, Xt_own, yt, recipes, thresholds, cv_seed, n_repeats, n_splits, n_own_seeds):
    prior = TEST_PRIORS[PANEL]
    n_target = FINALS_N[PANEL]
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=cv_seed
    )
    splits = list(rskf.split(Xt_own, yt))
    rows = []

    print(
        f"Running cv_seed={cv_seed}: {len(splits)} paired folds, "
        f"{len(recipes)} own-model recipes...",
        flush=True,
    )

    for fold_idx, (tr, te) in enumerate(splits, start=1):
        tr, te = np.asarray(tr), np.asarray(te)
        ytr = yt.iloc[tr].reset_index(drop=True)
        yte = yt.iloc[te].reset_index(drop=True)
        Xtr = Xt_own.iloc[tr].reset_index(drop=True)
        Xte = Xt_own.iloc[te].reset_index(drop=True)

        gen_cal = fit_platt(g_all[tr], ytr.values)
        gen_te = apply_platt(g_all[te], gen_cal)

        for recipe_idx, recipe in enumerate(recipes, start=1):
            selector_seed = cv_seed + fold_idx * 1000 + recipe_idx
            own_te, _own_tr, n_features = cancer_own_recipe_probs(
                recipe,
                Xtr,
                ytr,
                Xte,
                n_own_seeds=n_own_seeds,
                selector_seed=selector_seed,
            )
            blend_te = WEIGHT * gen_te + (1.0 - WEIGHT) * own_te

            eval_items = [
                ("own_only", own_te, [FIXED_THRESHOLD]),
                ("general_transfer_only", gen_te, [FIXED_THRESHOLD]),
                ("final_blend", blend_te, thresholds),
            ]
            for score_source, probs, source_thresholds in eval_items:
                for threshold in source_thresholds:
                    values = metric_row(yte.values, probs, float(threshold), prior, n_target)
                    rows.append(
                        {
                            "panel": PANEL,
                            "cv_seed": cv_seed,
                            "fold": fold_idx,
                            "recipe": recipe.label,
                            "recipe_key": recipe.key,
                            "prep": recipe.prep,
                            "selector": recipe.selector,
                            "k": recipe.k,
                            "score_source": score_source,
                            "weight": WEIGHT if score_source == "final_blend" else np.nan,
                            "threshold": float(threshold),
                            "threshold_kind": (
                                "fixed" if float(threshold) == FIXED_THRESHOLD else "grid"
                            ),
                            "n_features": n_features,
                            **values,
                        }
                    )

        if fold_idx % n_splits == 0:
            print(f"  cv_seed={cv_seed}: completed {fold_idx}/{len(splits)} folds", flush=True)

    return pd.DataFrame(rows)


def summarize(fold_rows):
    metric_cols = [
        "f1_star",
        "finals_f1",
        "finals_mcc",
        "f1_positive",
        "precision",
        "recall",
        "mcc",
        "mcc_star",
        "roc_auc",
        "predicted_positives",
    ]
    group_cols = [
        "panel",
        "recipe",
        "recipe_key",
        "prep",
        "selector",
        "k",
        "score_source",
        "weight",
        "threshold",
        "threshold_kind",
    ]

    seed_summary = (
        fold_rows.groupby(group_cols + ["cv_seed"], as_index=False, dropna=False)
        .agg(
            n_folds=("fold", "nunique"),
            n_features_mean=("n_features", "mean"),
            predicted_positives_total=("predicted_positives", "sum"),
            **{f"{col}_mean": (col, "mean") for col in metric_cols},
        )
    )

    summary = (
        seed_summary.groupby(group_cols, as_index=False, dropna=False)
        .agg(
            n_runs=("cv_seed", "nunique"),
            n_folds_per_run=("n_folds", "first"),
            n_features_mean=("n_features_mean", "mean"),
            predicted_positives_total=("predicted_positives_total", "sum"),
            **{f"{col}_mean": (f"{col}_mean", "mean") for col in metric_cols},
            **{f"{col}_std_across_seeds": (f"{col}_mean", "std") for col in metric_cols},
        )
    )

    fold_stats = (
        fold_rows.groupby(group_cols, as_index=False, dropna=False)
        .agg(
            n_folds_total=("fold", "count"),
            f1_star_se_across_folds=(
                "f1_star",
                lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0,
            ),
        )
    )
    summary = summary.merge(fold_stats, on=group_cols, how="left")
    summary["delta_vs_current_stable_f1_star"] = (
        summary["f1_star_mean"] - CURRENT_STABLE_F1_STAR
    )
    summary["delta_vs_best_calibration_grid_f1_star"] = (
        summary["f1_star_mean"] - BEST_CALIBRATION_GRID_F1_STAR
    )

    baseline = summary[
        summary["recipe_key"].eq("A_median_mrmr_40")
        & summary["score_source"].eq("final_blend")
    ][["threshold", "f1_star_mean"]].rename(
        columns={"f1_star_mean": "baseline_final_blend_f1_star_same_threshold"}
    )
    summary = summary.merge(baseline, on="threshold", how="left")
    summary["delta_vs_mrmr_baseline_same_threshold"] = (
        summary["f1_star_mean"] - summary["baseline_final_blend_f1_star_same_threshold"]
    )

    return summary.sort_values(
        ["score_source", "f1_star_mean", "finals_f1_mean", "roc_auc_mean"],
        ascending=[True, False, False, False],
    )


def _select_recipes(recipe_arg):
    recipes = build_recipes()
    if recipe_arg == "all":
        return recipes
    wanted = {r.strip() for r in recipe_arg.split(",") if r.strip()}
    out = [r for r in recipes if r.key in wanted or r.key[0] in wanted]
    missing = sorted(wanted - {r.key for r in out} - {r.key[0] for r in out})
    if missing:
        raise ValueError(f"Unknown recipe selectors: {missing}")
    return out


def print_summary(summary):
    blend = summary[summary["score_source"].eq("final_blend")].copy()
    display = pd.concat(
        [
            blend[blend["threshold_kind"].eq("fixed")],
            blend.sort_values(
                ["recipe_key", "f1_star_mean", "finals_f1_mean"],
                ascending=[True, False, False],
            )
            .groupby("recipe_key", as_index=False, dropna=False)
            .head(1),
        ],
        ignore_index=True,
    ).drop_duplicates(["recipe_key", "threshold"])
    display = display.sort_values(
        ["f1_star_mean", "finals_f1_mean", "roc_auc_mean"],
        ascending=[False, False, False],
    )

    cols = [
        "recipe",
        "threshold",
        "threshold_kind",
        "f1_star_mean",
        "delta_vs_mrmr_baseline_same_threshold",
        "delta_vs_current_stable_f1_star",
        "delta_vs_best_calibration_grid_f1_star",
        "finals_f1_mean",
        "f1_positive_mean",
        "precision_mean",
        "recall_mean",
        "mcc_star_mean",
        "roc_auc_mean",
        "predicted_positives_mean",
        "n_features_mean",
    ]
    print("\nSorted final-blend summary by mean F1*:", flush=True)
    print(
        display[cols].to_string(
            index=False,
            formatters={
                "threshold": "{:.2f}".format,
                "f1_star_mean": "{:.4f}".format,
                "delta_vs_mrmr_baseline_same_threshold": "{:+.4f}".format,
                "delta_vs_current_stable_f1_star": "{:+.4f}".format,
                "delta_vs_best_calibration_grid_f1_star": "{:+.4f}".format,
                "finals_f1_mean": "{:.4f}".format,
                "f1_positive_mean": "{:.4f}".format,
                "precision_mean": "{:.4f}".format,
                "recall_mean": "{:.4f}".format,
                "mcc_star_mean": "{:.4f}".format,
                "roc_auc_mean": "{:.4f}".format,
                "predicted_positives_mean": "{:.2f}".format,
                "n_features_mean": "{:.1f}".format,
            },
        ),
        flush=True,
    )

    fixed = blend[blend["threshold"].eq(FIXED_THRESHOLD)].sort_values(
        ["f1_star_mean", "finals_f1_mean"], ascending=[False, False]
    )
    best = blend.iloc[0] if not blend.empty else None
    best_fixed = fixed.iloc[0] if not fixed.empty else None
    baseline_fixed = fixed[fixed["recipe_key"].eq("A_median_mrmr_40")]

    if best is not None:
        print(
            f"\nBest final blend overall: {best.recipe} threshold={best.threshold:.2f} "
            f"F1*={best.f1_star_mean:.4f}",
            flush=True,
        )
    if best_fixed is not None:
        print(
            f"Best final blend at fixed threshold {FIXED_THRESHOLD:.2f}: "
            f"{best_fixed.recipe} F1*={best_fixed.f1_star_mean:.4f}",
            flush=True,
        )
    if not baseline_fixed.empty and best_fixed is not None:
        base = baseline_fixed.iloc[0]
        print(
            f"Fixed-threshold delta vs current mRMR baseline: "
            f"{best_fixed.f1_star_mean - base.f1_star_mean:+.4f}",
            flush=True,
        )


def run(recipe_arg="all", seeds=None, n_repeats=6, n_splits=5, n_own_seeds=5, include_grid=True):
    os.makedirs(OUT, exist_ok=True)
    if seeds is None:
        seeds = [SEED]
    recipes = _select_recipes(recipe_arg)
    thresholds = [FIXED_THRESHOLD]
    if include_grid:
        thresholds = sorted(set(thresholds) | set(float(t) for t in GRID_THRESHOLDS))

    print(
        f"\n=== {PANEL} own-model feature-selection comparison "
        f"(w={WEIGHT:.1f}, BLEND_MODE=prob, seeds={seeds}) ===",
        flush=True,
    )
    print(
        "Transfer scorer: existing general-minus-cancer BLEND "
        "(LGBM + RF + CatBoost probability average)",
        flush=True,
    )
    print("Own recipes:", flush=True)
    for recipe in recipes:
        print(f"  {recipe.key}: {recipe.label}", flush=True)
    print(
        f"Final-blend thresholds: fixed {FIXED_THRESHOLD:.2f}"
        + (f" plus grid {GRID_THRESHOLDS[0]:.2f}-{GRID_THRESHOLDS[-1]:.2f}" if include_grid else ""),
        flush=True,
    )

    t0 = time.time()
    g_all, yt_g = build_probability_blend_scorer(PANEL)
    Xt_own, yt = load_target_engineered(PANEL)
    if not (yt.values == yt_g.values).all():
        raise RuntimeError("Target labels from transfer and own matrices do not align.")
    if "Variant_ID" in Xt_own.columns:
        raise RuntimeError("Variant_ID unexpectedly present in cancer-own features.")
    print(
        f"Raw transfer BLEND AUC on cancer rows: {roc_auc_score(yt, g_all):.4f} "
        f"({time.time() - t0:.1f}s)",
        flush=True,
    )

    fold_tables = []
    for seed in seeds:
        fold_tables.append(
            evaluate_seed(
                g_all,
                Xt_own,
                yt,
                recipes,
                thresholds,
                cv_seed=seed,
                n_repeats=n_repeats,
                n_splits=n_splits,
                n_own_seeds=n_own_seeds,
            )
        )

    fold_rows = pd.concat(fold_tables, ignore_index=True)
    summary = summarize(fold_rows)
    out_csv = os.path.join(OUT, "cancer_own_feature_selection_compare.csv")
    summary.to_csv(out_csv, index=False)
    print_summary(summary)
    print(f"\nSaved: {out_csv}", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seeds", default=str(SEED))
    parser.add_argument("--n-own-seeds", type=int, default=5)
    parser.add_argument(
        "--recipes",
        default="all",
        help="all, comma-separated recipe keys, or A/B/C/D family letters",
    )
    parser.add_argument("--no-grid", action="store_true")
    args = parser.parse_args()

    run(
        recipe_arg=args.recipes,
        seeds=[int(s.strip()) for s in args.seeds.split(",") if s.strip()],
        n_repeats=args.repeats,
        n_splits=args.splits,
        n_own_seeds=args.n_own_seeds,
        include_grid=not args.no_grid,
    )
