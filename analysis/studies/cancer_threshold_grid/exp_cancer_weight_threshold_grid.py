"""Cancer transfer-blend weight/threshold grid.

This is a controlled follow-up to exp_transfer_blend_validate.py. It keeps the
same cancer-own deployed pipeline and the same general-minus-cancer BLEND transfer
scorer, then evaluates a finer blend-weight grid with threshold selection inside
each repeated-CV training fold.
"""
import os
import sys
import time
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "6")
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold

from config import SEED
from utils import TEST_PRIORS
from harness import (
    FINALS_N,
    bootstrap_f1_mcc,
    f1_at_prior,
    mcc_at_prior_analytic,
    tpr_fpr,
)
from exp_transfer import load_target_engineered, target_deployed_fn
from exp_transfer_blend import build_matrices, cat_score, lgb_score, rf_score
from calibration import apply_platt, fit_platt


OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
PANEL = "cancer"
WEIGHTS = np.round(np.arange(0.0, 1.0 + 0.001, 0.1), 1)
THRESHOLDS = np.round(np.arange(0.40, 0.95 + 0.001, 0.01), 2)
FIXED_THRESHOLDS = [0.71, 0.75, 0.78, 0.79, 0.80, 0.82]
STABILITY_THRESHOLDS = [0.71, 0.78, 0.79, 0.80, 0.81, 0.82, 0.83, 0.84, 0.85]
REPRO_F1_STAR = 0.7077
REPRO_FINALS_F1 = 0.7086
OPTIMIZED_GRID_F1_STAR = 0.7117


def pick_threshold_on_train(y_true, probs, prior):
    """Pick the threshold maximizing analytic F1* on train-side probabilities."""
    best_t, best_f1 = float(THRESHOLDS[0]), -1.0
    for t in THRESHOLDS:
        tpr, fpr = tpr_fpr(y_true, (probs >= t).astype(int))
        f1_star = f1_at_prior(tpr, fpr, prior)
        if f1_star > best_f1:
            best_t, best_f1 = float(t), float(f1_star)
    return best_t, best_f1


def safe_auc(y_true, probs):
    try:
        return float(roc_auc_score(y_true, probs))
    except ValueError:
        return np.nan


def build_probability_blend_scorer(panel):
    """Reproduce the BLEND transfer scorer in probability-averaging mode."""
    mode = os.environ.get("BLEND_MODE", "prob")
    if mode != "prob":
        raise ValueError(
            "This experiment is defined for BLEND_MODE=prob. "
            f"Got BLEND_MODE={mode!r}."
        )

    Xg, yg, Xt, yt_g, cats = build_matrices(panel)
    g_lgb = lgb_score(Xg, yg, Xt, cats)
    g_rf = rf_score(Xg, yg, Xt, cats)
    g_cat = cat_score(Xg, yg, Xt, cats)
    return (g_lgb + g_rf + g_cat) / 3.0, yt_g


def evaluate_fixed_thresholds(
    g_all,
    Xt_own,
    yt,
    thresholds,
    *,
    cv_seed=SEED,
    n_repeats=6,
    n_splits=5,
    progress_label="fixed thresholds",
):
    prior = TEST_PRIORS[PANEL]
    n_target = FINALS_N[PANEL]
    weight = 0.5
    own_fn = target_deployed_fn(PANEL)
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=cv_seed
    )
    per_threshold = {
        float(t): {
            "f1_positive": [],
            "f1_star": [],
            "precision": [],
            "recall": [],
            "mcc": [],
            "mcc_star": [],
            "roc_auc": [],
            "finals_f1": [],
            "finals_mcc": [],
            "predicted_positives": [],
        }
        for t in thresholds
    }

    splits = list(rskf.split(Xt_own, yt))
    print(f"Running {len(splits)} paired outer folds for {progress_label}...", flush=True)

    for fold_idx, (tr, te) in enumerate(splits, start=1):
        tr, te = np.asarray(tr), np.asarray(te)
        ytr = yt.iloc[tr].reset_index(drop=True)
        yte = yt.iloc[te].reset_index(drop=True)

        own_te, _ = own_fn(
            Xt_own.iloc[tr].reset_index(drop=True),
            ytr,
            Xt_own.iloc[te].reset_index(drop=True),
        )

        platt = fit_platt(g_all[tr], ytr.values)
        gen_te = apply_platt(g_all[te], platt)
        blend_te = weight * gen_te + (1.0 - weight) * own_te

        for threshold, bucket in per_threshold.items():
            preds = (blend_te >= threshold).astype(int)
            tpr, fpr = tpr_fpr(yte.values, preds)
            finals_f1, finals_mcc = bootstrap_f1_mcc(
                yte.values, preds, prior, n_target
            )

            bucket["f1_positive"].append(
                f1_score(yte.values, preds, zero_division=0)
            )
            bucket["f1_star"].append(f1_at_prior(tpr, fpr, prior))
            bucket["precision"].append(
                precision_score(yte.values, preds, zero_division=0)
            )
            bucket["recall"].append(recall_score(yte.values, preds, zero_division=0))
            bucket["mcc"].append(matthews_corrcoef(yte.values, preds))
            bucket["mcc_star"].append(mcc_at_prior_analytic(tpr, fpr, prior))
            bucket["roc_auc"].append(safe_auc(yte.values, blend_te))
            bucket["finals_f1"].append(finals_f1)
            bucket["finals_mcc"].append(finals_mcc)
            bucket["predicted_positives"].append(int(preds.sum()))

        if fold_idx % n_splits == 0:
            print(f"  completed {fold_idx}/{len(splits)} folds", flush=True)

    rows = []
    for threshold, vals in per_threshold.items():
        rows.append(
            {
                "cv_seed": cv_seed,
                "threshold": threshold,
                "n_folds": len(vals["f1_star"]),
                "f1_positive": float(np.mean(vals["f1_positive"])),
                "f1_star": float(np.mean(vals["f1_star"])),
                "f1_star_se": float(
                    np.std(vals["f1_star"], ddof=1) / np.sqrt(len(vals["f1_star"]))
                ),
                "finals_f1": float(np.mean(vals["finals_f1"])),
                "precision": float(np.mean(vals["precision"])),
                "recall": float(np.mean(vals["recall"])),
                "mcc": float(np.mean(vals["mcc"])),
                "mcc_star": float(np.mean(vals["mcc_star"])),
                "roc_auc": float(np.nanmean(vals["roc_auc"])),
                "predicted_positives": float(np.mean(vals["predicted_positives"])),
                "predicted_positives_total": int(np.sum(vals["predicted_positives"])),
            }
        )
    return pd.DataFrame(rows)


def run(n_repeats=6, n_splits=5):
    os.makedirs(OUT, exist_ok=True)
    prior = TEST_PRIORS[PANEL]
    n_target = FINALS_N[PANEL]

    print(
        f"\n=== {PANEL} weight/threshold grid "
        f"(prior {prior:.3f}, BLEND_MODE=prob) ===",
        flush=True,
    )
    print(
        "Transfer scorer: general-minus-cancer BLEND "
        "(LGBM + RF + CatBoost probability average)",
        flush=True,
    )
    print(
        f"Weights: {WEIGHTS[0]:.1f}..{WEIGHTS[-1]:.1f} step 0.1 | "
        f"thresholds: {THRESHOLDS[0]:.2f}..{THRESHOLDS[-1]:.2f} step 0.01",
        flush=True,
    )

    t0 = time.time()
    g_all, yt_g = build_probability_blend_scorer(PANEL)
    Xt_own, yt = load_target_engineered(PANEL)
    if not (yt.values == yt_g.values).all():
        raise RuntimeError("Target labels from transfer and own matrices do not align.")

    print(
        f"Transfer BLEND AUC on cancer rows: {roc_auc_score(yt, g_all):.4f} "
        f"({time.time() - t0:.1f}s)",
        flush=True,
    )

    own_fn = target_deployed_fn(PANEL)
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=SEED
    )
    per_weight = {
        float(w): {
            "threshold": [],
            "train_f1_star": [],
            "f1_positive": [],
            "f1_star": [],
            "precision": [],
            "recall": [],
            "mcc": [],
            "mcc_star": [],
            "roc_auc": [],
            "finals_f1": [],
            "finals_mcc": [],
        }
        for w in WEIGHTS
    }

    splits = list(rskf.split(Xt_own, yt))
    print(f"Running {len(splits)} paired outer folds...", flush=True)

    for fold_idx, (tr, te) in enumerate(splits, start=1):
        tr, te = np.asarray(tr), np.asarray(te)
        ytr = yt.iloc[tr].reset_index(drop=True)
        yte = yt.iloc[te].reset_index(drop=True)

        own_te, own_tr = own_fn(
            Xt_own.iloc[tr].reset_index(drop=True),
            ytr,
            Xt_own.iloc[te].reset_index(drop=True),
        )

        platt = fit_platt(g_all[tr], ytr.values)
        gen_tr = apply_platt(g_all[tr], platt)
        gen_te = apply_platt(g_all[te], platt)

        for w in WEIGHTS:
            w = float(w)
            blend_tr = w * gen_tr + (1.0 - w) * own_tr
            blend_te = w * gen_te + (1.0 - w) * own_te
            threshold, train_f1_star = pick_threshold_on_train(
                ytr.values, blend_tr, prior
            )
            preds = (blend_te >= threshold).astype(int)
            tpr, fpr = tpr_fpr(yte.values, preds)
            finals_f1, finals_mcc = bootstrap_f1_mcc(
                yte.values, preds, prior, n_target
            )

            bucket = per_weight[w]
            bucket["threshold"].append(threshold)
            bucket["train_f1_star"].append(train_f1_star)
            bucket["f1_positive"].append(
                f1_score(yte.values, preds, zero_division=0)
            )
            bucket["f1_star"].append(f1_at_prior(tpr, fpr, prior))
            bucket["precision"].append(
                precision_score(yte.values, preds, zero_division=0)
            )
            bucket["recall"].append(recall_score(yte.values, preds, zero_division=0))
            bucket["mcc"].append(matthews_corrcoef(yte.values, preds))
            bucket["mcc_star"].append(mcc_at_prior_analytic(tpr, fpr, prior))
            bucket["roc_auc"].append(safe_auc(yte.values, blend_te))
            bucket["finals_f1"].append(finals_f1)
            bucket["finals_mcc"].append(finals_mcc)

        if fold_idx % n_splits == 0:
            print(f"  completed {fold_idx}/{len(splits)} folds", flush=True)

    rows = []
    base_f1 = float(np.mean(per_weight[0.0]["f1_star"]))
    for w, vals in per_weight.items():
        row = {
            "panel": PANEL,
            "weight": w,
            "n_folds": len(vals["f1_star"]),
            "threshold_mean": float(np.mean(vals["threshold"])),
            "threshold_median": float(np.median(vals["threshold"])),
            "threshold_min": float(np.min(vals["threshold"])),
            "threshold_max": float(np.max(vals["threshold"])),
            "train_f1_star_mean": float(np.mean(vals["train_f1_star"])),
            "f1_positive_mean": float(np.mean(vals["f1_positive"])),
            "f1_star_mean": float(np.mean(vals["f1_star"])),
            "f1_star_se": float(
                np.std(vals["f1_star"], ddof=1) / np.sqrt(len(vals["f1_star"]))
            ),
            "precision_mean": float(np.mean(vals["precision"])),
            "recall_mean": float(np.mean(vals["recall"])),
            "mcc_mean": float(np.mean(vals["mcc"])),
            "mcc_star_mean": float(np.mean(vals["mcc_star"])),
            "roc_auc_mean": float(np.nanmean(vals["roc_auc"])),
            "finals_f1_mean": float(np.mean(vals["finals_f1"])),
            "finals_mcc_mean": float(np.mean(vals["finals_mcc"])),
            "delta_vs_w0_f1_star": float(np.mean(vals["f1_star"]) - base_f1),
            "delta_vs_reproduced_f1_star": float(
                np.mean(vals["f1_star"]) - REPRO_F1_STAR
            ),
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(
        ["f1_star_mean", "finals_f1_mean", "roc_auc_mean"],
        ascending=[False, False, False],
    )
    out_csv = os.path.join(OUT, "cancer_weight_threshold_grid.csv")
    df.to_csv(out_csv, index=False)

    print("\nSorted summary by F1*:", flush=True)
    display_cols = [
        "weight",
        "threshold_mean",
        "threshold_median",
        "f1_star_mean",
        "finals_f1_mean",
        "f1_positive_mean",
        "precision_mean",
        "recall_mean",
        "mcc_star_mean",
        "roc_auc_mean",
        "delta_vs_reproduced_f1_star",
    ]
    print(
        df[display_cols].to_string(
            index=False,
            formatters={
                "weight": "{:.1f}".format,
                "threshold_mean": "{:.3f}".format,
                "threshold_median": "{:.2f}".format,
                "f1_star_mean": "{:.4f}".format,
                "finals_f1_mean": "{:.4f}".format,
                "f1_positive_mean": "{:.4f}".format,
                "precision_mean": "{:.4f}".format,
                "recall_mean": "{:.4f}".format,
                "mcc_star_mean": "{:.4f}".format,
                "roc_auc_mean": "{:.4f}".format,
                "delta_vs_reproduced_f1_star": "{:+.4f}".format,
            },
        ),
        flush=True,
    )

    w05 = df.loc[np.isclose(df["weight"], 0.5)].iloc[0]
    best = df.iloc[0]
    print(
        f"\nReproduced reference: w=0.5 F1*={REPRO_F1_STAR:.4f} "
        f"finals={REPRO_FINALS_F1:.4f}",
        flush=True,
    )
    print(
        f"Grid w=0.5: threshold_mean={w05.threshold_mean:.3f} "
        f"threshold_median={w05.threshold_median:.2f} "
        f"F1*={w05.f1_star_mean:.4f} finals={w05.finals_f1_mean:.4f} "
        f"delta={w05.delta_vs_reproduced_f1_star:+.4f}",
        flush=True,
    )
    print(
        f"Best grid row: w={best.weight:.1f} "
        f"threshold_mean={best.threshold_mean:.3f} "
        f"threshold_median={best.threshold_median:.2f} "
        f"F1*={best.f1_star_mean:.4f} finals={best.finals_f1_mean:.4f} "
        f"delta={best.delta_vs_reproduced_f1_star:+.4f}",
        flush=True,
    )
    print(f"Saved: {out_csv}", flush=True)
    return df


def run_fixed_threshold_comparison(n_repeats=6, n_splits=5):
    os.makedirs(OUT, exist_ok=True)
    prior = TEST_PRIORS[PANEL]
    n_target = FINALS_N[PANEL]
    weight = 0.5

    print(
        f"\n=== {PANEL} fixed-threshold comparison "
        f"(w={weight:.1f}, prior {prior:.3f}, BLEND_MODE=prob) ===",
        flush=True,
    )
    print(
        "Transfer scorer: general-minus-cancer BLEND "
        "(LGBM + RF + CatBoost probability average)",
        flush=True,
    )
    print(
        "Fixed thresholds: " + ", ".join(f"{t:.2f}" for t in FIXED_THRESHOLDS),
        flush=True,
    )

    t0 = time.time()
    g_all, yt_g = build_probability_blend_scorer(PANEL)
    Xt_own, yt = load_target_engineered(PANEL)
    if not (yt.values == yt_g.values).all():
        raise RuntimeError("Target labels from transfer and own matrices do not align.")

    print(
        f"Transfer BLEND AUC on cancer rows: {roc_auc_score(yt, g_all):.4f} "
        f"({time.time() - t0:.1f}s)",
        flush=True,
    )

    fixed = evaluate_fixed_thresholds(
        g_all,
        Xt_own,
        yt,
        FIXED_THRESHOLDS,
        cv_seed=SEED,
        n_repeats=n_repeats,
        n_splits=n_splits,
    )

    rows = []
    for _, vals in fixed.iterrows():
        rows.append(
            {
                "panel": PANEL,
                "method": "fixed_threshold",
                "weight": weight,
                "threshold": float(vals["threshold"]),
                "n_folds": int(vals["n_folds"]),
                "f1_positive_mean": float(vals["f1_positive"]),
                "f1_star_mean": float(vals["f1_star"]),
                "f1_star_se": float(vals["f1_star_se"]),
                "finals_f1_mean": float(vals["finals_f1"]),
                "precision_mean": float(vals["precision"]),
                "recall_mean": float(vals["recall"]),
                "mcc_mean": float(vals["mcc"]),
                "mcc_star_mean": float(vals["mcc_star"]),
                "roc_auc_mean": float(vals["roc_auc"]),
                "predicted_positives_mean": float(vals["predicted_positives"]),
                "predicted_positives_total": int(vals["predicted_positives_total"]),
            }
        )

    previous_path = os.path.join(OUT, "cancer_weight_threshold_grid.csv")
    previous = None
    if os.path.exists(previous_path):
        prev = pd.read_csv(previous_path)
        hit = prev[np.isclose(prev["weight"], weight)]
        if not hit.empty:
            previous = hit.iloc[0]

    rows.append(
        {
            "panel": PANEL,
            "method": "optimized_threshold_grid_previous",
            "weight": weight,
            "threshold": np.nan,
            "n_folds": int(previous["n_folds"]) if previous is not None else np.nan,
            "f1_positive_mean": (
                float(previous["f1_positive_mean"]) if previous is not None else np.nan
            ),
            "f1_star_mean": (
                float(previous["f1_star_mean"])
                if previous is not None
                else OPTIMIZED_GRID_F1_STAR
            ),
            "f1_star_se": float(previous["f1_star_se"]) if previous is not None else np.nan,
            "finals_f1_mean": (
                float(previous["finals_f1_mean"]) if previous is not None else np.nan
            ),
            "precision_mean": (
                float(previous["precision_mean"]) if previous is not None else np.nan
            ),
            "recall_mean": float(previous["recall_mean"]) if previous is not None else np.nan,
            "mcc_mean": float(previous["mcc_mean"]) if previous is not None else np.nan,
            "mcc_star_mean": (
                float(previous["mcc_star_mean"]) if previous is not None else np.nan
            ),
            "roc_auc_mean": (
                float(previous["roc_auc_mean"]) if previous is not None else np.nan
            ),
            "predicted_positives_mean": np.nan,
            "predicted_positives_total": np.nan,
        }
    )

    df = pd.DataFrame(rows).sort_values(
        ["f1_star_mean", "finals_f1_mean"],
        ascending=[False, False],
        na_position="last",
    )
    out_csv = os.path.join(OUT, "cancer_fixed_threshold_comparison.csv")
    df.to_csv(out_csv, index=False)

    print("\nFixed-threshold summary by F1*:", flush=True)
    print(
        df[
            [
                "method",
                "threshold",
                "f1_star_mean",
                "finals_f1_mean",
                "f1_positive_mean",
                "precision_mean",
                "recall_mean",
                "mcc_star_mean",
                "predicted_positives_mean",
                "predicted_positives_total",
            ]
        ].to_string(
            index=False,
            formatters={
                "threshold": lambda x: "" if pd.isna(x) else f"{x:.2f}",
                "f1_star_mean": "{:.4f}".format,
                "finals_f1_mean": lambda x: "" if pd.isna(x) else f"{x:.4f}",
                "f1_positive_mean": lambda x: "" if pd.isna(x) else f"{x:.4f}",
                "precision_mean": lambda x: "" if pd.isna(x) else f"{x:.4f}",
                "recall_mean": lambda x: "" if pd.isna(x) else f"{x:.4f}",
                "mcc_star_mean": lambda x: "" if pd.isna(x) else f"{x:.4f}",
                "predicted_positives_mean": lambda x: "" if pd.isna(x) else f"{x:.2f}",
                "predicted_positives_total": lambda x: "" if pd.isna(x) else f"{int(x)}",
            },
        ),
        flush=True,
    )
    print(f"\nPrevious optimized-threshold grid F1*={OPTIMIZED_GRID_F1_STAR:.4f}", flush=True)
    print(f"Saved: {out_csv}", flush=True)
    return df


def run_threshold_stability(seeds, n_repeats=6, n_splits=5):
    os.makedirs(OUT, exist_ok=True)
    prior = TEST_PRIORS[PANEL]
    weight = 0.5

    print(
        f"\n=== {PANEL} threshold stability "
        f"(w={weight:.1f}, prior {prior:.3f}, BLEND_MODE=prob) ===",
        flush=True,
    )
    print(
        "Transfer scorer: general-minus-cancer BLEND "
        "(LGBM + RF + CatBoost probability average)",
        flush=True,
    )
    print("CV seeds: " + ", ".join(str(s) for s in seeds), flush=True)
    print(
        "Thresholds: " + ", ".join(f"{t:.2f}" for t in STABILITY_THRESHOLDS),
        flush=True,
    )

    t0 = time.time()
    g_all, yt_g = build_probability_blend_scorer(PANEL)
    Xt_own, yt = load_target_engineered(PANEL)
    if not (yt.values == yt_g.values).all():
        raise RuntimeError("Target labels from transfer and own matrices do not align.")

    print(
        f"Transfer BLEND AUC on cancer rows: {roc_auc_score(yt, g_all):.4f} "
        f"({time.time() - t0:.1f}s)",
        flush=True,
    )

    run_rows = []
    for i, seed in enumerate(seeds, start=1):
        print(f"\n--- stability run {i}/{len(seeds)}: cv_seed={seed} ---", flush=True)
        run_rows.append(
            evaluate_fixed_thresholds(
                g_all,
                Xt_own,
                yt,
                STABILITY_THRESHOLDS,
                cv_seed=seed,
                n_repeats=n_repeats,
                n_splits=n_splits,
                progress_label=f"cv_seed={seed}",
            )
        )

    per_run = pd.concat(run_rows, ignore_index=True)
    metric_cols = [
        "f1_star",
        "finals_f1",
        "f1_positive",
        "precision",
        "recall",
        "mcc_star",
        "predicted_positives",
    ]
    summary = per_run.groupby("threshold", as_index=False).agg(
        n_runs=("cv_seed", "nunique"),
        n_folds_per_run=("n_folds", "first"),
        **{
            f"{metric}_mean": (metric, "mean")
            for metric in metric_cols
        },
        **{
            f"{metric}_std": (metric, "std")
            for metric in metric_cols
        },
    )
    summary.insert(0, "panel", PANEL)
    summary.insert(2, "weight", weight)
    summary["included_for_comparison"] = summary["threshold"].eq(0.71)
    summary = summary.sort_values(
        ["f1_star_mean", "finals_f1_mean"],
        ascending=[False, False],
    )

    out_csv = os.path.join(OUT, "cancer_threshold_stability.csv")
    summary.to_csv(out_csv, index=False)

    print("\nThreshold stability summary by mean F1*:", flush=True)
    display_cols = [
        "threshold",
        "f1_star_mean",
        "f1_star_std",
        "finals_f1_mean",
        "finals_f1_std",
        "f1_positive_mean",
        "f1_positive_std",
        "precision_mean",
        "recall_mean",
        "mcc_star_mean",
        "predicted_positives_mean",
        "predicted_positives_std",
    ]
    print(
        summary[display_cols].to_string(
            index=False,
            formatters={
                "threshold": "{:.2f}".format,
                "f1_star_mean": "{:.4f}".format,
                "f1_star_std": "{:.4f}".format,
                "finals_f1_mean": "{:.4f}".format,
                "finals_f1_std": "{:.4f}".format,
                "f1_positive_mean": "{:.4f}".format,
                "f1_positive_std": "{:.4f}".format,
                "precision_mean": "{:.4f}".format,
                "recall_mean": "{:.4f}".format,
                "mcc_star_mean": "{:.4f}".format,
                "predicted_positives_mean": "{:.2f}".format,
                "predicted_positives_std": "{:.2f}".format,
            },
        ),
        flush=True,
    )

    def f1_for(threshold):
        hit = summary[np.isclose(summary["threshold"], threshold)]
        return None if hit.empty else float(hit.iloc[0]["f1_star_mean"])

    f082 = f1_for(0.82)
    comparisons = []
    for other in (0.71, 0.78, 0.80):
        f_other = f1_for(other)
        if f082 is not None and f_other is not None:
            comparisons.append(f"0.82 vs {other:.2f}: {f082 - f_other:+.4f}")
    print("\nKey comparisons: " + " | ".join(comparisons), flush=True)
    print(f"Saved: {out_csv}", flush=True)
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--fixed-thresholds", action="store_true")
    parser.add_argument("--stability", action="store_true")
    parser.add_argument("--seeds", default="42,43,44")
    args = parser.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if args.stability:
        run_threshold_stability(seeds, n_repeats=args.repeats, n_splits=args.splits)
    elif args.fixed_thresholds:
        run_fixed_threshold_comparison(n_repeats=args.repeats, n_splits=args.splits)
    else:
        run(n_repeats=args.repeats, n_splits=args.splits)
