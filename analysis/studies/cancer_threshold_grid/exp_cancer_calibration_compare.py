"""Cancer calibration-method comparison for the fixed transfer blend.

Controlled setup:
  * cancer-own deployed model + general-minus-cancer BLEND transfer scorer
  * BLEND_MODE=prob
  * fixed blend weight w=0.5
  * calibration fitted inside each outer training fold only
  * no Variant_ID feature use
"""
import os
import sys
import time
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "6")
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines")
)

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold

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
from run_pipeline import get_model_params, preprocess_data
from utils import TEST_PRIORS


OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
PANEL = "cancer"
WEIGHT = 0.5
CURRENT_BEST_STABLE_F1_STAR = 0.7136
FIXED_THRESHOLDS = [0.82, 0.80]
GRID_THRESHOLDS = np.round(np.arange(0.70, 0.90 + 0.001, 0.01), 2)
DEFAULT_METHODS = ["platt", "none", "isotonic", "temperature"]
METHOD_LABELS = {
    "platt": "platt_sigmoid",
    "none": "no_calibration",
    "isotonic": "isotonic",
    "temperature": "temperature_scaling",
}

_EPS = 1e-7


def _logit(probs):
    probs = np.clip(np.asarray(probs, dtype=float), _EPS, 1.0 - _EPS)
    return np.log(probs / (1.0 - probs))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _fit_temperature(probs, y):
    """Fit a scalar temperature by grid-search NLL, dependency-free."""
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=int)
    logits = _logit(probs)
    best_t, best_loss = 1.0, np.inf
    for t in np.linspace(0.25, 5.0, 192):
        pred = np.clip(_sigmoid(logits / t), _EPS, 1.0 - _EPS)
        loss = -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))
        if loss < best_loss:
            best_t, best_loss = float(t), float(loss)
    return {"temperature": best_t}


def _apply_temperature(probs, fitted):
    return _sigmoid(_logit(probs) / fitted["temperature"])


def fit_calibrator(method, probs, y):
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=int)
    if method == "none":
        return {"method": method}
    if method == "platt":
        return {"method": method, "fitted": fit_platt(probs, y)}
    if method == "isotonic":
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(probs, y)
        return {"method": method, "fitted": iso}
    if method == "temperature":
        return {"method": method, "fitted": _fit_temperature(probs, y)}
    raise ValueError(f"Unknown calibration method: {method}")


def apply_calibrator(probs, calibrator):
    method = calibrator["method"]
    probs = np.asarray(probs, dtype=float)
    if method == "none":
        return probs
    if method == "platt":
        return apply_platt(probs, calibrator["fitted"])
    if method == "isotonic":
        return calibrator["fitted"].predict(probs)
    if method == "temperature":
        return _apply_temperature(probs, calibrator["fitted"])
    raise ValueError(f"Unknown calibration method: {method}")


def safe_auc(y_true, probs):
    try:
        return float(roc_auc_score(y_true, probs))
    except ValueError:
        return np.nan


def cancer_own_raw_fn(X_tr, y_tr, X_te):
    """Cancer deployed model through raw probabilities before final calibration."""
    Xtr, Xte = _drop_leaky(X_tr, y_tr, X_te)
    Xtr_pp, _, Xte_pp, _ = preprocess_data(
        Xtr.copy(), Xtr.copy(), Xte.copy(), corr_threshold=0.95
    )
    saf = fit_score_agreement(Xtr_pp, y_tr, top_k=7)
    Xtr_pp = apply_score_agreement(Xtr_pp, saf)
    Xte_pp = apply_score_agreement(Xte_pp, saf)
    Xtr_pp, _, Xte_pp, _ = fast_mrmr(Xtr_pp, y_tr, Xte_pp, Xte_pp)
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    params = get_model_params(len(X_tr), spw)
    nf = 3 if len(X_tr) < 200 else 5
    test_probs, oof = lgb_oof_and_test(
        Xtr_pp, y_tr, Xte_pp, params, n_seeds=5, n_folds=nf
    )
    return np.asarray(test_probs, dtype=float), np.asarray(oof, dtype=float)


def metric_row(y_true, probs, threshold, prior, n_target):
    preds = (probs >= threshold).astype(int)
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


def evaluate_seed(g_all, Xt_own, yt, methods, thresholds, cv_seed, n_repeats, n_splits):
    prior = TEST_PRIORS[PANEL]
    n_target = FINALS_N[PANEL]
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=cv_seed
    )
    splits = list(rskf.split(Xt_own, yt))
    rows = []

    print(
        f"Running cv_seed={cv_seed}: {len(splits)} paired outer folds, "
        f"{len(methods)} calibration methods...",
        flush=True,
    )
    for fold_idx, (tr, te) in enumerate(splits, start=1):
        tr, te = np.asarray(tr), np.asarray(te)
        ytr = yt.iloc[tr].reset_index(drop=True)
        yte = yt.iloc[te].reset_index(drop=True)
        Xtr = Xt_own.iloc[tr].reset_index(drop=True)
        Xte = Xt_own.iloc[te].reset_index(drop=True)

        own_te_raw, own_tr_oof_raw = cancer_own_raw_fn(Xtr, ytr, Xte)
        gen_tr_raw = np.asarray(g_all[tr], dtype=float)
        gen_te_raw = np.asarray(g_all[te], dtype=float)

        for method in methods:
            own_cal = fit_calibrator(method, own_tr_oof_raw, ytr.values)
            gen_cal = fit_calibrator(method, gen_tr_raw, ytr.values)
            own_te = apply_calibrator(own_te_raw, own_cal)
            gen_te = apply_calibrator(gen_te_raw, gen_cal)
            blend_te = WEIGHT * gen_te + (1.0 - WEIGHT) * own_te

            for threshold in thresholds:
                values = metric_row(yte.values, blend_te, threshold, prior, n_target)
                rows.append(
                    {
                        "panel": PANEL,
                        "cv_seed": cv_seed,
                        "fold": fold_idx,
                        "method": METHOD_LABELS[method],
                        "method_key": method,
                        "weight": WEIGHT,
                        "threshold": float(threshold),
                        "threshold_kind": (
                            "fixed" if float(threshold) in FIXED_THRESHOLDS else "grid"
                        ),
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
    seed_summary = (
        fold_rows.groupby(
            ["panel", "method", "method_key", "weight", "threshold", "threshold_kind", "cv_seed"],
            as_index=False,
        )
        .agg(
            n_folds=("fold", "nunique"),
            predicted_positives_total=("predicted_positives", "sum"),
            **{f"{col}_mean": (col, "mean") for col in metric_cols},
        )
    )
    summary = (
        seed_summary.groupby(
            ["panel", "method", "method_key", "weight", "threshold", "threshold_kind"],
            as_index=False,
        )
        .agg(
            n_runs=("cv_seed", "nunique"),
            n_folds_per_run=("n_folds", "first"),
            predicted_positives_total=("predicted_positives_total", "sum"),
            **{f"{col}_mean": (f"{col}_mean", "mean") for col in metric_cols},
            **{f"{col}_std_across_seeds": (f"{col}_mean", "std") for col in metric_cols},
        )
    )

    fold_stats = (
        fold_rows.groupby(
            ["panel", "method", "method_key", "weight", "threshold", "threshold_kind"],
            as_index=False,
        )
        .agg(
            n_folds_total=("fold", "count"),
            f1_star_se_across_folds=("f1_star", lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
        )
    )
    summary = summary.merge(
        fold_stats,
        on=["panel", "method", "method_key", "weight", "threshold", "threshold_kind"],
        how="left",
    )
    summary["delta_vs_current_best_stable_f1_star"] = (
        summary["f1_star_mean"] - CURRENT_BEST_STABLE_F1_STAR
    )
    return summary.sort_values(
        ["f1_star_mean", "finals_f1_mean", "roc_auc_mean"],
        ascending=[False, False, False],
    )


def run(methods=None, seeds=None, n_repeats=6, n_splits=5, include_grid=True):
    os.makedirs(OUT, exist_ok=True)
    if methods is None:
        methods = DEFAULT_METHODS
    if seeds is None:
        seeds = [SEED, SEED + 1, SEED + 2]
    bad = sorted(set(methods) - set(DEFAULT_METHODS))
    if bad:
        raise ValueError(f"Unknown methods: {bad}")

    thresholds = sorted(set(FIXED_THRESHOLDS) | (set(GRID_THRESHOLDS) if include_grid else set()))
    print(
        f"\n=== {PANEL} calibration comparison "
        f"(w={WEIGHT:.1f}, BLEND_MODE=prob, seeds={seeds}) ===",
        flush=True,
    )
    print(
        "Transfer scorer: general-minus-cancer BLEND "
        "(LGBM + RF + CatBoost probability average)",
        flush=True,
    )
    print(
        "Calibration methods: " + ", ".join(METHOD_LABELS[m] for m in methods),
        flush=True,
    )
    print(
        "Thresholds: " + ", ".join(f"{t:.2f}" for t in thresholds),
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
                methods,
                thresholds,
                cv_seed=seed,
                n_repeats=n_repeats,
                n_splits=n_splits,
            )
        )

    fold_rows = pd.concat(fold_tables, ignore_index=True)
    summary = summarize(fold_rows)
    out_csv = os.path.join(OUT, "cancer_calibration_compare.csv")
    summary.to_csv(out_csv, index=False)

    fixed = summary[summary["threshold_kind"].eq("fixed")].copy()
    best_grid = (
        summary.sort_values(
            ["method", "f1_star_mean", "finals_f1_mean"],
            ascending=[True, False, False],
        )
        .groupby("method", as_index=False)
        .head(1)
    )
    display = pd.concat([fixed, best_grid], ignore_index=True).drop_duplicates(
        ["method", "threshold"]
    )
    display = display.sort_values(
        ["f1_star_mean", "finals_f1_mean"], ascending=[False, False]
    )

    print("\nSorted summary by mean F1* (fixed thresholds plus best grid row per method):", flush=True)
    display_cols = [
        "method",
        "threshold",
        "threshold_kind",
        "f1_star_mean",
        "delta_vs_current_best_stable_f1_star",
        "finals_f1_mean",
        "f1_positive_mean",
        "precision_mean",
        "recall_mean",
        "mcc_star_mean",
        "roc_auc_mean",
        "predicted_positives_mean",
        "predicted_positives_total",
    ]
    print(
        display[display_cols].to_string(
            index=False,
            formatters={
                "threshold": "{:.2f}".format,
                "f1_star_mean": "{:.4f}".format,
                "delta_vs_current_best_stable_f1_star": "{:+.4f}".format,
                "finals_f1_mean": "{:.4f}".format,
                "f1_positive_mean": "{:.4f}".format,
                "precision_mean": "{:.4f}".format,
                "recall_mean": "{:.4f}".format,
                "mcc_star_mean": "{:.4f}".format,
                "roc_auc_mean": "{:.4f}".format,
                "predicted_positives_mean": "{:.2f}".format,
                "predicted_positives_total": "{:d}".format,
            },
        ),
        flush=True,
    )

    best = summary.iloc[0]
    best_fixed = fixed.iloc[0] if not fixed.empty else None
    print(f"\nBest overall: {best.method} threshold={best.threshold:.2f} F1*={best.f1_star_mean:.4f}", flush=True)
    if best_fixed is not None:
        print(
            f"Best fixed-threshold row: {best_fixed.method} threshold={best_fixed.threshold:.2f} "
            f"F1*={best_fixed.f1_star_mean:.4f}",
            flush=True,
        )
    print(
        f"Current best stable reference: w=0.5 threshold=0.82 F1*="
        f"{CURRENT_BEST_STABLE_F1_STAR:.4f}",
        flush=True,
    )
    print(f"Saved: {out_csv}", flush=True)
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated subset of: platt,none,isotonic,temperature",
    )
    parser.add_argument("--no-grid", action="store_true")
    args = parser.parse_args()

    run(
        methods=[m.strip() for m in args.methods.split(",") if m.strip()],
        seeds=[int(s.strip()) for s in args.seeds.split(",") if s.strip()],
        n_repeats=args.repeats,
        n_splits=args.splits,
        include_grid=not args.no_grid,
    )
