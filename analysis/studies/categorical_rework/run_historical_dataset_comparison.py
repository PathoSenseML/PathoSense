"""Compare dataset variants under the historical ~0.73 CANCER pipeline.

This preserves the historical setup that produced the ~0.73 results:
  * old sentinel normalization + all-NaN row drop
  * old categorical schema/native LightGBM categoricals
  * AA/interactions feature engineering
  * leakage drop + preprocess_data
  * score-agreement features
  * mRMR 40
  * LightGBM + Platt/sigmoid
  * MASTER/general-minus-cancer probability blend at w=0.5
  * threshold grid 0.70-0.90 including fixed 0.82
  * no PCA

Only the input dataset source changes:
  * raw_initial: data/YARISMA_TRAIN_MASTER/KANSER.csv
  * new_cat_only: missingsiz new_cat_only datasets with CAT_new as categorical
  * all_cats_kept: missingsiz all_cats_kept datasets with CAT_new as categorical
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "6")

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from experiment_logger import log_experiment_result


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent / "denemeler" / "feature_selection"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
PIPELINES_DIR = PROJECT_ROOT / "pipelines"
for p in (str(ANALYSIS_DIR), str(PIPELINES_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import schema  # noqa: E402
from calibration import apply_platt, fit_platt  # noqa: E402
from config import SEED  # noqa: E402
from exp_transfer_blend import cat_score, lgb_score, rf_score  # noqa: E402
from feature_engineering import (  # noqa: E402
    apply_score_agreement,
    engineer_aa_features,
    engineer_acmg_features,
    engineer_interaction_features,
    fit_score_agreement,
)
from harness import (  # noqa: E402
    FINALS_N,
    _drop_leaky,
    bootstrap_f1_mcc,
    f1_at_prior,
    fast_mrmr,
    lgb_oof_and_test,
    mcc_at_prior_analytic,
    tpr_fpr,
)
from preprocess_v2 import apply_keep_nan, fit_keep_nan  # noqa: E402
from run_pipeline import get_model_params, preprocess_data  # noqa: E402
from schema import (  # noqa: E402
    drop_all_nan_rows,
    normalize_sentinels,
    partition_columns,
    to_categorical,
)
from utils import ACMG_PANELS, KNOWN_LEAKY, TEST_PRIORS, detect_leaky_columns  # noqa: E402


if "CAT_new" not in schema.CATEGORICAL_COLS:
    schema.CATEGORICAL_COLS.append("CAT_new")


OUT_DIR = ROOT / "outputs" / "historical_dataset_comparison"
FOLD_CSV = OUT_DIR / "historical_dataset_comparison_folds.csv"
SUMMARY_CSV = OUT_DIR / "historical_dataset_comparison.csv"
SUMMARY_MD = OUT_DIR / "historical_dataset_comparison.md"
TRANSFER_CACHE_TEMPLATE = OUT_DIR / "transfer_scores_{dataset_source}.csv"

DATASET_PATHS = {
    "raw_initial": {
        "general": ROOT / "data" / "YARISMA_TRAIN_MASTER.csv",
        "cancer": ROOT / "data" / "YARISMA_TRAIN_KANSER.csv",
    },
    "new_cat_only": {
        "general": ROOT / "outputs" / "new_cat_datasets" / "missingsiz" / "new_cat_only" / "YARISMA_TRAIN_MASTER_new_cat_missingsiz_new_cat_only.csv",
        "cancer": ROOT / "outputs" / "new_cat_datasets" / "missingsiz" / "new_cat_only" / "YARISMA_TRAIN_KANSER_new_cat_missingsiz_new_cat_only.csv",
    },
    "all_cats_kept": {
        "general": ROOT / "outputs" / "new_cat_datasets" / "missingsiz" / "all_cats_kept" / "YARISMA_TRAIN_MASTER_new_cat_missingsiz_all_cats_kept.csv",
        "cancer": ROOT / "outputs" / "new_cat_datasets" / "missingsiz" / "all_cats_kept" / "YARISMA_TRAIN_KANSER_new_cat_missingsiz_all_cats_kept.csv",
    },
}

PANEL = "cancer"
WEIGHT = 0.5
FIXED_THRESHOLD = 0.82
GRID_THRESHOLDS = [round(float(x), 2) for x in np.arange(0.70, 0.90 + 0.001, 0.01)]


def load_historical_panel(panel: str, dataset_source: str) -> pd.DataFrame:
    path = DATASET_PATHS[dataset_source][panel]
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df = normalize_sentinels(df)
    df = drop_all_nan_rows(df, verbose=False)
    df = to_categorical(df)
    if "CAT_new" in df.columns and not isinstance(df["CAT_new"].dtype, pd.CategoricalDtype):
        df["CAT_new"] = df["CAT_new"].astype("category")
    df = engineer_aa_features(df)
    if panel in ACMG_PANELS:
        df = engineer_acmg_features(df)
    if panel in ("cancer", "cftr"):
        df = engineer_interaction_features(df)
    return df.reset_index(drop=True)


def load_target_engineered(dataset_source: str):
    df = load_historical_panel(PANEL, dataset_source)
    _, n, c, t = partition_columns(df)
    return df[n + c].reset_index(drop=True), df[t].reset_index(drop=True).astype(int)


def build_matrices(dataset_source: str):
    target = load_historical_panel(PANEL, dataset_source)
    target_ids = set(target["Variant_ID"])
    general = load_historical_panel("general", dataset_source)
    general = general[~general["Variant_ID"].isin(target_ids)].reset_index(drop=True)
    _, gn, gc, gt = partition_columns(general)
    yg = general[gt].reset_index(drop=True).astype(int)
    Xg = general[gn + gc].reset_index(drop=True)
    leaky = sorted(set(detect_leaky_columns(Xg, yg, 0.99, verbose=False)) | (KNOWN_LEAKY & set(Xg.columns)))
    Xg = Xg.drop(columns=leaky, errors="ignore")

    yt = target["Label"].reset_index(drop=True).astype(int)
    Xt = target.reindex(columns=Xg.columns)
    Xg_pp, fitted = fit_keep_nan(Xg)
    Xt_pp = apply_keep_nan(Xt, fitted)
    cats = [c for c in Xg_pp.columns if isinstance(Xg_pp[c].dtype, pd.CategoricalDtype)]
    for c in cats:
        Xt_pp[c] = pd.Categorical(Xt_pp[c].astype(object), categories=Xg_pp[c].cat.categories)
    return Xg_pp, yg, Xt_pp, yt, cats


def build_transfer_scores(dataset_source: str):
    cache_path = Path(str(TRANSFER_CACHE_TEMPLATE).format(dataset_source=dataset_source))
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        return cached["general_transfer_prob_raw"].to_numpy(dtype=float), cached["y_true"].astype(int), {
            "transfer_cache": str(cache_path),
            "loaded_from_cache": True,
        }

    Xg, yg, Xt, yt, cats = build_matrices(dataset_source)
    t0 = time.time()
    g_lgb = lgb_score(Xg, yg, Xt, cats)
    g_rf = rf_score(Xg, yg, Xt, cats)
    g_cat = cat_score(Xg, yg, Xt, cats)
    scores = (g_lgb + g_rf + g_cat) / 3.0
    cache = pd.DataFrame({
        "sample_id": np.arange(len(yt)),
        "y_true": yt.to_numpy(),
        "general_transfer_prob_raw": scores,
        "dataset_source": dataset_source,
        "source_rows": len(Xg),
        "source_features": Xg.shape[1],
        "target_features": Xt.shape[1],
        "categorical_features": len(cats),
        "runtime_seconds": round(time.time() - t0, 1),
    })
    cache.to_csv(cache_path, index=False)
    return scores, yt, {
        "transfer_cache": str(cache_path),
        "loaded_from_cache": False,
        "source_rows": len(Xg),
        "source_features": Xg.shape[1],
        "target_features": Xt.shape[1],
        "categorical_features": len(cats),
    }


def safe_auc(y_true, probs):
    try:
        return float(roc_auc_score(y_true, probs))
    except ValueError:
        return np.nan


def metric_row(y_true, probs, threshold, prior, n_target):
    preds = (np.asarray(probs, dtype=float) >= float(threshold)).astype(int)
    tpr, fpr = tpr_fpr(y_true, preds)
    finals_f1, finals_mcc = bootstrap_f1_mcc(y_true, preds, prior, n_target)
    return {
        "f1_star": f1_at_prior(tpr, fpr, prior),
        "finals_adjusted_f1": finals_f1,
        "finals_mcc": finals_mcc,
        "f1_positive": f1_score(y_true, preds, zero_division=0),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "mcc": matthews_corrcoef(y_true, preds),
        "mcc_star": mcc_at_prior_analytic(tpr, fpr, prior),
        "auc": safe_auc(y_true, probs),
        "pred_positive_mean": float(preds.mean()),
        "predicted_positives": int(preds.sum()),
    }


def cancer_own_historical_probs(X_tr_raw, y_tr, X_te_raw, n_own_seeds):
    Xtr, Xte = _drop_leaky(X_tr_raw, y_tr, X_te_raw)
    Xtr_pp, _, Xte_pp, _ = preprocess_data(Xtr.copy(), Xtr.copy(), Xte.copy(), corr_threshold=0.95)
    features_before_selection = int(Xtr_pp.shape[1])
    saf = fit_score_agreement(Xtr_pp, y_tr, top_k=7)
    Xtr_pp = apply_score_agreement(Xtr_pp, saf)
    Xte_pp = apply_score_agreement(Xte_pp, saf)
    Xtr_pp, _, Xte_pp, fs_info = fast_mrmr(Xtr_pp, y_tr, Xte_pp, Xte_pp, max_features=40)
    selected_features = list(Xtr_pp.columns)
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    params = get_model_params(len(X_tr_raw), spw)
    nf = 3 if len(X_tr_raw) < 200 else 5
    test_probs, oof = lgb_oof_and_test(Xtr_pp, y_tr, Xte_pp, params, n_seeds=n_own_seeds, n_folds=nf)
    calibrator = fit_platt(oof, y_tr.values)
    return {
        "test_probs": apply_platt(test_probs, calibrator),
        "train_oof": apply_platt(oof, calibrator),
        "features_before_selection": features_before_selection,
        "selected_features": selected_features,
        "selected_feature_count": len(selected_features),
        "fs_info": fs_info,
    }


def completed_folds(dataset_source: str) -> set[int]:
    if not FOLD_CSV.exists():
        return set()
    df = pd.read_csv(FOLD_CSV, usecols=["dataset_source", "fold", "status"])
    done = df[df["dataset_source"].eq(dataset_source) & df["status"].eq("complete")]
    return set(done["fold"].astype(int).unique())


def append_rows(rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(FOLD_CSV, mode="a", header=not FOLD_CSV.exists(), index=False)


def evaluate_source(dataset_source: str, n_repeats: int, n_splits: int, n_own_seeds: int, max_seconds: float | None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g_all, yt_g, transfer_info = build_transfer_scores(dataset_source)
    Xt_own, yt = load_target_engineered(dataset_source)
    if not (yt.values == yt_g.values).all():
        raise RuntimeError(f"{dataset_source}: transfer and own labels do not align")
    print(f"[{dataset_source}] rows={len(yt)} own_features={Xt_own.shape[1]} transfer={transfer_info}", flush=True)

    prior = TEST_PRIORS[PANEL]
    n_target = FINALS_N[PANEL]
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    done = completed_folds(dataset_source)
    started = time.time()
    total_folds = n_splits * n_repeats
    for fold, (tr, te) in enumerate(splitter.split(Xt_own, yt), start=1):
        if fold in done:
            print(f"[{dataset_source}] fold {fold}/{total_folds} cached; skipping", flush=True)
            continue
        if max_seconds is not None and time.time() - started > max_seconds:
            print(f"[{dataset_source}] stopping early after {round(time.time() - started, 1)}s", flush=True)
            return

        tr = np.asarray(tr)
        te = np.asarray(te)
        ytr = yt.iloc[tr].reset_index(drop=True)
        yte = yt.iloc[te].reset_index(drop=True)
        Xtr = Xt_own.iloc[tr].reset_index(drop=True)
        Xte = Xt_own.iloc[te].reset_index(drop=True)
        own = cancer_own_historical_probs(Xtr, ytr, Xte, n_own_seeds=n_own_seeds)

        gen_cal = fit_platt(g_all[tr], ytr.values)
        gen_te = apply_platt(g_all[te], gen_cal)
        blend_te = WEIGHT * gen_te + (1.0 - WEIGHT) * own["test_probs"]
        selected_json = json.dumps(own["selected_features"])
        fold_rows = []
        for threshold in GRID_THRESHOLDS:
            values = metric_row(yte.values, blend_te, threshold, prior, n_target)
            fold_rows.append({
                "dataset_source": dataset_source,
                "panel": "CANCER",
                "fold": fold,
                "repeat_id": (fold - 1) // n_splits + 1,
                "split_id": (fold - 1) % n_splits + 1,
                "n_splits": n_splits,
                "n_repeats": n_repeats,
                "row_count": len(yt),
                "train_rows": len(tr),
                "validation_rows": len(te),
                "feature_count_before_selection": own["features_before_selection"],
                "selected_feature_count": own["selected_feature_count"],
                "selected_feature_names": selected_json,
                "blend_weight": WEIGHT,
                "threshold": threshold,
                "threshold_kind": "fixed_0.82" if float(threshold) == FIXED_THRESHOLD else "grid_0.70_0.90",
                "status": "complete",
                **values,
            })
        append_rows(fold_rows)
        print(f"[{dataset_source}] saved fold {fold}/{total_folds}", flush=True)


def summarize_results(n_repeats: int, n_splits: int, sources: list[str]) -> pd.DataFrame:
    if not FOLD_CSV.exists():
        raise FileNotFoundError(FOLD_CSV)
    folds = pd.read_csv(FOLD_CSV)
    expected = n_repeats * n_splits
    complete = folds[folds["status"].eq("complete")].copy()
    counts = complete.groupby("dataset_source")["fold"].nunique().to_dict()
    missing = {s: expected - int(counts.get(s, 0)) for s in sources}
    if any(v > 0 for v in missing.values()):
        print(f"Partial comparison; missing folds: {missing}", flush=True)

    metric_cols = [
        "f1_star",
        "finals_adjusted_f1",
        "precision",
        "recall",
        "mcc_star",
        "auc",
        "pred_positive_mean",
        "predicted_positives",
        "feature_count_before_selection",
        "selected_feature_count",
    ]
    summary = complete.groupby(["dataset_source", "threshold"], as_index=False).agg(
        n_folds=("fold", "nunique"),
        row_count=("row_count", "first"),
        blend_weight=("blend_weight", "first"),
        **{f"{c}_mean": (c, "mean") for c in metric_cols},
        f1_star_se=("f1_star", lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0),
    )
    best = summary.sort_values(
        ["dataset_source", "f1_star_mean", "finals_adjusted_f1_mean", "auc_mean"],
        ascending=[True, False, False, False],
    )
    best = best.groupby("dataset_source", as_index=False, dropna=False).head(1).copy()
    fixed = summary[np.isclose(summary["threshold"].astype(float), FIXED_THRESHOLD)].copy()
    selected = pd.concat([best.assign(selection_type="best_grid"), fixed.assign(selection_type="fixed_0.82")], ignore_index=True)
    selected = selected.sort_values(["selection_type", "dataset_source"]).drop_duplicates(["selection_type", "dataset_source"])

    selected = selected.merge(feature_name_summary(complete), on="dataset_source", how="left")
    selected["status"] = selected["dataset_source"].map(lambda s: "complete" if missing.get(s, 0) == 0 else "partial")
    selected["pipeline"] = "old_historical_mRMR40_score_agreement_50_50_blend_no_pca"
    selected.to_csv(SUMMARY_CSV, index=False)
    SUMMARY_MD.write_text(markdown_report(selected, summary), encoding="utf-8")
    log_selected_results(selected)
    return selected


def feature_name_summary(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset_source, g in folds.groupby("dataset_source"):
        counter = Counter()
        first = []
        for text in g.drop_duplicates("fold")["selected_feature_names"]:
            try:
                names = json.loads(text)
            except Exception:
                names = []
            if not first and names:
                first = names
            counter.update(names)
        common = [name for name, _ in counter.most_common(40)]
        rows.append({
            "dataset_source": dataset_source,
            "selected_feature_names": ";".join(common),
            "first_fold_selected_feature_names": ";".join(first),
        })
    return pd.DataFrame(rows)


def log_selected_results(selected: pd.DataFrame) -> None:
    for row in selected.itertuples(index=False):
        source = getattr(row, "dataset_source")
        selection = getattr(row, "selection_type")
        log_experiment_result({
            "panel": "CANCER",
            "experiment_name": f"historical_dataset_{source}_{selection}",
            "pipeline_role": "primary" if source == "new_cat_only" and selection == "best_grid" else "ablation",
            "preprocess_mode": "old_historical",
            "row_drop_enabled": "old_drop_all_nan_rows_only",
            "pca_enabled": False,
            "pca_variant": "none",
            "scaling_enabled": False,
            "outlier_control_enabled": "old_preprocess_data",
            "imputation_method": "old_preprocess_data",
            "encoding_method": "old native categorical + CAT_new registered categorical when present",
            "feature_selection_method": "mRMR_40",
            "model_name": "historical LightGBM cancer-own + general-minus-cancer blend",
            "calibration_method": "Platt/sigmoid",
            "threshold_strategy": "fixed_grid_0.70_0.90" if selection == "best_grid" else "fixed_0.82",
            "selected_threshold": getattr(row, "threshold"),
            "blend_enabled": True,
            "blend_weight": WEIGHT,
            "general_transfer_weight": WEIGHT,
            "cancer_own_weight": 1.0 - WEIGHT,
            "cv_strategy": "RepeatedStratifiedKFold historical harness",
            "n_folds": getattr(row, "n_folds"),
            "input_rows": getattr(row, "row_count"),
            "output_rows": getattr(row, "row_count"),
            "dropped_rows": "old_drop_all_nan_rows_only",
            "input_features": "old historical engineered",
            "output_features": getattr(row, "feature_count_before_selection_mean"),
            "dropped_features": "fold-local",
            "positive_ratio_before": "n/a",
            "positive_ratio_after": "n/a",
            "auc": getattr(row, "auc_mean"),
            "f1_star": getattr(row, "f1_star_mean"),
            "finals_adjusted_f1": getattr(row, "finals_adjusted_f1_mean"),
            "precision": getattr(row, "precision_mean"),
            "recall": getattr(row, "recall_mean"),
            "pred_positive_mean": getattr(row, "pred_positive_mean_mean"),
            "delta_f1_vs_baseline": "n/a",
            "notes": f"Historical ~0.73 pipeline; dataset_source={source}; selection={selection}.",
            "output_file": SUMMARY_CSV,
        })


def markdown_report(selected: pd.DataFrame, all_thresholds: pd.DataFrame) -> str:
    lines = [
        "# Historical Dataset Comparison",
        "",
        "Historical setup held fixed: old preprocessing, score-agreement, mRMR 40, Platt/sigmoid, general-minus-cancer probability blend, w=0.5, fixed threshold grid 0.70-0.90, no PCA.",
        "",
        "Compared dataset sources: `raw_initial`, `new_cat_only`, `all_cats_kept`.",
        "",
        "## Selected Rows",
        "",
        markdown_table(selected),
        "",
        "## All Thresholds",
        "",
        markdown_table(all_thresholds.sort_values(["dataset_source", "threshold"])),
    ]
    return "\n".join(lines) + "\n"


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    show = df.copy()
    for col in show.columns:
        show[col] = show[col].astype(str).str.replace("\n", " ", regex=False)
    cols = list(show.columns)
    rows = [list(row) for row in show.itertuples(index=False, name=None)]
    widths = [max(len(str(c)), *(len(str(r[i])) for r in rows)) for i, c in enumerate(cols)]

    def fmt(vals):
        return "| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(vals)) + " |"

    return "\n".join([fmt(cols), "| " + " | ".join("-" * w for w in widths) + " |"] + [fmt(r) for r in rows])


def run(sources: list[str], n_repeats: int, n_splits: int, n_own_seeds: int, max_seconds: float | None) -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for source in sources:
        evaluate_source(source, n_repeats, n_splits, n_own_seeds, max_seconds)
    return summarize_results(n_repeats, n_splits, sources)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run historical pipeline on selected dataset sources.")
    parser.add_argument("--sources", nargs="+", choices=sorted(DATASET_PATHS), default=["new_cat_only", "raw_initial", "all_cats_kept"])
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--n-own-seeds", type=int, default=5)
    parser.add_argument("--max-seconds", type=float, default=None)
    args = parser.parse_args()
    summary = run(args.sources, args.repeats, args.splits, args.n_own_seeds, args.max_seconds)
    print(summary.to_string(index=False), flush=True)
    print(f"Saved: {SUMMARY_CSV}", flush=True)
    print(f"Saved: {SUMMARY_MD}", flush=True)


if __name__ == "__main__":
    main()
