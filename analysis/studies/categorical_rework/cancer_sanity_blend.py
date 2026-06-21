"""CANCER primary sanity check and controlled blend scan.

Frozen CANCER decisions:
  * primary preprocessing: no_rowdrop
  * strict row-drop: ablation only
  * PCA: rejected / ablation only
  * primary own pipeline: updated CAT features + no_rowdrop + safe encoding +
    median imputation + outlier control + no PCA

This script runs fold-local preprocessing and a fixed LightGBM evaluator. It then
scans blend weights against a general-transfer scorer without searching models.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from experiment_logger import log_experiment_result, write_decision_log
from pca_experiments import (
    OUTPUT_DIR,
    apply_platt,
    bootstrap_f1_mcc,
    f1_at_prior,
    fit_platt,
    get_model_params,
    lgb_oof_and_test,
    load_updated_panel,
    mcc_at_prior_analytic,
    pick_threshold_at_prior,
    prepare_fold,
    tpr_fpr,
)


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent / "denemeler" / "feature_selection"
PIPELINES_DIR = PROJECT_ROOT / "pipelines"
if str(PIPELINES_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINES_DIR))

from calibration import apply_platt as _apply_platt, fit_platt as _fit_platt  # noqa: E402
from run_pipeline import get_model_params as _get_model_params  # noqa: E402
from utils import TEST_PRIORS  # noqa: E402

try:
    from general_blend import _cat_model, _cat_repr, _rf_repr  # noqa: E402
    CATBOOST_AVAILABLE = True
except ModuleNotFoundError:
    from sklearn.preprocessing import OrdinalEncoder

    CATBOOST_AVAILABLE = False

    def _rf_repr(X, cat_feats, enc=None, fit=False):
        Xr = X.copy()
        if cat_feats:
            s = X[cat_feats].astype(object).where(X[cat_feats].notna(), "NA")
            if fit:
                enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-2)
                Xr[cat_feats] = enc.fit_transform(s)
            else:
                Xr[cat_feats] = enc.transform(s)
        return Xr.fillna(-999.0), enc

# FINALS_N lives in analysis/harness, imported through pca_experiments above only
# indirectly in function helpers. Keep the explicit constant local for clarity.
FINALS_N_LOCAL = {"cancer": 600}
OUT_DIR = OUTPUT_DIR / "cancer_primary"
TARGET_COL = "Label"
WEIGHTS = [round(x, 1) for x in np.arange(0.0, 1.01, 0.1)]
THRESHOLDS = [round(x, 2) for x in np.arange(0.70, 0.901, 0.01)]
TRACK_THRESHOLDS = [0.80, 0.81, 0.82]


def _metric_row(y_true, probs, threshold, prior, n_target):
    preds = (np.asarray(probs) >= threshold).astype(int)
    tpr, fpr = tpr_fpr(np.asarray(y_true), preds)
    f1 = f1_at_prior(tpr, fpr, prior)
    mcc = mcc_at_prior_analytic(tpr, fpr, prior)
    f1b, _ = bootstrap_f1_mcc(np.asarray(y_true), preds, prior, n_target)
    try:
        auc = roc_auc_score(y_true, probs)
        ap = average_precision_score(y_true, probs)
    except ValueError:
        auc = 0.5
        ap = float(np.mean(y_true))
    return {
        "f1_star": float(f1),
        "finals_f1": float(f1b),
        "auc": float(auc),
        "average_precision": float(ap),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "mcc_star": float(mcc),
        "pred_pos": int(preds.sum()),
        "threshold": float(threshold),
    }


def _mean_se(values):
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0


def _summarize(rows: list[dict], group_cols: list[str]) -> pd.DataFrame:
    out = []
    for key, g in pd.DataFrame(rows).groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        item = dict(zip(group_cols, key))
        for col in ["f1_star", "finals_f1", "auc", "average_precision", "precision", "recall", "mcc_star", "pred_pos"]:
            mean, se = _mean_se(g[col])
            item[f"{col}_mean"] = mean
            if col == "f1_star":
                item["f1_star_se"] = se
        item["n_folds"] = len(g)
        if "selected_threshold" in g.columns:
            item["selected_threshold_mean"] = float(g["selected_threshold"].mean())
            item["selected_threshold_min"] = float(g["selected_threshold"].min())
            item["selected_threshold_max"] = float(g["selected_threshold"].max())
        out.append(item)
    return pd.DataFrame(out)


def _central_result_row(
    *,
    experiment_name: str,
    pipeline_role: str,
    threshold_strategy: str,
    output_file: Path,
    summary_row: dict,
    input_rows: int,
    input_features: int,
    selected_threshold="n/a",
    blend_enabled=False,
    blend_weight="n/a",
    general_transfer_weight="n/a",
    cancer_own_weight="n/a",
    delta_f1_vs_baseline="n/a",
    notes="n/a",
):
    return {
        "panel": "CANCER",
        "experiment_name": experiment_name,
        "pipeline_role": pipeline_role,
        "preprocess_mode": "no_rowdrop",
        "row_drop_enabled": False,
        "pca_enabled": False,
        "pca_variant": "none",
        "scaling_enabled": False,
        "outlier_control_enabled": True,
        "imputation_method": "median",
        "encoding_method": "safe ordinal / label encoding",
        "feature_selection_method": "none",
        "model_name": "fixed current LightGBM",
        "calibration_method": "Platt/sigmoid",
        "threshold_strategy": threshold_strategy,
        "selected_threshold": selected_threshold,
        "blend_enabled": blend_enabled,
        "blend_weight": blend_weight,
        "general_transfer_weight": general_transfer_weight,
        "cancer_own_weight": cancer_own_weight,
        "cv_strategy": "RepeatedStratifiedKFold",
        "n_folds": summary_row.get("n_folds", "n/a"),
        "input_rows": input_rows,
        "output_rows": input_rows,
        "dropped_rows": 0,
        "input_features": input_features,
        "output_features": "fold-local",
        "dropped_features": "fold-local",
        "positive_ratio_before": "fold-local",
        "positive_ratio_after": "fold-local",
        "auc": summary_row.get("auc_mean", "n/a"),
        "f1_star": summary_row.get("f1_star_mean", "n/a"),
        "finals_adjusted_f1": summary_row.get("finals_f1_mean", "n/a"),
        "precision": summary_row.get("precision_mean", "n/a"),
        "recall": summary_row.get("recall_mean", "n/a"),
        "pred_positive_mean": summary_row.get("pred_pos_mean", "n/a"),
        "delta_f1_vs_baseline": delta_f1_vs_baseline,
        "notes": notes,
        "output_file": output_file,
    }


def _train_own_fold(train_df, test_df, n_seeds):
    prepared = prepare_fold(
        train_df,
        test_df,
        impute=True,
        outlier_control=True,
        drop_rows=False,
        no_imputation_pca_safe=False,
    )
    if prepared is None:
        raise RuntimeError("no_rowdrop own preprocessing unexpectedly returned no fold")
    Xtr, ytr, Xte, yte = prepared.X_train, prepared.y_train, prepared.X_test, prepared.y_test
    spw = (ytr == 0).sum() / max(int((ytr == 1).sum()), 1)
    params = get_model_params(len(Xtr), spw)
    nf = min(5, max(2, int(ytr.sum()), int((ytr == 0).sum())))
    test_probs, train_oof = lgb_oof_and_test(Xtr, ytr, Xte, params, n_seeds=n_seeds, n_folds=nf)
    platt = fit_platt(train_oof, ytr.values)
    return {
        "train_oof": apply_platt(train_oof, platt),
        "test_probs": apply_platt(test_probs, platt),
        "y_train": ytr,
        "y_test": yte,
        "metadata": prepared.metadata,
    }


def _fit_general_transfer_scores(cancer_df, n_seeds=1):
    """Train a source-only transfer scorer on MASTER excluding all CANCER IDs."""
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    master = load_updated_panel("MASTER")
    cancer_ids = set(cancer_df["Variant_ID"])
    source = master[~master["Variant_ID"].isin(cancer_ids)].reset_index(drop=True)
    target = cancer_df.reset_index(drop=True)

    source_prep = prepare_fold(
        source,
        target,
        impute=True,
        outlier_control=True,
        drop_rows=False,
        no_imputation_pca_safe=False,
    )
    if source_prep is None:
        raise RuntimeError("transfer preprocessing unexpectedly returned no source matrix")
    Xs, ys, Xt = source_prep.X_train, source_prep.y_train, source_prep.X_test
    spw = (ys == 0).sum() / max(int((ys == 1).sum()), 1)
    params = _get_model_params(len(Xs), spw)
    params["n_jobs"] = 1
    cat_feats: list[str] = []

    i2, ies = train_test_split(np.arange(len(ys)), test_size=0.15, random_state=42, stratify=ys)
    lgbm = LGBMClassifier(**params, random_state=42)
    lgbm.fit(
        Xs.iloc[i2],
        ys.iloc[i2],
        eval_set=[(Xs.iloc[ies], ys.iloc[ies])],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
    )
    lgbm_scores = lgbm.predict_proba(Xt)[:, 1]

    # Keep the current transfer spirit: heterogeneous source scorer, probability average.
    X_ord, enc = _rf_repr(Xs, cat_feats, fit=True)
    Xt_ord, _ = _rf_repr(Xt, cat_feats, enc=enc, fit=False)
    rf = RandomForestClassifier(n_estimators=300, max_features="sqrt", n_jobs=1, class_weight="balanced", random_state=42)
    rf.fit(X_ord, ys)
    rf_scores = rf.predict_proba(Xt_ord)[:, 1]

    components = ["lgbm", "rf"]
    all_scores = [lgbm_scores, rf_scores]
    if CATBOOST_AVAILABLE:
        X_cat = _cat_repr(Xs, cat_feats)
        Xt_cat = _cat_repr(Xt, cat_feats)
        cb = _cat_model([], spw)
        cb.fit(X_cat.iloc[i2], ys.iloc[i2], eval_set=(X_cat.iloc[ies], ys.iloc[ies]))
        all_scores.append(cb.predict_proba(Xt_cat)[:, 1])
        components.append("catboost")

    scores = np.mean(np.vstack(all_scores), axis=0)
    return scores, {
        "source_rows": len(source),
        "excluded_cancer_ids": len(cancer_ids),
        "source_features": Xs.shape[1],
        "target_scored_rows": len(target),
        "transfer_scorer": "updated_MASTER_excluding_all_CANCER_ids_probability_average",
        "components": components,
        "catboost_available": CATBOOST_AVAILABLE,
    }


def run_sanity_and_blend(n_splits=5, n_repeats=6, n_seeds=1):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cancer = load_updated_panel("KANSER")
    y_all = cancer[TARGET_COL].astype(int).reset_index(drop=True)
    prior = TEST_PRIORS["cancer"]
    n_target = FINALS_N_LOCAL["cancer"]

    transfer_scores_raw, transfer_info = _fit_general_transfer_scores(cancer, n_seeds=n_seeds)
    print(f"Transfer scorer: {transfer_info}", flush=True)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    sanity_rows = []
    blend_rows = []
    threshold_grid_rows = []
    fold_cache = []
    t0 = time.time()
    for fold, (tr, te) in enumerate(rskf.split(cancer, y_all), start=1):
        print(f"Fold {fold}/{n_splits * n_repeats}", flush=True)
        train_df = cancer.iloc[tr].reset_index(drop=True)
        test_df = cancer.iloc[te].reset_index(drop=True)
        own = _train_own_fold(train_df, test_df, n_seeds=n_seeds)
        ytr = own["y_train"].reset_index(drop=True)
        yte = own["y_test"].reset_index(drop=True)

        g_platt = _fit_platt(transfer_scores_raw[tr], ytr.values)
        g_train = _apply_platt(transfer_scores_raw[tr], g_platt)
        g_test = _apply_platt(transfer_scores_raw[te], g_platt)
        fold_cache.append({"fold": fold, "own": own, "g_train": g_train, "g_test": g_test})

        own_thr = pick_threshold_at_prior(ytr.values, own["train_oof"], prior)
        own_metrics = _metric_row(yte.values, own["test_probs"], own_thr, prior, n_target)
        own_metrics.update({"fold": fold, "experiment": "sanity_cancer_own_no_rowdrop_no_pca", "selected_threshold": own_thr})
        sanity_rows.append(own_metrics)

        for w in WEIGHTS:
            b_train = w * g_train + (1 - w) * own["train_oof"]
            b_test = w * g_test + (1 - w) * own["test_probs"]
            # Conservative threshold tuning only within train fold.
            best_t, best_f1 = 0.80, -1.0
            for t in THRESHOLDS:
                tpr, fpr = tpr_fpr(ytr.values, (b_train >= t).astype(int))
                f1 = f1_at_prior(tpr, fpr, prior)
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t
            row = _metric_row(yte.values, b_test, best_t, prior, n_target)
            row.update({"fold": fold, "weight": w, "selected_threshold": best_t, "threshold_kind": "train_optimized_0.70_0.90"})
            blend_rows.append(row)

            for t in THRESHOLDS:
                grid_row = _metric_row(yte.values, b_test, t, prior, n_target)
                grid_row.update({
                    "fold": fold,
                    "weight": w,
                    "fixed_threshold": t,
                    "tracked_conservative_threshold": t in TRACK_THRESHOLDS,
                })
                threshold_grid_rows.append(grid_row)

    sanity = _summarize(sanity_rows, ["experiment"])
    sanity["preprocess_mode"] = "no_rowdrop"
    sanity["pipeline_role"] = "primary"
    sanity["pca"] = "rejected_ablation_only"
    sanity["calibration"] = "platt_sigmoid_unchanged"
    sanity["comparison_previous_reproduced_baseline_f1_star"] = 0.7077
    sanity["comparison_previous_threshold_stability_f1_star"] = 0.7136
    sanity["comparison_previous_calibration_threshold_grid_f1_star"] = 0.7162
    sanity["delta_vs_previous_reproduced_baseline"] = sanity["f1_star_mean"] - 0.7077
    sanity["note"] = (
        "This is cancer-own only with updated CAT/no_rowdrop/no-PCA/no feature selection; "
        "previous ~0.71 references include the general-transfer blend at weight 0.5 "
        "and/or threshold/calibration grid runs."
    )

    blend = _summarize(blend_rows, ["weight", "threshold_kind"]).sort_values("weight")
    baseline_f1 = float(sanity["f1_star_mean"].iloc[0])
    blend["delta_vs_cancer_own_sanity"] = blend["f1_star_mean"] - baseline_f1
    grid = _summarize(threshold_grid_rows, ["weight", "fixed_threshold", "tracked_conservative_threshold"]).sort_values(["weight", "fixed_threshold"])
    grid["delta_vs_cancer_own_sanity"] = grid["f1_star_mean"] - baseline_f1

    sanity_path = OUT_DIR / "cancer_primary_sanity_check.csv"
    blend_path = OUT_DIR / "cancer_primary_blend_weight_scan.csv"
    grid_path = OUT_DIR / "cancer_primary_blend_threshold_grid.csv"
    sanity.to_csv(sanity_path, index=False)
    blend.to_csv(blend_path, index=False)
    grid.to_csv(grid_path, index=False)

    best = blend.sort_values(["f1_star_mean", "finals_f1_mean"], ascending=False).iloc[0].to_dict()
    tracked = grid[grid["tracked_conservative_threshold"].eq(True)].copy()
    best_tracked = tracked.sort_values(["f1_star_mean", "finals_f1_mean"], ascending=False).iloc[0].to_dict()
    decision = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "frozen_decisions": [
            "strict row-drop rejected for CANCER; ablation only",
            "no_rowdrop selected as primary CANCER preprocessing",
            "PCA rejected for CANCER; ablation only",
            "Platt/sigmoid calibration unchanged for now",
        ],
        "primary_pipeline": "updated CAT features + no_rowdrop + safe encoding + median imputation + outlier control + no PCA",
        "sanity_result": sanity.iloc[0].to_dict(),
        "transfer_info": transfer_info,
        "next_selected_blend_threshold_result": best,
        "best_tracked_0.80_0.82_result": best_tracked,
        "runtime_seconds": round(time.time() - t0, 1),
        "outputs": {
            "sanity": str(sanity_path),
            "blend_scan": str(blend_path),
            "threshold_grid": str(grid_path),
        },
    }
    decision_path = OUT_DIR / "cancer_decision_log.json"
    with open(decision_path, "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2, default=str)
    _write_markdown(sanity, blend, grid, decision, OUT_DIR / "cancer_primary_sanity_and_blend_report.md")

    input_rows = len(cancer)
    input_features = cancer.shape[1] - 1
    sanity_summary = sanity.iloc[0].to_dict()
    log_experiment_result(_central_result_row(
        experiment_name="cancer_primary_sanity_no_rowdrop_no_pca",
        pipeline_role="sanity_check",
        threshold_strategy="prior_matched",
        output_file=sanity_path,
        summary_row=sanity_summary,
        input_rows=input_rows,
        input_features=input_features,
        delta_f1_vs_baseline=sanity_summary.get("delta_vs_previous_reproduced_baseline", "n/a"),
        notes=sanity_summary.get("note", "n/a"),
    ))
    for _, row in blend.iterrows():
        row_dict = row.to_dict()
        weight = float(row_dict["weight"])
        log_experiment_result(_central_result_row(
            experiment_name=f"cancer_blend_weight_{weight:.1f}_grid_0.70_0.90",
            pipeline_role="sanity_check",
            threshold_strategy=row_dict.get("threshold_kind", "grid_0.70_0.90"),
            selected_threshold=row_dict.get("selected_threshold_mean", "n/a"),
            output_file=blend_path,
            summary_row=row_dict,
            input_rows=input_rows,
            input_features=input_features,
            blend_enabled=True,
            blend_weight=weight,
            general_transfer_weight=weight,
            cancer_own_weight=round(1.0 - weight, 1),
            delta_f1_vs_baseline=row_dict.get("delta_vs_cancer_own_sanity", "n/a"),
            notes="Completed controlled CANCER blend weight scan with fold-local threshold tuning.",
        ))
    write_decision_log([
        "Strict row-drop was rejected as the primary CANCER preprocessing because it dropped 158/388 rows and changed the positive class ratio from 69.07% to 59.13%.",
        "CANCER primary preprocessing is now no_rowdrop.",
        "PCA was rejected for the primary CANCER pipeline because all PCA variants underperformed the no-PCA baseline.",
        "Primary CANCER pipeline is currently updated CAT features + no_rowdrop + safe encoding + median imputation + outlier control + no PCA.",
        "Calibration strategy remains Platt/sigmoid for now.",
        "Threshold strategy remains aligned with the previous conservative range around 0.80-0.82 for later blend experiments.",
        (
            "Latest selected blend result: "
            f"general_transfer_weight={best['weight']}, cancer_own_weight={round(1.0 - float(best['weight']), 1)}, "
            f"F1*={best['f1_star_mean']}, finals-adjusted F1={best['finals_f1_mean']}."
        ),
    ])

    print("\nSanity:")
    print(sanity.to_string(index=False))
    print("\nBlend scan:")
    print(blend.to_string(index=False))
    print("\nBest tracked thresholds 0.80-0.82:")
    print(tracked.sort_values(["f1_star_mean", "finals_f1_mean"], ascending=False).head(10).to_string(index=False))
    print(f"\nSaved: {sanity_path}")
    print(f"Saved: {blend_path}")
    print(f"Saved: {grid_path}")
    print(f"Saved: {decision_path}")
    return sanity, blend, grid, decision


def _write_markdown(sanity, blend, grid, decision, path):
    lines = [
        "# CANCER Primary Sanity And Blend Report",
        "",
        "Primary CANCER preprocessing is `no_rowdrop`; strict row-drop and PCA are ablations only.",
        "",
        "## Sanity Check",
        "",
        _markdown_table(sanity),
        "",
        "## Blend Weight Scan",
        "",
        _markdown_table(blend),
        "",
        "## Conservative Threshold Track 0.80-0.82",
        "",
        _markdown_table(grid[grid["tracked_conservative_threshold"].eq(True)]),
        "",
        "## Decision Log",
        "",
    ]
    for item in decision["frozen_decisions"]:
        lines.append(f"- {item}")
    best = decision["next_selected_blend_threshold_result"]
    lines.append(
        f"- Next selected blend/threshold result: w={best['weight']}, "
        f"threshold_mean={best.get('selected_threshold_mean')}, F1*={best['f1_star_mean']:.4f}, "
        f"finals F1={best['finals_f1_mean']:.4f}."
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(df):
    cols = list(df.columns)
    rows = [[str(v) for v in row] for row in df.astype(object).itertuples(index=False, name=None)]
    widths = [max(len(str(c)), *(len(r[i]) for r in rows)) if rows else len(str(c)) for i, c in enumerate(cols)]

    def fmt(vals):
        return "| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(vals)) + " |"

    return "\n".join([fmt(cols), "| " + " | ".join("-" * w for w in widths) + " |"] + [fmt(r) for r in rows])


def main():
    parser = argparse.ArgumentParser(description="Run CANCER no-rowdrop sanity check and controlled blend scan")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--n-seeds", type=int, default=1)
    args = parser.parse_args()
    run_sanity_and_blend(args.splits, args.repeats, args.n_seeds)


if __name__ == "__main__":
    main()
