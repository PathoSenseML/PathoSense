"""Fast leakage-safe CANCER sanity + blend scan.

This is a controlled pre-final scan, not the full historical nested-OOF harness.
Each outer train fold is split into model-fit/calibration subsets. Encoders,
imputers, clipping, Platt calibration, and threshold selection are fit without
touching the outer test fold.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from pca_experiments import (
    OUTPUT_DIR,
    apply_platt,
    bootstrap_f1_mcc,
    f1_at_prior,
    fit_platt,
    get_model_params,
    load_updated_panel,
    mcc_at_prior_analytic,
    prepare_fold,
    tpr_fpr,
)
from experiment_logger import log_experiment_result, write_decision_log

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent / "denemeler" / "feature_selection"
PIPELINES_DIR = PROJECT_ROOT / "pipelines"
if str(PIPELINES_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINES_DIR))

from utils import TEST_PRIORS  # noqa: E402

OUT_DIR = OUTPUT_DIR / "cancer_primary"
TARGET_COL = "Label"
WEIGHTS = [round(x, 1) for x in np.arange(0.0, 1.01, 0.1)]
THRESHOLDS = [round(x, 2) for x in np.arange(0.70, 0.901, 0.01)]
TRACK = {0.80, 0.81, 0.82}
FINALS_N = 600


def metric_row(y_true, probs, threshold, prior):
    preds = (probs >= threshold).astype(int)
    tpr, fpr = tpr_fpr(y_true, preds)
    f1 = f1_at_prior(tpr, fpr, prior)
    mcc = mcc_at_prior_analytic(tpr, fpr, prior)
    f1b, _ = bootstrap_f1_mcc(y_true, preds, prior, FINALS_N)
    return {
        "f1_star": f1,
        "finals_f1": f1b,
        "auc": roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.5,
        "average_precision": average_precision_score(y_true, probs) if len(np.unique(y_true)) > 1 else float(np.mean(y_true)),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "mcc_star": mcc,
        "pred_pos": int(preds.sum()),
    }


def summarize(rows, group_cols):
    df = pd.DataFrame(rows)
    out = []
    for key, g in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        for col in ["f1_star", "finals_f1", "auc", "average_precision", "precision", "recall", "mcc_star", "pred_pos"]:
            vals = g[col].astype(float)
            row[f"{col}_mean"] = float(vals.mean())
            if col == "f1_star":
                row["f1_star_se"] = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        if "selected_threshold" in g:
            row["selected_threshold_mean"] = float(g["selected_threshold"].mean())
            row["selected_threshold_min"] = float(g["selected_threshold"].min())
            row["selected_threshold_max"] = float(g["selected_threshold"].max())
        row["n_folds"] = len(g)
        out.append(row)
    return pd.DataFrame(out)


def train_own_holdout(train_df, test_df):
    prep = prepare_fold(train_df, test_df, impute=True, outlier_control=True, drop_rows=False)
    Xtr, ytr, Xte, yte = prep.X_train, prep.y_train, prep.X_test, prep.y_test
    fit_idx, cal_idx = train_test_split(
        np.arange(len(ytr)),
        test_size=0.25,
        random_state=42,
        stratify=ytr,
    )
    spw = (ytr.iloc[fit_idx] == 0).sum() / max(int((ytr.iloc[fit_idx] == 1).sum()), 1)
    params = get_model_params(len(fit_idx), spw)
    params["n_jobs"] = -1
    params["n_estimators"] = min(int(params.get("n_estimators", 300)), 120)
    params["learning_rate"] = max(float(params.get("learning_rate", 0.03)), 0.05)
    model = LGBMClassifier(**params, random_state=42)
    model.fit(
        Xtr.iloc[fit_idx],
        ytr.iloc[fit_idx],
        eval_set=[(Xtr.iloc[cal_idx], ytr.iloc[cal_idx])],
        callbacks=[lgb.early_stopping(15, verbose=False), lgb.log_evaluation(0)],
    )
    cal_raw = model.predict_proba(Xtr.iloc[cal_idx])[:, 1]
    test_raw = model.predict_proba(Xte)[:, 1]
    platt = fit_platt(cal_raw, ytr.iloc[cal_idx].values)
    return {
        "own_cal": apply_platt(cal_raw, platt),
        "own_test": apply_platt(test_raw, platt),
        "y_cal": ytr.iloc[cal_idx].reset_index(drop=True),
        "y_test": yte.reset_index(drop=True),
        "cal_original_index": cal_idx,
    }


def transfer_scores_lgbm(cancer):
    master = load_updated_panel("MASTER")
    source = master[~master["Variant_ID"].isin(set(cancer["Variant_ID"]))].reset_index(drop=True)
    prep = prepare_fold(source, cancer.reset_index(drop=True), impute=True, outlier_control=True, drop_rows=False)
    Xs, ys, Xt = prep.X_train, prep.y_train, prep.X_test
    spw = (ys == 0).sum() / max(int((ys == 1).sum()), 1)
    params = get_model_params(len(Xs), spw)
    params["n_jobs"] = -1
    params["n_estimators"] = min(int(params.get("n_estimators", 300)), 160)
    params["learning_rate"] = max(float(params.get("learning_rate", 0.03)), 0.05)
    fit_idx, es_idx = train_test_split(np.arange(len(ys)), test_size=0.15, random_state=42, stratify=ys)
    model = LGBMClassifier(**params, random_state=42)
    model.fit(
        Xs.iloc[fit_idx],
        ys.iloc[fit_idx],
        eval_set=[(Xs.iloc[es_idx], ys.iloc[es_idx])],
        callbacks=[lgb.early_stopping(15, verbose=False), lgb.log_evaluation(0)],
    )
    return model.predict_proba(Xt)[:, 1], {
        "transfer_scorer": "LGBM source-only updated MASTER excluding all CANCER IDs",
        "source_rows": len(source),
        "source_features": Xs.shape[1],
        "excluded_cancer_ids": int(cancer["Variant_ID"].nunique()),
    }


def run(splits=5, repeats=6):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cancer = load_updated_panel("KANSER")
    y = cancer[TARGET_COL].astype(int).reset_index(drop=True)
    prior = TEST_PRIORS["cancer"]
    g_raw, transfer_info = transfer_scores_lgbm(cancer)
    print(f"Transfer: {transfer_info}", flush=True)

    sanity_rows, blend_rows, grid_rows = [], [], []
    rskf = RepeatedStratifiedKFold(n_splits=splits, n_repeats=repeats, random_state=42)
    t0 = time.time()
    for fold, (tr, te) in enumerate(rskf.split(cancer, y), start=1):
        print(f"Fold {fold}/{splits * repeats}", flush=True)
        own = train_own_holdout(cancer.iloc[tr].reset_index(drop=True), cancer.iloc[te].reset_index(drop=True))
        ycal = own["y_cal"].values
        ytest = own["y_test"].values
        # Map calibration subset positions back to original row ids in this outer train fold.
        cal_global = tr[own["cal_original_index"]]
        g_platt = fit_platt(g_raw[cal_global], ycal)
        g_cal = apply_platt(g_raw[cal_global], g_platt)
        g_test = apply_platt(g_raw[te], g_platt)

        best_t, best_f1 = 0.80, -1.0
        for t in THRESHOLDS:
            tr_tpr, tr_fpr = tpr_fpr(ycal, (own["own_cal"] >= t).astype(int))
            f1 = f1_at_prior(tr_tpr, tr_fpr, prior)
            if f1 > best_f1:
                best_t, best_f1 = t, f1
        row = metric_row(ytest, own["own_test"], best_t, prior)
        row.update({"fold": fold, "experiment": "sanity_cancer_own_no_rowdrop_no_pca", "selected_threshold": best_t})
        sanity_rows.append(row)

        for w in WEIGHTS:
            b_cal = w * g_cal + (1 - w) * own["own_cal"]
            b_test = w * g_test + (1 - w) * own["own_test"]
            best_t, best_f1 = 0.80, -1.0
            for t in THRESHOLDS:
                tr_tpr, tr_fpr = tpr_fpr(ycal, (b_cal >= t).astype(int))
                f1 = f1_at_prior(tr_tpr, tr_fpr, prior)
                if f1 > best_f1:
                    best_t, best_f1 = t, f1
            row = metric_row(ytest, b_test, best_t, prior)
            row.update({"fold": fold, "weight": w, "threshold_kind": "calibration_optimized_0.70_0.90", "selected_threshold": best_t})
            blend_rows.append(row)
            for t in THRESHOLDS:
                grow = metric_row(ytest, b_test, t, prior)
                grow.update({"fold": fold, "weight": w, "fixed_threshold": t, "tracked_conservative_threshold": t in TRACK})
                grid_rows.append(grow)

    sanity = summarize(sanity_rows, ["experiment"])
    blend = summarize(blend_rows, ["weight", "threshold_kind"]).sort_values("weight")
    grid = summarize(grid_rows, ["weight", "fixed_threshold", "tracked_conservative_threshold"]).sort_values(["weight", "fixed_threshold"])
    sanity["preprocess_mode"] = "no_rowdrop"
    sanity["pipeline_role"] = "primary"
    sanity["pca"] = "rejected_ablation_only"
    sanity["calibration"] = "platt_sigmoid_unchanged"
    sanity["comparison_previous_reproduced_baseline_f1_star"] = 0.7077
    sanity["comparison_previous_threshold_stability_f1_star"] = 0.7136
    sanity["comparison_previous_calibration_threshold_grid_f1_star"] = 0.7162
    sanity["delta_vs_previous_reproduced_baseline"] = sanity["f1_star_mean"] - 0.7077
    sanity["note"] = "Fast holdout-calibration sanity scan; previous ~0.71 references used general-transfer blend and historical nested/threshold experiments."
    blend["delta_vs_own_sanity"] = blend["f1_star_mean"] - float(sanity["f1_star_mean"].iloc[0])
    grid["delta_vs_own_sanity"] = grid["f1_star_mean"] - float(sanity["f1_star_mean"].iloc[0])

    sanity_path = OUT_DIR / "cancer_primary_sanity_check_fast.csv"
    blend_path = OUT_DIR / "cancer_primary_blend_weight_scan_fast.csv"
    grid_path = OUT_DIR / "cancer_primary_blend_threshold_grid_fast.csv"
    sanity.to_csv(sanity_path, index=False)
    blend.to_csv(blend_path, index=False)
    grid.to_csv(grid_path, index=False)
    sanity_row = sanity.iloc[0]
    log_experiment_result({
        "panel": "CANCER",
        "experiment_name": "cancer_fast_sanity_no_rowdrop_no_pca",
        "pipeline_role": "sanity_check",
        "preprocess_mode": "no_rowdrop",
        "row_drop_enabled": False,
        "pca_enabled": False,
        "pca_variant": "none",
        "scaling_enabled": False,
        "outlier_control_enabled": True,
        "imputation_method": "median",
        "encoding_method": "safe ordinal / label encoding",
        "feature_selection_method": "none",
        "model_name": "capped fixed LightGBM cancer-own",
        "calibration_method": "Platt/sigmoid",
        "threshold_strategy": "calibration split grid_0.70_0.90",
        "selected_threshold": sanity_row.get("selected_threshold_mean", "per_fold"),
        "blend_enabled": False,
        "blend_weight": "n/a",
        "general_transfer_weight": 0.0,
        "cancer_own_weight": 1.0,
        "cv_strategy": "RepeatedStratifiedKFold holdout-calibration",
        "n_folds": sanity_row["n_folds"],
        "input_rows": 388,
        "output_rows": 388,
        "dropped_rows": 0,
        "input_features": 364,
        "output_features": 344,
        "dropped_features": 19,
        "positive_ratio_before": 0.690722,
        "positive_ratio_after": 0.690722,
        "auc": sanity_row["auc_mean"],
        "f1_star": sanity_row["f1_star_mean"],
        "finals_adjusted_f1": sanity_row["finals_f1_mean"],
        "precision": sanity_row["precision_mean"],
        "recall": sanity_row["recall_mean"],
        "pred_positive_mean": sanity_row["pred_pos_mean"],
        "delta_f1_vs_baseline": "n/a",
        "notes": "Fast holdout-calibration scan; use as pre-final signal only.",
        "output_file": sanity_path,
    })
    for _, blend_log in blend.iterrows():
        weight = float(blend_log["weight"])
        log_experiment_result({
            "panel": "CANCER",
            "experiment_name": f"cancer_fast_blend_weight_{weight:.1f}_grid_0.70_0.90",
            "pipeline_role": "sanity_check",
            "preprocess_mode": "no_rowdrop",
            "row_drop_enabled": False,
            "pca_enabled": False,
            "pca_variant": "none",
            "scaling_enabled": False,
            "outlier_control_enabled": True,
            "imputation_method": "median",
            "encoding_method": "safe ordinal / label encoding",
            "feature_selection_method": "none",
            "model_name": "capped fixed LightGBM + general transfer",
            "calibration_method": "Platt/sigmoid",
            "threshold_strategy": blend_log.get("threshold_kind", "grid_0.70_0.90"),
            "selected_threshold": blend_log.get("selected_threshold_mean", "per_fold"),
            "blend_enabled": True,
            "blend_weight": weight,
            "general_transfer_weight": weight,
            "cancer_own_weight": round(1.0 - weight, 1),
            "cv_strategy": "RepeatedStratifiedKFold holdout-calibration",
            "n_folds": blend_log["n_folds"],
            "input_rows": 388,
            "output_rows": 388,
            "dropped_rows": 0,
            "input_features": 364,
            "output_features": 344,
            "dropped_features": 19,
            "positive_ratio_before": 0.690722,
            "positive_ratio_after": 0.690722,
            "auc": blend_log["auc_mean"],
            "f1_star": blend_log["f1_star_mean"],
            "finals_adjusted_f1": blend_log["finals_f1_mean"],
            "precision": blend_log["precision_mean"],
            "recall": blend_log["recall_mean"],
            "pred_positive_mean": blend_log["pred_pos_mean"],
            "delta_f1_vs_baseline": blend_log["delta_vs_own_sanity"],
            "notes": "Fast blend scan row; not full final blend harness.",
            "output_file": blend_path,
        })

    best = blend.sort_values(["f1_star_mean", "finals_f1_mean"], ascending=False).iloc[0].to_dict()
    best_tracked = grid[grid["tracked_conservative_threshold"]].sort_values(["f1_star_mean", "finals_f1_mean"], ascending=False).iloc[0].to_dict()
    decision = {
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
    }
    decision_path = OUT_DIR / "cancer_decision_log.json"
    decision_path.write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")
    write_decision_log([
        "Strict row-drop rejected for CANCER; ablation only.",
        "no_rowdrop selected as primary CANCER preprocessing.",
        "PCA rejected for CANCER; ablation only.",
        "Primary CANCER pipeline remains updated CAT features + no_rowdrop + safe encoding + median imputation + outlier control + no PCA.",
        "Platt/sigmoid calibration unchanged for now.",
        f"Latest fast blend scan best weight={best['weight']} threshold_mean={best.get('selected_threshold_mean')}.",
    ])
    report_path = OUT_DIR / "cancer_primary_fast_report.md"
    report_path.write_text(
        "# CANCER Primary Fast Sanity And Blend Scan\n\n"
        "Strict row-drop and PCA are ablations only. This scan uses leakage-safe outer-fold holdout calibration.\n\n"
        "## Sanity\n\n" + markdown_table(sanity) + "\n\n"
        "## Blend\n\n" + markdown_table(blend) + "\n\n"
        "## Tracked 0.80-0.82\n\n" + markdown_table(grid[grid["tracked_conservative_threshold"]]),
        encoding="utf-8",
    )
    print("\nSanity")
    print(sanity.to_string(index=False))
    print("\nBlend")
    print(blend.to_string(index=False))
    print(f"\nSaved {sanity_path}\nSaved {blend_path}\nSaved {grid_path}\nSaved {decision_path}\nSaved {report_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=6)
    args = parser.parse_args()
    run(args.splits, args.repeats)


def markdown_table(df):
    cols = list(df.columns)
    rows = [[str(v) for v in row] for row in df.astype(object).itertuples(index=False, name=None)]
    widths = [max(len(str(c)), *(len(r[i]) for r in rows)) if rows else len(str(c)) for i, c in enumerate(cols)]

    def fmt(vals):
        return "| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(vals)) + " |"

    return "\n".join([fmt(cols), "| " + " | ".join("-" * w for w in widths) + " |"] + [fmt(r) for r in rows])


if __name__ == "__main__":
    main()
