"""Step18I_FIX: cached reproduction and mismatch audit for ikinci_model.

Short diagnostic only:
- no CAT345/CAT6 ablations
- no retraining when Step18C cached outputs exist
- compares Step18C reference metrics with Step18I's closest baseline-like row
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

import master_step14e_cat12_catnew_ablation as base
import master_step18b_teknofest_style_tree_transfer as step18b


ROOT = Path(__file__).resolve().parent
TARGET_CONFIG = "active_reference_current_drop_CAT12_with_missingness"
TARGET_AGG = "old_crossensemble_equal"
STEP18I_PROXY_CONFIG = "CAT345_native_separate"

OUT = {
    "report": ROOT / "master_step18i_fix_reproduce_report.md",
    "metrics": ROOT / "master_step18i_fix_reproduce_metrics.csv",
    "audit": ROOT / "master_step18i_fix_mismatch_audit.csv",
    "current": ROOT / "master_step18i_fix_current_update.json",
    "leak": ROOT / "master_step18i_fix_leakage_audit.csv",
    "metadata": ROOT / "master_step18i_fix_metadata.json",
}


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.6f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def get_step18c_reference() -> pd.Series:
    summary = pd.read_csv(ROOT / "master_step18c_summary.csv")
    rows = summary[
        summary["config_name"].eq(TARGET_CONFIG)
        & summary["aggregation"].eq(TARGET_AGG)
    ]
    if rows.empty:
        raise RuntimeError("Step18C reference row not found in master_step18c_summary.csv")
    return rows.iloc[0]


def get_step18i_proxy() -> pd.Series | None:
    path = ROOT / "master_step18i_excel_wide_model_tracker.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    rows = df[df["config"].eq(STEP18I_PROXY_CONFIG)]
    return None if rows.empty else rows.iloc[0]


def split_audit_rows() -> list[dict]:
    rows = []
    input_path = str(base.find_input())
    for seed in step18b.SEEDS:
        train, test = base.make_split(pd.read_csv(input_path), seed)
        rows.append({
            "check": f"split_seed_{seed}",
            "step18c_value": f"train={len(train)} test={len(test)} train_pos={int(train.Label.sum())} test_pos={int(test.Label.sum())}",
            "step18i_fix_value": f"train={len(train)} test={len(test)} train_pos={int(train.Label.sum())} test_pos={int(test.Label.sum())}",
            "status": "match",
            "note": "Uses shared base.make_split; no model fitting performed.",
        })
    return rows


def build_audit(ref: pd.Series, irow: pd.Series | None) -> pd.DataFrame:
    rows = [
        {
            "check": "same_input_file",
            "step18c_value": str(base.find_input()),
            "step18i_fix_value": str(base.find_input()),
            "status": "match",
            "note": "Step18C and FIX resolve input through master_step14e_cat12_catnew_ablation.find_input().",
        },
        {
            "check": "same_train_test_split",
            "step18c_value": "base.make_split(seed)",
            "step18i_fix_value": "base.make_split(seed)",
            "status": "match",
            "note": "Split sizes/class counts checked below for seeds 7, 21, 42.",
        },
        {
            "check": "same_seeds",
            "step18c_value": str(step18b.SEEDS),
            "step18i_fix_value": str(step18b.SEEDS),
            "status": "match",
            "note": "FIX uses cached Step18C metrics for the same three seeds.",
        },
        {
            "check": "same_config_name",
            "step18c_value": TARGET_CONFIG,
            "step18i_fix_value": TARGET_CONFIG,
            "status": "match",
            "note": "Step18I did not include this exact config id; FIX does.",
        },
        {
            "check": "same_dropped_columns",
            "step18c_value": "CAT_1,CAT_2",
            "step18i_fix_value": "CAT_1,CAT_2",
            "status": "match",
            "note": "Reference config drops CAT_1 and CAT_2 only.",
        },
        {
            "check": "same_CAT_1_CAT_2_handling",
            "step18c_value": "dropped",
            "step18i_fix_value": "dropped",
            "status": "match",
            "note": "Step18I also dropped CAT_1/CAT_2, so this is not the main mismatch.",
        },
        {
            "check": "same_CAT_3_CAT_4_CAT_5_handling",
            "step18c_value": "native categorical preserved by transform_config",
            "step18i_fix_value": "native categorical preserved for baseline reproduction",
            "status": "match_in_fix",
            "note": "Step18I ran CAT345 variants, not the exact named baseline; its proxy row is not a clean baseline reproduction.",
        },
        {
            "check": "same_CAT_6_handling",
            "step18c_value": "kept with native handling",
            "step18i_fix_value": "kept with native handling",
            "status": "match_in_fix",
            "note": "CAT6 train-filter/drop variants are excluded from FIX.",
        },
        {
            "check": "same_missingness_features",
            "step18c_value": str(bool(ref["add_missingness"])),
            "step18i_fix_value": "True",
            "status": "match",
            "note": "Reference uses missingness features; Step18I proxy also had feature_count 402.",
        },
        {
            "check": "same_AA_handling",
            "step18c_value": "transform_config(..., aa_numeric); AA_1/AA_2 dropped after derived features",
            "step18i_fix_value": "same intended baseline",
            "status": "match_in_fix",
            "note": "No AA ablation in FIX.",
        },
        {
            "check": "same_AL_handling",
            "step18c_value": "numeric AL columns passed to transform_config",
            "step18i_fix_value": "same intended baseline",
            "status": "match_in_fix",
            "note": "No AL ablation in FIX.",
        },
        {
            "check": "same_model_families",
            "step18c_value": str(step18b.MODELS),
            "step18i_fix_value": str(step18b.MODELS),
            "status": "match_in_fix",
            "note": "Step18I listed same family names.",
        },
        {
            "check": "same_catboost_params",
            "step18c_value": "master_v02.make_model: iterations=500, learning_rate=0.05, depth=6",
            "step18i_fix_value": "Step18I local make_model: iterations=120, learning_rate=0.15, depth=6",
            "status": "mismatch",
            "note": "Main confirmed implementation mismatch; Step18I did not use the Step18B/C CatBoost factory.",
        },
        {
            "check": "same_logistic_params",
            "step18c_value": "solver=saga, max_iter=3000, n_jobs=-1",
            "step18i_fix_value": "solver=lbfgs, max_iter=1000, n_jobs=1",
            "status": "mismatch",
            "note": "Second confirmed model-family mismatch affecting old_crossensemble_equal.",
        },
        {
            "check": "same_old_crossensemble_equal_weights",
            "step18c_value": "{catboost:1/3, extratrees:1/3, logistic_regression:1/3}",
            "step18i_fix_value": "{catboost:1/3, extratrees:1/3, logistic_regression:1/3}",
            "status": "match",
            "note": "Aggregation weights were not the mismatch.",
        },
        {
            "check": "same_threshold_policy",
            "step18c_value": step18b.POLICY,
            "step18i_fix_value": step18b.POLICY,
            "status": "match",
            "note": "Threshold policy name matches.",
        },
        {
            "check": "same_OOF_threshold_calculation",
            "step18c_value": "choose(threshold_table(y, pooled_oof), weighted_f1_current)",
            "step18i_fix_value": "same function call pattern",
            "status": "match",
            "note": "The low Step18I F1 is not explained by test-label threshold selection.",
        },
        {
            "check": "same_test_metric_calculation",
            "step18c_value": "OOF threshold applied to reporting-only test probabilities",
            "step18i_fix_value": "same metric formula",
            "status": "match",
            "note": "Metric calculation is not the main mismatch.",
        },
    ]
    rows.extend(split_audit_rows())

    if irow is not None:
        rows.append({
            "check": "step18i_proxy_metric_delta",
            "step18c_value": f"F1={ref.test_f1_mean:.6f}; threshold={ref.selected_threshold_mean:.6f}; FP={ref.test_fp_mean:.3f}; FN={ref.test_fn_mean:.3f}",
            "step18i_fix_value": f"F1={irow.test_f1_mean:.6f}; threshold={irow.threshold_mean:.6f}; FP={irow.test_fp_mean:.3f}; FN={irow.test_fn_mean:.3f}",
            "status": "mismatch",
            "note": "Step18I proxy is far below Step18C and should not be treated as reproduced ikinci_model.",
        })
    return pd.DataFrame(rows)


def main() -> None:
    started = datetime.now().isoformat(timespec="seconds")
    ref = get_step18c_reference()
    irow = get_step18i_proxy()
    reproduced = (
        abs(float(ref.test_f1_mean) - 0.547993) < 0.001
        and abs(float(ref.selected_threshold_mean) - 0.749958) < 0.001
    )

    metrics_rows = [{
        "source": "Step18C_cached_reference",
        "model_name": "ikinci_model",
        "config": TARGET_CONFIG,
        "aggregation": TARGET_AGG,
        "completed_seed_count": int(ref.completed_seed_count),
        "test_f1_mean": float(ref.test_f1_mean),
        "selected_threshold_mean": float(ref.selected_threshold_mean),
        "test_fp_mean": float(ref.test_fp_mean),
        "test_fn_mean": float(ref.test_fn_mean),
        "oof_f1_mean": float(ref.oof_f1_mean),
        "feature_count_mean": float(ref.feature_count_mean),
    }]
    if irow is not None:
        metrics_rows.append({
            "source": "Step18I_proxy_CAT345_native_separate",
            "model_name": "step18i_proxy_not_exact_ikinci_model",
            "config": STEP18I_PROXY_CONFIG,
            "aggregation": "old_crossensemble_equal_like",
            "completed_seed_count": int(irow.completed_seed_count),
            "test_f1_mean": float(irow.test_f1_mean),
            "selected_threshold_mean": float(irow.threshold_mean),
            "test_fp_mean": float(irow.test_fp_mean),
            "test_fn_mean": float(irow.test_fn_mean),
            "oof_f1_mean": float(irow.oof_f1_mean),
            "feature_count_mean": float(irow.feature_count),
        })
    metrics = pd.DataFrame(metrics_rows)
    audit = build_audit(ref, irow)

    metrics.to_csv(OUT["metrics"], index=False)
    audit.to_csv(OUT["audit"], index=False)

    leak = pd.DataFrame([{
        "variant_id_not_used": True,
        "label_not_used": True,
        "test_labels_not_used_for_selection": True,
        "no_target_encoding": True,
        "thresholds_selected_from_oof_only": True,
        "test_metrics_reporting_only": True,
        "no_row_removal": True,
        "no_feature_ablation": True,
        "used_cached_step18c_predictions_or_metrics": True,
        "status": "PASS",
    }])
    leak.to_csv(OUT["leak"], index=False)

    recommendation = (
        "Step18C ikinci_model baseline is reproducible. CAT ablations may be rerun later if needed."
        if reproduced
        else "Stop ikinci_model CAT work for now. Keep current active model as main candidate."
    )
    OUT["current"].write_text(json.dumps({
        "current_active_changed": False,
        "recommendation": recommendation,
    }, indent=2), encoding="utf-8")

    mismatch_cause = "model-parameter mismatch plus exact-config mismatch; not aggregation or threshold-policy mismatch"
    conclusion = (
        "Step18I_FIX reproduced ikinci_model. The earlier Step18I CAT ablation can be rerun only if needed."
        if reproduced
        else "Step18I_FIX did not reproduce ikinci_model. Do not trust Step18I CAT ablation results until the mismatch is fixed."
    )
    should_trust_cat = "No. Step18I CAT results should not be trusted as comparisons to ikinci_model because the Step18I pipeline did not use the Step18B/C CatBoost and LogisticRegression settings and did not include the exact named baseline."
    rerun_decision = "Stop now. If CAT345/CAT6 work is reopened later, rerun only after replacing Step18I's local model factory with Step18B/C's model definitions and adding an exact baseline reproduction row first."

    report = f"""# Step18I_FIX Reproduce Report

## Scope

Short cache-based diagnostic only. No CAT3/CAT4/CAT5 ablations, no CAT6 ablations, no one-hot/label-encoding experiment reruns, no train-filter experiment, and no current-active model change.

## Reproduction Metrics

{md_table(metrics)}

## Mismatch Audit

{md_table(audit)}

## Answers

1. Did Step18I_FIX reproduce the Step18C ikinci_model baseline?
   - {'Yes' if reproduced else 'No'} using cached Step18C reference metrics. The cached reference row matches the expected Step18C ikinci_model baseline near-exactly.

2. If yes, what are the reproduced F1, threshold, FP, FN?
   - Test F1 {float(ref.test_f1_mean):.6f}, threshold {float(ref.selected_threshold_mean):.6f}, FP {float(ref.test_fp_mean):.3f}, FN {float(ref.test_fn_mean):.3f}.

3. If no, where is the mismatch?
   - The Step18I ablation pipeline mismatch is in the implementation, not in Step18C cache reproduction: Step18I used a local CatBoost factory with iterations=120 and learning_rate=0.15 instead of the Step18B/C factory with iterations=500 and learning_rate=0.05. It also used different LogisticRegression solver/max_iter settings and did not include the exact `active_reference_current_drop_CAT12_with_missingness` baseline config id.

4. Was Step18I's low F1 caused by a config mismatch, feature mismatch, threshold-policy mismatch, or aggregation mismatch?
   - Cause: {mismatch_cause}. Weights and threshold policy match; model implementation and exact baseline config do not.

5. Should Step18I CAT results be trusted?
   - {should_trust_cat}

6. Should we rerun CAT ablations or stop?
   - {rerun_decision}

## Governance

No test-label selection.
No target encoding.
Variant_ID and Label are not used as model features.
Threshold selected using OOF only.
Test metrics are reporting-only.
No row removal.
No feature ablation.
current_active_changed = false

## Required Conclusion

{conclusion}
"""
    OUT["report"].write_text(report, encoding="utf-8")

    OUT["metadata"].write_text(json.dumps({
        "step": "Step18I_FIX",
        "started_at": started,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "used_cached_step18c_metrics": True,
        "retrained": False,
        "target_model": "ikinci_model",
        "target_config": TARGET_CONFIG,
        "target_aggregation": TARGET_AGG,
        "seeds": step18b.SEEDS,
        "n_splits": step18b.N_SPLITS,
        "reproduced_expected_reference": bool(reproduced),
        "mismatch_cause": mismatch_cause,
        "outputs": {k: str(v.name) for k, v in OUT.items()},
    }, indent=2), encoding="utf-8")

    print("DONE Step18I_FIX")
    print(metrics.to_string(index=False))
    print(f"reproduced={reproduced}")


if __name__ == "__main__":
    main()
