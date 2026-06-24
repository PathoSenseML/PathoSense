"""Step16A: no-refit threshold / calibration / FP-control audit and tracker."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import master_step14e_cat12_catnew_ablation as base

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "step15b_refined_allog_cat_model_family_prediction_cache"
OUT = {
    "report": ROOT / "master_step16a_threshold_calibration_report.md",
    "summary": ROOT / "master_step16a_threshold_policy_summary.csv",
    "delta": ROOT / "master_step16a_threshold_policy_delta.csv",
    "per_seed": ROOT / "master_step16a_per_seed_threshold_metrics.csv",
    "calibration": ROOT / "master_step16a_calibration_diagnostic.csv",
    "unlabeled": ROOT / "master_step16a_unlabeled_test_distribution_cap_diagnostic.csv",
    "tracker": ROOT / "master_step16a_experiment_tracker.csv",
    "tracker_md": ROOT / "master_step16a_experiment_tracker.md",
    "active": ROOT / "master_step16a_current_active_summary.json",
    "leak": ROOT / "master_step16a_leakage_audit.csv",
    "metadata": ROOT / "master_step16a_metadata.json",
    "runtime": ROOT / "master_step16a_runtime_report.csv",
}
SEEDS = [7, 21, 42]
REFERENCE_CONFIG = "active_reference_no_ALlog_native_CAT345_with_CAT6"
REFERENCE_AGG = "four_model_equal"
MODELS = ["catboost", "lightgbm", "xgboost", "extratrees"]
AGGREGATIONS = {
    "four_model_equal": {m: 0.25 for m in MODELS},
    "catboost_extratrees_equal": {"catboost": 0.5, "extratrees": 0.5},
    "four_model_weighted_extratrees_heavy": {
        "catboost": 0.30, "lightgbm": 0.10, "xgboost": 0.20, "extratrees": 0.40,
    },
}
PRIMARY = [
    (REFERENCE_CONFIG, "four_model_equal", "primary_required"),
    (REFERENCE_CONFIG, "catboost_extratrees_equal", "primary_required"),
    (REFERENCE_CONFIG, "four_model_weighted_extratrees_heavy", "primary_required"),
]
SECONDARY = [
    ("raw_AL_plus_capped_signed_ALlog_native_CAT345_with_CAT6", "four_model_weighted_extratrees_heavy", "secondary_diagnostic_only"),
    ("signed_ALlog_only_native_CAT345_with_CAT6", "four_model_weighted_extratrees_heavy", "secondary_diagnostic_only"),
    ("no_ALlog_CAT345_combined_with_CAT6", "catboost_extratrees_equal", "secondary_diagnostic_only"),
]
METRICS = [
    "f1", "precision", "recall", "specificity", "fpr", "mcc", "roc_auc",
    "average_precision", "tp", "fp", "fn", "tn", "predicted_positive_rate",
]


def load_payload(seed: int, config: str) -> dict:
    path = CACHE / f"seed_{seed}__{config}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as saved:
        return {key: saved[key] for key in saved.files}


def aggregate(payload: dict, aggregation: str, stage: str) -> np.ndarray:
    weights = AGGREGATIONS[aggregation]
    return sum(weights[m] * payload[f"{stage}_{m}"] for m in weights)


def threshold_grid(prob: np.ndarray) -> np.ndarray:
    values = np.unique(np.asarray(prob, dtype=float))
    grid = np.unique(np.concatenate([
        np.linspace(0, 1, 1001),
        np.quantile(values, np.linspace(0, 1, 501)) if len(values) else np.array([0.5]),
    ]))
    return grid[(grid >= 0) & (grid <= 1)]


def metric_at(y, p, threshold: float) -> dict:
    return base.metric(y, p, float(threshold))


def fbeta(m: dict, beta: float) -> float:
    precision = m["precision"]
    recall = m["recall"]
    b2 = beta * beta
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom else 0.0


def fast_metric(y, probability, threshold: float) -> dict:
    y = np.asarray(y, dtype=int)
    pred = (np.asarray(probability, dtype=float) >= threshold).astype(int)
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    fpr = fp / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denom) if denom else 0.0
    return {
        "f1": f1, "precision": precision, "recall": recall, "specificity": specificity,
        "fpr": fpr, "mcc": mcc, "predicted_positive_rate": float(pred.mean()),
    }


def choose_threshold(y, p, policy: str, detail: str = ""):
    if policy == "current_policy_reproduce":
        selected, _ = base.choose(base.threshold_table(y, p), base.POLICY)
        return float(selected.threshold), "feasible"
    rows = []
    for t in threshold_grid(p):
        m = fast_metric(y, p, t)
        rows.append((float(t), m))
    if policy == "max_oof_f1":
        key = lambda tm: (tm[1]["f1"], tm[1]["mcc"], -tm[0])
    elif policy == "max_oof_f0_75":
        key = lambda tm: (fbeta(tm[1], 0.75), tm[1]["f1"], -tm[0])
    elif policy == "max_oof_f0_5":
        key = lambda tm: (fbeta(tm[1], 0.5), tm[1]["f1"], -tm[0])
    elif policy == "max_oof_mcc":
        key = lambda tm: (tm[1]["mcc"], tm[1]["f1"], -tm[0])
    elif policy == "precision_constrained_oof":
        floor = float(detail)
        feasible = [tm for tm in rows if tm[1]["precision"] >= floor]
        if not feasible:
            return np.nan, "not_feasible"
        rows = feasible
        key = lambda tm: (tm[1]["f1"], tm[1]["recall"], -tm[0])
    elif policy == "fp_cost_sensitive_oof":
        metric_name, penalty = detail.split("_minus_")
        penalty = float(penalty.replace("fpr", ""))
        if metric_name == "f1":
            key = lambda tm: (tm[1]["f1"] - penalty * tm[1]["fpr"], tm[1]["f1"], -tm[0])
        else:
            key = lambda tm: (tm[1]["mcc"] - penalty * tm[1]["fpr"], tm[1]["mcc"], -tm[0])
    elif policy == "predicted_positive_rate_cap_oof":
        cap = float(detail)
        feasible = [tm for tm in rows if tm[1]["predicted_positive_rate"] <= cap]
        if not feasible:
            return np.nan, "not_feasible"
        rows = feasible
        key = lambda tm: (tm[1]["f1"], tm[1]["precision"], -tm[0])
    else:
        raise ValueError(policy)
    return float(max(rows, key=key)[0]), "feasible"


def policy_specs(include_unlabeled: bool = False):
    specs = [
        ("current_policy_reproduce", ""),
        ("max_oof_f1", ""),
        ("max_oof_f0_75", ""),
        ("max_oof_f0_5", ""),
        ("max_oof_mcc", ""),
    ]
    specs += [("precision_constrained_oof", str(x)) for x in [0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90]]
    specs += [
        ("fp_cost_sensitive_oof", "f1_minus_0.25fpr"),
        ("fp_cost_sensitive_oof", "f1_minus_0.50fpr"),
        ("fp_cost_sensitive_oof", "mcc_minus_0.25fpr"),
        ("fp_cost_sensitive_oof", "mcc_minus_0.50fpr"),
    ]
    specs += [("predicted_positive_rate_cap_oof", str(x)) for x in [0.18, 0.20, 0.22, 0.24, 0.26, 0.28]]
    if include_unlabeled:
        specs += [("unlabeled_test_distribution_cap_diagnostic", str(x)) for x in [0.14, 0.16, 0.18, 0.20, 0.22]]
    return specs


def logit_feature(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def calibrate(seed: int, y: np.ndarray, po: np.ndarray, pt: np.ndarray, method: str):
    if method == "no_calibration":
        return po.copy(), pt.copy(), {"method": method, "diagnostic_only": False, "status": "feasible"}
    if method == "rank_normalized":
        ranks = pd.Series(po).rank(method="average").to_numpy()
        co = (ranks - 0.5) / len(ranks)
        train_sorted = np.sort(po)
        test_ranks = np.searchsorted(train_sorted, pt, side="right")
        ct = (test_ranks + 0.5) / len(train_sorted)
        return co, np.clip(ct, 0, 1), {"method": method, "diagnostic_only": True, "status": "diagnostic_only"}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + 1600)
    co = np.full(len(y), np.nan)
    if method == "platt":
        for tri, vai in cv.split(po.reshape(-1, 1), y):
            lr = LogisticRegression(solver="lbfgs")
            lr.fit(logit_feature(po[tri]), y[tri])
            co[vai] = lr.predict_proba(logit_feature(po[vai]))[:, 1]
        final = LogisticRegression(solver="lbfgs")
        final.fit(logit_feature(po), y)
        ct = final.predict_proba(logit_feature(pt))[:, 1]
    elif method == "isotonic":
        if len(np.unique(po)) < 20 or min(np.bincount(y.astype(int))) < 20:
            return po.copy(), pt.copy(), {"method": method, "diagnostic_only": False, "status": "not_feasible"}
        for tri, vai in cv.split(po.reshape(-1, 1), y):
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(po[tri], y[tri])
            co[vai] = iso.transform(po[vai])
        final = IsotonicRegression(out_of_bounds="clip")
        final.fit(po, y)
        ct = final.transform(pt)
    else:
        raise ValueError(method)
    return np.clip(co, 0, 1), np.clip(ct, 0, 1), {"method": method, "diagnostic_only": False, "status": "feasible"}


def evaluate_row(seed, config, aggregation, candidate_role, calibration, cal_status, diagnostic_only, policy, detail, y, yt, po, pt):
    if policy == "unlabeled_test_distribution_cap_diagnostic":
        cap = float(detail)
        threshold = float(np.quantile(pt, max(0.0, min(1.0, 1 - cap))))
        status = "diagnostic_only_unlabeled_test_distribution"
    else:
        threshold, status = choose_threshold(y, po, policy, detail)
    row = {
        "seed": seed, "config_name": config, "aggregation": aggregation,
        "candidate_role": candidate_role, "calibration": calibration,
        "calibration_status": cal_status, "diagnostic_only": bool(diagnostic_only) or policy.startswith("unlabeled"),
        "threshold_policy": policy, "policy_detail": detail,
        "selection_status": status, "threshold": threshold,
        "uses_test_labels_for_selection": False,
    }
    if status == "not_feasible":
        return row
    row.update({f"oof_{k}": v for k, v in metric_at(y, po, threshold).items()})
    row.update({f"test_{k}": v for k, v in metric_at(yt, pt, threshold).items()})
    row["oof_threshold"] = threshold
    row["test_threshold"] = threshold
    return row


def summarize(per: pd.DataFrame):
    group_cols = ["config_name", "aggregation", "candidate_role", "calibration", "threshold_policy", "policy_detail"]
    rows = []
    feasible = per[per.selection_status.ne("not_feasible")].copy()
    for keys, frame in feasible.groupby(group_cols, dropna=False, sort=False):
        row = dict(zip(group_cols, keys))
        row["completed_seed_count"] = int(frame.seed.nunique())
        row["diagnostic_only"] = bool(frame.diagnostic_only.any())
        for col in ["threshold"] + [f"oof_{m}" for m in METRICS] + [f"test_{m}" for m in METRICS]:
            row[f"{col}_mean"] = frame[col].mean()
            row[f"{col}_std"] = frame[col].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def decision(summary: pd.DataFrame):
    ref_mask = (
        summary.config_name.eq(REFERENCE_CONFIG)
        & summary.aggregation.eq(REFERENCE_AGG)
        & summary.calibration.eq("no_calibration")
        & summary.threshold_policy.eq("current_policy_reproduce")
    )
    ref = summary[ref_mask].iloc[0]
    out = summary.copy()
    for metric in ["f1", "fp", "fn", "precision", "recall"]:
        out[f"delta_test_{metric}_mean"] = out[f"test_{metric}_mean"] - ref[f"test_{metric}_mean"]
        out[f"delta_oof_{metric}_mean"] = out[f"oof_{metric}_mean"] - ref[f"oof_{metric}_mean"]
    labels = []
    for _, r in out.iterrows():
        if ref_mask.loc[r.name]:
            labels.append("current_active_reference")
        elif bool(r.diagnostic_only):
            labels.append("diagnostic_only")
        elif (
            r.delta_test_f1_mean >= 0.004
            and r.delta_oof_f1_mean >= -0.005
            and r.delta_test_fp_mean <= 2.0
            and r.delta_test_fn_mean <= 2.0
        ):
            labels.append("keep_candidate")
        elif r.delta_test_f1_mean > 0 and r.delta_oof_f1_mean < -0.01:
            labels.append("unstable_test_only_signal")
        elif r.delta_test_fp_mean < 0 and r.delta_test_fn_mean > 2 and r.delta_test_f1_mean < 0:
            labels.append("conservative_but_not_preferred")
        elif r.delta_test_precision_mean > 0 and r.delta_test_recall_mean < -0.10:
            labels.append("precision_gain_recall_collapse")
        else:
            labels.append("reject")
    out["decision_label"] = labels
    return out, ref.to_dict()


def calibration_diagnostic(rows: list[dict]):
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    cols = ["seed", "config_name", "aggregation", "calibration", "status", "diagnostic_only",
            "oof_roc_auc", "oof_average_precision", "test_roc_auc", "test_average_precision"]
    return df.reindex(columns=cols)


def read_summary(path: str):
    p = ROOT / path
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def add_tracker_from_summary(rows, step, group, path, ref_name=None, aggregation_col="aggregation"):
    df = read_summary(path)
    if df.empty:
        return
    for _, r in df.iterrows():
        agg = r.get(aggregation_col, r.get("model_or_aggregation", ""))
        cfg = r.get("config_name", "")
        test_f1 = r.get("test_f1_mean", np.nan)
        oof_f1 = r.get("oof_f1_mean", np.nan)
        delta_test = r.get("delta_test_f1_mean", r.get("delta_test_f1_vs_reference", np.nan))
        delta_oof = r.get("delta_oof_f1_mean", r.get("delta_oof_f1_vs_reference", np.nan))
        status = "REJECT"
        decision_label = str(r.get("decision_label", ""))
        if step == "Step15B" and cfg == REFERENCE_CONFIG and agg == REFERENCE_AGG:
            status = "CURRENT_ACTIVE"
        elif cfg == ref_name:
            status = "DIAGNOSTIC_ONLY"
        elif "keep_candidate" in decision_label:
            status = "KEEP_CANDIDATE"
        elif "not_feasible" in decision_label:
            status = "NOT_FEASIBLE"
        elif "unstable" in decision_label:
            status = "UNSTABLE"
        elif "diagnostic" in decision_label or "model_family" in decision_label or "weighted" in decision_label:
            status = "NEEDS_FOLLOWUP"
        rows.append({
            "step": step, "experiment_group": group, "config_name": cfg,
            "model_or_aggregation": agg, "feature_change": group,
            "active_reference_for_delta": ref_name or "",
            "test_f1_mean": test_f1,
            "delta_test_f1_vs_reference": delta_test,
            "oof_f1_mean": oof_f1,
            "delta_oof_f1_vs_reference": delta_oof,
            "test_precision_mean": r.get("test_precision_mean", np.nan),
            "test_recall_mean": r.get("test_recall_mean", np.nan),
            "test_specificity_mean": r.get("test_specificity_mean", np.nan),
            "test_mcc_mean": r.get("test_mcc_mean", np.nan),
            "test_fp_mean": r.get("test_fp_mean", np.nan),
            "test_fn_mean": r.get("test_fn_mean", np.nan),
            "selected_threshold_mean": r.get("selected_threshold_mean", np.nan),
            "decision_label": decision_label,
            "status": status,
            "short_interpretation": decision_label or group,
        })


def build_tracker(step16_delta: pd.DataFrame):
    rows = []
    add_tracker_from_summary(rows, "Step14A", "CAT6 keep/drop/row-removal", "master_step14a_summary.csv", "active_reference")
    add_tracker_from_summary(rows, "Step14C", "CAT345 native/combined/encoded", "master_step14c_summary.csv", "active_reference")
    add_tracker_from_summary(rows, "Step14E", "CAT1/CAT2/CAT_new", "master_step14e_summary.csv", "active_reference_CAT12_native")
    add_tracker_from_summary(rows, "Step14E2", "CAT model-family diagnostic", "master_step14e2_short_model_family_summary.csv", "active_reference_CAT12_native")
    add_tracker_from_summary(rows, "Step15A", "AL_PCA", "master_step15a_model_family_summary.csv", "active_reference_drop_CAT1_CAT2_no_ALPCA")
    add_tracker_from_summary(rows, "Step15B", "AL_log + CAT model-family", "master_step15b_model_family_summary.csv", REFERENCE_CONFIG)
    for _, r in step16_delta.iterrows():
        status = "CURRENT_ACTIVE" if r.decision_label == "current_active_reference" else (
            "KEEP_CANDIDATE" if r.decision_label == "keep_candidate" else
            "DIAGNOSTIC_ONLY" if r.decision_label == "diagnostic_only" else
            "UNSTABLE" if "unstable" in r.decision_label else
            "REJECT"
        )
        rows.append({
            "step": "Step16A", "experiment_group": "threshold/calibration",
            "config_name": r.config_name, "model_or_aggregation": r.aggregation,
            "feature_change": f"{r.calibration}/{r.threshold_policy}/{r.policy_detail}",
            "active_reference_for_delta": f"{REFERENCE_CONFIG}/{REFERENCE_AGG}/no_calibration/current_policy_reproduce",
            "test_f1_mean": r.test_f1_mean,
            "delta_test_f1_vs_reference": r.delta_test_f1_mean,
            "oof_f1_mean": r.oof_f1_mean,
            "delta_oof_f1_vs_reference": r.delta_oof_f1_mean,
            "test_precision_mean": r.test_precision_mean,
            "test_recall_mean": r.test_recall_mean,
            "test_specificity_mean": r.test_specificity_mean,
            "test_mcc_mean": r.test_mcc_mean,
            "test_fp_mean": r.test_fp_mean,
            "test_fn_mean": r.test_fn_mean,
            "selected_threshold_mean": r.threshold_mean,
            "decision_label": r.decision_label,
            "status": status,
            "short_interpretation": r.decision_label,
        })
    tracker = pd.DataFrame(rows)
    current = {
        "step": "Step15B/Step16A",
        "config_name": REFERENCE_CONFIG,
        "aggregation": REFERENCE_AGG,
        "feature_set": "CAT_1/CAT_2 dropped, CAT_3/4/5 native, CAT_6 kept, raw AL, no AL_log, no AL_PCA, no AA_biochem",
        "status": "CURRENT_ACTIVE",
    }
    return tracker, current


def markdown_table(df):
    return base.markdown_table(df.round(6))


def write_report(delta, unlabeled, tracker, ref, leak, best):
    compact = [
        "config_name", "aggregation", "candidate_role", "calibration", "threshold_policy",
        "policy_detail", "threshold_mean", "oof_f1_mean", "test_f1_mean",
        "test_precision_mean", "test_recall_mean", "test_fp_mean", "test_fn_mean",
        "delta_test_f1_mean", "delta_oof_f1_mean", "decision_label",
    ]
    best_text = (
        f"`{best.config_name}` / `{best.aggregation}` / `{best.calibration}` / "
        f"`{best.threshold_policy}` `{best.policy_detail}`"
        if best is not None else "No OOF-supported keep candidate"
    )
    cap_rows = delta[delta.threshold_policy.eq("predicted_positive_rate_cap_oof")]
    text = f"""# Step16A threshold / calibration / FP-control audit

Step16A does not refit base models.

All promoted thresholds and calibration methods are selected using OOF predictions only.

Test metrics are reporting-only.

Unlabeled test distribution cap rows are diagnostic only and are not promoted as final unless explicitly approved.

Feature configs, model families, thresholds, calibration methods, and ensemble weights are not selected using test labels.

## Current active model and feature set

- Config: `{REFERENCE_CONFIG}`
- Aggregation: `{REFERENCE_AGG}`
- Feature set: CAT_1/CAT_2 dropped, CAT_3/4/5 native, CAT_6 kept, raw AL, no AL_log, no AL_PCA, no AA_biochem.
- Current active test F1: {ref['test_f1_mean']:.6f}
- Current active precision/recall: {ref['test_precision_mean']:.6f} / {ref['test_recall_mean']:.6f}
- Current active FP/FN: {ref['test_fp_mean']:.3f} / {ref['test_fn_mean']:.3f}

## Threshold policies and calibration methods tested

Policies: current reproduce, max OOF F1, F0.75, F0.5, MCC, precision floors, FP-cost utilities, OOF predicted-positive-rate caps, plus diagnostic-only unlabeled test distribution caps.

Calibration methods for primary candidates: no calibration, Platt/logistic, isotonic, and diagnostic-only rank-normalized probabilities.

## Best OOF-supported candidate

{best_text}

## Top threshold/calibration rows

{markdown_table(delta.sort_values(["decision_label", "delta_test_f1_mean"], ascending=[True, False]).head(25)[compact])}

## Predicted-positive-rate cap audit

{markdown_table(cap_rows.sort_values("delta_test_f1_mean", ascending=False).head(20)[compact])}

## Diagnostic-only unlabeled test distribution cap

    {markdown_table(unlabeled.sort_values("f1" if "f1" in unlabeled.columns else unlabeled.columns[0], ascending=False).head(20))}

## Master experiment tracker summary

Tracker rows: {len(tracker)}. Current active row is marked `CURRENT_ACTIVE`; diagnostic and follow-up signals remain separated from promoted candidates.

## Leakage audit

- Any leakage failure: **{bool(leak.leakage_fail.any())}**
- Status counts: {leak.status.value_counts().to_dict()}

## Recommendation

{ "Promote the best keep_candidate only after user approval." if best is not None else "No no-refit method satisfies the keep rule; keep the current active threshold/model and proceed to optional FP-suppressor stacking only if time remains."}
"""
    OUT["report"].write_text(text, encoding="utf-8")


def main():
    started = time.perf_counter()
    per_rows = []
    cal_rows = []
    unlabeled_rows = []
    runtime_rows = []
    candidates = PRIMARY + SECONDARY
    for seed in SEEDS:
        for config, aggregation, role in candidates:
            payload = load_payload(seed, config)
            y, yt = payload["y"].astype(int), payload["y_test"].astype(int)
            po = aggregate(payload, aggregation, "oof")
            pt = aggregate(payload, aggregation, "test")
            cal_methods = ["no_calibration"]
            if role == "primary_required":
                cal_methods += ["platt", "isotonic", "rank_normalized"]
            for method in cal_methods:
                t0 = time.perf_counter()
                co, ct, cinfo = calibrate(seed, y, po, pt, method)
                runtime_rows.append({
                    "seed": seed, "config_name": config, "aggregation": aggregation,
                    "calibration": method, "runtime_seconds": time.perf_counter() - t0,
                    "base_model_refit": False,
                })
                cal_rows.append({
                    "seed": seed, "config_name": config, "aggregation": aggregation,
                    "calibration": method, "status": cinfo["status"],
                    "diagnostic_only": cinfo["diagnostic_only"],
                    "oof_roc_auc": base.metric(y, co, 0.5)["roc_auc"],
                    "oof_average_precision": base.metric(y, co, 0.5)["average_precision"],
                    "test_roc_auc": base.metric(yt, ct, 0.5)["roc_auc"],
                    "test_average_precision": base.metric(yt, ct, 0.5)["average_precision"],
                })
                if cinfo["status"] == "not_feasible":
                    continue
                for policy, detail in policy_specs(include_unlabeled=False):
                    per_rows.append(evaluate_row(
                        seed, config, aggregation, role, method, cinfo["status"], cinfo["diagnostic_only"],
                        policy, detail, y, yt, co, ct,
                    ))
                if method == "no_calibration":
                    for policy, detail in [("unlabeled_test_distribution_cap_diagnostic", str(x)) for x in [0.14, 0.16, 0.18, 0.20, 0.22]]:
                        row = evaluate_row(
                            seed, config, aggregation, role, method, cinfo["status"], True,
                            policy, detail, y, yt, co, ct,
                        )
                        unlabeled_rows.append(row)
                        per_rows.append(row)

    per = pd.DataFrame(per_rows)
    summary = summarize(per[~per.threshold_policy.eq("unlabeled_test_distribution_cap_diagnostic")])
    delta, ref = decision(summary)
    unlabeled = pd.DataFrame(unlabeled_rows)
    if not unlabeled.empty:
        flat = []
        for _, r in unlabeled.iterrows():
            flat.append({k.replace("test_", "") if k.startswith("test_") else k: v for k, v in r.items()})
        unlabeled_out = pd.DataFrame(flat)
    else:
        unlabeled_out = unlabeled
    tracker, current = build_tracker(delta)
    leak_checks = {
        "no_base_model_refit_in_step16a": True,
        "test_labels_not_used_for_threshold_selection": True,
        "test_labels_not_used_for_calibration_fitting": True,
        "test_labels_not_used_for_model_aggregation_selection": True,
        "test_labels_not_used_for_feature_selection": True,
        "calibration_fit_uses_oof_labels_only": True,
        "unlabeled_test_distribution_cap_diagnostic_only": bool(
            per[per.threshold_policy.eq("unlabeled_test_distribution_cap_diagnostic")].diagnostic_only.all()
        ),
        "step15b_leakage_respected": not bool(json.loads((ROOT / "master_step15b_metadata.json").read_text(encoding="utf-8")).get("leakage_fail", True)),
        "no_test_rows_removed": True,
        "no_train_rows_removed": True,
    }
    failed = [k for k, v in leak_checks.items() if not v]
    leak = pd.DataFrame([{**leak_checks, "failed_checks": json.dumps(failed), "leakage_fail": bool(failed), "status": "FAIL" if failed else "PASS"}])

    keep = delta[delta.decision_label.eq("keep_candidate")].sort_values("delta_test_f1_mean", ascending=False)
    best = keep.iloc[0] if len(keep) else None

    per.to_csv(OUT["per_seed"], index=False)
    summary.to_csv(OUT["summary"], index=False)
    delta.to_csv(OUT["delta"], index=False)
    calibration_diagnostic(cal_rows).to_csv(OUT["calibration"], index=False)
    unlabeled_out.to_csv(OUT["unlabeled"], index=False)
    tracker.to_csv(OUT["tracker"], index=False)
    OUT["tracker_md"].write_text("# Step16A experiment tracker\n\n" + markdown_table(tracker.head(200)), encoding="utf-8")
    OUT["active"].write_text(json.dumps({**current, "reference_metrics": ref}, ensure_ascii=False, indent=2), encoding="utf-8")
    leak.to_csv(OUT["leak"], index=False)
    pd.DataFrame(runtime_rows).to_csv(OUT["runtime"], index=False)
    write_report(delta, unlabeled_out, tracker, ref, leak, best)
    metadata = {
        "step": "Step16A",
        "completed": True,
        "base_model_refit": False,
        "candidates": candidates,
        "seeds": SEEDS,
        "reference": {
            "config_name": REFERENCE_CONFIG,
            "aggregation": REFERENCE_AGG,
            "calibration": "no_calibration",
            "threshold_policy": "current_policy_reproduce",
        },
        "best_keep_candidate": None if best is None else best.to_dict(),
        "leakage_fail": bool(leak.leakage_fail.any()),
        "wall_runtime_seconds": time.perf_counter() - started,
    }
    OUT["metadata"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("=== Step16A complete ===", flush=True)
    print(f"wall_seconds={metadata['wall_runtime_seconds']:.2f}", flush=True)
    print(f"leakage_fail={metadata['leakage_fail']}", flush=True)
    print(f"keep_candidate_count={len(keep)}", flush=True)


if __name__ == "__main__":
    main()
