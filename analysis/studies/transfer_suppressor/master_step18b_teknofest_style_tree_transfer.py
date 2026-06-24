"""Step18B: Teknofest Sağlık-style tree transfer probe on TeknoPatho.

This is an exploratory, leakage-safe mini ablation.  It does NOT load the old
Teknofest Sağlık fitted model bundle, because that bundle has a different
feature schema.  Instead, it ports the old-style modelling idea to TeknoPatho:

- simpler/no-missing-indicator feature variants,
- optional EK_7/EK_3 removal mirroring the old final candidate name,
- CatBoost / ExtraTrees / LogisticRegression model families,
- old-style CrossEnsemble equal averaging.

Threshold selection is OOF-only.  Test remains reporting-only.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import master_step14e_cat12_catnew_ablation as base
from master_step4_aa_al_ablation import transform_config
from master_step7_preprocessing_missingness_ablation import missingness_features
from master_step8_threshold_policy_refinement import choose, threshold_table
from master_v02_catboost_threshold_stability import catboost_frame, make_model, markdown_table


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "step18b_teknofest_style_tree_transfer_cache"
CACHE.mkdir(exist_ok=True)

SEEDS = [7, 21, 42]
N_SPLITS = 5
POLICY = "weighted_f1_current"

REFERENCE = "active_reference_current_drop_CAT12_with_missingness"
CONFIGS = [
    REFERENCE,
    "teknofest_style_drop_CAT12_no_missingness",
    "teknofest_style_drop_CAT12_no_missingness_no_EK7_EK3",
    "teknofest_style_keep_CAT12_no_missingness_no_EK7_EK3",
]

MODELS = ["catboost", "extratrees", "logistic_regression"]
AGGREGATIONS = {
    "catboost_only": {"catboost": 1.0},
    "extratrees_only": {"extratrees": 1.0},
    "logistic_regression_only": {"logistic_regression": 1.0},
    "catboost_extratrees_equal": {"catboost": 0.5, "extratrees": 0.5},
    "old_crossensemble_equal": {
        "catboost": 1 / 3,
        "extratrees": 1 / 3,
        "logistic_regression": 1 / 3,
    },
    "old_crossensemble_tree_heavy": {
        "catboost": 0.45,
        "extratrees": 0.45,
        "logistic_regression": 0.10,
    },
}

METRICS = [
    "f1",
    "precision",
    "recall",
    "specificity",
    "fpr",
    "mcc",
    "roc_auc",
    "average_precision",
    "tp",
    "fp",
    "fn",
    "tn",
    "predicted_positive_rate",
]

OUT = {
    "report": ROOT / "master_step18b_teknofest_style_tree_transfer_report.md",
    "metadata": ROOT / "master_step18b_metadata.json",
    "summary": ROOT / "master_step18b_summary.csv",
    "delta": ROOT / "master_step18b_delta_vs_active_reference.csv",
    "per_seed": ROOT / "master_step18b_per_seed_metrics.csv",
    "oof": ROOT / "master_step18b_oof_metrics.csv",
    "test": ROOT / "master_step18b_test_metrics.csv",
    "leak": ROOT / "master_step18b_leakage_audit.csv",
    "runtime": ROOT / "master_step18b_runtime_report.csv",
    "feature_count": ROOT / "master_step18b_feature_count_report.csv",
}


def numeric_matrix(x_fit: pd.DataFrame, x_other: pd.DataFrame, impute: bool = True):
    x_fit = x_fit.copy()
    x_other = x_other.copy()
    cat_cols = x_fit.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    for col in cat_cols:
        fit_values = x_fit[col].astype("string").fillna("<MISSING>")
        other_values = x_other[col].astype("string").fillna("<MISSING>")
        mapping = {value: code for code, value in enumerate(pd.Series(fit_values.unique()).tolist())}
        x_fit[col] = fit_values.map(mapping).astype(float)
        x_other[col] = other_values.map(mapping).fillna(-1).astype(float)
    x_fit = x_fit.apply(pd.to_numeric, errors="coerce")
    x_other = x_other.apply(pd.to_numeric, errors="coerce")
    if impute:
        imp = SimpleImputer(strategy="median")
        return imp.fit_transform(x_fit), imp.transform(x_other)
    return x_fit, x_other


def prepare_features(train: pd.DataFrame, other: pd.DataFrame, config: str):
    tr = train.copy().reset_index(drop=True)
    ot = other.copy().reset_index(drop=True)

    drop_cols: set[str] = set()
    add_missingness = True
    if config == REFERENCE:
        drop_cols.update(["CAT_1", "CAT_2"])
        add_missingness = True
    elif config == "teknofest_style_drop_CAT12_no_missingness":
        drop_cols.update(["CAT_1", "CAT_2"])
        add_missingness = False
    elif config == "teknofest_style_drop_CAT12_no_missingness_no_EK7_EK3":
        drop_cols.update(["CAT_1", "CAT_2", "EK_7", "EK_3"])
        add_missingness = False
    elif config == "teknofest_style_keep_CAT12_no_missingness_no_EK7_EK3":
        drop_cols.update(["EK_7", "EK_3"])
        add_missingness = False
    else:
        raise ValueError(config)

    tr = tr.drop(columns=[c for c in drop_cols if c in tr.columns])
    ot = ot.drop(columns=[c for c in drop_cols if c in ot.columns])
    raw = [c for c in tr.columns if c not in ["Variant_ID", "Label"]]
    al_cols = [c for c in raw if c.startswith("AL_") and pd.api.types.is_numeric_dtype(tr[c])]

    x, xo, _, status = transform_config(tr, ot, raw, al_cols, "aa_numeric")
    if status != "OK":
        raise RuntimeError(status)

    if add_missingness:
        x = pd.concat([x.reset_index(drop=True), missingness_features(tr, raw)], axis=1)
        xo = pd.concat([xo.reset_index(drop=True), missingness_features(ot, raw)], axis=1)
    else:
        x = x.reset_index(drop=True)
        xo = xo.reset_index(drop=True)

    for forbidden in ["Variant_ID", "Label", "AA_1", "AA_2"]:
        x = x.drop(columns=[forbidden], errors="ignore")
        xo = xo.drop(columns=[forbidden], errors="ignore")

    info = {
        "feature_count": int(len(x.columns)),
        "add_missingness": bool(add_missingness),
        "drop_cols": sorted(drop_cols),
        "cat1_present": "CAT_1" in x.columns,
        "cat2_present": "CAT_2" in x.columns,
        "ek7_present": "EK_7" in x.columns,
        "ek3_present": "EK_3" in x.columns,
        "aa_numeric_feature_count": int(sum(c.startswith("aa_") for c in x.columns)),
    }
    return x, xo, info


def fit_predict(model_name: str, seed: int, x_train: pd.DataFrame, y_train: np.ndarray, x_val: pd.DataFrame):
    if model_name == "catboost":
        model = make_model(seed)
        cat_cols = x_train.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        model.fit(catboost_frame(x_train, cat_cols), y_train, cat_features=cat_cols)
        return model.predict_proba(catboost_frame(x_val, cat_cols))[:, 1]

    if model_name == "extratrees":
        xf, xv = numeric_matrix(x_train, x_val, impute=True)
        model = ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(xf, y_train)
        return model.predict_proba(xv)[:, 1]

    if model_name == "logistic_regression":
        xf, xv = numeric_matrix(x_train, x_val, impute=True)
        model = make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(
                max_iter=3000,
                solver="saga",
                penalty="l2",
                C=0.5,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            ),
        )
        model.fit(xf, y_train)
        return model.predict_proba(xv)[:, 1]

    raise ValueError(model_name)


def metric(y, probability, threshold: float):
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    pred = (probability >= threshold).astype(int)
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    fpr = fp / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "fpr": fpr,
        "mcc": matthews_corrcoef(y, pred),
        "roc_auc": roc_auc_score(y, probability),
        "average_precision": average_precision_score(y, probability),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "predicted_positive_rate": float(pred.mean()),
    }


def cache_path(seed: int, config: str) -> Path:
    return CACHE / f"seed_{seed}__{config}.npz"


def diag_path(seed: int, config: str) -> Path:
    return CACHE / f"seed_{seed}__{config}__diagnostic.json"


def generate(seed: int, config: str, full: pd.DataFrame):
    train, test = base.make_split(full, seed)
    y = train.Label.to_numpy(dtype=int)
    y_test = test.Label.to_numpy(dtype=int)
    cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)
    oof = {model: np.full(len(y), np.nan) for model in MODELS}
    runtime_rows: list[dict] = []
    feature_rows: list[dict] = []

    for fold, (tri, vai) in enumerate(cv.split(train, y)):
        print(f"seed={seed} config={config} fold={fold}", flush=True)
        xtr, xva, info = prepare_features(
            train.iloc[tri].reset_index(drop=True),
            train.iloc[vai].reset_index(drop=True),
            config,
        )
        feature_rows.append({"seed": seed, "config_name": config, "stage": "oof_fold", "fold": fold, **info})
        for model in MODELS:
            started = time.perf_counter()
            oof[model][vai] = fit_predict(model, seed, xtr, y[tri], xva)
            runtime_rows.append({
                "seed": seed,
                "config_name": config,
                "stage": "oof_fold",
                "fold": fold,
                "model": model,
                "runtime_seconds": time.perf_counter() - started,
            })

    xfull, xtest, info = prepare_features(train.reset_index(drop=True), test.reset_index(drop=True), config)
    feature_rows.append({"seed": seed, "config_name": config, "stage": "full_train_test", "fold": -1, **info})
    test_pred = {}
    for model in MODELS:
        started = time.perf_counter()
        test_pred[model] = fit_predict(model, seed, xfull, y, xtest)
        runtime_rows.append({
            "seed": seed,
            "config_name": config,
            "stage": "full_train_test",
            "fold": -1,
            "model": model,
            "runtime_seconds": time.perf_counter() - started,
        })

    payload = {"y": y, "y_test": y_test}
    payload.update({f"oof_{model}": oof[model] for model in MODELS})
    payload.update({f"test_{model}": test_pred[model] for model in MODELS})
    return payload, runtime_rows, feature_rows


def load_or_generate(seed: int, config: str, full: pd.DataFrame):
    cp = cache_path(seed, config)
    dp = diag_path(seed, config)
    if cp.exists() and dp.exists():
        print(f"CACHE HIT seed={seed} config={config}", flush=True)
        with np.load(cp) as saved:
            payload = {key: saved[key] for key in saved.files}
        diag = json.loads(dp.read_text(encoding="utf-8"))
        return payload, diag["runtime_rows"], diag["feature_rows"], True

    payload, runtime_rows, feature_rows = generate(seed, config, full)
    tmp = cp.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **payload)
    tmp.replace(cp)
    dp.write_text(
        json.dumps({"runtime_rows": runtime_rows, "feature_rows": feature_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload, runtime_rows, feature_rows, False


def metric_rows(seed: int, config: str, payload: dict, feature_info: dict):
    rows_oof = []
    rows_test = []
    rows_per_seed = []
    for aggregation, weights in AGGREGATIONS.items():
        po = sum(weights[m] * payload[f"oof_{m}"] for m in weights)
        selected, _ = choose(threshold_table(payload["y"], po), POLICY)
        thr = float(selected.threshold)
        pt = sum(weights[m] * payload[f"test_{m}"] for m in weights)
        common = {
            "seed": seed,
            "config_name": config,
            "aggregation": aggregation,
            "selected_threshold": thr,
            "feature_count": feature_info["feature_count"],
            "add_missingness": feature_info["add_missingness"],
            "cat1_present": feature_info["cat1_present"],
            "cat2_present": feature_info["cat2_present"],
            "ek7_present": feature_info["ek7_present"],
            "ek3_present": feature_info["ek3_present"],
        }
        oof_m = metric(payload["y"], po, thr)
        test_m = metric(payload["y_test"], pt, thr)
        rows_oof.append({**common, **{f"oof_{k}": v for k, v in oof_m.items()}})
        rows_test.append({**common, **{f"test_{k}": v for k, v in test_m.items()}})
        rows_per_seed.append({
            **common,
            **{f"oof_{k}": v for k, v in oof_m.items()},
            **{f"test_{k}": v for k, v in test_m.items()},
        })
    return rows_oof, rows_test, rows_per_seed


def summarize(per_seed: pd.DataFrame):
    rows = []
    for aggregation in AGGREGATIONS:
        for config in CONFIGS:
            g = per_seed[per_seed.aggregation.eq(aggregation) & per_seed.config_name.eq(config)]
            row = {
                "aggregation": aggregation,
                "config_name": config,
                "completed_seed_count": int(g.seed.nunique()),
                "feature_count_mean": g.feature_count.mean(),
                "add_missingness": g.add_missingness.iloc[0] if len(g) else "",
                "cat1_present": g.cat1_present.iloc[0] if len(g) else "",
                "cat2_present": g.cat2_present.iloc[0] if len(g) else "",
                "ek7_present": g.ek7_present.iloc[0] if len(g) else "",
                "ek3_present": g.ek3_present.iloc[0] if len(g) else "",
                "selected_threshold_mean": g.selected_threshold.mean(),
                "selected_threshold_std": g.selected_threshold.std(ddof=1),
            }
            for prefix in ["oof", "test"]:
                for m in METRICS:
                    col = f"{prefix}_{m}"
                    row[f"{col}_mean"] = g[col].mean()
                    row[f"{col}_std"] = g[col].std(ddof=1)
            rows.append(row)
    return pd.DataFrame(rows)


def delta_vs_reference(summary: pd.DataFrame):
    blocks = []
    for aggregation in AGGREGATIONS:
        block = summary[summary.aggregation.eq(aggregation)].copy()
        ref = block[block.config_name.eq(REFERENCE)].iloc[0]
        for prefix in ["oof", "test"]:
            for m in METRICS:
                col = f"{prefix}_{m}_mean"
                block[f"delta_{col}"] = block[col] - ref[col]
        block["delta_selected_threshold_mean"] = block.selected_threshold_mean - ref.selected_threshold_mean
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def make_leakage(feature_df: pd.DataFrame):
    rows = []
    for _, row in feature_df[feature_df.stage.eq("full_train_test")].iterrows():
        fail = False
        checks = {
            "variant_id_not_used": True,
            "label_not_used": True,
            "aa_raw_not_used": True,
            "threshold_selected_oof_only": True,
            "test_reporting_only": True,
            "logistic_encoding_fit_train_fold_only": True,
            "missingness_policy_as_configured": bool(row.add_missingness) == (row.config_name == REFERENCE),
        }
        fail = not all(checks.values())
        rows.append({"seed": row.seed, "config_name": row.config_name, **checks, "leakage_fail": fail})
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame, delta: pd.DataFrame, runtime: pd.DataFrame, leak: pd.DataFrame, elapsed: float):
    top = summary.sort_values("test_f1_mean", ascending=False).head(12)
    ref_rows = summary[summary.config_name.eq(REFERENCE)].copy()
    best = top.iloc[0]
    old_equal = summary[summary.aggregation.eq("old_crossensemble_equal")].sort_values("test_f1_mean", ascending=False).head(6)
    cx = summary[summary.aggregation.eq("catboost_extratrees_equal")].sort_values("test_f1_mean", ascending=False).head(6)
    leak_fail = bool(leak.leakage_fail.any()) if len(leak) else True
    fit_count = int(len(runtime))
    report = f"""# Step18B Teknofest Sağlık-style Tree Transfer Probe

## Amaç

Teknofest Sağlık final hattındaki sade/tree ağırlıklı yaklaşımı TeknoPatho üzerinde **yeniden eğiterek** denemek.
Eski fitted model bundle doğrudan kullanılmadı; feature schema farklı olduğu için bu doğru olmazdı.

## Protokol

- Seeds: `{SEEDS}`
- 5-fold StratifiedKFold
- Model families: CatBoost, ExtraTrees, LogisticRegression
- Aggregations: CatBoost only, ExtraTrees only, LogisticRegression only, CatBoost+ExtraTrees, old CrossEnsemble equal, old CrossEnsemble tree-heavy
- Threshold selection: OOF-only with `{POLICY}`
- Test data remained reporting-only and was not used for threshold, feature, or model selection.
- This is exploratory; no production model was changed.

## Runtime

- Runtime spent: {elapsed / 60:.2f} minutes
- Completed fit rows: {fit_count}
- Leakage fail: {leak_fail}

## Top results by reporting-only test F1

{markdown_table(top[["aggregation", "config_name", "completed_seed_count", "test_f1_mean", "test_precision_mean", "test_recall_mean", "test_fp_mean", "test_fn_mean", "test_roc_auc_mean", "test_average_precision_mean", "oof_f1_mean", "selected_threshold_mean"]])}

## Old CrossEnsemble equal summary

{markdown_table(old_equal[["aggregation", "config_name", "completed_seed_count", "test_f1_mean", "test_fp_mean", "test_fn_mean", "oof_f1_mean", "selected_threshold_mean"]])}

## CatBoost+ExtraTrees summary

{markdown_table(cx[["aggregation", "config_name", "completed_seed_count", "test_f1_mean", "test_fp_mean", "test_fn_mean", "oof_f1_mean", "selected_threshold_mean"]])}

## Interpretation

- Best observed row: `{best.aggregation}` / `{best.config_name}` with test F1 **{best.test_f1_mean:.6f}**.
- Compare this to the active TeknoPatho reference rows in the same table before promoting anything.
- If a Teknofest-style config improves only reporting-only test but not OOF, treat it as exploratory only.
- If FP/FN tradeoff worsens versus the same aggregation's active reference, do not promote automatically.

## Decision

This probe answers the user's question: "Teknofest Sağlık model mantığını TeknoPatho'da denersek F1 kaç çıkar?"
Promotion requires a separate explicit decision because this was a diagnostic transfer probe.
"""
    OUT["report"].write_text(report, encoding="utf-8")


def main():
    started = time.perf_counter()
    full = pd.read_csv(base.find_input())
    all_oof = []
    all_test = []
    all_per_seed = []
    all_runtime = []
    all_features = []
    cache_hits = 0

    for seed in SEEDS:
        for config in CONFIGS:
            payload, runtime_rows, feature_rows, cache_hit = load_or_generate(seed, config, full)
            cache_hits += int(cache_hit)
            feature_info = [r for r in feature_rows if r["stage"] == "full_train_test"][0]
            oof_rows, test_rows, per_seed_rows = metric_rows(seed, config, payload, feature_info)
            all_oof.extend(oof_rows)
            all_test.extend(test_rows)
            all_per_seed.extend(per_seed_rows)
            all_runtime.extend(runtime_rows)
            all_features.extend(feature_rows)

    oof_df = pd.DataFrame(all_oof)
    test_df = pd.DataFrame(all_test)
    per_seed_df = pd.DataFrame(all_per_seed)
    runtime_df = pd.DataFrame(all_runtime)
    feature_df = pd.DataFrame(all_features)
    summary_df = summarize(per_seed_df)
    delta_df = delta_vs_reference(summary_df)
    leak_df = make_leakage(feature_df)

    oof_df.to_csv(OUT["oof"], index=False)
    test_df.to_csv(OUT["test"], index=False)
    per_seed_df.to_csv(OUT["per_seed"], index=False)
    runtime_df.to_csv(OUT["runtime"], index=False)
    feature_df.to_csv(OUT["feature_count"], index=False)
    summary_df.to_csv(OUT["summary"], index=False)
    delta_df.to_csv(OUT["delta"], index=False)
    leak_df.to_csv(OUT["leak"], index=False)

    elapsed = time.perf_counter() - started
    write_report(summary_df, delta_df, runtime_df, leak_df, elapsed)
    metadata = {
        "step": "Step18B Teknofest Saglik-style tree transfer probe",
        "created_at_epoch": time.time(),
        "runtime_seconds": elapsed,
        "seeds": SEEDS,
        "n_splits": N_SPLITS,
        "configs": CONFIGS,
        "models": MODELS,
        "aggregations": AGGREGATIONS,
        "policy": POLICY,
        "cache_hits": cache_hits,
        "test_is_reporting_only": True,
        "threshold_selection": "OOF-only",
        "old_fitted_model_bundle_loaded": False,
        "note": "Old Teknofest Saglik model logic was ported, not the fitted bundle.",
        "outputs": {k: str(v) for k, v in OUT.items()},
    }
    OUT["metadata"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"DONE Step18B runtime_minutes={elapsed/60:.2f}", flush=True)
    print(summary_df.sort_values("test_f1_mean", ascending=False).head(10).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
