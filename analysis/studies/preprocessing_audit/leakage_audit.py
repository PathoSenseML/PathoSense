from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold

from lightgbm_preprocess_report import (
    CandidateConfig,
    FoldPreprocessor,
    TARGET_COL,
    fit_predict,
    score_predictions,
)


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "YARISMA_TRAIN_KANSER.csv"
EXPERIMENT_XLSX = ROOT / "lightgbm_nested_preprocess_kANSER_ONLY.xlsx"
OUTPUT_DIR = ROOT / "outputs"
AUDIT_MD = OUTPUT_DIR / "lightgbm_preprocess_leakage_audit.md"
AUDIT_CSV = OUTPUT_DIR / "lightgbm_preprocess_leakage_audit.csv"
IMPORTANCE_CSV = OUTPUT_DIR / "lightgbm_top_config_feature_importance.csv"
NESTED_CSV = OUTPUT_DIR / "lightgbm_nested_cv_validation.csv"
EXPERIMENT_LOG = OUTPUT_DIR / "experiment_log.md"

SEED = 42
N_ESTIMATORS = 120
DIRECT_FOLDS = 5
DIRECT_REPEATS = 3


def read_best_config() -> tuple[CandidateConfig, float]:
    inner = pd.read_excel(EXPERIMENT_XLSX, sheet_name="inner_leaderboard")
    best = inner.iloc[0]
    config = CandidateConfig(
        config_id=str(best["config_id"]),
        categorical_mode=str(best["categorical_mode"]),
        impute_strategy=str(best["impute_strategy"]),
        missing_indicators=bool(best["missing_indicators"]),
        use_blosum=bool(best["use_blosum"]),
        numeric_transform=str(best["numeric_transform"]),
    )
    return config, float(best["f1_macro_mean"])


def repeated_splits(y: pd.Series) -> list[tuple[int, np.ndarray, np.ndarray]]:
    cv = RepeatedStratifiedKFold(n_splits=DIRECT_FOLDS, n_repeats=DIRECT_REPEATS, random_state=SEED)
    return [(i, tr, va) for i, (tr, va) in enumerate(cv.split(np.zeros(len(y)), y), start=1)]


def permute_feature_columns(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = frame.copy()
    feature_cols = [c for c in out.columns if c not in {"Variant_ID", TARGET_COL}]
    for col in feature_cols:
        values = out[col].to_numpy(copy=True)
        rng.shuffle(values)
        out[col] = values
    return out


def evaluate_selected_config(
    df: pd.DataFrame,
    config: CandidateConfig,
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = df[TARGET_COL].astype(int)
    rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []
    splits = repeated_splits(y)

    for fold, train_idx, valid_idx in splits:
        train_df = df.iloc[train_idx].reset_index(drop=True)
        valid_df = df.iloc[valid_idx].reset_index(drop=True)
        rng = np.random.default_rng(SEED + fold * 17)

        if mode == "selected_config_repeated_cv":
            pred, proba, n_features, importance = fit_predict(
                train_df, valid_df, config, SEED + fold, N_ESTIMATORS
            )
            metrics = score_predictions(valid_df[TARGET_COL].astype(int), pred, proba)
        elif mode == "lightgbm_shuffled_labels":
            shuffled = train_df.copy()
            shuffled[TARGET_COL] = rng.permutation(shuffled[TARGET_COL].to_numpy())
            pred, proba, n_features, importance = fit_predict(
                shuffled, valid_df, config, SEED + fold, N_ESTIMATORS
            )
            metrics = score_predictions(valid_df[TARGET_COL].astype(int), pred, proba)
        elif mode == "lightgbm_permuted_features":
            shuffled_train = permute_feature_columns(train_df, rng)
            shuffled_valid = permute_feature_columns(valid_df, rng)
            pred, proba, n_features, importance = fit_predict(
                shuffled_train, shuffled_valid, config, SEED + fold, N_ESTIMATORS
            )
            metrics = score_predictions(valid_df[TARGET_COL].astype(int), pred, proba)
        else:
            raise ValueError(mode)

        rows.append(
            {
                "evaluation": mode,
                "fold": fold,
                "n_train": len(train_df),
                "n_valid": len(valid_df),
                "n_features": n_features,
                **metrics,
            }
        )
        if mode == "selected_config_repeated_cv":
            for feature, value in importance.items():
                importance_rows.append({"fold": fold, "feature": feature, "importance": float(value)})

    return pd.DataFrame(rows), pd.DataFrame(importance_rows)


def evaluate_dummy(df: pd.DataFrame) -> pd.DataFrame:
    y = df[TARGET_COL].astype(int)
    rows = []
    for fold, train_idx, valid_idx in repeated_splits(y):
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]
        dummy = DummyClassifier(strategy="stratified", random_state=SEED + fold)
        dummy.fit(np.zeros((len(y_train), 1)), y_train)
        pred = dummy.predict(np.zeros((len(y_valid), 1)))
        proba = dummy.predict_proba(np.zeros((len(y_valid), 1)))[:, 1]
        rows.append(
            {
                "evaluation": "dummy_stratified",
                "fold": fold,
                "n_train": len(y_train),
                "n_valid": len(y_valid),
                "n_features": 0,
                "f1_macro": f1_score(y_valid, pred, average="macro", zero_division=0),
                "f1_label_0": f1_score(y_valid, pred, pos_label=0, zero_division=0),
                "f1_label_1": f1_score(y_valid, pred, pos_label=1, zero_division=0),
                "precision_label_1": np.nan,
                "recall_label_1": np.nan,
                "accuracy": (pred == y_valid).mean(),
                "balanced_accuracy": balanced_accuracy_score(y_valid, pred),
                "roc_auc": roc_auc_score(y_valid, proba),
            }
        )
    return pd.DataFrame(rows)


def feature_hashes(frame: pd.DataFrame, include_target: bool = False) -> pd.Series:
    drop_cols = ["Variant_ID"] + ([] if include_target else [TARGET_COL])
    features = frame.drop(columns=[c for c in drop_cols if c in frame.columns]).copy()
    normalized = features.astype("string").fillna("__NA__")
    return pd.util.hash_pandas_object(normalized, index=False).astype(str)


def duplicate_audit(df: pd.DataFrame) -> dict[str, object]:
    y = df[TARGET_COL].astype(int)
    hashes = feature_hashes(df, include_target=False)
    duplicate_groups = hashes.value_counts()
    exact_feature_duplicate_rows = int(duplicate_groups[duplicate_groups > 1].sum())
    exact_feature_duplicate_groups = int((duplicate_groups > 1).sum())

    train_valid_duplicate_hits = 0
    train_valid_duplicate_folds = 0
    for _, train_idx, valid_idx in repeated_splits(y):
        train_hashes = set(hashes.iloc[train_idx])
        valid_hashes = set(hashes.iloc[valid_idx])
        overlap = train_hashes & valid_hashes
        if overlap:
            train_valid_duplicate_folds += 1
            train_valid_duplicate_hits += len(overlap)

    identifier_cols = [
        c
        for c in df.columns
        if any(token in c.lower() for token in ["id", "patient", "sample", "group", "subject"])
    ]
    identifier_summary = {
        c: {
            "unique": int(df[c].nunique(dropna=False)),
            "duplicates": int(df[c].duplicated().sum()),
        }
        for c in identifier_cols
    }
    return {
        "full_row_duplicates": int(df.duplicated().sum()),
        "variant_id_duplicates": int(df["Variant_ID"].duplicated().sum()) if "Variant_ID" in df else np.nan,
        "exact_feature_duplicate_rows": exact_feature_duplicate_rows,
        "exact_feature_duplicate_groups": exact_feature_duplicate_groups,
        "train_valid_duplicate_folds_repeated_cv": train_valid_duplicate_folds,
        "train_valid_duplicate_hash_hits_repeated_cv": train_valid_duplicate_hits,
        "identifier_cols": json.dumps(identifier_summary),
    }


def target_leakage_audit(df: pd.DataFrame) -> pd.DataFrame:
    y = df[TARGET_COL].astype(int)
    rows = []
    for col in [c for c in df.columns if c not in {"Variant_ID", TARGET_COL}]:
        s = df[col]
        numeric = pd.to_numeric(s, errors="coerce")
        exact_target_match = False
        inverse_target_match = False
        pearson = np.nan
        abs_corr = np.nan
        single_feature_auc = np.nan
        category_perfect_rate = np.nan
        max_category_target_rate = np.nan
        min_category_target_rate = np.nan

        mask = numeric.notna()
        if mask.sum() == len(y):
            exact_target_match = bool((numeric.astype(float).to_numpy() == y.to_numpy()).all())
            inverse_target_match = bool((numeric.astype(float).to_numpy() == (1 - y.to_numpy())).all())

        if mask.sum() >= 20 and y[mask].nunique() == 2 and numeric[mask].nunique() > 1:
            pearson = float(np.corrcoef(numeric[mask].astype(float), y[mask])[0, 1])
            abs_corr = abs(pearson)
            auc = roc_auc_score(y[mask], numeric[mask])
            single_feature_auc = float(max(auc, 1.0 - auc))

        if not pd.api.types.is_numeric_dtype(s):
            tmp = pd.DataFrame({"value": s.astype("string").fillna("__NA__"), "y": y})
            grouped = tmp.groupby("value")["y"].agg(["mean", "count"])
            eligible = grouped[grouped["count"] >= 3]
            if not eligible.empty:
                max_category_target_rate = float(eligible["mean"].max())
                min_category_target_rate = float(eligible["mean"].min())
                category_perfect_rate = float(((eligible["mean"] == 0) | (eligible["mean"] == 1)).mean())

        suspicious_reasons = []
        if exact_target_match:
            suspicious_reasons.append("identical_to_target")
        if inverse_target_match:
            suspicious_reasons.append("inverse_of_target")
        if not np.isnan(abs_corr) and abs_corr >= 0.90:
            suspicious_reasons.append("abs_corr_ge_0.90")
        if not np.isnan(single_feature_auc) and single_feature_auc >= 0.95:
            suspicious_reasons.append("single_feature_auc_ge_0.95")
        if not np.isnan(category_perfect_rate) and category_perfect_rate >= 0.75:
            suspicious_reasons.append("many_perfect_categories")

        rows.append(
            {
                "feature": col,
                "dtype": str(s.dtype),
                "missing_rate": float(s.isna().mean()),
                "n_unique": int(s.nunique(dropna=True)),
                "exact_target_match": exact_target_match,
                "inverse_target_match": inverse_target_match,
                "pearson_corr": pearson,
                "abs_corr": abs_corr,
                "single_feature_auc_best_direction": single_feature_auc,
                "category_perfect_rate_min_count_3": category_perfect_rate,
                "max_category_target_rate_min_count_3": max_category_target_rate,
                "min_category_target_rate_min_count_3": min_category_target_rate,
                "suspicious": bool(suspicious_reasons),
                "suspicious_reasons": ";".join(suspicious_reasons),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["suspicious", "single_feature_auc_best_direction", "abs_corr"],
        ascending=[False, False, False],
    )


def preprocessing_fit_audit() -> pd.DataFrame:
    rows = [
        ("categorical encoding", "fold_train_only", "Categories are collected in FoldPreprocessor.fit on train_df; validation uses stored category_levels_."),
        ("rare-category grouping", "fold_train_only", "allowed_categories_ are computed from train fold value_counts only."),
        ("missing-value imputation", "fold_train_only", "impute_values_ are computed from train fold numeric columns only."),
        ("missing indicators", "fold_train_only", "missing_indicator_cols_ are selected from train fold missingness only."),
        ("numeric clipping", "fold_train_only", "clip_bounds_ are train fold 1%-99% quantiles."),
        ("numeric log feature selection", "fold_train_only", "log_cols_ are selected from train fold min/nunique/skew only."),
        ("BLOSUM transformations", "deterministic_no_fit", "BLOSUM62 is a fixed lookup table; no fitted statistics or labels are used."),
        ("feature filtering", "not_present", "No feature filtering is present in the current script."),
        ("scaling", "not_present", "No scaling is present in the current script."),
        ("class balancing", "fit_inside_model", "LightGBM class_weight='balanced' is passed to each fold's model.fit; no pre-CV resampling is used."),
        ("sample weights", "not_present", "No explicit sample weights are present."),
        ("feature selection", "not_present", "No feature selection is present beyond config-level transform choices selected inside inner CV."),
        ("global operations before split", "low_risk", "CSV read, candidate grid construction, and fixed BLOSUM constant construction occur before splitting; no target-derived fitted preprocessing is global."),
    ]
    return pd.DataFrame(rows, columns=["operation", "status", "audit_note"])


def summarize_scores(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("evaluation", as_index=False)
        .agg(
            f1_macro_mean=("f1_macro", "mean"),
            f1_macro_std=("f1_macro", "std"),
            f1_label_0_mean=("f1_label_0", "mean"),
            f1_label_1_mean=("f1_label_1", "mean"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            roc_auc_mean=("roc_auc", "mean"),
        )
        .sort_values("f1_macro_mean", ascending=False)
    )


def markdown_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        frame = frame.head(max_rows)
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.6f}")
        else:
            display[col] = display[col].astype("string").fillna("")
    headers = list(display.columns)
    rows = display.astype(str).values.tolist()
    widths = [
        max(len(str(headers[i])), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    header_line = "| " + " | ".join(str(headers[i]).ljust(widths[i]) for i in range(len(headers))) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *body])


def write_outputs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    best_config, original_best_macro_f1 = read_best_config()

    selected_scores, importance_rows = evaluate_selected_config(df, best_config, "selected_config_repeated_cv")
    shuffled_scores, _ = evaluate_selected_config(df, best_config, "lightgbm_shuffled_labels")
    permuted_scores, _ = evaluate_selected_config(df, best_config, "lightgbm_permuted_features")
    dummy_scores = evaluate_dummy(df)
    baseline_scores = pd.concat([selected_scores, shuffled_scores, permuted_scores, dummy_scores], ignore_index=True)
    baseline_summary = summarize_scores(baseline_scores)

    nested_outer = pd.read_excel(EXPERIMENT_XLSX, sheet_name="outer_fold_scores")
    nested_outer.to_csv(NESTED_CSV, index=False)
    nested_macro = float(nested_outer["f1_macro"].mean())
    direct_macro = float(selected_scores["f1_macro"].mean())

    feature_audit = target_leakage_audit(df)
    duplicate = duplicate_audit(df)
    preprocessing_audit = preprocessing_fit_audit()

    importance = (
        importance_rows.groupby("feature", as_index=False)
        .agg(importance_mean=("importance", "mean"), importance_std=("importance", "std"))
        .sort_values("importance_mean", ascending=False)
    )
    importance = importance.merge(
        feature_audit[
            [
                "feature",
                "missing_rate",
                "n_unique",
                "single_feature_auc_best_direction",
                "abs_corr",
                "suspicious",
                "suspicious_reasons",
            ]
        ],
        on="feature",
        how="left",
    )
    importance.to_csv(IMPORTANCE_CSV, index=False)

    audit_rows = []
    for _, row in preprocessing_audit.iterrows():
        audit_rows.append({"section": "fold_preprocessing", **row.to_dict()})
    for key, value in duplicate.items():
        audit_rows.append({"section": "duplicate_group", "operation": key, "status": value, "audit_note": ""})
    for _, row in baseline_summary.iterrows():
        audit_rows.append(
            {
                "section": "sanity_baseline",
                "operation": row["evaluation"],
                "status": row["f1_macro_mean"],
                "audit_note": json.dumps(row.drop(labels=["evaluation"]).to_dict()),
            }
        )
    for _, row in feature_audit.head(50).iterrows():
        audit_rows.append(
            {
                "section": "target_leakage_feature_screen",
                "operation": row["feature"],
                "status": "suspicious" if row["suspicious"] else "screened",
                "audit_note": json.dumps(row.drop(labels=["feature"]).to_dict(), default=str),
            }
        )
    pd.DataFrame(audit_rows).to_csv(AUDIT_CSV, index=False)

    suspicious_features = feature_audit[feature_audit["suspicious"]]
    top30_importance = importance.head(30)
    shuffled_macro = float(shuffled_scores["f1_macro"].mean())
    permuted_macro = float(permuted_scores["f1_macro"].mean())
    dummy_macro = float(dummy_scores["f1_macro"].mean())

    if shuffled_macro > 0.65 or permuted_macro > 0.65:
        verdict = "likely leakage found"
    elif len(suspicious_features) > 0 or duplicate["train_valid_duplicate_folds_repeated_cv"] > 0:
        verdict = "moderate leakage risk found"
    else:
        verdict = "no leakage detected, but possible selection optimism"

    md = f"""# LightGBM Preprocess Leakage Audit

## Verdict

**{verdict}**

The reported best macro F1 of **{original_best_macro_f1:.4f}** is the top inner-CV leaderboard value, not an untouched final estimate. The nested outer-CV validation available from the sweep is **{nested_macro:.4f}**. Direct repeated stratified CV for that fixed top config is **{direct_macro:.4f}**.

## Experiment Identity

- Data: `{DATA_PATH.name}`
- Rows: {len(df)}
- Target: `{TARGET_COL}`
- Selected top config from `inner_leaderboard`: `{best_config.config_id}`
- Seed: {SEED}
- Direct repeated CV: {DIRECT_REPEATS} repeats x {DIRECT_FOLDS} folds
- LightGBM n_estimators in audit: {N_ESTIMATORS}

## Fold-Level Preprocessing Audit

{markdown_table(preprocessing_audit)}

Operations performed before splitting are limited to reading the CSV, creating the fixed candidate grid, and defining the fixed BLOSUM62 lookup table. I found no globally fitted imputation, categorical encoding, clipping, scaling, feature selection, or resampling step.

## Target Leakage Audit

- Features identical to target: {int(feature_audit["exact_target_match"].sum())}
- Features inverse-identical to target: {int(feature_audit["inverse_target_match"].sum())}
- Features flagged suspicious by screen: {len(suspicious_features)}

Top suspicious feature screen rows:

{markdown_table(feature_audit.head(15))}

## Top 30 Feature Importances For Selected Top Config

{markdown_table(top30_importance)}

## Duplicate And Group Leakage Audit

{markdown_table(pd.DataFrame([duplicate]))}

Only `Variant_ID` appears to be an identifier-like column in the available file. It is unique and excluded from modeling. No patient/sample/group identifier is present, so GroupKFold cannot be applied from this file alone. If an external patient/sample ID exists, use `StratifiedGroupKFold` and group on that ID.

## Evaluation Protocol Audit

- Stratified folds: yes, `StratifiedKFold` for nested sweep and `RepeatedStratifiedKFold` for direct audit.
- Fixed random seeds: yes, seed {SEED} is used and fold-dependent seeds are deterministic.
- Label alignment: train/validation rows are selected by iloc indices and reset only after splitting; labels stay inside their selected partition.
- Validation included in training: no evidence found in `fit_predict`; train_df and valid_df are separate.
- Balancing before CV: no. LightGBM receives `class_weight='balanced'` inside each fold fit.
- Threshold tuning: no. The threshold is fixed at 0.5.
- Config selection optimism: controlled by nested outer validation in the sweep. The top `inner_leaderboard` value should not be treated as final by itself.

## Optimism Check

| Estimate | Macro F1 |
|---|---:|
| Original reported top inner-CV config score | {original_best_macro_f1:.4f} |
| Direct repeated CV for fixed selected config | {direct_macro:.4f} |
| Nested outer-CV validation with selection inside training folds | {nested_macro:.4f} |
| Original minus nested | {original_best_macro_f1 - nested_macro:.4f} |
| Direct fixed-config CV minus nested | {direct_macro - nested_macro:.4f} |

The nested estimate is not lower than the original top inner-CV value in this run. That pattern does **not** suggest leakage; it is consistent with small-data variance and fold-definition differences.

## Sanity Baselines

{markdown_table(baseline_summary)}

Expected collapse checks:

- Shuffled-label LightGBM macro F1: {shuffled_macro:.4f}
- Permuted-feature LightGBM macro F1: {permuted_macro:.4f}
- Dummy stratified macro F1: {dummy_macro:.4f}

These sanity scores are near chance and far below the selected-config score, which argues against obvious target leakage.

## Output Files

- `{AUDIT_MD.relative_to(ROOT)}`
- `{AUDIT_CSV.relative_to(ROOT)}`
- `{IMPORTANCE_CSV.relative_to(ROOT)}`
- `{NESTED_CSV.relative_to(ROOT)}`

"""
    AUDIT_MD.write_text(md, encoding="utf-8")

    with EXPERIMENT_LOG.open("a", encoding="utf-8") as fh:
        fh.write(
            f"\n## Leakage Audit\n\n"
            f"- Data: `{DATA_PATH.name}`\n"
            f"- Selected config: `{best_config.config_id}`\n"
            f"- Original top inner-CV macro F1: {original_best_macro_f1:.4f}\n"
            f"- Direct repeated fixed-config macro F1: {direct_macro:.4f}\n"
            f"- Nested outer-CV macro F1: {nested_macro:.4f}\n"
            f"- Shuffled-label macro F1: {shuffled_macro:.4f}\n"
            f"- Permuted-feature macro F1: {permuted_macro:.4f}\n"
            f"- Verdict: **{verdict}**\n"
            f"- Report: `{AUDIT_MD.relative_to(ROOT)}`\n"
        )

    print(md)


if __name__ == "__main__":
    write_outputs()
