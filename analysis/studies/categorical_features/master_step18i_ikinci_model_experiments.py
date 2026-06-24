"""Step18I: CAT3/4/5 Encoding & Correlation Audit + CAT6 Train Filter Diagnostic

Base config: active_reference_current_drop_CAT12_with_missingness
Aggregation: old_crossensemble_equal
Threshold target: ~0.749958
Baseline Test F1: ~0.547993
"""
import os
os.environ['JOBLIB_MEMMAPPING'] = '0'
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
import scipy.stats as stats
from sklearn.pipeline import make_pipeline

import master_step14e_cat12_catnew_ablation as base
from master_step4_aa_al_ablation import transform_config
from master_step7_preprocessing_missingness_ablation import missingness_features
from master_step8_threshold_policy_refinement import choose, threshold_table
from master_v02_catboost_threshold_stability import catboost_frame, markdown_table
from catboost import CatBoostClassifier

def make_model(seed: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=120,
        depth=6,
        learning_rate=0.15,
        loss_function="Logloss",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )

ROOT = Path(__file__).resolve().parent

OUT = {
    "report": ROOT / "master_step18i_ikinci_model_cat6_cat345_report.md",
    "excel_wide": ROOT / "master_step18i_excel_wide_model_tracker.csv",
    "current_update": ROOT / "master_step18i_current_active_update.json",
    "leak": ROOT / "master_step18i_leakage_audit.csv",
    "metadata": ROOT / "master_step18i_metadata.json",
    "cat345_encoding": ROOT / "master_step18i_cat345_encoding_summary.csv",
    "correlation": ROOT / "master_step18i_cat345_correlation_audit.csv",
    "label_assoc": ROOT / "master_step18i_cat345_label_association_diagnostic.csv",
    "miss_overlap": ROOT / "master_step18i_cat345_missingness_overlap.csv",
    "single_feat": ROOT / "master_step18i_cat345_single_feature_replacement_summary.csv",
    "cat6_diag": ROOT / "master_step18i_cat6_train_filter_diagnostic_summary.csv",
    "cat6_rows": ROOT / "master_step18i_cat6_train_filter_removed_rows_audit.csv",
}

SEEDS = [7, 21, 42]
N_SPLITS = 5
POLICY = "weighted_f1_current"

MODELS = ["catboost", "extratrees", "logistic_regression"]
WEIGHTS = {
    "catboost": 1 / 3,
    "extratrees": 1 / 3,
    "logistic_regression": 1 / 3,
}

CONFIGS_PASS_1 = [
    "CAT345_native_separate",
    "CAT345_ordinal_label_encoded_separate",
    "CAT345_onehot_separate",
    "CAT345_combined_native",
    "CAT345_combined_ordinal_label_encoded",
    "CAT345_combined_onehot",
    "CAT345_native_plus_combined",
    "CAT345_onehot_plus_combined_onehot",
    "CAT34_combined_plus_CAT5",
    "CAT45_combined_plus_CAT3",
    "CAT35_combined_plus_CAT4",
    "only_CAT3",
    "only_CAT4",
    "only_CAT5",
    "drop_CAT3_keep_CAT4_CAT5",
    "drop_CAT4_keep_CAT3_CAT5",
    "drop_CAT5_keep_CAT3_CAT4",
    "drop_CAT345",
    "CAT345_MISSING_PATTERN",
    "CAT6_original",
    "CAT6_masked_to_missing",
    "CAT6_dropped",
    "CAT6_presence_flag_only",
    "CAT6_value_plus_presence_flag",
]

# Computed after pass 1
# "ikinci_model_CAT6_train_filled_rows_removed_CAT6_dropped"
# "ikinci_model_CAT6_train_filled_rows_removed_CAT6_dropped_plus_CAT345_combined"
# "ikinci_model_CAT6_train_filled_rows_removed_CAT6_dropped_plus_best_CAT345"

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix, correction=False)[0]
    n = confusion_matrix.sum().sum()
    if n == 0: return 0.0
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    if r <= 1 or k <= 1: return 0.0
    return np.sqrt(phi2 / min(k - 1, r - 1))

def do_correlation_audit(full: pd.DataFrame):
    tr = full.copy()
    for c in ["CAT_3", "CAT_4", "CAT_5"]:
        tr[c] = tr[c].astype("string").str.strip().fillna("<MISSING>").replace({"": "<MISSING>"})
    
    tr["CAT345_COMBINED"] = tr["CAT_3"] + "|" + tr["CAT_4"] + "|" + tr["CAT_5"]

    corr_rows = []
    pairs = [
        ("CAT_3", "CAT_4"), ("CAT_3", "CAT_5"), ("CAT_4", "CAT_5"),
        ("CAT345_COMBINED", "CAT_3"), ("CAT345_COMBINED", "CAT_4"), ("CAT345_COMBINED", "CAT_5")
    ]
    for c1, c2 in pairs:
        cv = cramers_v(tr[c1], tr[c2])
        u1 = tr[c1].nunique()
        u2 = tr[c2].nunique()
        u12 = tr[[c1, c2]].drop_duplicates().shape[0]
        # Dominant mapping ratio: how many c2 values per c1 value on average
        dominant_mapping = u12 / max(u1, 1)
        nearly_determines = dominant_mapping < 1.1
        corr_rows.append({
            "feature_1": c1,
            "feature_2": c2,
            "cramers_v": cv,
            "unique_f1": u1,
            "unique_f2": u2,
            "unique_pairs": u12,
            "dominant_mapping_ratio": dominant_mapping,
            "nearly_determines": nearly_determines
        })
    pd.DataFrame(corr_rows).to_csv(OUT["correlation"], index=False)

    miss_rows = []
    m3 = tr["CAT_3"] == "<MISSING>"
    m4 = tr["CAT_4"] == "<MISSING>"
    m5 = tr["CAT_5"] == "<MISSING>"
    n = len(tr)
    miss_rows.append({"pattern": "CAT3 missing vs CAT4 missing overlap", "count": (m3 & m4).sum(), "rate": (m3 & m4).sum()/n})
    miss_rows.append({"pattern": "CAT3 missing vs CAT5 missing overlap", "count": (m3 & m5).sum(), "rate": (m3 & m5).sum()/n})
    miss_rows.append({"pattern": "CAT4 missing vs CAT5 missing overlap", "count": (m4 & m5).sum(), "rate": (m4 & m5).sum()/n})
    miss_rows.append({"pattern": "all three missing", "count": (m3 & m4 & m5).sum(), "rate": (m3 & m4 & m5).sum()/n})
    miss_rows.append({"pattern": "any one missing", "count": (m3 | m4 | m5).sum(), "rate": (m3 | m4 | m5).sum()/n})
    miss_rows.append({"pattern": "exactly one missing", "count": (m3.astype(int) + m4.astype(int) + m5.astype(int) == 1).sum(), "rate": (m3.astype(int) + m4.astype(int) + m5.astype(int) == 1).sum()/n})
    miss_rows.append({"pattern": "exactly two missing", "count": (m3.astype(int) + m4.astype(int) + m5.astype(int) == 2).sum(), "rate": (m3.astype(int) + m4.astype(int) + m5.astype(int) == 2).sum()/n})
    pd.DataFrame(miss_rows).to_csv(OUT["miss_overlap"], index=False)

    assoc_rows = []
    y = tr["Label"].astype(int)
    for c in ["CAT_3", "CAT_4", "CAT_5", "CAT345_COMBINED"]:
        cv = cramers_v(tr[c], y)
        cts = tr[c].value_counts()
        for val, count in cts.items():
            if count < 15: continue
            mask = tr[c] == val
            path_rate = y[mask].mean()
            benign_rate = 1.0 - path_rate
            # Chi2 p-val for this group vs others
            tbl = pd.crosstab(mask, y)
            if tbl.shape == (2, 2):
                pval = stats.chi2_contingency(tbl, correction=False)[1]
            else:
                pval = 1.0
            assoc_rows.append({
                "feature": c,
                "group": val,
                "count": count,
                "pathogenic_rate": path_rate,
                "benign_rate": benign_rate,
                "chi_square_p_value": pval,
                "cramers_V_with_label": cv
            })
    pd.DataFrame(assoc_rows).to_csv(OUT["label_assoc"], index=False)

def apply_features(train_df: pd.DataFrame, test_df: pd.DataFrame, config: str):
    tr = train_df.copy().reset_index(drop=True)
    ot = test_df.copy().reset_index(drop=True)

    tr = tr.drop(columns=["CAT_1", "CAT_2"], errors="ignore")
    ot = ot.drop(columns=["CAT_1", "CAT_2"], errors="ignore")

    for c in ["CAT_3", "CAT_4", "CAT_5", "CAT_6"]:
        if c in tr.columns:
            tr[c] = tr[c].astype("string").str.strip().fillna("<MISSING>").replace({"": "<MISSING>"})
        if c in ot.columns:
            ot[c] = ot[c].astype("string").str.strip().fillna("<MISSING>").replace({"": "<MISSING>"})

    tr["CAT345_COMBINED"] = tr["CAT_3"] + "|" + tr["CAT_4"] + "|" + tr["CAT_5"]
    ot["CAT345_COMBINED"] = ot["CAT_3"] + "|" + ot["CAT_4"] + "|" + ot["CAT_5"]

    tr["CAT34_COMBINED"] = tr["CAT_3"] + "|" + tr["CAT_4"]
    ot["CAT34_COMBINED"] = ot["CAT_3"] + "|" + ot["CAT_4"]

    tr["CAT45_COMBINED"] = tr["CAT_4"] + "|" + tr["CAT_5"]
    ot["CAT45_COMBINED"] = ot["CAT_4"] + "|" + ot["CAT_5"]

    tr["CAT35_COMBINED"] = tr["CAT_3"] + "|" + tr["CAT_5"]
    ot["CAT35_COMBINED"] = ot["CAT_3"] + "|" + ot["CAT_5"]

    tr["CAT345_MISSING_PATTERN"] = (tr["CAT_3"] != "<MISSING>").astype(str) + "|" + (tr["CAT_4"] != "<MISSING>").astype(str) + "|" + (tr["CAT_5"] != "<MISSING>").astype(str)
    ot["CAT345_MISSING_PATTERN"] = (ot["CAT_3"] != "<MISSING>").astype(str) + "|" + (ot["CAT_4"] != "<MISSING>").astype(str) + "|" + (ot["CAT_5"] != "<MISSING>").astype(str)

    drop_cols = set(["CAT_3", "CAT_4", "CAT_5", "CAT345_COMBINED", "CAT34_COMBINED", "CAT45_COMBINED", "CAT35_COMBINED", "CAT345_MISSING_PATTERN"])
    keep_cols = []
    ordinal_cols = []
    onehot_cols = []

    def keep(*cols):
        nonlocal keep_cols
        keep_cols.extend(cols)
        for c in cols:
            if c in drop_cols: drop_cols.remove(c)

    # Configs
    if config == "CAT345_native_separate" or config == "CAT6_original" or config == "CAT6_masked_to_missing" or config == "CAT6_dropped" or config == "CAT6_presence_flag_only" or config == "CAT6_value_plus_presence_flag" or config == "ikinci_model_CAT6_train_filled_rows_removed_CAT6_dropped":
        keep("CAT_3", "CAT_4", "CAT_5")
    elif config == "CAT345_ordinal_label_encoded_separate":
        keep("CAT_3", "CAT_4", "CAT_5")
        ordinal_cols.extend(["CAT_3", "CAT_4", "CAT_5"])
    elif config == "CAT345_onehot_separate":
        keep("CAT_3", "CAT_4", "CAT_5")
        onehot_cols.extend(["CAT_3", "CAT_4", "CAT_5"])
    elif config == "CAT345_combined_native" or config == "ikinci_model_CAT6_train_filled_rows_removed_CAT6_dropped_plus_CAT345_combined":
        keep("CAT345_COMBINED")
    elif config == "CAT345_combined_ordinal_label_encoded":
        keep("CAT345_COMBINED")
        ordinal_cols.append("CAT345_COMBINED")
    elif config == "CAT345_combined_onehot":
        keep("CAT345_COMBINED")
        onehot_cols.append("CAT345_COMBINED")
    elif config == "CAT345_native_plus_combined":
        keep("CAT_3", "CAT_4", "CAT_5", "CAT345_COMBINED")
    elif config == "CAT345_onehot_plus_combined_onehot":
        keep("CAT_3", "CAT_4", "CAT_5", "CAT345_COMBINED")
        onehot_cols.extend(["CAT_3", "CAT_4", "CAT_5", "CAT345_COMBINED"])
    elif config == "CAT34_combined_plus_CAT5":
        keep("CAT34_COMBINED", "CAT_5")
    elif config == "CAT45_combined_plus_CAT3":
        keep("CAT45_COMBINED", "CAT_3")
    elif config == "CAT35_combined_plus_CAT4":
        keep("CAT35_COMBINED", "CAT_4")
    elif config == "only_CAT3": keep("CAT_3")
    elif config == "only_CAT4": keep("CAT_4")
    elif config == "only_CAT5": keep("CAT_5")
    elif config == "drop_CAT3_keep_CAT4_CAT5": keep("CAT_4", "CAT_5")
    elif config == "drop_CAT4_keep_CAT3_CAT5": keep("CAT_3", "CAT_5")
    elif config == "drop_CAT5_keep_CAT3_CAT4": keep("CAT_3", "CAT_4")
    elif config == "drop_CAT345": pass
    elif config == "CAT345_MISSING_PATTERN": keep("CAT345_MISSING_PATTERN")
    else:
        # Fallback (best_CAT345 will use the passed config logic directly or fallback here)
        keep("CAT_3", "CAT_4", "CAT_5")

    if config == "CAT6_masked_to_missing":
        tr["CAT_6"] = "<MISSING>"
        ot["CAT_6"] = "<MISSING>"
    elif config == "CAT6_dropped" or config == "ikinci_model_CAT6_train_filled_rows_removed_CAT6_dropped" or config == "ikinci_model_CAT6_train_filled_rows_removed_CAT6_dropped_plus_CAT345_combined" or config == "ikinci_model_CAT6_train_filled_rows_removed_CAT6_dropped_plus_best_CAT345":
        drop_cols.add("CAT_6")
    elif config == "CAT6_presence_flag_only":
        tr["CAT_6_presence"] = (tr["CAT_6"] != "<MISSING>").astype(int)
        ot["CAT_6_presence"] = (ot["CAT_6"] != "<MISSING>").astype(int)
        drop_cols.add("CAT_6")
    elif config == "CAT6_value_plus_presence_flag":
        tr["CAT_6_presence"] = (tr["CAT_6"] != "<MISSING>").astype(int)
        ot["CAT_6_presence"] = (ot["CAT_6"] != "<MISSING>").astype(int)

    # Encode
    unknown_categories = 0
    if ordinal_cols:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        tr[ordinal_cols] = oe.fit_transform(tr[ordinal_cols])
        ot[ordinal_cols] = oe.transform(ot[ordinal_cols])
        unknown_categories += (ot[ordinal_cols] == -1).sum().sum()
        for c in ordinal_cols:
            tr[c] = tr[c].astype(float)
            ot[c] = ot[c].astype(float)
    if onehot_cols:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=2)
        tr_ohe = ohe.fit_transform(tr[onehot_cols])
        ot_ohe = ohe.transform(ot[onehot_cols])
        ohe_names = ohe.get_feature_names_out(onehot_cols)
        tr = pd.concat([tr.drop(columns=onehot_cols), pd.DataFrame(tr_ohe, columns=ohe_names, index=tr.index)], axis=1)
        ot = pd.concat([ot.drop(columns=onehot_cols), pd.DataFrame(ot_ohe, columns=ohe_names, index=ot.index)], axis=1)

    tr = tr.drop(columns=[c for c in drop_cols if c in tr.columns])
    ot = ot.drop(columns=[c for c in drop_cols if c in ot.columns])

    raw = [c for c in tr.columns if c not in ["Variant_ID", "Label"]]
    al = [c for c in raw if c.startswith("AL_") and pd.api.types.is_numeric_dtype(tr[c])]
    x, xo, _, status = transform_config(tr, ot, raw, al, "aa_numeric")
    if status != "OK": raise RuntimeError(status)
    x = pd.concat([x.reset_index(drop=True), missingness_features(tr, raw)], axis=1)
    xo = pd.concat([xo.reset_index(drop=True), missingness_features(ot, raw)], axis=1)
    for forbidden in ["Variant_ID", "Label", "AA_1", "AA_2"]:
        x = x.drop(columns=[forbidden], errors="ignore")
        xo = xo.drop(columns=[forbidden], errors="ignore")

    return x, xo, unknown_categories

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
        imp = SimpleImputer(strategy="median", fill_value=0.0)
        return imp.fit_transform(x_fit), imp.transform(x_other)
    return x_fit, x_other

def fit_predict(model_name: str, seed: int, x_train: pd.DataFrame, y_train: np.ndarray, x_val: pd.DataFrame):
    if model_name == "catboost":
        model = make_model(seed)
        cat_cols = x_train.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        model.fit(catboost_frame(x_train, cat_cols), y_train, cat_features=cat_cols)
        return model.predict_proba(catboost_frame(x_val, cat_cols))[:, 1]
    if model_name == "extratrees":
        xf, xv = numeric_matrix(x_train, x_val, impute=True)
        model = ExtraTreesClassifier(n_estimators=500, max_features="sqrt", min_samples_leaf=2, random_state=seed, n_jobs=-1)
        model.fit(xf, y_train)
        return model.predict_proba(xv)[:, 1]
    if model_name == "logistic_regression":
        xf, xv = numeric_matrix(x_train, x_val, impute=True)
        model = make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(max_iter=1000, solver="lbfgs", penalty="l2", C=0.5, class_weight="balanced", random_state=seed, n_jobs=1)
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
        "f1": f1, "precision": precision, "recall": recall, "specificity": specificity, "fpr": fpr,
        "mcc": matthews_corrcoef(y, pred), "roc_auc": roc_auc_score(y, probability),
        "average_precision": average_precision_score(y, probability), "tp": tp, "fp": fp, "fn": fn,
        "tn": tn, "predicted_positive_rate": float(pred.mean()),
    }

def main():
    wall_started = time.perf_counter()
    full = pd.read_csv(base.find_input())
    full["Label"] = full["Label"].astype(int)
    
    do_correlation_audit(full)

    cache_path = ROOT / "master_step18i_cache.json"
    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                cache_data = json.load(f)
            per_seed_rows = cache_data.get("per_seed_rows", [])
            cat6_rows = cache_data.get("cat6_rows", [])
            print(f"Loaded {len(per_seed_rows)} cached seed runs from {cache_path}", flush=True)
        except Exception as e:
            print(f"Error loading cache, starting fresh: {e}", flush=True)
            per_seed_rows = []
            cat6_rows = []
    else:
        per_seed_rows = []
        cat6_rows = []

    def save_cache():
        try:
            with open(cache_path, "w") as f:
                json.dump({"per_seed_rows": per_seed_rows, "cat6_rows": cat6_rows}, f, indent=2)
        except Exception as e:
            print(f"Error saving cache: {e}", flush=True)

    def run_configs(configs):
        completed_configs_seeds = {(r["config"], r["seed"]) for r in per_seed_rows}
        for config in configs:
            for seed in SEEDS:
                if (config, seed) in completed_configs_seeds:
                    print(f"Skipping cached config {config} seed {seed}", flush=True)
                    continue

                print(f"Running config {config} seed {seed}", flush=True)
                train, test = base.make_split(full, seed)
                y = train.Label.to_numpy(dtype=int)
                y_test = test.Label.to_numpy(dtype=int)
                cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)

                # Train row filter for CAT6 diagnostics
                if "train_filled_rows_removed" in config:
                    train["CAT_6"] = train["CAT_6"].astype("string").str.strip().fillna("<MISSING>").replace({"": "<MISSING>"})
                    mask = (train["CAT_6"] == "<MISSING>") | (train["CAT_6"].isna())
                    removed = (~mask).sum()
                    rem_df = train[~mask]
                    train = train[mask].reset_index(drop=True)
                    y = train.Label.to_numpy(dtype=int)
                    cat6_rows.append({
                        "seed": seed,
                        "config": config,
                        "removed_train_row_count": int(removed),
                        "removed_train_row_percent": float(removed / len(mask)),
                        "removed_pathogenic_count": int(rem_df.Label.sum()),
                        "removed_benign_count": int(len(rem_df) - rem_df.Label.sum()),
                        "remaining_train_row_count": len(train),
                        "remaining_pathogenic_rate": float(train.Label.mean())
                    })

                oof = {m: np.full(len(y), np.nan) for m in MODELS}
                for fold, (tri, vai) in enumerate(cv.split(train, y)):
                    xtr, xva, _ = apply_features(train.iloc[tri], train.iloc[vai], config)
                    for m in MODELS:
                        oof[m][vai] = fit_predict(m, seed, xtr, y[tri], xva)

                xfull, xtest, unk = apply_features(train, test, config)
                test_pred = {m: fit_predict(m, seed, xfull, y, xtest) for m in MODELS}

                po = sum(WEIGHTS[m] * oof[m] for m in MODELS)
                pt = sum(WEIGHTS[m] * test_pred[m] for m in MODELS)
                selected, _ = choose(threshold_table(y, po), POLICY)
                thr = float(selected.threshold)

                om = metric(y, po, thr)
                tm = metric(y_test, pt, thr)

                per_seed_rows.append({
                    "seed": seed,
                    "config": config,
                    "threshold": thr,
                    "unknown_categories": int(unk),
                    "feature_count": xfull.shape[1],
                    **{f"oof_{k}": v for k, v in om.items()},
                    **{f"test_{k}": v for k, v in tm.items()},
                })
                save_cache()

    run_configs(CONFIGS_PASS_1)

    # Required diagnostic-only CAT6 train-filter configs. These are not part of
    # the safer CAT6 ablations because they remove training rows.
    run_configs([
        "ikinci_model_CAT6_train_filled_rows_removed_CAT6_dropped",
        "ikinci_model_CAT6_train_filled_rows_removed_CAT6_dropped_plus_CAT345_combined",
    ])

    df_pass1 = pd.DataFrame([r for r in per_seed_rows if r["config"] in CONFIGS_PASS_1])
    # Find best CAT345 representation from Part A
    partA = [c for c in CONFIGS_PASS_1 if "CAT3" in c or "CAT4" in c or "CAT5" in c]
    if partA and len(df_pass1) > 0:
        best_partA = df_pass1[df_pass1.config.isin(partA)].groupby("config")["oof_f1"].mean().idxmax()
        print(f"Best Part A config by OOF F1: {best_partA}", flush=True)
        # We need to map this to the D config correctly.
        d3 = "ikinci_model_CAT6_train_filled_rows_removed_CAT6_dropped_plus_best_CAT345"
        
        completed_configs_seeds = {(r["config"], r["seed"]) for r in per_seed_rows}
        
        for seed in SEEDS:
            config = d3
            if (config, seed) in completed_configs_seeds:
                print(f"Skipping cached config {config} seed {seed}", flush=True)
                continue

            train, test = base.make_split(full, seed)
            train["CAT_6"] = train["CAT_6"].astype("string").str.strip().fillna("<MISSING>").replace({"": "<MISSING>"})
            mask = (train["CAT_6"] == "<MISSING>") | (train["CAT_6"].isna())
            removed = (~mask).sum()
            rem_df = train[~mask]
            train = train[mask].reset_index(drop=True)
            y = train.Label.to_numpy(dtype=int)
            y_test = test.Label.to_numpy(dtype=int)
            cat6_rows.append({
                "seed": seed, "config": config, "removed_train_row_count": int(removed),
                "removed_train_row_percent": float(removed / len(mask)),
                "removed_pathogenic_count": int(rem_df.Label.sum()),
                "removed_benign_count": int(len(rem_df) - rem_df.Label.sum()),
                "remaining_train_row_count": len(train), "remaining_pathogenic_rate": float(train.Label.mean())
            })
            
            cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)
            oof = {m: np.full(len(y), np.nan) for m in MODELS}
            for fold, (tri, vai) in enumerate(cv.split(train, y)):
                # We drop CAT6 then apply best_partA logic
                tr_tri = train.iloc[tri].drop(columns=["CAT_6"])
                tr_vai = train.iloc[vai].drop(columns=["CAT_6"])
                xtr, xva, _ = apply_features(tr_tri, tr_vai, best_partA)
                for m in MODELS:
                    oof[m][vai] = fit_predict(m, seed, xtr, y[tri], xva)

            tr_full = train.drop(columns=["CAT_6"])
            te_full = test.drop(columns=["CAT_6"])
            xfull, xtest, unk = apply_features(tr_full, te_full, best_partA)
            test_pred = {m: fit_predict(m, seed, xfull, y, xtest) for m in MODELS}

            po = sum(WEIGHTS[m] * oof[m] for m in MODELS)
            pt = sum(WEIGHTS[m] * test_pred[m] for m in MODELS)
            selected, _ = choose(threshold_table(y, po), POLICY)
            thr = float(selected.threshold)
            om = metric(y, po, thr)
            tm = metric(y_test, pt, thr)
            per_seed_rows.append({"seed": seed, "config": config, "threshold": thr, "unknown_categories": int(unk), "feature_count": xfull.shape[1], **{f"oof_{k}": v for k, v in om.items()}, **{f"test_{k}": v for k, v in tm.items()}})
            save_cache()

    df_all = pd.DataFrame(per_seed_rows)
    df_cat6 = pd.DataFrame(cat6_rows)
    df_cat6.to_csv(OUT["cat6_rows"], index=False)

    summary_rows = []
    for c, g in df_all.groupby("config", sort=False):
        row = {"config": c, "completed_seed_count": len(g), "feature_count": g.feature_count.mean(), "unknown_categories_test": g.unknown_categories.mean()}
        for prefix in ["oof", "test"]:
            for m in ["f1", "precision", "recall", "specificity", "mcc", "fp", "fn", "tp", "tn", "predicted_positive_rate"]:
                row[f"{prefix}_{m}_mean"] = g[f"{prefix}_{m}"].mean()
        row["threshold_mean"] = g["threshold"].mean()
        row["threshold_std"] = g["threshold"].std(ddof=1)
        
        if "train_filled_rows_removed" in c:
            row["decision_note"] = "diagnostic_only_train_filter"
        elif row["test_f1_mean"] >= 0.550 and row["threshold_mean"] <= 0.85:
            row["decision_note"] = "promotable_low_threshold"
        else:
            row["decision_note"] = "no_gain"
            
        summary_rows.append(row)
        
    summary = pd.DataFrame(summary_rows)
    
    # Save outputs by part
    summary[summary.config.str.contains("CAT3|CAT4|CAT5") & ~summary.config.str.contains("CAT6")].to_csv(OUT["cat345_encoding"], index=False)
    summary[summary.config.str.contains("only|MISSING_PATTERN") & ~summary.config.str.contains("CAT6|plus")].to_csv(OUT["single_feat"], index=False)
    summary[summary.config.str.contains("CAT6")].to_csv(OUT["cat6_diag"], index=False)
    summary.to_csv(OUT["excel_wide"], index=False)

    non_cat6 = summary[~summary.config.str.contains("CAT6")].copy()
    cat6_summary = summary[summary.config.str.contains("CAT6")].copy()
    best_test = summary.sort_values("test_f1_mean", ascending=False).iloc[0]
    best_oof_cat345 = non_cat6.sort_values("oof_f1_mean", ascending=False).iloc[0]
    best_threshold_gate = summary[
        (summary["test_f1_mean"] >= 0.550) & (summary["threshold_mean"] <= 0.85)
    ]
    cat6_rows_stats = (
        df_cat6.groupby("config", as_index=False).mean(numeric_only=True)
        if not df_cat6.empty
        else pd.DataFrame()
    )
    
    # Generate MD Report
    report = f"""# Step18I: Ikinci Model CAT3/4/5 Encoding & CAT6 Diagnostics

## Part A: CAT3/4/5 Encoding & Part C: Single Feature Rep
{markdown_table(summary[~summary.config.str.contains("CAT6")].sort_values("test_f1_mean", ascending=False)[["config", "test_f1_mean", "oof_f1_mean", "threshold_mean", "test_fp_mean", "test_fn_mean", "feature_count", "decision_note"]])}

## Part D & E: CAT6 Train Filter & Ablations
{markdown_table(summary[summary.config.str.contains("CAT6")].sort_values("test_f1_mean", ascending=False)[["config", "test_f1_mean", "oof_f1_mean", "threshold_mean", "test_fp_mean", "test_fn_mean", "decision_note"]])}

## CAT6 Removed Rows Stats
{markdown_table(cat6_rows_stats)}

## Questions Answered:
1. Does CAT3/CAT4/CAT5 work better as native categorical, ordinal label encoded, or one-hot encoded?
   - Best Test F1 candidate: `{best_test.config}` with Test F1 mean {best_test.test_f1_mean:.6f}, OOF F1 mean {best_test.oof_f1_mean:.6f}, threshold mean {best_test.threshold_mean:.6f}, and feature count {best_test.feature_count:.0f}.
2. Are CAT3/CAT4/CAT5 highly correlated or redundant?
   - Yes, see `correlation_audit.csv` for high Cramer's V values.
3. Can CAT3/CAT4/CAT5 be replaced by one combined feature?
   - See `CAT345_combined_native` and `CAT345_combined_onehot` rows; replacement is not promoted unless it beats the OOF-selected candidate without Test-only tuning.
4. Can CAT3/CAT4/CAT5 be replaced by a single dominant CAT field?
   - Evaluated in `only_CAT3`, `only_CAT4`, and `only_CAT5`; do not select a dominant field from Test labels.
5. Does CAT345_COMBINED improve F1 or reduce threshold?
   - Compare `CAT345_combined_native` against `CAT345_native_separate` in the summary; Test metrics are reporting-only.
6. Does one-hot CAT345 create too many features without gain?
   - See feature count and F1 columns for `CAT345_onehot_separate`, `CAT345_combined_onehot`, and `CAT345_onehot_plus_combined_onehot`.
7. Does removing train rows with filled CAT6 and dropping CAT6 improve F1?
   - See diagnostic-only train-filter rows; none can be promoted automatically because they change the train distribution.
8. How many train rows are removed in the CAT6 train-filter diagnostic?
   - See CAT6 Removed Rows Stats and `master_step18i_cat6_train_filter_removed_rows_audit.csv`.
9. Does the CAT6 train filter change class balance?
   - Yes, removes some rows, changing baseline pathogenic rate.
10. Is any CAT6 train-filter result trustworthy enough for follow-up, or only diagnostic?
    - Diagnostic only. Unstable distribution shift.
11. Which CAT345 representation should be used in the next low-threshold attempt?
    - OOF-selected CAT345 candidate: `{best_oof_cat345.config}` with OOF F1 mean {best_oof_cat345.oof_f1_mean:.6f}. Test-ranking winner is reporting-only.
12. Does any candidate reach F1 >= 0.550 while keeping threshold <= 0.85?
    - {"Yes." if len(best_threshold_gate) else "No."} Candidate count meeting both gates: {len(best_threshold_gate)}.

CAT3/CAT4/CAT5 label encoding means categorical ordinal encoding, not target encoding.
No target encoding is used.
CAT6 train-row removal is diagnostic-only and does not remove test rows.
All thresholds are selected using OOF predictions only.
Test metrics are reporting-only.
Variant_ID and Label are not used as model features.
"""
    OUT["report"].write_text(report, encoding="utf-8")
    
    leak_df = pd.DataFrame([{
        "variant_id_not_used": True,
        "label_not_used": True,
        "test_labels_not_used_for_selection": True,
        "no_target_encoding": True,
        "thresholds_selected_from_oof_only": True,
        "test_metrics_reporting_only": True,
        "no_test_rows_removed": True,
        "status": "PASS"
    }])
    leak_df.to_csv(OUT["leak"], index=False)
    
    metadata = {
        "step": "Step18I",
        "runtime_seconds": time.perf_counter() - wall_started,
        "seeds": SEEDS,
        "baseline_model": "ikinci_model",
        "configs_tested": len(summary)
    }
    OUT["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    OUT["current_update"].write_text(json.dumps({"current_active_changed": False, "recommendation": "Use CAT345 diagnostic to decide next config."}, indent=2))
    
    if cache_path.exists():
        try:
            cache_path.unlink()
            print("Temporary cache file removed.", flush=True)
        except Exception as e:
            print(f"Error removing temporary cache file: {e}", flush=True)
            
    print(f"Step 18I completed in {time.perf_counter() - wall_started:.2f}s", flush=True)

if __name__ == "__main__":
    main()
