"""MASTER Step 4: controlled AA numeric / AL signed-log CatBoost ablation."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from master_step3b_repeated_split_catboost import (
    error_types,
    find_input,
    leakage_audit,
    make_split,
)
from master_v02_catboost_threshold_stability import (
    catboost_frame,
    classification_metrics,
    make_model,
    markdown_table,
    select_weighted_f1_threshold,
)


ROOT = Path(__file__).resolve().parent
FAST_SEEDS = [7, 21, 42, 101, 2024]
CONFIGS = ["raw_baseline", "aa_numeric", "al_log", "aa_numeric_plus_al_log"]
TARGET_PRIOR = {0: 0.85, 1: 0.15}
N_SPLITS = 5
ESTIMATED_TOTAL_RUNTIME_MINUTES = 90
CHECKPOINT_SCHEMA_VERSION = 2
PARTIALS = {
    "oof": ROOT / ".master_step4_partial_oof.csv",
    "test": ROOT / ".master_step4_partial_test.csv",
    "leakage": ROOT / ".master_step4_partial_leakage.csv",
    "runtime": ROOT / ".master_step4_partial_runtime.csv",
    "all_nan": ROOT / ".master_step4_partial_all_nan.csv",
}
OUTPUTS = {
    "split": ROOT / "master_step4_seed_split_summary.csv",
    "oof": ROOT / "master_step4_oof_metrics_by_seed_config.csv",
    "test": ROOT / "master_step4_test_metrics_by_seed_config.csv",
    "summary": ROOT / "master_step4_repeated_test_summary_by_config.csv",
    "delta": ROOT / "master_step4_ablation_delta_vs_raw.csv",
    "leakage": ROOT / "master_step4_leakage_audit.csv",
    "all_nan": ROOT / "master_step4_all_nan_analysis.csv",
    "runtime": ROOT / "master_step4_runtime_report.csv",
    "ranking": ROOT / "master_step4_config_ranking.csv",
    "report": ROOT / "master_step4_report.md",
    "metadata": ROOT / "master_step4_metadata.json",
}


AA_PROPERTIES = {
    "hydrophobicity": dict(zip(
        list("ARNDCQEGHILKMFPSTWYV"),
        [1.8, -4.5, -3.5, -3.5, 2.5, -3.5, -3.5, -0.4, -3.2, 4.5,
         3.8, -3.9, 1.9, 2.8, -1.6, -0.8, -0.7, -0.9, -1.3, 4.2],
    )),
    "volume": dict(zip(
        list("ARNDCQEGHILKMFPSTWYV"),
        [88.6, 173.4, 114.1, 111.1, 108.5, 143.8, 138.4, 60.1, 153.2, 166.7,
         166.7, 168.6, 162.9, 189.9, 112.7, 89.0, 116.1, 227.8, 193.6, 140.0],
    )),
    "molecular_weight": dict(zip(
        list("ARNDCQEGHILKMFPSTWYV"),
        [89.09, 174.20, 132.12, 133.10, 121.16, 146.15, 147.13, 75.07, 155.16,
         131.17, 131.17, 146.19, 149.21, 165.19, 115.13, 105.09, 119.12,
         204.23, 181.19, 117.15],
    )),
    "polarity": dict(zip(
        list("ARNDCQEGHILKMFPSTWYV"),
        [8.1, 10.5, 11.6, 13.0, 5.5, 10.5, 12.3, 9.0, 10.4, 5.2,
         4.9, 11.3, 5.7, 5.2, 8.0, 9.2, 8.6, 5.4, 6.2, 5.9],
    )),
    "charge": {
        "A": 0.0, "R": 1.0, "N": 0.0, "D": -1.0, "C": 0.0,
        "Q": 0.0, "E": -1.0, "G": 0.0, "H": 1.0, "I": 0.0,
        "L": 0.0, "K": 1.0, "M": 0.0, "F": 0.0, "P": 0.0,
        "S": 0.0, "T": 0.0, "W": 0.0, "Y": 0.0, "V": 0.0,
    },
}
VALID_AA = set(AA_PROPERTIES["hydrophobicity"])
AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
BLOSUM62_ROWS = [
    [4,-1,-2,-2,0,-1,-1,0,-2,-1,-1,-1,-1,-2,-1,1,0,-3,-2,0],
    [-1,5,0,-2,-3,1,0,-2,0,-3,-2,2,-1,-3,-2,-1,-1,-3,-2,-3],
    [-2,0,6,1,-3,0,0,0,1,-3,-3,0,-2,-3,-2,1,0,-4,-2,-3],
    [-2,-2,1,6,-3,0,2,-1,-1,-3,-4,-1,-3,-3,-1,0,-1,-4,-3,-3],
    [0,-3,-3,-3,9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],
    [-1,1,0,0,-3,5,2,-2,0,-3,-2,1,0,-3,-1,0,-1,-2,-1,-2],
    [-1,0,0,2,-4,2,5,-2,0,-3,-3,1,-2,-3,-1,0,-1,-3,-2,-2],
    [0,-2,0,-1,-3,-2,-2,6,-2,-4,-4,-2,-3,-3,-2,0,-2,-2,-3,-3],
    [-2,0,1,-1,-3,0,0,-2,8,-3,-3,-1,-2,-1,-2,-1,-2,-2,2,-3],
    [-1,-3,-3,-3,-1,-3,-3,-4,-3,4,2,-3,1,0,-3,-2,-1,-3,-1,3],
    [-1,-2,-3,-4,-1,-2,-3,-4,-3,2,4,-2,2,0,-3,-2,-1,-2,-1,1],
    [-1,2,0,-1,-3,1,1,-2,-1,-3,-2,5,-1,-3,-1,0,-1,-3,-2,-2],
    [-1,-1,-2,-3,-1,0,-2,-3,-2,1,2,-1,5,0,-2,-1,-1,-1,-1,1],
    [-2,-3,-3,-3,-2,-3,-3,-3,-1,0,0,-3,0,6,-4,-2,-2,1,3,-1],
    [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4,7,-1,-1,-4,-3,-2],
    [1,-1,1,0,-1,0,0,0,-1,-2,-2,0,-1,-2,-1,4,1,-3,-2,-2],
    [0,-1,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1,1,5,-2,-2,0],
    [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1,1,-4,-3,-2,11,2,-3],
    [-2,-2,-2,-3,-2,-1,-2,-3,2,-1,-1,-2,-1,3,-3,-2,-2,2,7,-1],
    [0,-3,-3,-3,-1,-2,-2,-3,-3,3,1,-2,1,-1,-2,-2,0,-3,-1,4],
]
GRANTHAM_ROWS = [
    [0,112,111,126,195,91,107,60,86,94,96,106,84,113,27,99,58,148,112,64],
    [112,0,86,96,180,43,54,125,29,97,102,26,91,97,103,110,71,101,77,96],
    [111,86,0,23,139,46,42,80,68,149,153,94,142,158,91,46,65,174,143,133],
    [126,96,23,0,154,61,45,94,81,168,172,101,160,177,108,65,85,181,160,152],
    [195,180,139,154,0,154,170,159,174,198,198,202,196,205,169,112,149,215,194,192],
    [91,43,46,61,154,0,29,87,24,109,113,53,101,116,76,68,42,130,99,96],
    [107,54,42,45,170,29,0,98,40,134,138,56,126,140,93,80,65,152,122,121],
    [60,125,80,94,159,87,98,0,98,135,138,127,127,153,42,56,59,184,147,109],
    [86,29,68,81,174,24,40,98,0,94,99,32,87,100,77,89,47,115,83,84],
    [94,97,149,168,198,109,134,135,94,0,5,102,10,21,95,142,89,61,33,29],
    [96,102,153,172,198,113,138,138,99,5,0,107,15,22,98,145,92,61,36,32],
    [106,26,94,101,202,53,56,127,32,102,107,0,95,102,103,121,78,110,85,97],
    [84,91,142,160,196,101,126,127,87,10,15,95,0,28,87,135,81,67,36,21],
    [113,97,158,177,205,116,140,153,100,21,22,102,28,0,114,155,103,40,22,50],
    [27,103,91,108,169,76,93,42,77,95,98,103,87,114,0,74,38,147,110,68],
    [99,110,46,65,112,68,80,56,89,142,145,121,135,155,74,0,58,177,144,124],
    [58,71,65,85,149,42,65,59,47,89,92,78,81,103,38,58,0,128,92,69],
    [148,101,174,181,215,130,152,184,115,61,61,110,67,40,147,177,128,0,37,88],
    [112,77,143,160,194,99,122,147,83,33,36,85,36,22,110,144,92,37,0,55],
    [64,96,133,152,192,96,121,109,84,29,32,97,21,50,68,124,69,88,55,0],
]
BLOSUM62 = {(a, b): BLOSUM62_ROWS[i][j] for i, a in enumerate(AA_ORDER) for j, b in enumerate(AA_ORDER)}
GRANTHAM = {(a, b): GRANTHAM_ROWS[i][j] for i, a in enumerate(AA_ORDER) for j, b in enumerate(AA_ORDER)}
AA_GROUP = {
    **{aa: 0 for aa in "AVLIM"}, **{aa: 1 for aa in "FWY"},
    **{aa: 2 for aa in "STNQC"}, **{aa: 3 for aa in "KRH"},
    **{aa: 4 for aa in "DE"}, **{aa: 5 for aa in "GP"},
}


def normalized_aa(series: pd.Series) -> pd.Series:
    result = series.astype("string").str.strip().str.upper()
    return result.where(result.isin(VALID_AA))


def aa_features(frame: pd.DataFrame) -> pd.DataFrame:
    aa1 = normalized_aa(frame["AA_1"])
    aa2 = normalized_aa(frame["AA_2"])
    valid1, valid2 = aa1.notna(), aa2.notna()
    pair_valid = valid1 & valid2
    pair_keys = pd.Series(list(zip(aa1, aa2)), index=frame.index)
    values = {
        "aa1_valid": valid1.astype(float), "aa2_valid": valid2.astype(float),
        "aa_pair_valid": pair_valid.astype(float),
        "aa_same": (aa1 == aa2).astype("Float64").where(pair_valid).astype(float),
        "aa_blosum62_score": pair_keys.map(BLOSUM62).where(pair_valid),
        "aa_grantham_distance": pair_keys.map(GRANTHAM).where(pair_valid),
    }
    for name in ["hydrophobicity", "volume", "molecular_weight", "polarity"]:
        lookup = AA_PROPERTIES[name]
        left, right = aa1.map(lookup).astype(float), aa2.map(lookup).astype(float)
        values[f"aa_{name}_1"] = left
        values[f"aa_{name}_2"] = right
        values[f"aa_{name}_diff"] = (right-left).where(pair_valid)
        values[f"aa_{name}_absdiff"] = (right-left).abs().where(pair_valid)
    charge1 = aa1.map(AA_PROPERTIES["charge"]).astype(float)
    charge2 = aa2.map(AA_PROPERTIES["charge"]).astype(float)
    values.update({
        "aa_charge_1": charge1, "aa_charge_2": charge2,
        "aa_charge_diff": (charge2-charge1).where(pair_valid),
        "aa_charge_changed": (charge1 != charge2).astype(float).where(pair_valid),
    })
    for name, members in {
        "aromatic": set("FWY"), "sulfur": set("CM"),
        "proline": {"P"}, "glycine": {"G"},
    }.items():
        flag1, flag2 = aa1.isin(members).astype(float), aa2.isin(members).astype(float)
        values[f"aa_{name}_1"] = flag1.where(valid1)
        values[f"aa_{name}_2"] = flag2.where(valid2)
        values[f"aa_{name}_changed"] = (flag1 != flag2).astype(float).where(pair_valid)
    group1, group2 = aa1.map(AA_GROUP).astype(float), aa2.map(AA_GROUP).astype(float)
    values["aa_group_1_numeric"] = group1
    values["aa_group_2_numeric"] = group2
    values["aa_physicochemical_group_changed"] = (group1 != group2).astype(float).where(pair_valid)
    return pd.DataFrame(values, index=frame.index)


def al_log_features(frame: pd.DataFrame, al_columns: list[str]) -> pd.DataFrame:
    values = {}
    for column in al_columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        transformed = np.sign(numeric) * np.log1p(np.abs(numeric))
        values[f"LOG_AL_{column}"] = transformed.replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame(values, index=frame.index)


def transform_config(train: pd.DataFrame, test: pd.DataFrame, raw_features: list[str],
                     al_columns: list[str], config: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str], str]:
    if config in {"aa_numeric", "aa_numeric_plus_al_log"} and not {"AA_1", "AA_2"}.issubset(train.columns):
        return train[raw_features], test[raw_features], raw_features, "SKIPPED: AA_1/AA_2 not found"
    base_features = (
        [col for col in raw_features if col not in {"AA_1", "AA_2"}]
        if config in {"aa_numeric", "aa_numeric_plus_al_log"}
        else raw_features
    )
    train_parts, test_parts = [train[base_features].copy()], [test[base_features].copy()]
    if config in {"aa_numeric", "aa_numeric_plus_al_log"}:
        train_parts.append(aa_features(train))
        test_parts.append(aa_features(test))
    if config in {"al_log", "aa_numeric_plus_al_log"}:
        if not al_columns:
            return train[raw_features], test[raw_features], raw_features, "SKIPPED: no numeric AL_* columns"
        train_parts.append(al_log_features(train, al_columns))
        test_parts.append(al_log_features(test, al_columns))
    x_train = pd.concat(train_parts, axis=1)
    x_test = pd.concat(test_parts, axis=1)
    return x_train, x_test, x_train.columns.tolist(), "OK"


def load_partials(allowed_seeds: list[int]) -> tuple[dict[str, list[dict]], set[tuple[int, str]]]:
    if not all(path.exists() for path in PARTIALS.values()):
        return {name: [] for name in PARTIALS}, set()
    runtime_frame = pd.read_csv(PARTIALS["runtime"])
    if "checkpoint_schema" not in runtime_frame.columns or not runtime_frame.checkpoint_schema.eq(CHECKPOINT_SCHEMA_VERSION).all():
        print("Ignoring incompatible older Step 4 checkpoints.", flush=True)
        return {name: [] for name in PARTIALS}, set()
    rows = {name: pd.read_csv(path).to_dict("records") for name, path in PARTIALS.items()}
    key_sets = [
        {(int(row["seed"]), str(row["config_name"])) for row in values}
        for values in rows.values()
    ]
    completed = set.intersection(*key_sets) if key_sets else set()
    completed = {key for key in completed if key[0] in allowed_seeds}
    rows = {
        name: [row for row in values if (int(row["seed"]), str(row["config_name"])) in completed]
        for name, values in rows.items()
    }
    return rows, completed


def save_partials(rows: dict[str, list[dict]]) -> None:
    for name, path in PARTIALS.items():
        pd.DataFrame(rows[name]).to_csv(path, index=False)


def config_leakage(base: dict, seed: int, config: str, features: list[str],
                   status: str) -> dict:
    suspicious_tokens = ("label", "target", "outcome", "variant_id", "variantid")
    suspicious = [col for col in features if any(token in col.lower() for token in suspicious_tokens)]
    label_in = "Label" in features
    id_in = "Variant_ID" in features
    numeric_config = config in {"aa_numeric", "aa_numeric_plus_al_log"}
    aa_raw_retained = numeric_config and any(col in features for col in ["AA_1", "AA_2"])
    failures = []
    if base["variant_id_overlap_count"] > 0:
        failures.append("Variant_ID overlap")
    if label_in:
        failures.append("Label in config features")
    if id_in:
        failures.append("Variant_ID in config features")
    if aa_raw_retained:
        failures.append("AA_1/AA_2 retained in numeric config")
    if status.startswith("SKIPPED"):
        leakage_status = "SKIPPED"
    elif failures:
        leakage_status = "FAIL"
    elif base["status"] == "WARNING" or suspicious:
        leakage_status = "WARNING"
    else:
        leakage_status = "PASS"
    return {
        "seed": seed, "config_name": config, "status": leakage_status,
        "variant_id_overlap_count": base["variant_id_overlap_count"],
        "train_duplicate_variant_id_count": base["train_duplicate_variant_id_count"],
        "test_duplicate_variant_id_count": base["test_duplicate_variant_id_count"],
        "full_duplicate_row_count": json.loads(base["details"])["full_duplicate_rows"],
        "identical_raw_feature_hash_count": base["identical_feature_hash_count"],
        "identical_hash_all_nan_explained": base["identical_hash_all_nan_explained"],
        "config_feature_count": len(features), "label_in_config_features": label_in,
        "variant_id_in_config_features": id_in,
        "aa_raw_columns_retained_in_numeric_config": aa_raw_retained,
        "suspicious_config_feature_columns": json.dumps(suspicious),
        "transformation_used_label": False,
        "transformation_used_test_learned_statistics": False,
        "threshold_selected_on_test": False, "model_selected_on_test": False,
        "details": json.dumps({"base_split_audit": base["details"], "failures": failures}),
    }


def bootstrap_raw_from_step3b(seeds: list[int], rows: dict[str, list[dict]],
                              completed: set[tuple[int, str]], full: pd.DataFrame,
                              raw_features: list[str], full_duplicates: int) -> None:
    required = [
        ROOT / "master_step3b_oof_metrics_by_seed.csv",
        ROOT / "master_step3b_test_metrics_by_seed.csv",
        ROOT / "master_step3b_all_nan_analysis_by_seed.csv",
    ]
    if not all(path.exists() for path in required):
        return
    source_oof = pd.read_csv(required[0]).set_index("seed")
    source_test = pd.read_csv(required[1]).set_index("seed")
    source_nan = pd.read_csv(required[2]).set_index("seed")
    for seed in seeds:
        key = (seed, "raw_baseline")
        if key in completed or seed not in source_oof.index or seed not in source_test.index:
            continue
        train, test = make_split(full, seed)
        base_audit, _ = leakage_audit(full, train, test, raw_features, seed, full_duplicates)
        oof_row = source_oof.loc[seed].to_dict()
        oof_row.update({"seed": seed, "config_name": "raw_baseline", "status": "OK",
                        "feature_count": len(raw_features)})
        test_row = source_test.loc[seed].to_dict()
        test_row.update({"seed": seed, "config_name": "raw_baseline", "status": "OK"})
        rows["oof"].append(oof_row)
        rows["test"].append(test_row)
        rows["leakage"].append(config_leakage(
            base_audit, seed, "raw_baseline", raw_features, "OK"
        ))
        nan_row = source_nan.loc[seed].to_dict()
        nan_row.update({"seed": seed, "config_name": "raw_baseline"})
        rows["all_nan"].append(nan_row)
        rows["runtime"].append({
            "seed": seed, "config_name": "raw_baseline", "actual_runtime_seconds": 0.0,
            "split_seconds": 0.0, "feature_transform_seconds": 0.0,
            "leakage_seconds": 0.0, "oof_training_seconds": 0.0,
            "final_fit_test_seconds": 0.0, "reporting_seconds": 0.0,
            "status": "REUSED_STEP3B_VERIFIED", "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
        })
        completed.add(key)
    save_partials(rows)


def aggregate_by_config(test: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for config, group in test[test.status.eq("OK")].groupby("config_name", sort=False):
        thresholds = oof[(oof.config_name.eq(config)) & (oof.status.eq("OK"))].selected_threshold
        rows.append({
            "config_name": config, "n_successful_seeds": len(group),
            "test_f1_mean": group.test_f1.mean(), "test_f1_std": group.test_f1.std(ddof=1),
            "test_f1_min": group.test_f1.min(), "test_f1_q25": group.test_f1.quantile(.25),
            "test_f1_median": group.test_f1.median(), "test_f1_q75": group.test_f1.quantile(.75),
            "test_f1_max": group.test_f1.max(),
            "test_precision_mean": group.test_precision.mean(), "test_precision_std": group.test_precision.std(ddof=1),
            "test_recall_mean": group.test_recall.mean(), "test_recall_std": group.test_recall.std(ddof=1),
            "test_fpr_mean": group.test_fpr.mean(), "test_fpr_std": group.test_fpr.std(ddof=1),
            "test_balanced_accuracy_mean": group.test_balanced_accuracy.mean(),
            "test_balanced_accuracy_std": group.test_balanced_accuracy.std(ddof=1),
            "test_fp_mean": group.test_fp.mean(), "test_fp_std": group.test_fp.std(ddof=1),
            "test_fn_mean": group.test_fn.mean(), "test_fn_std": group.test_fn.std(ddof=1),
            "selected_threshold_mean": thresholds.mean(), "selected_threshold_std": thresholds.std(ddof=1),
            "feature_count_mean": oof[(oof.config_name.eq(config)) & (oof.status.eq("OK"))].feature_count.mean(),
        })
    return pd.DataFrame(rows)


def main() -> None:
    start = time.perf_counter()
    seeds = FAST_SEEDS
    print("=== MASTER Step 4 estimated runtime ===", flush=True)
    print("One seed + one config 5-fold OOF + final fit: approximately 3-5 minutes", flush=True)
    print("5 seeds x 4 configs = 20 runs; approximately 60-120 minutes", flush=True)
    print("A slow machine may approach 2 hours", flush=True)
    print("Runtime varies by machine. Verified Step 3b raw runs are reused where available.", flush=True)

    input_path = find_input()
    full = pd.read_csv(input_path)
    full["Label"] = full["Label"].astype(int)
    raw_features = [col for col in full.columns if col not in ["Variant_ID", "Label"]]
    numeric_al = [
        col for col in raw_features
        if col.startswith("AL_") and pd.api.types.is_numeric_dtype(full[col])
    ]
    negative_al = [col for col in numeric_al if (pd.to_numeric(full[col], errors="coerce") < 0).any()]
    nonnegative_al = [col for col in numeric_al if col not in negative_al]
    al_inf_input_count = int(np.isinf(full[numeric_al].to_numpy(dtype=float)).sum()) if numeric_al else 0
    full_duplicates = int(full.duplicated().sum())
    rows, completed = load_partials(seeds)
    bootstrap_raw_from_step3b(
        seeds, rows, completed, full, raw_features, full_duplicates
    )
    print(f"FAST_5_SEED; seeds={seeds}; resumed/reused pairs={len(completed)}", flush=True)

    split_rows = []
    for seed in seeds:
        split_start = time.perf_counter()
        train, test = make_split(full, seed)
        seed_split_seconds = time.perf_counter() - split_start
        split_rows.append({
            "seed": seed, "train_n": len(train), "test_n": len(test),
            "train_benign": int(train.Label.eq(0).sum()), "train_pathogenic": int(train.Label.eq(1).sum()),
            "test_benign": int(test.Label.eq(0).sum()), "test_pathogenic": int(test.Label.eq(1).sum()),
            "train_benign_share": train.Label.eq(0).mean(), "test_benign_share": test.Label.eq(0).mean(),
        })
        base_audit, _ = leakage_audit(full, train, test, raw_features, seed, full_duplicates)
        for config in CONFIGS:
            key = (seed, config)
            if key in completed:
                continue
            run_start = time.perf_counter()
            split_seconds = seed_split_seconds  # Same deterministic split is shared across configs.
            print(f"\nSeed={seed} config={config}", flush=True)
            transform_start = time.perf_counter()
            x_train_raw, x_test_raw, config_features, config_status = transform_config(
                train, test, raw_features, numeric_al, config
            )
            feature_transform_seconds = time.perf_counter() - transform_start
            leakage_start = time.perf_counter()
            leak_row = config_leakage(base_audit, seed, config, config_features, config_status)
            leakage_seconds = time.perf_counter() - leakage_start
            if config_status.startswith("SKIPPED"):
                rows["oof"].append({"seed": seed, "config_name": config, "status": config_status,
                                    "feature_count": len(config_features)})
                rows["test"].append({"seed": seed, "config_name": config, "status": config_status})
                rows["leakage"].append(leak_row)
                rows["all_nan"].append({"seed": seed, "config_name": config})
                rows["runtime"].append({
                    "seed": seed, "config_name": config, "actual_runtime_seconds": time.perf_counter()-run_start,
                    "split_seconds": split_seconds, "feature_transform_seconds": feature_transform_seconds,
                    "leakage_seconds": leakage_seconds, "oof_training_seconds": 0.0,
                    "final_fit_test_seconds": 0.0, "reporting_seconds": 0.0, "status": config_status,
                    "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
                })
                save_partials(rows)
                completed.add(key)
                continue

            categorical = x_train_raw.select_dtypes(include=["object", "category"]).columns.tolist()
            x_train = catboost_frame(x_train_raw, categorical)
            x_test = catboost_frame(x_test_raw, categorical)
            y_train = train.Label.to_numpy(dtype=int)
            y_test = test.Label.to_numpy(dtype=int)
            shares = pd.Series(y_train).value_counts(normalize=True).to_dict()
            sample_weights = np.where(y_train == 0, .85/shares[0], .15/shares[1])

            oof_start = time.perf_counter()
            cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
            oof_prob = np.full(len(train), np.nan)
            for fold, (fit_idx, valid_idx) in enumerate(cv.split(x_train, y_train), 1):
                model = make_model(seed)
                model.fit(x_train.iloc[fit_idx], y_train[fit_idx], cat_features=categorical)
                oof_prob[valid_idx] = model.predict_proba(x_train.iloc[valid_idx])[:, 1]
                print(f"  OOF fold {fold}/{N_SPLITS}", flush=True)
            threshold, weighted = select_weighted_f1_threshold(y_train, oof_prob, sample_weights)
            oof_raw = classification_metrics(y_train, oof_prob, threshold)
            oof_seconds = time.perf_counter() - oof_start

            final_start = time.perf_counter()
            final_model = make_model(seed)
            final_model.fit(x_train, y_train, cat_features=categorical)
            test_prob = final_model.predict_proba(x_test)[:, 1]
            test_raw = classification_metrics(y_test, test_prob, threshold)
            final_seconds = time.perf_counter() - final_start

            reporting_start = time.perf_counter()
            rows["oof"].append({
                "seed": seed, "config_name": config, "status": "OK", "feature_count": len(config_features),
                "selected_threshold": threshold, "oof_auc": roc_auc_score(y_train, oof_prob),
                "oof_average_precision": average_precision_score(y_train, oof_prob),
                "oof_f1": oof_raw["f1"], "oof_precision": oof_raw["precision"],
                "oof_recall": oof_raw["recall"], "oof_specificity": oof_raw["specificity"],
                "oof_fpr": oof_raw["fpr"], "oof_balanced_accuracy": oof_raw["balanced_accuracy"],
                "oof_predicted_positive_rate": oof_raw["predicted_positive_rate"], **weighted,
                "oof_tn": oof_raw["tn"], "oof_fp": oof_raw["fp"],
                "oof_fn": oof_raw["fn"], "oof_tp": oof_raw["tp"],
            })
            rows["test"].append({
                "seed": seed, "config_name": config, "status": "OK",
                "selected_threshold_from_oof": threshold, "threshold_source": "OOF train CV only",
                "test_usage": "repeated split reporting only",
                "test_auc": roc_auc_score(y_test, test_prob),
                "test_average_precision": average_precision_score(y_test, test_prob),
                "test_f1": test_raw["f1"], "test_precision": test_raw["precision"],
                "test_recall": test_raw["recall"], "test_specificity": test_raw["specificity"],
                "test_fpr": test_raw["fpr"], "test_balanced_accuracy": test_raw["balanced_accuracy"],
                "test_predicted_positive_rate": test_raw["predicted_positive_rate"],
                "test_tn": test_raw["tn"], "test_fp": test_raw["fp"],
                "test_fn": test_raw["fn"], "test_tp": test_raw["tp"],
            })
            rows["leakage"].append(leak_row)
            all_nan_mask = x_test_raw.isna().all(axis=1).to_numpy()
            pred = (test_prob >= threshold).astype(int)
            errors = pd.Series(error_types(y_test[all_nan_mask], pred[all_nan_mask])).value_counts().to_dict()
            rows["all_nan"].append({
                "seed": seed, "config_name": config, "all_nan_test_count": int(all_nan_mask.sum()),
                "all_nan_test_benign": int(((y_test == 0) & all_nan_mask).sum()),
                "all_nan_test_pathogenic": int(((y_test == 1) & all_nan_mask).sum()),
                "all_nan_test_tn": int(errors.get("TN", 0)), "all_nan_test_fp": int(errors.get("FP", 0)),
                "all_nan_test_fn": int(errors.get("FN", 0)), "all_nan_test_tp": int(errors.get("TP", 0)),
            })
            reporting_seconds = time.perf_counter() - reporting_start
            actual = time.perf_counter() - run_start
            rows["runtime"].append({
                "seed": seed, "config_name": config, "actual_runtime_seconds": actual,
                "split_seconds": split_seconds, "feature_transform_seconds": feature_transform_seconds,
                "leakage_seconds": leakage_seconds, "oof_training_seconds": oof_seconds,
                "final_fit_test_seconds": final_seconds, "reporting_seconds": reporting_seconds,
                "status": "COMPLETED",
                "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
            })
            save_partials(rows)
            completed.add(key)
            pending = len(seeds)*len(CONFIGS)-len(completed)
            completed_times = [float(row["actual_runtime_seconds"]) for row in rows["runtime"] if row["status"] == "COMPLETED"]
            eta = np.mean(completed_times)*pending/60 if completed_times else np.nan
            print(f"Completed {seed}/{config}: {actual/60:.2f} min, threshold={threshold:.6f}, "
                  f"test F1={test_raw['f1']:.6f}, FP/FN={test_raw['fp']}/{test_raw['fn']}, ETA={eta:.1f} min", flush=True)

    config_order = {name: i for i, name in enumerate(CONFIGS)}
    for values in rows.values():
        values.sort(key=lambda row: (seeds.index(int(row["seed"])), config_order[str(row["config_name"])]))
    oof = pd.DataFrame(rows["oof"])
    test_metrics = pd.DataFrame(rows["test"])
    leakage = pd.DataFrame(rows["leakage"])
    all_nan = pd.DataFrame(rows["all_nan"])
    runtime = pd.DataFrame(rows["runtime"])
    split = pd.DataFrame(split_rows)
    summary = aggregate_by_config(test_metrics, oof)
    raw = summary.set_index("config_name").loc["raw_baseline"]
    delta_rows = []
    for _, row in summary.iterrows():
        delta_f1 = row.test_f1_mean - raw.test_f1_mean
        if row.config_name == "raw_baseline":
            interpretation = "reference"
        elif delta_f1 >= .01:
            interpretation = "promising 5-seed fast-ablation evidence"
        elif delta_f1 <= -.01:
            interpretation = "not promising in 5-seed fast ablation"
        else:
            interpretation = "small/inconclusive 5-seed difference"
        delta_rows.append({
            "config_name": row.config_name, "delta_test_f1_mean": delta_f1,
            "delta_test_precision_mean": row.test_precision_mean-raw.test_precision_mean,
            "delta_test_recall_mean": row.test_recall_mean-raw.test_recall_mean,
            "delta_test_fpr_mean": row.test_fpr_mean-raw.test_fpr_mean,
            "delta_test_balanced_accuracy_mean": row.test_balanced_accuracy_mean-raw.test_balanced_accuracy_mean,
            "delta_test_fp_mean": row.test_fp_mean-raw.test_fp_mean,
            "delta_test_fn_mean": row.test_fn_mean-raw.test_fn_mean,
            "interpretation": interpretation,
        })
    delta = pd.DataFrame(delta_rows)
    ranking = summary.sort_values(
        ["test_f1_mean", "test_f1_std", "test_fpr_mean", "test_balanced_accuracy_mean"],
        ascending=[False, True, True, False],
    ).reset_index(drop=True)
    ranking.insert(0, "aggregate_rank", np.arange(1, len(ranking)+1))
    ranking["ranking_scope"] = "aggregate repeated split ranking; not single-seed selection"

    split.to_csv(OUTPUTS["split"], index=False)
    oof.to_csv(OUTPUTS["oof"], index=False)
    test_metrics.to_csv(OUTPUTS["test"], index=False)
    summary.to_csv(OUTPUTS["summary"], index=False)
    delta.to_csv(OUTPUTS["delta"], index=False)
    leakage.to_csv(OUTPUTS["leakage"], index=False)
    all_nan.to_csv(OUTPUTS["all_nan"], index=False)
    runtime.to_csv(OUTPUTS["runtime"], index=False)
    ranking.to_csv(OUTPUTS["ranking"], index=False)

    wall_minutes = (time.perf_counter()-start)/60
    actual_minutes = runtime.loc[runtime.status.eq("COMPLETED"), "actual_runtime_seconds"].sum()/60
    best = ranking.iloc[0]
    engineered = delta[~delta.config_name.eq("raw_baseline")]
    best_engineered_delta = engineered.delta_test_f1_mean.max()
    engineered_promising = best_engineered_delta >= .01 and not leakage.status.eq("FAIL").any()
    next_step = (
        "Use the promising feature set in an ensemble or model-family comparison; this 5-seed evidence is not final confirmation."
        if engineered_promising else
        "Keep raw CatBoost + threshold policy as the main baseline; next consider ensemble or threshold-policy revision."
    )
    metadata = {
        "project_folder": "TeknoPatho", "step": "Step 4 5-seed AA numeric and AL-log fast ablation",
        "seed_list": seeds,
        "split_rule": "same automatic prior-controlled split per seed across all configs",
        "target_train_prior": {"benign": .15, "pathogenic": .85},
        "target_test_prior": {"benign": .85, "pathogenic": .15},
        "model_family": "CatBoostClassifier",
        "threshold_policy": "weighted_final_like_oof_f1_threshold",
        "configs": CONFIGS, "aa_raw_columns_removed_in_numeric_configs": True,
        "al_log_transform_policy": "log1p(x) for non-negative values; sign(x)*log1p(abs(x)) for negative values; inf -> NaN",
        "test_usage": "repeated split reporting only",
        "no_single_seed_selection": True,
        "seed_count": 5, "eight_seed_pilot": False, "full_20_seed_confirmation": False,
        "estimated_total_runtime_minutes": ESTIMATED_TOTAL_RUNTIME_MINUTES,
        "actual_total_runtime_minutes": actual_minutes,
        "raw_baseline_source": "verified Step 3b artifacts reused for identical seed/split/model/policy runs",
        "created_files": [Path(__file__).name, *[p.name for p in OUTPUTS.values()], *[p.name for p in PARTIALS.values()]],
    }
    OUTPUTS["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    report = f"""# MASTER Step 4 5-seed AA numeric / AL-log fast ablation

## Purpose and protocol

AA numeric and AL signed-log features are added to all raw features to test incremental information. v0 established the raw CatBoost baseline; v0.1 introduced final-like weighted OOF thresholding; v0.2 tested same-split seed stability; Step 3b found repeated-split raw F1 near 0.50 with volatile FP control.

- This is a **5-seed fast ablation**; seeds={seeds}; configs={CONFIGS}.
- The 8-seed pilot was not run. The 20-seed confirmation was not run.
- Same seed uses the identical train/test split for every config.
- Thresholds are selected only from each train OOF prediction set. Test is repeated-split reporting only.
- No ensemble, SHAP, or hyperparameter tuning. Transformations are deterministic and never use Label.
- Verified Step 3b raw runs were reused because their splits, model, seeds, and policy are identical; transformed configs were trained from scratch.
- Estimated runtime=60-120 minutes (planning value {ESTIMATED_TOTAL_RUNTIME_MINUTES}); accumulated trained-config runtime={actual_minutes:.2f} minutes and current execution wall time={wall_minutes:.2f} minutes (reused raw compute excluded).

## Configs

- `raw_baseline`: {len(raw_features)} original features.
- `aa_numeric`: raw except `AA_1/AA_2`, plus 41 deterministic numeric AA features. Raw AA columns are explicitly removed.
- `al_log`: raw + {len(numeric_al)} deterministic signed-log AL features.
- `aa_numeric_plus_al_log`: both additions; raw `AA_1/AA_2` removed and original AL features retained.
- AA lookups: embedded BLOSUM62, embedded standard Grantham distance matrix, and Kyte-Doolittle hydrophobicity.
- LOG_AL formula: non-negative values use `log1p(x)`; negative values use `sign(x) * log1p(abs(x))`; missing remains missing and infinities become NaN.
- Numeric AL columns found / LOG_AL features produced: {len(numeric_al)} / {len(numeric_al)}. Signed-log columns={len(negative_al)}; non-negative log1p columns={len(nonnegative_al)}; source infinities converted to NaN={al_inf_input_count}.

## Leakage audit

- FAIL present: {leakage.status.eq('FAIL').any()}
- WARNING present: {leakage.status.eq('WARNING').any()}
- Status counts: {leakage.status.value_counts().to_dict()}

{markdown_table(leakage)}

## Repeated-split aggregate results

{markdown_table(summary)}

## Delta versus raw

{markdown_table(delta)}

## Aggregate ranking

{markdown_table(ranking)}

This is aggregate repeated-split ranking, not selection of a single seed. Five-seed fast-ablation evidence is not final proof.

## AA / AL interpretation

- AA numeric delta F1: {float(delta.loc[delta.config_name.eq('aa_numeric'), 'delta_test_f1_mean'].iloc[0]):+.6f}.
- AL log delta F1: {float(delta.loc[delta.config_name.eq('al_log'), 'delta_test_f1_mean'].iloc[0]):+.6f}.
- AA + AL log delta F1: {float(delta.loc[delta.config_name.eq('aa_numeric_plus_al_log'), 'delta_test_f1_mean'].iloc[0]):+.6f}.
- Best aggregate config: {best.config_name}, F1={best.test_f1_mean:.6f} ± {best.test_f1_std:.6f}.

## FP/FN and all-NaN diagnostics

{markdown_table(all_nan)}

Engineered features promising under the predeclared +0.01 aggregate F1 rule: {engineered_promising}.

## Next step

{next_step}

Do not claim definitive improvement from this 5-seed fast ablation. Aggregate repeated-split mean ± std is the evidence; no single seed is final evidence.
"""
    OUTPUTS["report"].write_text(report, encoding="utf-8")

    print("\n=== MASTER Step 4 final summary ===")
    print(f"Total seeds/configs: {len(seeds)}/{len(CONFIGS)}")
    print(f"Successful pairs: {len(test_metrics[test_metrics.status.eq('OK')])}")
    print(f"Leakage FAIL/WARNING: {leakage.status.eq('FAIL').any()}/{leakage.status.eq('WARNING').any()}")
    for config in CONFIGS:
        row = summary.set_index('config_name').loc[config]
        d = delta.set_index('config_name').loc[config, 'delta_test_f1_mean']
        print(f"{config}: mean F1={row.test_f1_mean:.6f}, delta={d:+.6f}")
    print(f"Best config: {best.config_name}; F1={best.test_f1_mean:.6f} ± {best.test_f1_std:.6f}")
    print(f"Best mean precision/recall/FP/FN: {best.test_precision_mean:.6f}/{best.test_recall_mean:.6f}/{best.test_fp_mean:.3f}/{best.test_fn_mean:.3f}")
    print(f"Accumulated trained-config runtime: {actual_minutes:.2f} minutes")
    print(f"Current execution wall time: {wall_minutes:.2f} minutes")
    trained = runtime[runtime.status.eq('COMPLETED')]
    print(f"Average trained seed/config runtime: {trained.actual_runtime_seconds.mean()/60:.2f} minutes")
    print("8-seed pilot performed: False")
    print("20-seed confirmation performed: False")
    print(f"Recommended next step: {next_step}")


if __name__ == "__main__":
    main()
