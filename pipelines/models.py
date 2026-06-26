"""PathoSense — deployed model & preprocessing helpers (extracted from run_pipeline.py 2026-06-26).

The CANONICAL home for the 5 functions the deployed pipeline (predict.py, transfer.py) and the
analysis/ scripts share. Importing these no longer drags in run_pipeline.py's legacy SHAP/plotting
imports or its 595 lines of legacy eval — just the model libs. run_pipeline.py re-exports them for
backward compatibility with the ~120 analysis/ scripts that still import them from there.

  preprocess_data            median+rank per-category preprocessing (small-panel own recipe)
  _encode_for_mi / select_features_mrmr   reference mRMR feature selection
  get_model_params           size-adaptive LightGBM hyperparameters
  train_multi_seed_ensemble  multi-seed LightGBM OOF + final ensemble (nested early-stopping)
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_selection import mutual_info_classif

from config import SEED
from schema import (partition_columns, classify_columns,
                    CAT_CONSTANT, CAT_SPARSE, CAT_POPULATION_FREQ,
                    CAT_CONSERVATION, CAT_SCORE_CENTERED)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_data(X_train, X_val, X_test, corr_threshold=0.95):
    """Per-category preprocessing for the YARISMA_TRAIN data.

    Replaces the old IQR/median/var/corr pipeline with one that applies
    a different transform per column category (population MAF gets rank
    transform; conservation gets clip + median impute; in-silico passes
    through; constants are dropped; etc.).

    Pipeline order:
        1. Classify every numeric column (schema.classify_columns)
        2. Drop columns that are constant or sparse (no information)
        3. Per-category transforms:
             a. population_frequency -> rank transform on training, then
                map val/test by interpolation; add missing-indicator;
                fill NaN with 0.5 (median rank).
             b. conservation/score_centered -> clip outliers at p1/p99;
                add missing-indicator; impute remaining NaN with training median.
             c. in_silico_prob / cadd_phred / bounded_positive / aa_derived
                / near0 / near1 -> as-is (LightGBM handles NaN); add
                missing-indicator if missing>5%; impute remaining NaN with
                training median.
        4. Drop near-zero-variance after transform (some columns may still be
           uninformative).
        5. Drop pairwise-correlated columns at |r| > corr_threshold.  Critical:
           uses NaN-aware pairwise Pearson, NOT post-imputation correlation
           (post-imputation correlation drops features that are different in
           reality but happen to collapse to the same median).
        6. Reassemble numeric+categorical, return.

    Categorical columns are passed through; LightGBM handles pd.Categorical
    with native NaN. A missing-indicator is added for any categorical with
    5-95% NaN.
    """
    info: dict = {}

    _, numeric_cols, categorical_cols, _ = partition_columns(X_train)

    Xn_train = X_train[numeric_cols].copy()
    Xn_val   = X_val[numeric_cols].copy()
    Xn_test  = X_test[numeric_cols].copy()

    # ---- 1. Classify every numeric column on training data ----
    col_categories = classify_columns(X_train)
    info["column_categories_count"] = (
        pd.Series(col_categories).value_counts().to_dict()
    )

    # ---- 2. Drop constants / sparse ----
    drop_constants = [c for c, cat in col_categories.items()
                      if cat in (CAT_CONSTANT, CAT_SPARSE)]
    Xn_train.drop(columns=drop_constants, inplace=True, errors="ignore")
    Xn_val.drop(columns=drop_constants, inplace=True, errors="ignore")
    Xn_test.drop(columns=drop_constants, inplace=True, errors="ignore")
    info["dropped_constants"] = drop_constants

    # ---- 3. Per-category transforms ----
    # Track which cols need each transform.
    pop_cols   = [c for c in Xn_train.columns
                  if col_categories.get(c) == CAT_POPULATION_FREQ]
    cons_cols  = [c for c in Xn_train.columns
                  if col_categories.get(c) in (CAT_CONSERVATION, CAT_SCORE_CENTERED)]
    # Everything else that's not categorical (in-silico bulk, cadd, bounded
    # positive, engineered AA, near-zero/one) is in the "passthrough" group.
    other_cols = [c for c in Xn_train.columns
                  if c not in pop_cols and c not in cons_cols]

    # 3a. Population MAF: rank-transform per-column on training.
    # Use pandas Series.rank(pct=True) on training -> mapping for val/test
    # via interpolated quantile lookup.
    pop_quantile_grids: dict = {}
    for c in pop_cols:
        s = Xn_train[c]
        ranks = s.rank(pct=True, na_option="keep")
        Xn_train[c] = ranks
        # Store the original values + their ranks so we can map val/test:
        non_nan_vals = s.dropna().sort_values().values
        non_nan_ranks = np.linspace(1 / len(non_nan_vals), 1.0, len(non_nan_vals)) if len(non_nan_vals) else np.array([])
        pop_quantile_grids[c] = (non_nan_vals, non_nan_ranks)
        for df in (Xn_val, Xn_test):
            vals = df[c].values.astype(float)
            mapped = np.full_like(vals, np.nan, dtype=float)
            present = ~np.isnan(vals)
            if present.any() and len(non_nan_vals) > 0:
                mapped[present] = np.interp(
                    vals[present], non_nan_vals, non_nan_ranks,
                    left=non_nan_ranks[0], right=non_nan_ranks[-1])
            df[c] = mapped

    # 3b. Conservation: clip at training p1/p99
    conservation_clip: dict = {}
    for c in cons_cols:
        s = Xn_train[c].dropna()
        if len(s) < 10:
            continue
        lo, hi = float(s.quantile(0.01)), float(s.quantile(0.99))
        conservation_clip[c] = (lo, hi)
        for df in (Xn_train, Xn_val, Xn_test):
            df[c] = df[c].clip(lo, hi)

    # 3c. Missing-indicators (for any numeric col with 5-95% NaN in train)
    miss_rate = Xn_train.isna().mean()
    informative_num = miss_rate[(miss_rate > 0.05) & (miss_rate < 0.95)].index.tolist()
    for c in informative_num:
        name = f"{c}_missing"
        for df in (Xn_train, Xn_val, Xn_test):
            df[name] = df[c].isna().astype(int)

    # 3d. Categorical missing-indicators (added to the numeric block)
    cat_informative: list = []
    if categorical_cols:
        cat_miss = X_train[categorical_cols].isna().mean()
        cat_informative = cat_miss[(cat_miss > 0.05) & (cat_miss < 0.95)].index.tolist()
        for c in cat_informative:
            name = f"{c}_missing"
            for src, dst in [(X_train, Xn_train), (X_val, Xn_val), (X_test, Xn_test)]:
                dst[name] = src[c].isna().astype(int).values

    # 3e. Impute remaining NaN with training median.  For population columns
    # which are now rank-percentiles in [0,1], median NaN-fill = 0.5 (neutral).
    medians = Xn_train.median()
    Xn_train = Xn_train.fillna(medians)
    Xn_val   = Xn_val.fillna(medians)
    Xn_test  = Xn_test.fillna(medians)

    # ---- 4. Drop low-variance after transform ----
    low_var_after = Xn_train.var()[lambda v: v < 1e-6].index.tolist()
    Xn_train.drop(columns=low_var_after, inplace=True, errors="ignore")
    Xn_val.drop(columns=low_var_after, inplace=True, errors="ignore")
    Xn_test.drop(columns=low_var_after, inplace=True, errors="ignore")
    info["low_var_after_transform"] = low_var_after

    # ---- 5. Pairwise-correlation drop, NaN-aware (using pre-imputation data) ----
    # We compute correlation on the RAW (pre-imputation) X_train slice to avoid
    # the artefact where two unrelated cols become identical after median-fill.
    # `df.corr(min_periods=...)` returns NaN for pairs with insufficient overlap.
    raw = X_train[[c for c in Xn_train.columns if c in X_train.columns]]
    if raw.shape[1] > 50:
        corr = raw.corr(min_periods=30).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        dropped_corr = [c for c in upper.columns
                        if (upper[c] > corr_threshold).any()]
        Xn_train.drop(columns=dropped_corr, inplace=True, errors="ignore")
        Xn_val.drop(columns=dropped_corr, inplace=True, errors="ignore")
        Xn_test.drop(columns=dropped_corr, inplace=True, errors="ignore")
    else:
        dropped_corr = []
    info["high_corr_removed"] = dropped_corr

    # ---- 6. Reassemble numeric + categorical ----
    def _reassemble(num_df, src_df):
        if not categorical_cols:
            return num_df
        return pd.concat([num_df, src_df[categorical_cols]], axis=1)

    out_train = _reassemble(Xn_train, X_train)
    out_val   = _reassemble(Xn_val,   X_val)
    out_test  = _reassemble(Xn_test,  X_test)

    info["final_features"] = len(out_train.columns)
    info["n_numeric"]      = len(Xn_train.columns)
    info["n_categorical"]  = len(categorical_cols)

    # ---- Fitted params for inference replay (predict.py) ----
    info["fitted"] = {
        "column_categories":      col_categories,
        "drop_constants":         drop_constants,
        "pop_quantile_grids":     pop_quantile_grids,
        "conservation_clip":      conservation_clip,
        "miss_indicator_cols":    informative_num,
        "cat_miss_indicator_cols": cat_informative,
        "medians":                medians,
        "low_var_after_transform": low_var_after,
        "high_corr_cols":         dropped_corr,
        "numeric_cols_kept":      list(Xn_train.columns),
        "categorical_cols":       list(categorical_cols),
    }
    return out_train, out_val, out_test, info


# ---------------------------------------------------------------------------
# Feature selection — mRMR
# ---------------------------------------------------------------------------

def _encode_for_mi(X):
    """Numeric matrix for MI / correlation computation.

    Categorical columns become integer codes (-1 for NaN). Numeric columns
    pass through. Returned frame has the same column order as X.
    """
    out = X.copy()
    discrete_mask = []
    for col in out.columns:
        if isinstance(out[col].dtype, pd.CategoricalDtype):
            out[col] = out[col].cat.codes  # -1 for NaN
            discrete_mask.append(True)
        else:
            discrete_mask.append(False)
    return out, discrete_mask


def select_features_mrmr(X_train, y_train, X_val, X_test, max_features=0):
    """
    Minimum Redundancy Maximum Relevance feature selection.

    Greedy algorithm:
        1. Pick the feature with highest MI(feature, target)
        2. Iteratively add the feature maximising
           MI(feature, target) - mean(|corr(feature, already_selected)|)

    The number of features adapts to dataset size:
        n < 200 -> 20,  200-500 -> 40,  500+ -> 60

    Categorical columns are temporarily encoded as integer codes for both the
    MI computation (with discrete_features set correctly) and the redundancy
    correlation. The returned frames preserve original dtypes (pd.Categorical
    stays Categorical so LightGBM consumes it natively).
    """
    n = len(X_train)
    if max_features <= 0:
        if n < 200:
            max_features = min(20, len(X_train.columns))
        elif n < 500:
            max_features = min(40, len(X_train.columns))
        else:
            max_features = min(60, len(X_train.columns))

    n_before = len(X_train.columns)

    if n_before <= max_features:
        return X_train, X_val, X_test, {
            "method": "none_needed", "n_before": n_before, "n_after": n_before,
        }

    # Encoded copy for MI + correlation; preserves column order
    X_enc, discrete_mask = _encode_for_mi(X_train)
    mi = mutual_info_classif(
        X_enc, y_train,
        discrete_features=discrete_mask,
        random_state=SEED,
        n_neighbors=max(3, min(5, n // 30)),
    )
    mi_map = dict(zip(X_train.columns, mi))

    selected = []
    candidates = list(X_train.columns)

    best = max(candidates, key=lambda c: mi_map[c])
    selected.append(best)
    candidates.remove(best)

    while len(selected) < max_features and candidates:
        scores = {}
        for c in candidates:
            relevance = mi_map[c]
            redundancy = X_enc[selected].corrwith(X_enc[c]).abs().mean()
            scores[c] = relevance - redundancy
        best = max(scores, key=scores.get)
        selected.append(best)
        candidates.remove(best)

    info = {
        "method": "mRMR",
        "n_before": n_before,
        "n_after": len(selected),
        "top_5": {c: round(mi_map[c], 4) for c in selected[:5]},
    }
    return X_train[selected], X_val[selected], X_test[selected], info


# ---------------------------------------------------------------------------
# Hyperparameters (size-adaptive)
# ---------------------------------------------------------------------------

def get_model_params(n_train, scale_pos_weight):
    """
    Return LightGBM hyperparameters adapted to training-set size.

    Three tiers:
        small  (n < 200) : conservative — fewer trees, more regularisation
        medium (200-400)  : balanced
        large  (400+)     : deeper trees, less regularisation
    """
    common = {
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "binary_logloss",
        "scale_pos_weight": scale_pos_weight,
        "verbose": -1,
    }

    if n_train < 200:
        return {**common,
            "n_estimators": 500, "learning_rate": 0.02,
            "num_leaves": 24, "max_depth": 5, "min_child_samples": 10,
            "subsample": 0.7, "colsample_bytree": 0.7,
            "reg_alpha": 0.3, "reg_lambda": 1.5,
        }
    elif n_train < 400:
        return {**common,
            "n_estimators": 800, "learning_rate": 0.03,
            "num_leaves": 31, "max_depth": 6, "min_child_samples": 10,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_alpha": 0.3, "reg_lambda": 1.5,
        }
    else:
        return {**common,
            "n_estimators": 2000, "learning_rate": 0.02,
            "num_leaves": 31, "max_depth": 6, "min_child_samples": 15,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_alpha": 0.3, "reg_lambda": 1.5,
        }


# ---------------------------------------------------------------------------
# Multi-seed ensemble
# ---------------------------------------------------------------------------

def train_multi_seed_ensemble(X_train_val, y_train_val, X_test,
                               params_base, n_seeds=5, n_folds=5):
    """
    Train *n_seeds* LightGBM models with different random seeds and average
    their predicted probabilities to reduce variance.

    For each seed:
        - Generate out-of-fold (OOF) predictions via K-fold CV
        - Train a final model on all training data (with 10 % holdout for ES)
        - Predict on X_test

    Returns:
        models       : list of final trained models
        avg_oof      : averaged OOF probabilities
        avg_test     : averaged test probabilities
        oof_std      : mean std of OOF predictions across seeds
    """
    models = []
    oof_all = np.zeros((n_seeds, len(y_train_val)))
    test_all = np.zeros((n_seeds, len(X_test)))
    patience = 50 if len(y_train_val) > 300 else 30

    for i in range(n_seeds):
        seed_i = SEED + i
        params = {**params_base, "random_state": seed_i}

        # OOF predictions
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed_i)
        oof = np.zeros(len(y_train_val))
        for tr_idx, va_idx in skf.split(X_train_val, y_train_val):
            # Nest the early-stopping split INSIDE the training fold so the OOF
            # fold (va_idx) stays fully held out. Using va_idx itself as the ES
            # eval_set tunes the tree count on the very rows we then predict —
            # mild optimism that inflates the OOF feeding Platt + threshold.
            # (Matches the nested protocol already used by the blend's CatBoost
            # OOF in general_blend.py.)
            in_tr, in_es = train_test_split(
                tr_idx, test_size=0.15, random_state=seed_i,
                stratify=y_train_val.iloc[tr_idx])
            m = lgb.LGBMClassifier(**params)
            m.fit(X_train_val.iloc[in_tr], y_train_val.iloc[in_tr],
                  eval_set=[(X_train_val.iloc[in_es], y_train_val.iloc[in_es])],
                  callbacks=[lgb.early_stopping(patience, verbose=False),
                             lgb.log_evaluation(0)])
            oof[va_idx] = m.predict_proba(X_train_val.iloc[va_idx])[:, 1]
        oof_all[i] = oof

        # Final model (small holdout for early stopping only)
        X_tr, X_es, y_tr, y_es = train_test_split(
            X_train_val, y_train_val, test_size=0.1,
            random_state=seed_i, stratify=y_train_val)
        final = lgb.LGBMClassifier(**params)
        final.fit(X_tr, y_tr,
                  eval_set=[(X_es, y_es)],
                  callbacks=[lgb.early_stopping(patience, verbose=False),
                             lgb.log_evaluation(0)])
        models.append(final)
        test_all[i] = final.predict_proba(X_test)[:, 1]

    return models, oof_all.mean(0), test_all.mean(0), oof_all.std(0).mean()
