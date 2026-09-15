"""Research-only ES-row reclamation comparison; never writes models/ or pipelines/.

A is the current early-stopped fit.  B selects A's count on exactly A's ES
partition, then fits that count on all eligible gradient rows. RF/XGB are held
identical. Target-side learned state is always fit inside its train partition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import catboost
from catboost import CatBoostClassifier
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (auc, average_precision_score, precision_recall_curve,
                             roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "pipelines")]

from calibration import apply_platt, fit_platt
from config import SEED
from feature_engineering import engineer_aa_features
from general_blend import (_cat_general, _cat_model, _cat_repr, _cats, _rf_general,
                           _rf_repr, _xgb_model)
from harness import f1_at_prior, mcc_at_prior_analytic, tpr_fpr
from models import get_model_params
from predict import (LGBM_PARAM_OVERRIDE, REDUN_PRUNE_LGBM, _apply_own_preprocess,
                     _engineer, _fit_own_preprocess, _lgbm_redun_keep)
from preprocess_v2 import apply_al, fit_al, fit_keep_nan
from recipe import PANEL_RECIPE
from schema import drop_all_nan_rows, normalize_sentinels, partition_columns, to_categorical
from transfer import (TRANSFER_AL_FEATURES, TRANSFER_BLEND, TRANSFER_BLEND_WEIGHTS,
                      TRANSFER_TARGETRANK, fit_target_pop_grids, project_target,
                      transfer_score_matrix)
from utils import (KNOWN_LEAKY, PANELS, TEST_PRIORS, detect_leaky_columns,
                   optimize_threshold_f1_balanced)

OUT = ROOT / "analysis" / "output" / "final_refit"
PANEL_ORDER = ("general", "cancer", "pah", "cftr")
FROZEN_THRESHOLD = {"general": 0.81, "cancer": 0.80}


def sha(values):
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()[:16]


def file_hashes():
    paths = list((ROOT / "models").rglob("*")) + list((ROOT / "pipelines").rglob("*"))
    paths.append(ROOT / "analysis" / "output" / "PDR_CANONICAL_RESULTS.csv")
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths) if p.is_file()}


def target_prior_scores(y, score, prior):
    """Target-prior AP and PR-AUC, delegated to sklearn so ties are grouped correctly."""
    y = np.asarray(y, dtype=int); score = np.asarray(score, dtype=float)
    n_pos, n_neg = max(int(y.sum()), 1), max(int((y == 0).sum()), 1)
    weights = np.where(y == 1, prior / n_pos, (1 - prior) / n_neg)
    ap = float(average_precision_score(y, score, sample_weight=weights))
    precision, recall, _ = precision_recall_curve(y, score, sample_weight=weights)
    return ap, float(auc(recall[::-1], precision[::-1]))


def metrics_at(y, probs, threshold, prior):
    pred = np.asarray(probs >= threshold, dtype=int)
    tpr, fpr = tpr_fpr(y, pred)
    f1, mcc = f1_at_prior(tpr, fpr, prior), mcc_at_prior_analytic(tpr, fpr, prior)
    ap, pr_auc = target_prior_scores(y, probs, prior)
    return {"f1": f1, "mcc": mcc, "composite": (f1 + mcc) / 2,
            "auc": roc_auc_score(y, probs), "ap_target_prior": ap,
            "pr_auc_trapezoid_target_prior": pr_auc}


def composite_threshold(y, probs, prior):
    choices = np.arange(0.02, 0.98, 0.005)
    return float(max(choices, key=lambda t: metrics_at(y, probs, t, prior)["composite"]))


def remove_unused_categories(frame):
    frame = frame.copy()
    for col in frame.columns:
        if isinstance(frame[col].dtype, pd.CategoricalDtype):
            frame[col] = frame[col].cat.remove_unused_categories()
    return frame


def prepare_own(X_train, y_train, X_apply, cfg):
    """Fold-local leak screen, vocabularies, preprocessing, score agreement and mRMR."""
    X_train = remove_unused_categories(X_train)
    numeric = [c for c in X_train if not isinstance(X_train[c].dtype, pd.CategoricalDtype)]
    leaky = sorted(set(detect_leaky_columns(X_train[numeric], y_train, 0.99, verbose=False)) |
                   (KNOWN_LEAKY & set(X_train.columns)))
    fitted, art = _fit_own_preprocess(X_train.drop(columns=leaky, errors="ignore"), y_train, cfg)
    applied = _apply_own_preprocess(X_apply.drop(columns=leaky, errors="ignore"), art)
    return fitted, applied.reindex(columns=fitted.columns)


def make_lgb(params, model_seed):
    return lgb.LGBMClassifier(**{**params, "random_state": model_seed, "n_jobs": 4})


def fit_lgb_pair(X, y, params, *, model_seed, split_seed, holdout, patience, records, tag):
    """One ES selection produces both arms; records observed fitted tree counts/timings."""
    idx = np.arange(len(y))
    gradient_idx, es_idx = train_test_split(idx, test_size=holdout, random_state=split_seed, stratify=y)
    started = time.time()
    current = make_lgb(params, model_seed)
    current.fit(X.iloc[gradient_idx], y.iloc[gradient_idx],
                eval_set=[(X.iloc[es_idx], y.iloc[es_idx])],
                callbacks=[lgb.early_stopping(patience, verbose=False), lgb.log_evaluation(0)])
    count = max(int(current.best_iteration_ or params["n_estimators"]), 1)
    observed = int(current.booster_.num_trees())
    records.append({"tag": tag, "kind": "lgb", "arm": "A", "eligible_rows": len(y),
                    "gradient_rows": len(gradient_idx), "es_rows": len(es_idx), "tree_count": count,
                    "observed_trees": observed, "model_seed": model_seed, "split_seed": split_seed,
                    "holdout": holdout, "seconds": time.time() - started})
    started = time.time()
    refit = make_lgb({**params, "n_estimators": count}, model_seed)
    refit.fit(X, y)  # intentionally no eval_set/callback: B's exact no-ES semantics
    observed = int(refit.booster_.num_trees())
    if observed != count:
        raise AssertionError(f"B LGB tree mismatch {observed} != {count}")
    records.append({"tag": tag, "kind": "lgb", "arm": "B", "eligible_rows": len(y),
                    "gradient_rows": len(y), "es_rows": 0, "tree_count": count,
                    "observed_trees": observed, "model_seed": model_seed, "split_seed": split_seed,
                    "holdout": holdout, "seconds": time.time() - started})
    return current, refit


def fit_cat_pair(X, y, prototype, *, split_seed, records, tag):
    """CatBoost keeps its production model seed; only its ES partition varies."""
    def fresh(**overrides):
        params = prototype.get_params(); params.update(overrides)
        return CatBoostClassifier(**params)
    idx = np.arange(len(y))
    gradient_idx, es_idx = train_test_split(idx, test_size=0.15, random_state=split_seed, stratify=y)
    started = time.time()
    current = fresh()
    current.fit(X.iloc[gradient_idx], y.iloc[gradient_idx], eval_set=(X.iloc[es_idx], y.iloc[es_idx]))
    count = max(int(current.get_best_iteration()) + 1, 1)
    observed = int(current.tree_count_)
    records.append({"tag": tag, "kind": "cat", "arm": "A", "eligible_rows": len(y),
                    "gradient_rows": len(gradient_idx), "es_rows": len(es_idx), "tree_count": count,
                    "observed_trees": observed, "model_seed": SEED, "split_seed": split_seed,
                    "holdout": 0.15, "seconds": time.time() - started})
    started = time.time()
    refit = fresh(iterations=count, early_stopping_rounds=None)
    refit.fit(X, y)  # no eval_set; model's fixed random_seed remains SEED
    observed = int(refit.tree_count_)
    if observed != count:
        raise AssertionError(f"B Cat tree mismatch {observed} != {count}")
    records.append({"tag": tag, "kind": "cat", "arm": "B", "eligible_rows": len(y),
                    "gradient_rows": len(y), "es_rows": 0, "tree_count": count,
                    "observed_trees": observed, "model_seed": SEED, "split_seed": split_seed,
                    "holdout": 0.15, "seconds": time.time() - started})
    return current, refit


def own_lgb_pair(Xraw, y, Xtest, cfg, panel, records, tag):
    """Five-seed deployed LGBM with strict inner locality. Model tier/SPW stay outer-local."""
    n_folds = 3 if len(y) < 200 else 5
    outer_spw = (y == 0).sum() / max((y == 1).sum(), 1)
    outer_params = get_model_params(len(y), outer_spw)
    if panel in LGBM_PARAM_OVERRIDE:
        outer_params = {**outer_params, **LGBM_PARAM_OVERRIDE[panel]}
    outer_patience = 50 if len(y) > 300 else 30
    oof = {arm: np.zeros(len(y)) for arm in ("A", "B")}
    tests = {arm: [] for arm in ("A", "B")}
    final_X, final_test = prepare_own(Xraw, y, Xtest, cfg)  # shared safely across seeds
    final_keep = _lgbm_redun_keep(final_X) if panel in REDUN_PRUNE_LGBM else list(final_X.columns)
    for seed_offset in range(5):
        model_seed = SEED + seed_offset
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=model_seed)
        for fold, (a, b) in enumerate(skf.split(Xraw, y)):
            inner_X, inner_val = prepare_own(Xraw.iloc[a].reset_index(drop=True), y.iloc[a].reset_index(drop=True),
                                              Xraw.iloc[b].reset_index(drop=True), cfg)
            keep = _lgbm_redun_keep(inner_X) if panel in REDUN_PRUNE_LGBM else list(inner_X.columns)
            current, refit = fit_lgb_pair(inner_X[keep], y.iloc[a].reset_index(drop=True), outer_params,
                                          model_seed=model_seed, split_seed=model_seed, holdout=0.15,
                                          patience=outer_patience, records=records,
                                          tag=f"{tag}:lgb_oof:s{seed_offset}:f{fold}")
            oof["A"][b] += current.predict_proba(inner_val[keep])[:, 1] / 5
            oof["B"][b] += refit.predict_proba(inner_val[keep])[:, 1] / 5
        current, refit = fit_lgb_pair(final_X[final_keep], y, outer_params,
                                      model_seed=model_seed, split_seed=model_seed, holdout=0.10,
                                      patience=outer_patience, records=records,
                                      tag=f"{tag}:lgb_final:s{seed_offset}")
        tests["A"].append(current.predict_proba(final_test[final_keep])[:, 1])
        tests["B"].append(refit.predict_proba(final_test[final_keep])[:, 1])
    return oof, {arm: np.mean(tests[arm], axis=0) for arm in tests}


def general_own_pair(Xraw, y, Xtest, records, tag):
    cfg = PANEL_RECIPE["general"]
    lgb_oof, lgb_test = own_lgb_pair(Xraw, y, Xtest, cfg, "general", records, tag)
    raw = {key: np.zeros(len(y)) for key in ("rf", "xgb", "catA", "catB")}
    outer_spw = (y == 0).sum() / max((y == 1).sum(), 1)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (a, b) in enumerate(skf.split(Xraw, y)):
        inner_X, inner_val = prepare_own(Xraw.iloc[a].reset_index(drop=True), y.iloc[a].reset_index(drop=True),
                                          Xraw.iloc[b].reset_index(drop=True), cfg)
        cats = _cats(inner_X); xord, enc = _rf_repr(inner_X, cats, fit=True); vord, _ = _rf_repr(inner_val, cats, enc=enc)
        xcat, vcat = _cat_repr(inner_X, cats), _cat_repr(inner_val, cats)
        raw["rf"][b] = _rf_general().fit(xord, y.iloc[a]).predict_proba(vord)[:, 1]
        raw["xgb"][b] = _xgb_model().fit(xord, y.iloc[a]).predict_proba(vord)[:, 1]
        cat_idx = [xcat.columns.get_loc(c) for c in cats]
        current, refit = fit_cat_pair(xcat, y.iloc[a].reset_index(drop=True), _cat_general(cat_idx, outer_spw),
                                      split_seed=SEED + fold, records=records, tag=f"{tag}:cat_oof:f{fold}")
        raw["catA"][b], raw["catB"][b] = current.predict_proba(vcat)[:, 1], refit.predict_proba(vcat)[:, 1]
    final_X, final_test = prepare_own(Xraw, y, Xtest, cfg)
    cats = _cats(final_X); xord, enc = _rf_repr(final_X, cats, fit=True); tord, _ = _rf_repr(final_test, cats, enc=enc)
    xcat, tcat = _cat_repr(final_X, cats), _cat_repr(final_test, cats)
    rf_test = _rf_general().fit(xord, y).predict_proba(tord)[:, 1]
    xgb_test = _xgb_model().fit(xord, y).predict_proba(tord)[:, 1]
    cat_idx = [xcat.columns.get_loc(c) for c in cats]
    current, refit = fit_cat_pair(xcat, y, _cat_general(cat_idx, outer_spw), split_seed=SEED + 99,
                                  records=records, tag=f"{tag}:cat_final")
    result = {}
    for arm, cat_key, cat_test in (("A", "catA", current.predict_proba(tcat)[:, 1]),
                                   ("B", "catB", refit.predict_proba(tcat)[:, 1])):
        calibrators = [fit_platt(lgb_oof[arm], y), fit_platt(raw["rf"], y),
                       fit_platt(raw[cat_key], y), fit_platt(raw["xgb"], y)]
        train = np.mean([apply_platt(lgb_oof[arm], calibrators[0]), apply_platt(raw["rf"], calibrators[1]),
                         apply_platt(raw[cat_key], calibrators[2]), apply_platt(raw["xgb"], calibrators[3])], axis=0)
        test = np.mean([apply_platt(lgb_test[arm], calibrators[0]), apply_platt(rf_test, calibrators[1]),
                        apply_platt(cat_test, calibrators[2]), apply_platt(xgb_test, calibrators[3])], axis=0)
        result[arm] = (train, test)
    return result


def cancer_own_pair(Xraw, y, Xtest, records, tag):
    oof, test = own_lgb_pair(Xraw, y, Xtest, PANEL_RECIPE["cancer"], "cancer", records, tag)
    return {arm: (apply_platt(oof[arm], fit_platt(oof[arm], y)),
                  apply_platt(test[arm], fit_platt(oof[arm], y))) for arm in ("A", "B")}


def load_raw(panel):
    df = drop_all_nan_rows(normalize_sentinels(pd.read_csv(PANELS[panel])), verbose=False).reset_index(drop=True)
    if df["Variant_ID"].duplicated().any():
        raise AssertionError(f"{panel} has duplicate Variant_ID")
    return df


def fit_source_pair(panel, records):
    """Fit source once per panel/arm policy; source is independent of all target rows."""
    target = load_raw(panel); target_ids = set(target["Variant_ID"])
    all_source = load_raw("general")
    source = all_source.loc[~all_source["Variant_ID"].isin(target_ids)].reset_index(drop=True)
    if set(source["Variant_ID"]) & target_ids:
        raise AssertionError("target IDs leaked into transfer source")
    source = engineer_aa_features(to_categorical(source))
    _, nums, cats, label = partition_columns(source)
    y = source[label].reset_index(drop=True).astype(int); X = source[nums + cats].reset_index(drop=True)
    leaky = sorted(set(detect_leaky_columns(X, y, 0.99, verbose=False)) | (KNOWN_LEAKY & set(X.columns)))
    X = X.drop(columns=leaky, errors="ignore"); raw_cols = list(X.columns)
    drop_al = panel == "cancer"; al_features = TRANSFER_AL_FEATURES.get(panel, ())
    alf = fit_al(X, al_features) if al_features else None
    model_X = X.drop(columns=[c for c in X if str(c).startswith("AL_")], errors="ignore") if drop_al else X
    Xpp, ppf = fit_keep_nan(model_X)
    if alf is not None:
        Xpp = pd.concat([Xpp.reset_index(drop=True), apply_al(X, alf).reset_index(drop=True)], axis=1)
        Xpp = Xpp.loc[:, ~Xpp.columns.duplicated()]
    feats = list(Xpp.columns); cat_feats = [c for c in feats if isinstance(Xpp[c].dtype, pd.CategoricalDtype)]
    cat_cats = {c: list(Xpp[c].cat.categories) for c in cat_feats}
    spw = (y == 0).sum() / max((y == 1).sum(), 1); params = get_model_params(len(y), spw)
    current_lgb, refit_lgb = fit_lgb_pair(Xpp, y, params, model_seed=SEED, split_seed=SEED + 7,
                                          holdout=0.15, patience=40, records=records,
                                          tag=f"source:{panel}:lgb_final")
    blend_models = None
    if panel in TRANSFER_BLEND:
        xord, enc = _rf_repr(Xpp, cat_feats, fit=True)
        rf = RandomForestClassifier(n_estimators=600, max_features="sqrt", n_jobs=4,
                                    class_weight="balanced", random_state=SEED).fit(xord, y)
        xcat = _cat_repr(Xpp, cat_feats); cat_idx = [xcat.columns.get_loc(c) for c in cat_feats]
        current_cat, refit_cat = fit_cat_pair(xcat, y, _cat_model(cat_idx, spw), split_seed=SEED + 5,
                                               records=records, tag=f"source:{panel}:cat_final")
        blend_models = (rf, enc, current_cat, refit_cat)
    result = {}
    for arm, lgb_model in (("A", current_lgb), ("B", refit_lgb)):
        blend = None if blend_models is None else {"rf_model": blend_models[0], "encoder": blend_models[1],
                 "cat_model": blend_models[2] if arm == "A" else blend_models[3], "cat_feats": cat_feats}
        result[arm] = {"model": lgb_model, "models": [lgb_model], "raw_cols": raw_cols, "ppf": ppf,
                       "alf": alf, "feats": feats, "cat_cats": cat_cats, "blend": blend,
                       "blend_weights": TRANSFER_BLEND_WEIGHTS.get(panel, (1/3, 1/3, 1/3)),
                       "n_train": len(y), "source_rows_before": len(all_source),
                       "excluded_target_rows": len(all_source) - len(source)}
    return result


def evaluate_panel(panel, splits, sources, records, arrays):
    raw = load_raw(panel); engineered = _engineer(raw.copy(), PANEL_RECIPE[panel])
    _, nums, cats, label = partition_columns(engineered)
    X, y = engineered[nums + cats].reset_index(drop=True), engineered[label].reset_index(drop=True).astype(int)
    prior, rows = TEST_PRIORS[panel], []
    for repeat, fold, tr, te in splits:
        tr, te = np.asarray(tr), np.asarray(te)
        if set(tr) & set(te): raise AssertionError("outer train/test overlap")
        tag = f"{panel}:r{repeat}:f{fold}"
        if panel == "general":
            probs = general_own_pair(X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True),
                                     X.iloc[te].reset_index(drop=True), records, tag)
        else:
            # PAH grids use all outer-train feature values: train-only but transductive within the
            # calibration sample. This is recorded as an inner-rank nuance, never fitted on outer test.
            grids = ({arm: fit_target_pop_grids(raw.iloc[tr].reset_index(drop=True), sources[panel][arm])
                      for arm in ("A", "B")} if panel in TRANSFER_TARGETRANK else None)
            transfer = {}
            for arm in ("A", "B"):
                art = sources[panel][arm]
                gtr = transfer_score_matrix(project_target(raw.iloc[tr].reset_index(drop=True), art,
                                                           None if grids is None else grids[arm]), art)
                gte = transfer_score_matrix(project_target(raw.iloc[te].reset_index(drop=True), art,
                                                           None if grids is None else grids[arm]), art)
                platt = fit_platt(gtr, y.iloc[tr]); transfer[arm] = (apply_platt(gtr, platt), apply_platt(gte, platt))
            if panel == "cancer":
                own = cancer_own_pair(X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True),
                                      X.iloc[te].reset_index(drop=True), records, tag)
                probs = {arm: (0.5 * transfer[arm][0] + 0.5 * own[arm][0],
                               0.5 * transfer[arm][1] + 0.5 * own[arm][1]) for arm in ("A", "B")}
            else:
                probs = transfer
        for arm, (ptr, pte) in probs.items():
            arrays[f"{tag}_{arm}_train_indices"] = tr; arrays[f"{tag}_{arm}_test_indices"] = te
            arrays[f"{tag}_{arm}_train_y"] = y.iloc[tr].values; arrays[f"{tag}_{arm}_test_y"] = y.iloc[te].values
            arrays[f"{tag}_{arm}_train_probs"] = ptr; arrays[f"{tag}_{arm}_test_probs"] = pte
            choices = {"current_rule": FROZEN_THRESHOLD.get(panel),
                       "train_composite": composite_threshold(y.iloc[tr], ptr, prior)}
            if choices["current_rule"] is None:
                choices["current_rule"] = optimize_threshold_f1_balanced(y.iloc[tr], ptr, target_prior=prior,
                                                                             n_bootstrap=300)[0]
            for rule, threshold in choices.items():
                row = {"panel": panel, "repeat": repeat, "fold": fold, "arm": arm, "rule": rule,
                       "threshold": threshold, "n_train": len(tr), "n_test": len(te),
                       "train_split_sha256": sha(tr), "test_split_sha256": sha(te)}
                row.update(metrics_at(y.iloc[te], pte, threshold, prior)); rows.append(row)
        # Survive an interrupted bounded run with raw probabilities, exact indices and tree records.
        pd.DataFrame(rows).to_csv(OUT / f"checkpoint_{panel}_fold_metrics.csv", index=False)
        pd.DataFrame(records).to_csv(OUT / "tree_counts.csv", index=False)
        np.savez_compressed(OUT / f"checkpoint_{panel}_probabilities_and_splits.npz", **arrays)
    return rows


def corrected_summary(metrics, requested_panels):
    frame = pd.DataFrame(metrics); out = []
    for rule in ("current_rule", "train_composite"):
        sub = frame[frame.rule == rule]
        names = list(requested_panels)
        if set(PANEL_ORDER).issubset(requested_panels): names.append("equal_four_panel_mean")
        for panel in names:
            if panel == "equal_four_panel_mean":
                a = sub[sub.arm == "A"].groupby(["repeat", "fold"])["composite"].mean()
                b = sub[sub.arm == "B"].groupby(["repeat", "fold"])["composite"].mean()
                ratio = 1 / (max(len(a), 1) // max(sub.repeat.nunique(), 1) - 1)
            else:
                a = sub[(sub.panel == panel) & (sub.arm == "A")].set_index(["repeat", "fold"])["composite"]
                b = sub[(sub.panel == panel) & (sub.arm == "B")].set_index(["repeat", "fold"])["composite"]
                r = sub[(sub.panel == panel) & (sub.arm == "A")].iloc[0]
                ratio = float(r.n_test / r.n_train)
            d = (b - a).sort_index(); by_repeat = d.groupby(level=0).mean()
            se = float(np.sqrt((1 / len(d) + ratio) * d.var(ddof=1))) if len(d) > 1 else 0.0
            out.append({"rule": rule, "panel": panel, "n_paired_folds": len(d), "A_composite": float(a.mean()),
                        "B_composite": float(b.mean()), "delta_composite": float(d.mean()),
                        "corrected_repeated_cv_se": se, "corrected_repeated_cv_sigma": float(d.mean()/se) if se else 0.0,
                        "repeat_delta_min": float(by_repeat.min()), "repeat_delta_max": float(by_repeat.max()),
                        "paired_delta_min": float(d.min()), "paired_delta_max": float(d.max())})
    return pd.DataFrame(out)


def self_test():
    import tempfile
    OUT = Path(tempfile.mkdtemp(prefix="final_refit_selftest_"))   # scratch, never analysis/output
    rng = np.random.RandomState(9)
    X = pd.DataFrame(rng.normal(size=(48, 3))); y = pd.Series([0] * 24 + [1] * 24); records = []
    current, refit = fit_lgb_pair(X, y, {**get_model_params(len(y), 1.0), "n_estimators": 20},
                                   model_seed=11, split_seed=13, holdout=.10, patience=5, records=records, tag="self")
    assert records[-1]["gradient_rows"] == len(y) and records[-1]["es_rows"] == 0
    path = OUT / "selftest_refit.joblib"; joblib.dump(refit, path)
    assert np.array_equal(refit.predict_proba(X), joblib.load(path).predict_proba(X))
    cat = _cat_general([], 1.0).set_params(iterations=20, depth=3)
    ca, cb = fit_cat_pair(X, y, cat, split_seed=17, records=records, tag="self_cat")
    path = OUT / "selftest_cat_refit.joblib"; joblib.dump(cb, path)
    assert cb.tree_count_ == [r for r in records if r["tag"] == "self_cat" and r["arm"] == "B"][0]["tree_count"]
    assert np.array_equal(cb.predict_proba(X), joblib.load(path).predict_proba(X))
    tied_y, tied_p, prior = np.array([1, 0, 1, 0]), np.array([.8, .8, .2, .2]), .2
    ap, trap = target_prior_scores(tied_y, tied_p, prior); w = np.where(tied_y == 1, prior/2, (1-prior)/2)
    assert np.isclose(ap, average_precision_score(tied_y, tied_p, sample_weight=w)) and trap >= ap
    tr, te = next(RepeatedStratifiedKFold(n_splits=3, random_state=SEED).split(X, y)); assert not (set(tr) & set(te))
    _, before = prepare_own(X.iloc[tr], y.iloc[tr], X.iloc[te], PANEL_RECIPE["general"])
    # Outer test labels are deliberately absent from prepare_own's signature; perturbing them cannot alter state.
    _perturbed_outer_labels = 1 - y.iloc[te]
    _, after = prepare_own(X.iloc[tr], y.iloc[tr], X.iloc[te], PANEL_RECIPE["general"])
    assert before.equals(after) and len(_perturbed_outer_labels) == len(te)
    print("self-test passed: observed B tree counts, no-ES refit, Cat reload, tied weighted AP, split/outer-label isolation")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repeats", type=int, default=2); ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--panels", nargs="*", choices=PANEL_ORDER, default=list(PANEL_ORDER)); ap.add_argument("--max-folds", type=int)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test: self_test(); return
    OUT.mkdir(parents=True, exist_ok=True); before = file_hashes(); started = time.time(); records = []; arrays = {}; metrics = []
    census = []
    general_ids = set(load_raw("general")["Variant_ID"])
    for panel in args.panels:
        raw_census = load_raw(panel)
        census.append({"panel": panel, "usable_rows": len(raw_census), "pathogenic": int(raw_census.Label.sum()),
                       "benign": int((raw_census.Label == 0).sum()),
                       "target_ids_overlapping_general": len(set(raw_census.Variant_ID) & general_ids) if panel != "general" else 0})
    pd.DataFrame(census).to_csv(OUT / "census.csv", index=False)
    sources = {panel: fit_source_pair(panel, records) for panel in args.panels if panel != "general"}
    pd.DataFrame([{"target_panel": panel, "source_rows_before": art["A"]["source_rows_before"],
                   "source_rows_after": art["A"]["n_train"], "excluded_target_rows": art["A"]["excluded_target_rows"]}
                  for panel, art in sources.items()]).to_csv(OUT / "source_census.csv", index=False)
    for panel in args.panels:
        raw = load_raw(panel); splitter = RepeatedStratifiedKFold(n_splits=args.splits, n_repeats=args.repeats, random_state=SEED)
        splits = [(i // args.splits, i % args.splits, tr, te) for i, (tr, te) in enumerate(splitter.split(raw, raw.Label))]
        if args.max_folds:
            splits = splits[:args.max_folds]
        print(f"{panel}: {len(raw)} usable rows, {len(splits)} paired outer folds", flush=True)
        metrics.extend(evaluate_panel(panel, splits, sources, records, arrays))
    after = file_hashes()
    if before != after: raise AssertionError("production model/pipeline/canonical hash changed")
    pd.DataFrame(metrics).to_csv(OUT / "fold_metrics.csv", index=False); pd.DataFrame(records).to_csv(OUT / "tree_counts.csv", index=False)
    summary = corrected_summary(metrics, args.panels); summary.to_csv(OUT / "summary.csv", index=False); np.savez_compressed(OUT / "probabilities_and_splits.npz", **arrays)
    manifest = {"git_head": os.popen("git rev-parse HEAD").read().strip(), "args": vars(args), "seconds": time.time()-started,
                "versions": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "lightgbm": lgb.__version__, "catboost": catboost.__version__, "xgboost": xgboost.__version__},
                "production_hashes_before": before, "production_hashes_after": after, "no_deployment": True,
                "policy_note": "A/B apply the same ES policy to every affected LGB/Cat final and inner-OOF fit; RF/XGB are unchanged. Thus calibration itself is compared under the uniform policy, not held fixed.",
                "target_rank_note": "PAH rank grids are outer-train-only; calibration scores share that train grid (transductive within train, never outer test)."}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("\n" + summary.to_string(index=False)); print(f"\nSaved research-only results to {OUT} in {manifest['seconds']:.1f}s; models/ unchanged.")


if __name__ == "__main__": main()
