"""Bounded, deterministic finalization diagnostics; never writes production artifacts.

Reports row eligibility, duplicate/conflict status, split-representation coverage, and
ES count/prediction sensitivity.  It intentionally does not score any outer holdout or
select a candidate.  Outputs stay under analysis/output/finalization/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "pipelines")]

from config import SEED
from feature_engineering import engineer_aa_features
from general_blend import _cat_general, _cat_model, _cat_repr, _cats
from models import get_model_params
from predict import (LGBM_PARAM_OVERRIDE, REDUN_PRUNE_LGBM, _engineer,
                     _fit_own_preprocess, _lgbm_redun_keep)
from recipe import PANEL_RECIPE
from schema import (drop_all_nan_rows, normalize_sentinels, partition_columns,
                    to_categorical)
from preprocess_v2 import apply_al, fit_al, fit_keep_nan
from transfer import (TRANSFER_AL_FEATURES, TRANSFER_BLEND,
                      TRANSFER_DROP_AL)
from utils import KNOWN_LEAKY, PANELS, detect_leaky_columns

OUT = ROOT / "analysis" / "output" / "finalization"
PANEL_ORDER = ("general", "cancer", "pah", "cftr")
# These are fixed before inspection; the deployed baseline split is included first.
OWN_SEEDS = (42, 43, 44, 45, 46)
SOURCE_LGB_SPLITS = (49, 50, 51, 52, 53)  # deployed source LGB split is SEED + 7
SOURCE_CAT_SPLITS = (47, 48, 49, 50, 51)  # deployed source Cat split is SEED + 5
GENERAL_CAT_SPLITS = (141, 142, 143, 144, 145)  # deployed final Cat split is SEED + 99
NULL_SEEDS = tuple(range(10_000, 10_100))
EXPECTED_SOURCE_CENSUS = {"cancer": (2571, 1853, 718), "pah": (2562, 1834, 728), "cftr": (2738, 2001, 737)}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tracked_hashes() -> dict[str, str]:
    paths = [HERE / "exp_finalization_diagnostics.py"]
    paths += [Path(p) for p in PANELS.values()]
    paths += sorted((ROOT / "models").rglob("artifacts.joblib"))
    paths += [ROOT / "pipelines" / name for name in (
        "predict.py", "transfer.py", "models.py", "general_blend.py", "preprocess_v2.py",
        "schema.py", "feature_engineering.py", "recipe.py", "utils.py", "config.py")]
    return {str(p.relative_to(ROOT)): digest(p) for p in paths if p.is_file()}


def load_raw(panel: str, *, usable: bool) -> pd.DataFrame:
    df = normalize_sentinels(pd.read_csv(PANELS[panel]))
    return drop_all_nan_rows(df, verbose=False).reset_index(drop=True) if usable else df.reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("Variant_ID", "Label")]


def family_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    cols = feature_columns(df)
    groups = {
        "AL": [c for c in cols if c.startswith("AL_")],
        "EK": [c for c in cols if c.startswith("EK_")],
        "AA": [c for c in cols if c.startswith("AA_")],
        "CAT": [c for c in cols if c.startswith("CAT_")],
    }
    assigned = set().union(*map(set, groups.values()))
    groups["other"] = [c for c in cols if c not in assigned]
    return {k: v for k, v in groups.items() if v}


def duplicate_summary(df: pd.DataFrame) -> tuple[int, int, int]:
    mask = df.duplicated(subset=feature_columns(df), keep=False)
    dups = df.loc[mask]
    if dups.empty:
        return 0, 0, 0
    # `duplicated` already gave exact pandas equality semantics (including NaN); group
    # on the same columns with dropna=False rather than using an approximate hash key.
    groups = list(dups.groupby(feature_columns(df), dropna=False, sort=False))
    return len(groups), len(dups), sum(g.Label.nunique() > 1 for _, g in groups)


def census_and_duplicates() -> pd.DataFrame:
    rows = []
    usable_by_panel = {}
    for panel in PANEL_ORDER:
        raw = load_raw(panel, usable=False)
        usable = load_raw(panel, usable=True)
        usable_by_panel[panel] = usable
        all_nan = len(raw) - len(usable)
        groups, dup_rows, conflicts = duplicate_summary(raw)
        if usable.Variant_ID.duplicated().any():
            raise AssertionError(f"usable {panel} has duplicate Variant_ID values")
        rows.append({"panel": panel, "raw_rows": len(raw), "raw_pathogenic": int(raw.Label.sum()),
                     "raw_benign": int((raw.Label == 0).sum()), "all_feature_nan": all_nan,
                     "usable_rows": len(usable), "usable_pathogenic": int(usable.Label.sum()),
                     "usable_benign": int((usable.Label == 0).sum()), "duplicate_groups_raw": groups,
                     "duplicate_rows_raw": dup_rows, "conflicted_duplicate_groups_raw": conflicts})
    all_ids = pd.concat([df.assign(panel=panel)[["Variant_ID", "Label", "panel"]]
                         for panel, df in usable_by_panel.items()], ignore_index=True)
    cross = all_ids.groupby("Variant_ID")
    shared = cross.filter(lambda g: len(g) > 1)
    conflicts = int(shared.groupby("Variant_ID").Label.nunique().gt(1).sum()) if not shared.empty else 0
    if conflicts:
        raise AssertionError(f"cross-panel label conflicts: {conflicts}")
    overlaps = {panel: len(set(usable_by_panel[panel].Variant_ID) & set(usable_by_panel["general"].Variant_ID))
                for panel in PANEL_ORDER if panel != "general"}
    frame = pd.DataFrame(rows)
    for panel, n in overlaps.items():
        frame.loc[frame.panel == panel, "ids_overlapping_general"] = n
    frame["ids_overlapping_general"] = frame["ids_overlapping_general"].fillna(0).astype(int)
    pd.DataFrame([{ "shared_variant_ids": int(shared.Variant_ID.nunique()),
                    "cross_panel_label_conflicts": conflicts}]).to_csv(OUT / "cross_panel_id_audit.csv", index=False)
    return frame


def _safe_z(observed: float, values: list[float]) -> float:
    sd = float(np.std(values, ddof=1))
    return 0.0 if sd == 0 else (observed - float(np.mean(values))) / sd


def source_eligible_frame(target_panel: str) -> pd.DataFrame:
    target_ids = set(load_raw(target_panel, usable=True).Variant_ID)
    source = load_raw("general", usable=True)
    source = source.loc[~source.Variant_ID.isin(target_ids)].reset_index(drop=True)
    observed = (len(source), int(source.Label.sum()), int((source.Label == 0).sum()))
    if observed != EXPECTED_SOURCE_CENSUS[target_panel] or set(source.Variant_ID) & target_ids:
        raise AssertionError(f"unexpected eligible source census for {target_panel}: {observed}")
    return source


def split_effects(df: pd.DataFrame, holdout: float, split_seed: int) -> list[dict]:
    """Family-level coverage of a stratified ES holdout, without feature-wise tests."""
    idx = np.arange(len(df)); y = df.Label.reset_index(drop=True)
    gradient, es = train_test_split(idx, test_size=holdout, random_state=split_seed, stratify=y)
    if set(gradient) & set(es) or y.iloc[gradient].nunique() != 2 or y.iloc[es].nunique() != 2:
        raise AssertionError("invalid stratified ES split")
    records = []
    # Convert each small family block once per split; vectorized quantiles avoid a
    # Python DataFrame slice for every numeric column × null split.
    for cls in ("all", "0", "1"):
        full_idx = idx if cls == "all" else idx[y.to_numpy() == int(cls)]
        grad_idx = gradient if cls == "all" else gradient[y.iloc[gradient].to_numpy() == int(cls)]
        es_idx = es if cls == "all" else es[y.iloc[es].to_numpy() == int(cls)]
        for family, cols in family_columns(df).items():
            block = df[cols]
            matrix = block.isna().to_numpy(dtype=float)
            miss_full = matrix[full_idx].mean(axis=1); miss_es = matrix[es_idx].mean(axis=1)
            miss_scale = max(float(miss_full.std(ddof=0)), 1e-9)
            records.append({"class": cls, "family": family, "metric": "missingness_abs_std_diff",
                            "value": abs(float(miss_es.mean() - miss_full.mean())) / miss_scale})
            cats = [c for c in cols if not pd.api.types.is_numeric_dtype(block[c])]
            if cats:
                unseen = []
                for c in cats:
                    train_levels = set(block[c].iloc[grad_idx].dropna().astype(str))
                    held = block[c].iloc[es_idx].dropna().astype(str)
                    unseen.append(float((~held.isin(train_levels)).mean()) if len(held) else 0.0)
                records.append({"class": cls, "family": family, "metric": "categorical_unseen_rate",
                                "value": float(np.mean(unseen))})
            nums = [c for c in cols if pd.api.types.is_numeric_dtype(block[c])]
            if nums:
                values = block[nums].to_numpy(dtype=float, na_value=np.nan)
                train, held = values[grad_idx], values[es_idx]
                valid_train = np.isfinite(train).sum(axis=0) >= 20
                valid_held = np.isfinite(held).sum(axis=0) > 0
                valid = valid_train & valid_held
                if valid.any():
                    lo = np.nanquantile(train[:, valid], .01, axis=0)
                    hi = np.nanquantile(train[:, valid], .99, axis=0)
                    held_valid = held[:, valid]
                    observed = np.isfinite(held_valid)
                    tails = observed & ((held_valid < lo) | (held_valid > hi))
                    tail_count, observed_count = int(tails.sum()), int(observed.sum())
                    records.append({"class": cls, "family": family, "metric": "numeric_outside_gradient_q01_q99",
                                    "value": tail_count / observed_count})
                    records.append({"class": cls, "family": family, "metric": "numeric_tail_cell_count",
                                    "value": float(tail_count)})
    return records


def coverage_diagnostics(census: pd.DataFrame) -> pd.DataFrame:
    """Compare actual production stopping seeds with 100 same-size stratified null splits."""
    specs = []
    for panel in ("general", "cancer"):
        specs.append((panel, "own_final_lgb", .10, OWN_SEEDS, load_raw(panel, usable=True)))
    # Only seed 141 is a production split; 142..145 are sensitivity-only seeds.
    specs.append(("general", "general_final_cat", .15, (GENERAL_CAT_SPLITS[0],), load_raw("general", usable=True)))
    for panel in ("cancer", "pah", "cftr"):
        specs.append((panel, "source_lgb", .15, (SOURCE_LGB_SPLITS[0],), source_eligible_frame(panel)))
    for panel in ("cancer", "cftr"):
        specs.append((panel, "source_cat", .15, (SOURCE_CAT_SPLITS[0],), source_eligible_frame(panel)))
    all_rows, effect_cache = [], {}
    for panel, role, holdout, seeds, raw in specs:
        data_key = f"source:{panel}" if role.startswith("source_") else f"target:{panel}"
        def cached_effects(seed: int) -> list[dict]:
            key = (data_key, holdout, seed)
            if key not in effect_cache:
                effect_cache[key] = split_effects(raw, holdout, seed)
            return effect_cache[key]
        actual = []
        for seed in seeds:
            for r in cached_effects(seed):
                actual.append({**r, "panel": panel, "role": role, "holdout": holdout, "split_seed": seed,
                               "split_kind": "production"})
        null = []
        for seed in NULL_SEEDS:
            for r in cached_effects(seed):
                null.append({**r, "panel": panel, "role": role, "holdout": holdout, "split_seed": seed,
                             "split_kind": "stratified_null"})
        actual_df, null_df = pd.DataFrame(actual), pd.DataFrame(null)
        for key, group in actual_df.groupby(["class", "family", "metric"], sort=False):
            candidates = null_df[(null_df["class"] == key[0]) & (null_df.family == key[1]) &
                                 (null_df.metric == key[2])].value.tolist()
            for _, r in group.iterrows():
                all_rows.append({**r.to_dict(), "null_median": float(np.median(candidates)),
                                 "null_p05": float(np.quantile(candidates, .05)),
                                 "null_p95": float(np.quantile(candidates, .95)),
                                 "null_standardized_offset": _safe_z(float(r.value), candidates)})
    frame = pd.DataFrame(all_rows)
    frame.to_csv(OUT / "coverage_effects.csv", index=False)
    return frame


def own_matrix(panel: str, *, lgb_only: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    raw = load_raw(panel, usable=True)
    eng = _engineer(raw.copy(), PANEL_RECIPE[panel])
    _, nums, cats, label = partition_columns(eng)
    X, y = eng[nums + cats].reset_index(drop=True), eng[label].astype(int).reset_index(drop=True)
    numeric = [c for c in X if not isinstance(X[c].dtype, pd.CategoricalDtype)]
    leaky = sorted(set(detect_leaky_columns(X[numeric], y, .99, verbose=False)) | (KNOWN_LEAKY & set(X.columns)))
    Xpp, _ = _fit_own_preprocess(X.drop(columns=leaky, errors="ignore"), y, PANEL_RECIPE[panel])
    if lgb_only and panel in REDUN_PRUNE_LGBM:
        Xpp = Xpp[_lgbm_redun_keep(Xpp)]
    return Xpp, y


def source_matrix(target_panel: str) -> tuple[pd.DataFrame, pd.Series]:
    target_ids = set(load_raw(target_panel, usable=True).Variant_ID)
    source = load_raw("general", usable=True)
    source = source.loc[~source.Variant_ID.isin(target_ids)].reset_index(drop=True)
    observed = (len(source), int(source.Label.sum()), int((source.Label == 0).sum()))
    if observed != EXPECTED_SOURCE_CENSUS[target_panel] or set(source.Variant_ID) & target_ids:
        raise AssertionError(f"unexpected eligible source census for {target_panel}: {observed}")
    source = engineer_aa_features(to_categorical(source))
    _, nums, cats, label = partition_columns(source)
    X, y = source[nums + cats].reset_index(drop=True), source[label].astype(int).reset_index(drop=True)
    leaky = sorted(set(detect_leaky_columns(X, y, .99, verbose=False)) | (KNOWN_LEAKY & set(X.columns)))
    X = X.drop(columns=leaky, errors="ignore")
    drop_al = TRANSFER_DROP_AL.get(target_panel, False)
    al_features = TRANSFER_AL_FEATURES.get(target_panel, ())
    alf = fit_al(X, al_features) if al_features else None
    model_X = X.drop(columns=[c for c in X if str(c).startswith("AL_")], errors="ignore") if drop_al else X
    Xpp, _ = fit_keep_nan(model_X)
    if alf is not None:
        Xpp = pd.concat([Xpp.reset_index(drop=True), apply_al(X, alf).reset_index(drop=True)], axis=1)
        Xpp = Xpp.loc[:, ~Xpp.columns.duplicated()]
    return Xpp, y


def lgb_runs(scope: str, X: pd.DataFrame, y: pd.Series, *, holdout: float, patience: int,
             model_seeds: tuple[int, ...], split_seeds: tuple[int, ...], params: dict) -> list[dict]:
    rows, baseline = [], None
    # If the counts differ, fixed model seed isolates split sensitivity; equal-length pairs record deployed coupling.
    pairs = list(zip(model_seeds, split_seeds)) if len(model_seeds) == len(split_seeds) else [(model_seeds[0], s) for s in split_seeds]
    for model_seed, split_seed in pairs:
        idx = np.arange(len(y)); grad, es = train_test_split(idx, test_size=holdout, random_state=split_seed, stratify=y)
        model = lgb.LGBMClassifier(**params, random_state=model_seed, n_jobs=4)
        started = time.time()
        model.fit(X.iloc[grad], y.iloc[grad], eval_set=[(X.iloc[es], y.iloc[es])],
                  callbacks=[lgb.early_stopping(patience, verbose=False), lgb.log_evaluation(0)])
        tree_count = int(model.best_iteration_ or params["n_estimators"])
        observed = int(model.booster_.num_trees())
        if tree_count != observed or tree_count < 1:
            raise AssertionError(f"{scope} LGB tree-count mismatch")
        prediction = model.predict_proba(X)[:, 1]
        if baseline is None:
            baseline = prediction
        rows.append({"scope": scope, "model_kind": "lgb", "model_seed": model_seed, "split_seed": split_seed,
                     "holdout": holdout, "patience": patience, "gradient_rows": len(grad), "es_rows": len(es),
                     "tree_count": tree_count, "seconds": time.time() - started,
                     "mean_abs_prediction_delta_from_baseline": float(np.mean(abs(prediction - baseline))),
                     "prediction_spearman_from_baseline": float(spearmanr(prediction, baseline).statistic)})
    return rows


def cat_runs(scope: str, X: pd.DataFrame, y: pd.Series, *, prototype, split_seeds: tuple[int, ...]) -> list[dict]:
    cats = _cats(X); Xcat = _cat_repr(X, cats); cat_idx = [Xcat.columns.get_loc(c) for c in cats]
    rows, baseline = [], None
    for split_seed in split_seeds:
        idx = np.arange(len(y)); grad, es = train_test_split(idx, test_size=.15, random_state=split_seed, stratify=y)
        params = prototype(cat_idx, (y == 0).sum() / max((y == 1).sum(), 1)).get_params()
        model = CatBoostClassifier(**params)
        started = time.time(); model.fit(Xcat.iloc[grad], y.iloc[grad], eval_set=(Xcat.iloc[es], y.iloc[es]))
        tree_count = int(model.get_best_iteration()) + 1
        observed = int(model.tree_count_)
        if tree_count != observed or tree_count < 1:
            raise AssertionError(f"{scope} Cat tree-count mismatch")
        prediction = model.predict_proba(Xcat)[:, 1]
        if baseline is None:
            baseline = prediction
        rows.append({"scope": scope, "model_kind": "cat", "model_seed": SEED, "split_seed": split_seed,
                     "holdout": .15, "patience": 40, "gradient_rows": len(grad), "es_rows": len(es),
                     "tree_count": tree_count, "seconds": time.time() - started,
                     "mean_abs_prediction_delta_from_baseline": float(np.mean(abs(prediction - baseline))),
                     "prediction_spearman_from_baseline": float(spearmanr(prediction, baseline).statistic)})
    return rows


def es_sensitivity() -> pd.DataFrame:
    rows = []
    for panel in ("general", "cancer"):
        X, y = own_matrix(panel); spw = (y == 0).sum() / max((y == 1).sum(), 1)
        params = get_model_params(len(y), spw)
        if panel in LGBM_PARAM_OVERRIDE:
            params = {**params, **LGBM_PARAM_OVERRIDE[panel]}
        rows += lgb_runs(f"own:{panel}:final_lgb", X, y, holdout=.10,
                         patience=50 if len(y) > 300 else 30, model_seeds=OWN_SEEDS,
                         split_seeds=OWN_SEEDS, params=params)
    X, y = own_matrix("general", lgb_only=False)
    rows += cat_runs("own:general:final_cat", X, y, prototype=_cat_general, split_seeds=GENERAL_CAT_SPLITS)
    # Separates ES-split sensitivity from the deployed member-seed coupling above.
    X, y = own_matrix("general")
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    params = {**get_model_params(len(y), spw), **LGBM_PARAM_OVERRIDE["general"]}
    rows += lgb_runs("own:general:final_lgb_fixed_model", X, y, holdout=.10, patience=50,
                     model_seeds=(SEED,), split_seeds=OWN_SEEDS, params=params)
    for panel in ("cancer", "pah", "cftr"):
        X, y = source_matrix(panel); spw = (y == 0).sum() / max((y == 1).sum(), 1)
        rows += lgb_runs(f"source:{panel}:lgb", X, y, holdout=.15, patience=40,
                         model_seeds=(SEED,), split_seeds=SOURCE_LGB_SPLITS,
                         params=get_model_params(len(y), spw))
        if panel in TRANSFER_BLEND:
            rows += cat_runs(f"source:{panel}:cat", X, y, prototype=_cat_model, split_seeds=SOURCE_CAT_SPLITS)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "es_sensitivity_runs.csv", index=False)
    return frame


def as_markdown(frame: pd.DataFrame) -> str:
    """Small local formatter; diagnostics do not require pandas' optional tabulate extra."""
    columns = list(frame.columns)
    def cell(value):
        return str(value).replace("|", "\\|")
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(lines)


def markdown(census: pd.DataFrame, source_census: pd.DataFrame, coverage: pd.DataFrame, sensitivity: pd.DataFrame, started: float) -> str:
    lines = ["# Finalization diagnostic evidence ledger", "", "Research-only diagnostic run; no outer-held-out score or candidate selection was run.", "",
             "## Row eligibility and identity", "", as_markdown(census), "",
             "- All-feature-NaN rows: **unsafe to reclaim** (no learnable features; mixed-label duplicate groups).",
             "- Target-ID-overlapping general rows: **unsafe to reclaim** into transfer source (would make target calibration in-sample).",
             "- Eligible non-overlap general rows: **retained** by current transfer source policy.",
             "- Same-label exact duplicates: **not tested for removal**; no deduplication policy was introduced.", "",
             "Eligible transfer-source census (general after target-ID exclusion):", "", as_markdown(source_census), "",
             "## Split representation", "",
             "`coverage_effects.csv` compares each production ES split to 100 stratified null splits of the same size. Effects are family-level only: class-conditional missingness standardized difference, categorical unseen-rate, and numeric held-out values outside gradient q01–q99. No feature-wise p-values or selection follows from these diagnostics.", ""]
    top = coverage.reindex(coverage.null_standardized_offset.abs().sort_values(ascending=False).index).head(12)
    lines += ["Largest absolute family-level null offsets:", "", as_markdown(top[["panel", "role", "split_seed", "class", "family", "metric", "value", "null_median", "null_standardized_offset"]]), ""]
    summary = sensitivity.groupby(["scope", "model_kind"], as_index=False).agg(
        n=("split_seed", "size"), tree_min=("tree_count", "min"), tree_max=("tree_count", "max"),
        tree_median=("tree_count", "median"), max_prediction_mad=("mean_abs_prediction_delta_from_baseline", "max"),
        min_prediction_spearman=("prediction_spearman_from_baseline", "min"), seconds=("seconds", "sum"))
    lines += ["## ES count and prediction sensitivity", "", "Counts/predictions use training matrices only; they quantify stopping-split sensitivity, not generalization or a candidate win.", "", as_markdown(summary), "",
              "## Status", "", "- [x] Census, all-NaN, duplicate/conflict and cross-panel-ID audit", "- [x] Production-vs-stratified-null coverage audit", "- [x] Five predeclared ES split seeds for each affected final/source family", "- [ ] Outer comparison and package rehearsal — blocked pending supervisor approval", "",
              f"Observed diagnostic runtime: {time.time() - started:.1f}s."]
    return "\n".join(lines) + "\n"


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time(); before = tracked_hashes()
    (OUT / "preflight_hashes.json").write_text(json.dumps({"hashes": before, "seed": SEED,
        "own_split_seeds": OWN_SEEDS, "source_lgb_split_seeds": SOURCE_LGB_SPLITS,
        "source_cat_split_seeds": SOURCE_CAT_SPLITS, "general_cat_split_seeds": GENERAL_CAT_SPLITS}, indent=2) + "\n")
    census = census_and_duplicates(); census.to_csv(OUT / "census.csv", index=False)
    sources = [source_eligible_frame(panel) for panel in ("cancer", "pah", "cftr")]
    source_census = pd.DataFrame([{"target_panel": panel, "eligible_source_rows": len(source),
        "pathogenic": int(source.Label.sum()), "benign": int((source.Label == 0).sum())}
        for panel, source in zip(("cancer", "pah", "cftr"), sources)])
    source_census.to_csv(OUT / "source_census.csv", index=False)
    coverage = coverage_diagnostics(census)
    sensitivity = es_sensitivity()
    after = tracked_hashes()
    if before != after:
        raise AssertionError("production/data/model hash changed during diagnostics")
    sensitivity_summary = sensitivity.groupby(["scope", "model_kind"], as_index=False).agg(
        n_splits=("split_seed", "size"), tree_count_min=("tree_count", "min"), tree_count_max=("tree_count", "max"),
        tree_count_median=("tree_count", "median"), max_prediction_mad=("mean_abs_prediction_delta_from_baseline", "max"),
        min_prediction_spearman=("prediction_spearman_from_baseline", "min"), seconds=("seconds", "sum"))
    sensitivity_summary.to_csv(OUT / "es_sensitivity_summary.csv", index=False)
    manifest = {"git_head": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "seconds": time.time() - started, "no_deployment": True, "hashes_before": before,
                "hashes_after": after, "coverage_null_splits_per_spec": len(NULL_SEEDS),
                "notes": "No outer labels were used to select a policy; source predictions are sensitivity probes on training matrices."}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "EVIDENCE_LEDGER.md").write_text(markdown(census, source_census, coverage, sensitivity, started))
    print(f"Saved diagnostics to {OUT} in {manifest['seconds']:.1f}s; production artifacts unchanged.")


if __name__ == "__main__":
    main()
