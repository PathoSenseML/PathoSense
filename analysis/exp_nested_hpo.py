"""Idea #4 — hyperparameter optimization done PROPERLY (nested CV).

The deployed LGBM uses a fixed heuristic (get_model_params). Prior HPO was rejected for overfitting the
source — almost always because tuning saw the evaluation fold. Here HPO is NESTED:
  outer RepeatedStratifiedKFold (evaluation) — never seen by tuning.
  per outer-train fold: inner 3-fold random search over LGBM params (objective = inner-OOF AUC),
                        pick best, train on the full outer-train, predict the outer-test fold.
  default config (get_model_params) evaluated on the SAME outer folds -> paired Δ.
Preprocessing (fit_keep_nan) is fit per outer fold; the search varies only model hyperparameters.
Run: .venv/bin/python -u analysis/exp_nested_hpo.py [panel ...]
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from config import SEED
from utils import TEST_PRIORS
from harness import _drop_leaky, pick_threshold_at_prior, tpr_fpr, f1_at_prior
from preprocess_v2 import fit_keep_nan, apply_keep_nan
from calibration import fit_platt, apply_platt
from run_pipeline import get_model_params
from schema import partition_columns
from harness import load_panel

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
N_REPEATS = {"general": 2, "cancer": 5, "pah": 5, "cftr": 6}

GRID = {"num_leaves": [15, 31, 63], "min_child_samples": [10, 20, 40],
        "learning_rate": [0.03, 0.05, 0.1], "reg_lambda": [0.0, 1.0, 5.0],
        "feature_fraction": [0.7, 0.9], "reg_alpha": [0.0, 0.5]}
N_CONFIGS = 15


def sample_configs(rng):
    cfgs = []
    for _ in range(N_CONFIGS):
        cfgs.append({k: v[rng.randint(len(v))] for k, v in GRID.items()})
    return cfgs


def _cats(X):
    return [c for c in X.columns if hasattr(X[c], "cat")] and [c for c in X.columns if str(X[c].dtype) == "category"]


def fit_pred(Xtr, ytr, Xte, params, cats, spw):
    p = {"n_estimators": 1000, "scale_pos_weight": spw, "random_state": SEED, "n_jobs": 4,
         "verbosity": -1, **params}
    from sklearn.model_selection import train_test_split
    i2, ies = train_test_split(np.arange(len(ytr)), test_size=0.12, random_state=SEED, stratify=ytr)
    m = LGBMClassifier(**p)
    m.fit(Xtr.iloc[i2], ytr.iloc[i2], eval_set=[(Xtr.iloc[ies], ytr.iloc[ies])],
          callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
          categorical_feature=cats or "auto")
    return m.predict_proba(Xte)[:, 1]


def inner_auc(Xtr, ytr, params, cats, spw):
    skf = StratifiedKFold(3, shuffle=True, random_state=SEED)
    oof = np.zeros(len(ytr))
    for a, b in skf.split(Xtr, ytr):
        oof[b] = fit_pred(Xtr.iloc[a], ytr.iloc[a], Xtr.iloc[b], params, cats, spw)
    return roc_auc_score(ytr, oof)


def run(panel):
    prior = TEST_PRIORS[panel]; df = load_panel(panel)
    _, num, cat, tgt = partition_columns(df)
    X = df[num + cat].reset_index(drop=True); y = df[tgt].astype(int).reset_index(drop=True)
    rng = np.random.RandomState(SEED); configs = sample_configs(rng)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=N_REPEATS[panel], random_state=SEED)
    per = {"default": [], "tuned": []}; picks = []
    t0 = time.time()
    for tr, te in rskf.split(X, y):
        Xtr0, Xte0 = _drop_leaky(X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True),
                                 X.iloc[te].reset_index(drop=True))
        Xtr, ppf = fit_keep_nan(Xtr0); Xte = apply_keep_nan(Xte0, ppf)
        ytr, yte = y.iloc[tr].reset_index(drop=True), y.iloc[te].reset_index(drop=True)
        cats = [c for c in Xtr.columns if str(Xtr[c].dtype) == "category"]
        spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
        # inner HPO on outer-train ONLY
        best, best_auc = None, -1
        for cfg in configs:
            a = inner_auc(Xtr, ytr, cfg, cats, spw)
            if a > best_auc:
                best_auc, best = a, cfg
        picks.append(tuple(sorted(best.items())))
        # default vs tuned: train on outer-train, predict outer-test + train-oof for threshold
        for key, params in (("default", {k: v for k, v in get_model_params(len(tr), spw).items()
                                         if k in GRID or k in ("learning_rate",)}), ("tuned", best)):
            # OOF on outer-train for threshold
            skf = StratifiedKFold(5, shuffle=True, random_state=SEED); oof = np.zeros(len(ytr))
            for a2, b2 in skf.split(Xtr, ytr):
                oof[b2] = fit_pred(Xtr.iloc[a2], ytr.iloc[a2], Xtr.iloc[b2], params, cats, spw)
            test = fit_pred(Xtr, ytr, Xte, params, cats, spw)
            pf = fit_platt(oof, ytr.values); oofc, testc = apply_platt(oof, pf), apply_platt(test, pf)
            thr = pick_threshold_at_prior(ytr.values, oofc, prior)
            tpr, fpr = tpr_fpr(yte.values, (testc >= thr).astype(int))
            per[key].append(f1_at_prior(tpr, fpr, prior))
    base = np.array(per["default"]); v = np.array(per["tuned"]); d = v - base
    nf = len(base); se = d.std(ddof=1) / np.sqrt(nf); sig = d.mean() / se if se > 0 else 0.0
    vd = "DEPLOY" if d.mean() > 0 and sig >= 2 else "marginal" if d.mean() > 0 and sig >= 1 else "no gain"
    from collections import Counter
    common = Counter(picks).most_common(1)[0]
    print(f"\n[{panel}] default F1*={base.mean():.4f}  nested-HPO F1*={v.mean():.4f}  "
          f"Δ={d.mean():+.4f} ({sig:+.2f}σ)  -> {vd}  ({nf} folds, {time.time()-t0:.0f}s)", flush=True)
    print(f"   most-picked config ({common[1]}/{nf} folds): {dict(common[0])}", flush=True)
    return {"panel": panel, "default": base.mean(), "tuned": v.mean(), "delta": d.mean(), "sigma": sig}


if __name__ == "__main__":
    panels = sys.argv[1:] or ["cftr", "pah", "cancer", "general"]
    print("=== Idea #4: nested-CV HPO vs default — honest (no leakage) ===", flush=True)
    rows = []
    for pn in panels:
        rows.append(run(pn))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "nested_hpo.csv"), index=False)
    print(f"\nsaved -> nested_hpo.csv", flush=True)
