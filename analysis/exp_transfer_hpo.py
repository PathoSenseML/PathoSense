"""Exp 4 — HPO the transfer scorer's LightGBM (Optuna), incl. bagging (the multi-seed fix).

The scorer uses default get_model_params; `subsample` is inert (no bagging_freq) so the
Exp-1 multi-seed barely decorrelated. Here we search params INCLUDING subsample_freq
(=bagging_freq) so bagging actually engages, then check whether multi-seed helps once it does.

Honest, overfitting-guarded:
  * HPO objective = GENERAL CV F1* (the scorer's training panel) — fast (n_repeats=2, 1 seed).
  * Validate best params on GENERAL at full fidelity (paired vs default), AND on the held-out
    target panels (cftr/pah/cancer) via the deployed scorer — the targets are an honest outer
    test the HPO never saw. Deploy only if the win transfers to the targets at >=2sigma.

Run:  .venv/bin/python -u analysis/exp_transfer_hpo.py [--trials 40]
"""
import os, sys, time, argparse
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import RepeatedStratifiedKFold
from config import SEED
from utils import TEST_PRIORS
from calibration import fit_platt, apply_platt
from harness import (cv_evaluate, load_panel, compare, _drop_leaky, lgb_oof_and_test,
                     pick_threshold_at_prior, tpr_fpr, f1_at_prior, FINALS_N)
from preprocess_v2 import fit_keep_nan, apply_keep_nan
from run_pipeline import get_model_params
from exp_transfer import _load, load_target_engineered, target_deployed_fn
from transfer import fit_transfer_scorer, transfer_score, TRANSFER_PANELS

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def make_general_fn(extra, n_seeds=1):
    """General reworked recipe (native-NaN, all features) with LGBM param overrides."""
    def _fn(X_tr, y_tr, X_te):
        Xtr, Xte = _drop_leaky(X_tr, y_tr, X_te)
        Xtr_pp, ppf = fit_keep_nan(Xtr); Xte_pp = apply_keep_nan(Xte, ppf)
        spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        params = {**get_model_params(len(X_tr), spw), **extra}
        tp, oof = lgb_oof_and_test(Xtr_pp, y_tr, Xte_pp, params, n_seeds=n_seeds, n_folds=5)
        pf = fit_platt(oof, y_tr.values)
        return apply_platt(tp, pf), apply_platt(oof, pf)
    return _fn


def hpo_general(n_trials, df):
    def objective(trial):
        po = dict(
            num_leaves=trial.suggest_int("num_leaves", 15, 63),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 40),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            subsample_freq=trial.suggest_int("subsample_freq", 0, 5),   # bagging_freq -> activates subsample
            reg_alpha=trial.suggest_float("reg_alpha", 0.0, 2.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.0, 3.0),
            n_estimators=trial.suggest_int("n_estimators", 300, 2500, step=100),
            max_depth=trial.suggest_int("max_depth", 3, 8),
        )
        return cv_evaluate("general", make_general_fn(po), n_repeats=2, df=df, n_jobs=6).f1.mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    print(f"\n[general] HPO best ({time.time()-t0:.0f}s): {best}", flush=True)
    base = cv_evaluate("general", make_general_fn({}, n_seeds=3), n_repeats=6, df=df)
    cand = cv_evaluate("general", make_general_fn(best, n_seeds=3), n_repeats=6, df=df)
    c = compare(base, cand)
    print(f"[general] default F1*={c['a_f1']:.4f}  tuned F1*={c['b_f1']:.4f}  "
          f"Δ={c['delta_f1']:+.4f} ({c['sigma']:+.2f}σ)  {c['verdict']}", flush=True)
    return best, c


def eval_targets(best, repeats_cancer=8, repeats_other=12):
    rows = []
    for panel in ["pah", "cftr", "cancer"]:
        prior = TEST_PRIORS[panel]; w = TRANSFER_PANELS[panel]
        Xt, yt = load_target_engineered(panel); raw = _load(panel)
        t0 = time.time()
        configs = {
            "baseline": transfer_score(raw, fit_transfer_scorer(panel)),
            "hpo":      transfer_score(raw, fit_transfer_scorer(panel, model_params=best)),
            "hpo+ms5":  transfer_score(raw, fit_transfer_scorer(panel, model_params=best, n_seeds=5)),
        }
        fn = target_deployed_fn(panel)
        nr = repeats_cancer if panel == "cancer" else repeats_other
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=nr, random_state=SEED)
        f1 = {k: [] for k in configs}
        for tr, te in rskf.split(Xt, yt):
            tr, te = np.asarray(tr), np.asarray(te)
            ytr, yte = yt.iloc[tr].reset_index(drop=True), yt.iloc[te].reset_index(drop=True)
            if w >= 1.0:
                e_tr = np.zeros(len(tr)); e_te = np.zeros(len(te))
            else:
                e_te, e_tr = fn(Xt.iloc[tr].reset_index(drop=True), ytr,
                                Xt.iloc[te].reset_index(drop=True))
            for k, g in configs.items():
                pl = fit_platt(g[tr], ytr.values)
                gt, ge = apply_platt(g[tr], pl), apply_platt(g[te], pl)
                b_tr = w * gt + (1 - w) * e_tr; b_te = w * ge + (1 - w) * e_te
                thr = pick_threshold_at_prior(ytr.values, b_tr, prior)
                tpr, fpr = tpr_fpr(yte.values, (b_te >= thr).astype(int))
                f1[k].append(f1_at_prior(tpr, fpr, prior))
        base = np.array(f1["baseline"]); nf = len(base)
        print(f"\n[{panel}] target eval ({nf} folds, {time.time()-t0:.0f}s):", flush=True)
        for k in configs:
            arr = np.array(f1[k]); d = arr - base
            se = d.std(ddof=1) / np.sqrt(nf) if nf > 1 else 0.0
            sig = d.mean() / se if se > 0 else 0.0
            tag = "ref" if k == "baseline" else (f"Δ={d.mean():+.4f} ({sig:+.2f}σ)")
            print(f"    {k:9s} F1*={arr.mean():.4f}  {tag}", flush=True)
            rows.append({"panel": panel, "config": k, "f1": arr.mean(),
                         "delta": 0 if k == "baseline" else d.mean(),
                         "sigma": 0 if k == "baseline" else sig})
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--trials", type=int, default=40)
    a = ap.parse_args()
    dfg = load_panel("general")
    best, gen_c = hpo_general(a.trials, dfg)
    rows = eval_targets(best)
    rows.append({"panel": "general", "config": "hpo_vs_default", "f1": gen_c["b_f1"],
                 "delta": gen_c["delta_f1"], "sigma": gen_c["sigma"]})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "exp_transfer_hpo.csv"), index=False)
    print(f"\nbest_params = {best}", flush=True)
    print(f"saved -> {os.path.join(OUT, 'exp_transfer_hpo.csv')}", flush=True)
