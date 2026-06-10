"""Per-panel LightGBM hyperparameter optimization on the HONEST CV harness.

The deployed get_model_params is a hand-tuned size-tiered default. We search around
it with Optuna, objective = mean analytic F1* at the test prior over a fast CV
(n_repeats=2, n_seeds=1), then VALIDATE the best config at full fidelity (paired vs
deployed). Most relevant for GENERAL (cross-panel transfer can't help the source
panel) but run for all.
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from harness import cv_evaluate, load_panel, compare, deployed_pipeline_fn
from pipeline_v2 import make_fn as _make_recipe_fn
from utils import TEST_PRIORS

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
PANELS = ["general", "cancer", "pah", "cftr"]

# The DEPLOYED recipe per panel (mirrors pipelines/recipe.py) — HPO tunes the model
# on top of the actual deployed preprocessing/feature recipe (general: native-NaN +
# all features).
RECIPE = {
    "general": dict(impute="none", feature_select="none", al_features=(), score_agree=False),
    "cancer":  dict(impute="median", feature_select="mrmr", al_features=(), score_agree=True),
    "pah":     dict(impute="median", feature_select="mrmr", al_features=(), score_agree=False),
    "cftr":    dict(impute="median", feature_select="mrmr", al_features=(), score_agree=True),
}


def make_fn(panel, params_override, n_seeds=1):
    rec = dict(RECIPE[panel]); rec["model_params"] = params_override
    return _make_recipe_fn(panel, rec, n_seeds=n_seeds)


def run(panel, n_trials=40):
    df = load_panel(panel)
    rep_fast = 2
    rep_full = {"general": 6, "cancer": 10, "pah": 10, "cftr": 12}[panel]

    def objective(trial):
        po = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 3.0),
            "n_estimators": trial.suggest_int("n_estimators", 300, 2500, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
        }
        r = cv_evaluate(panel, make_fn(panel, po), n_repeats=rep_fast, df=df, n_jobs=6)
        return r.f1.mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    print(f"\n[{panel}] best params: {best}", flush=True)

    # validate at full fidelity, paired vs the NEW-recipe DEFAULT params (isolates HPO)
    base = cv_evaluate(panel, make_fn(panel, {}, n_seeds=3), n_repeats=rep_full, df=df)
    cand = cv_evaluate(panel, make_fn(panel, best, n_seeds=3), n_repeats=rep_full, df=df)
    c = compare(base, cand)
    print(f"[{panel}] recipe-default F1*={c['a_f1']:.4f}  tuned F1*={c['b_f1']:.4f}  "
          f"Δ={c['delta_f1']:+.4f} ({c['sigma']:+.2f}σ)  {c['verdict']}", flush=True)
    return {"panel": panel, **c, "best_params": best}


if __name__ == "__main__":
    panels = sys.argv[1:] or PANELS
    rows = [run(pn) for pn in panels]
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "exp_hpo.csv"), index=False)
    print(f"\nsaved -> {os.path.join(OUT, 'exp_hpo.csv')}")
