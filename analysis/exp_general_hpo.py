"""General LGBM HPO (Optuna) targeting AUC, around the now-corrected spw1_bag base. SEARCH: maximize
5-fold OOF AUC of the general LGBM over the param space. CONFIRM (honest, leakage-free): the winning
params vs the deployed spw1_bag default via the harness on HELD-OUT seeds (different from the search),
LGBM-only then the blend. Deploy only if non-negative on held-out seeds.
Run: .venv/bin/python -u analysis/exp_general_hpo.py [n_trials]
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.metrics import roc_auc_score
from config import SEED
from harness import cv_evaluate, load_panel, compare
from schema import partition_columns
from run_pipeline import get_model_params
from meeting_lib import native_nan_oof

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
df = load_panel("general"); _, _num, _cat, _tgt = partition_columns(df)
yv = df[_tgt].astype(int)
SPW = (yv == 0).sum() / max((yv == 1).sum(), 1)
BASE = {**get_model_params(2000, SPW), "scale_pos_weight": 1.0, "bagging_freq": 1, "bagging_fraction": 0.8}
Xa = df[_num + _cat].reset_index(drop=True); ya = yv.reset_index(drop=True)


def objective(trial):
    p = {**BASE,
         "num_leaves": trial.suggest_int("num_leaves", 16, 64),
         "max_depth": trial.suggest_int("max_depth", 4, 9),
         "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
         "n_estimators": trial.suggest_int("n_estimators", 800, 3000, step=200),
         "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
         "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
         "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 3.0),
         "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.2)}
    _, oof = native_nan_oof(Xa, ya, Xa, params=p, n_seeds=1)   # CV OOF (search objective)
    return roc_auc_score(ya.values, oof)


def mk(params):
    def fn(Xtr, ytr, Xte):
        return native_nan_oof(Xtr, ytr, Xte, params=params)
    return fn


if __name__ == "__main__":
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    base_auc = roc_auc_score(ya.values, native_nan_oof(Xa, ya, Xa, params=BASE, n_seeds=1)[1])
    print(f"=== general LGBM HPO (Optuna {n_trials} trials, target OOF AUC; spw1_bag base AUC={base_auc:.4f}) ===", flush=True)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = {**BASE, **study.best_params}
    print(f"  search best OOF AUC={study.best_value:.4f} (base {base_auc:.4f}, Δ={study.best_value-base_auc:+.4f})", flush=True)
    print(f"  best params: {study.best_params}", flush=True)

    # HONEST confirmation on HELD-OUT seeds (not the search seed)
    print("\n  honest paired CV (LGBM-only), held-out seeds:", flush=True)
    rows = []
    for seed in (SEED + 1000, SEED + 2000):
        b = cv_evaluate("general", mk(BASE), label="base", n_repeats=6, df=df, seed=seed)
        t = cv_evaluate("general", mk(best), label="hpo", n_repeats=6, df=df, seed=seed)
        c = compare(b, t); dauc = t.auc.mean() - b.auc.mean()
        da = t.auc - b.auc; sa = da.mean()/(da.std(ddof=1)/np.sqrt(len(da))) if da.std() > 0 else 0
        print(f"    seed{seed}: F1* {b.f1.mean():.4f}->{t.f1.mean():.4f} Δ={c['delta_f1']:+.4f} ({c['sigma']:+.2f}σ) | "
              f"AUC {b.auc.mean():.4f}->{t.auc.mean():.4f} Δ={dauc:+.4f} ({sa:+.2f}σ)", flush=True)
        rows.append({"seed": seed, "f1_base": b.f1.mean(), "f1_hpo": t.f1.mean(), "df1": c["delta_f1"],
                     "sig_f1": c["sigma"], "dauc": dauc, "sig_auc": sa})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "general_hpo.csv"), index=False)
    import json
    with open(os.path.join(OUT, "general_hpo_best.json"), "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"\nsaved -> general_hpo.csv, general_hpo_best.json", flush=True)
