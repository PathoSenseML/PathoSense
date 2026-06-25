"""NESTED-honest gate for the cftr scorer-HPO win (held-out-seed gate showed +0.0084 F1* but params were
selected on cftr's OWN AUC → selection-optimism). Proper test: OUTER CV over cftr; in each outer fold HPO
the scorer params on the OUTER-TRAIN cftr AUC only, then score deployed F1* on the OUTER-TEST cftr (which
param-selection never saw). Paired vs the base scorer on the same outer-test folds. If still positive →
the HPO generalizes honestly → deploy via TRANSFER_PARAMS. (Scorer always trains on general-minus-all-cftr;
only the 7 hyperparams are cftr-informed.)
Run: .venv/bin/python -u analysis/exp_cftr_scorer_nested.py [--trials N --outer K]
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from config import SEED
from utils import TEST_PRIORS
from calibration import fit_platt, apply_platt
from harness import pick_threshold_at_prior, tpr_fpr, f1_at_prior
from transfer import fit_transfer_scorer, transfer_score
from exp_transfer import _load

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def hpo_params(df, y_sel, idx_sel, n_trials):
    """HPO scorer LGBM params to maximize AUC on the SELECTION rows (idx_sel) only."""
    def obj(trial):
        p = {"num_leaves": trial.suggest_int("num_leaves", 16, 80),
             "max_depth": trial.suggest_int("max_depth", 3, 10),
             "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
             "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.06, log=True),
             "n_estimators": trial.suggest_int("n_estimators", 400, 2500, step=100),
             "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
             "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 4.0)}
        g = transfer_score(df, fit_transfer_scorer("cftr", model_params=p))
        return roc_auc_score(y_sel, g[idx_sel])
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(obj, n_trials=n_trials)
    return st.best_params


def main(n_trials, n_outer):
    prior = TEST_PRIORS["cftr"]; df = _load("cftr"); y = df["Label"].astype(int).values
    g_base = transfer_score(df, fit_transfer_scorer("cftr"))
    print(f"=== cftr scorer-HPO NESTED gate ({n_outer} outer folds, {n_trials} trials each) ===", flush=True)
    f_base, f_tun = [], []
    skf = StratifiedKFold(n_outer, shuffle=True, random_state=SEED + 7)
    for k, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
        t0 = time.time()
        bp = hpo_params(df, y[tr], tr, n_trials)               # select params on outer-TRAIN AUC only
        g_tun = transfer_score(df, fit_transfer_scorer("cftr", model_params=bp))
        # deployed scoring on outer-TEST: Platt+threshold fit on outer-train, applied to outer-test
        for g, acc in ((g_base, f_base), (g_tun, f_tun)):
            pl = fit_platt(g[tr], y[tr]); thr = pick_threshold_at_prior(y[tr], apply_platt(g[tr], pl), prior)
            acc.append(f1_at_prior(*tpr_fpr(y[te], (apply_platt(g[te], pl) >= thr).astype(int)), prior))
        print(f"  outer fold {k}: base F1*={f_base[-1]:.4f}  tuned F1*={f_tun[-1]:.4f}  "
              f"(Δ{f_tun[-1]-f_base[-1]:+.4f}, {time.time()-t0:.0f}s)  params={bp}", flush=True)
    b, t = np.array(f_base), np.array(f_tun); d = t - b
    se = d.std(ddof=1)/np.sqrt(len(d)) if len(d) > 1 else 0.0; sig = d.mean()/se if se > 0 else 0.0
    print(f"\n  NESTED: base F1*={b.mean():.4f}  tuned F1*={t.mean():.4f}  Δ={d.mean():+.4f} ({sig:+.2f}σ over {len(d)} folds)", flush=True)
    pd.DataFrame({"fold": range(len(b)), "base": b, "tuned": t}).to_csv(os.path.join(OUT, "cftr_scorer_nested.csv"), index=False)
    print(f"  verdict: {'GENERALIZES (deploy)' if d.mean() > 0 else 'selection-overfit (reject)'}", flush=True)


if __name__ == "__main__":
    nt = int(sys.argv[sys.argv.index("--trials") + 1]) if "--trials" in sys.argv else 25
    no = int(sys.argv[sys.argv.index("--outer") + 1]) if "--outer" in sys.argv else 5
    main(nt, no)
