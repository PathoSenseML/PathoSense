"""Overnight: Optuna HPO of the TRANSFER-SCORER LGBM params for the pure-transfer small panels (pah, cftr
— their whole deployed prediction IS the scorer). Search = scorer AUC on the target (out-of-sample; the
scorer excludes target Variant_IDs). Gate = deployed CV (pure transfer w=1.0): per fold Platt on g[tr],
F1-opt-at-prior threshold, F1* on g[te]; the scorer is fixed across folds. Held-out seeds for the gate.
Deploy only if robustly non-negative on F1* (prior: scorer is locked → pah collapses; test honestly).
Run: .venv/bin/python -u analysis/exp_scorer_hpo.py [panel ...] [--trials N]
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from config import SEED
from utils import TEST_PRIORS
from calibration import fit_platt, apply_platt
from harness import pick_threshold_at_prior, tpr_fpr, f1_at_prior
from transfer import fit_transfer_scorer, transfer_score

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
from exp_transfer import _load


def gate(panel, g, prior, seeds=(SEED + 1000, SEED + 2000), nr=10):
    """Deployed pure-transfer CV F1* (mean over seeds) for a fixed scorer-score vector g."""
    df = _load(panel); y = df["Label"].astype(int).values
    out = []
    for seed in seeds:
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=nr, random_state=seed)
        f = []
        for tr, te in rskf.split(np.zeros(len(y)), y):
            tr, te = np.asarray(tr), np.asarray(te)
            pl = fit_platt(g[tr], y[tr]); gtr, gte = apply_platt(g[tr], pl), apply_platt(g[te], pl)
            t = pick_threshold_at_prior(y[tr], gtr, prior)
            f.append(f1_at_prior(*tpr_fpr(y[te], (gte >= t).astype(int)), prior))
        out.append(np.array(f))
    return out  # list of per-fold arrays, one per seed


def run(panel, n_trials):
    prior = TEST_PRIORS[panel]; df = _load(panel); y = df["Label"].astype(int).values
    base_art = fit_transfer_scorer(panel); g_base = transfer_score(df, base_art)
    base_auc = roc_auc_score(y, g_base)
    print(f"\n[{panel}] base scorer AUC={base_auc:.4f}", flush=True)

    def obj(trial):
        p = {"num_leaves": trial.suggest_int("num_leaves", 16, 80),
             "max_depth": trial.suggest_int("max_depth", 3, 10),
             "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
             "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.06, log=True),
             "n_estimators": trial.suggest_int("n_estimators", 400, 2500, step=100),
             "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
             "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 4.0)}
        art = fit_transfer_scorer(panel, model_params=p)
        return roc_auc_score(y, transfer_score(df, art))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    t0 = time.time(); study.optimize(obj, n_trials=n_trials)
    best_p = study.best_params
    print(f"  search best scorer AUC={study.best_value:.4f} (Δ{study.best_value-base_auc:+.4f}, {time.time()-t0:.0f}s)  {best_p}", flush=True)
    g_best = transfer_score(df, fit_transfer_scorer(panel, model_params=best_p))

    # honest gate (held-out seeds), F1* paired
    base_g = gate(panel, g_base, prior); best_g = gate(panel, g_best, prior)
    rows = []
    for i, seed in enumerate((SEED + 1000, SEED + 2000)):
        b, t = base_g[i], best_g[i]; d = t - b; nf = len(b)
        sig = d.mean()/(d.std(ddof=1)/np.sqrt(nf)) if d.std() > 0 else 0
        print(f"  gate seed{seed}: base F1*={b.mean():.4f}  tuned F1*={t.mean():.4f}  Δ={d.mean():+.4f} ({sig:+.2f}σ)", flush=True)
        rows.append({"panel": panel, "seed": seed, "base_f1": b.mean(), "tuned_f1": t.mean(),
                     "delta": d.mean(), "sigma": sig, "base_auc": base_auc, "tuned_auc": roc_auc_score(y, g_best),
                     "best_params": str(best_p)})
    return rows


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    nt = 40
    if "--trials" in sys.argv:
        nt = int(sys.argv[sys.argv.index("--trials") + 1])
    panels = args or ["pah", "cftr"]
    print(f"=== scorer-LGBM Optuna HPO ({nt} trials) — pure-transfer panels ===", flush=True)
    allr = []
    for pn in panels:
        allr += run(pn, nt)
        pd.DataFrame(allr).to_csv(os.path.join(OUT, "scorer_hpo.csv"), index=False)
    print(f"\nsaved -> scorer_hpo.csv", flush=True)
