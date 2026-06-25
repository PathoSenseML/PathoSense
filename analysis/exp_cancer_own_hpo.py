"""Overnight: Optuna HPO of the CANCER OWN-model LGBM (median+score_agree+mRMR), which is 0.5 of the
deployed cancer prediction. Search = own-model 5-fold OOF AUC. Gate = deployed cancer (0.5·transfer[rfup]
+ 0.5·own_tuned) @ threshold 0.81, held-out seeds, paired vs current own (default params). Deploy if
robustly non-negative on F1*.
Run: .venv/bin/python -u analysis/exp_cancer_own_hpo.py [--trials N]
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from lightgbm import LGBMClassifier
from config import SEED
from utils import TEST_PRIORS
from calibration import fit_platt, apply_platt
from harness import pick_threshold_at_prior, tpr_fpr, f1_at_prior, lgb_oof_and_test
from run_pipeline import get_model_params
from transfer import fit_transfer_scorer, transfer_score
from exp_transfer import _load, load_target_engineered
from exp_cancer_own import own_preprocess

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
THR = 0.81


def main(nt):
    prior = TEST_PRIORS["cancer"]
    df = _load("cancer"); y = df["Label"].astype(int)
    g = transfer_score(df, fit_transfer_scorer("cancer"))         # deployed rfup transfer (fixed)
    Xt, yt = load_target_engineered("cancer")
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    # preprocess full cancer once for the SEARCH objective (own model matrix)
    Xpp_full, _ = own_preprocess(Xt, y, Xt)
    base_params = get_model_params(len(y), spw)

    def own_oof_auc(params):
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED); oof = np.zeros(len(y))
        for a, b in skf.split(Xpp_full, y):
            m = LGBMClassifier(**params)
            m.fit(Xpp_full.iloc[a], y.iloc[a],
                  categorical_feature=[c for c in Xpp_full.columns if str(Xpp_full[c].dtype) == "category"] or "auto")
            oof[b] = m.predict_proba(Xpp_full.iloc[b])[:, 1]
        return roc_auc_score(y.values, oof)

    base_auc = own_oof_auc(base_params)
    print(f"=== cancer OWN-model Optuna HPO ({nt} trials); base own OOF AUC={base_auc:.4f} ===", flush=True)

    def obj(trial):
        p = {**base_params,
             "num_leaves": trial.suggest_int("num_leaves", 8, 48),
             "max_depth": trial.suggest_int("max_depth", 3, 8),
             "min_child_samples": trial.suggest_int("min_child_samples", 3, 40),
             "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.06, log=True),
             "n_estimators": trial.suggest_int("n_estimators", 300, 2000, step=100),
             "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
             "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 4.0)}
        return own_oof_auc(p)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    t0 = time.time(); study.optimize(obj, n_trials=nt)
    best = {**base_params, **study.best_params}
    print(f"  search best own OOF AUC={study.best_value:.4f} (Δ{study.best_value-base_auc:+.4f}, {time.time()-t0:.0f}s)", flush=True)

    def own_oof_test(Xtr_pp, Xte_pp, ytr, params):
        tp, oof = lgb_oof_and_test(Xtr_pp, ytr, Xte_pp, params, n_seeds=5, n_folds=5)
        pf = fit_platt(oof, ytr.values); return apply_platt(tp, pf), apply_platt(oof, pf)

    rows = []
    for seed in (SEED + 1000, SEED + 2000):
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=8, random_state=seed)
        f_base, f_tun = [], []
        for tr, te in rskf.split(Xt, y):
            tr, te = np.asarray(tr), np.asarray(te); ytr, yte = y.iloc[tr], y.iloc[te]
            Xtr_pp, Xte_pp = own_preprocess(Xt.iloc[tr].reset_index(drop=True), ytr.reset_index(drop=True),
                                            Xt.iloc[te].reset_index(drop=True))
            pl = fit_platt(g[tr], ytr.values); gtr, gte = apply_platt(g[tr], pl), apply_platt(g[te], pl)
            for params, acc in ((base_params, f_base), (best, f_tun)):
                e_te, e_oof = own_oof_test(Xtr_pp, Xte_pp, ytr, params)
                b_te = 0.5 * gte + 0.5 * e_te
                acc.append(f1_at_prior(*tpr_fpr(yte.values, (b_te >= THR).astype(int)), prior))
        b, t = np.array(f_base), np.array(f_tun); d = t - b; nf = len(b)
        sig = d.mean()/(d.std(ddof=1)/np.sqrt(nf)) if d.std() > 0 else 0
        print(f"  gate seed{seed}: base F1*={b.mean():.4f}  tuned F1*={t.mean():.4f}  Δ={d.mean():+.4f} ({sig:+.2f}σ)", flush=True)
        rows.append({"seed": seed, "base_f1": b.mean(), "tuned_f1": t.mean(), "delta": d.mean(),
                     "sigma": sig, "best_params": str(study.best_params)})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "cancer_own_hpo.csv"), index=False)
    print(f"\nsaved -> cancer_own_hpo.csv", flush=True)


if __name__ == "__main__":
    nt = int(sys.argv[sys.argv.index("--trials") + 1]) if "--trials" in sys.argv else 40
    main(nt)
