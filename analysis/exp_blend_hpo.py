"""HPO the never-tuned blend members (RF, CatBoost) on general. (user: all params tested?)
RF/CatBoost run at hardcoded defaults in general_blend.py / transfer.py. Optuna-tune each by general
3-fold OOF AUC, then VALIDATE the equal-3 blend with tuned members vs default-member blend (honest
per-fold CV, F1* at prior, paired). Deploy only if >=2sigma.

Run:  .venv/bin/python -u analysis/exp_blend_hpo.py [--trials 25]
"""
import os, sys, time, argparse
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

from config import SEED
from harness import cv_evaluate, load_panel, compare, lgb_oof_and_test, _drop_leaky
from exp_general_blend import _cats
from preprocess_v2 import fit_keep_nan, apply_keep_nan
from calibration import fit_platt, apply_platt
from run_pipeline import get_model_params


def _ord(Xtr, Xte, cats):
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-2)
    a, b = Xtr.copy(), Xte.copy()
    if cats:
        a[cats] = enc.fit_transform(Xtr[cats].astype(object).where(Xtr[cats].notna(), "NA"))
        b[cats] = enc.transform(Xte[cats].astype(object).where(Xte[cats].notna(), "NA"))
    return a.fillna(-999), b.fillna(-999)


def _rf_oof_test(Xtr, ytr, Xte, spw, params, nf=5):
    cats = _cats(Xtr); a, b = _ord(Xtr, Xte, cats); oof = np.zeros(len(ytr))
    skf = StratifiedKFold(nf, shuffle=True, random_state=SEED)
    for i, j in skf.split(a, ytr):
        m = RandomForestClassifier(**params, n_jobs=4, class_weight="balanced", random_state=SEED)
        m.fit(a.iloc[i], ytr.iloc[i]); oof[j] = m.predict_proba(a.iloc[j])[:, 1]
    m = RandomForestClassifier(**params, n_jobs=4, class_weight="balanced", random_state=SEED).fit(a, ytr)
    return oof, m.predict_proba(b)[:, 1]


def _cat_oof_test(Xtr, ytr, Xte, spw, params, nf=5):
    cats = _cats(Xtr); a, b = Xtr.copy(), Xte.copy()
    for c in cats:
        a[c] = Xtr[c].astype(object).where(Xtr[c].notna(), "NA").astype(str)
        b[c] = Xte[c].astype(object).where(Xte[c].notna(), "NA").astype(str)
    idx = [a.columns.get_loc(c) for c in cats]; oof = np.zeros(len(ytr))
    skf = StratifiedKFold(nf, shuffle=True, random_state=SEED)
    for k, (i, j) in enumerate(skf.split(a, ytr)):
        i2, ies = train_test_split(i, test_size=0.15, random_state=SEED + k, stratify=ytr.iloc[i])
        m = CatBoostClassifier(**params, scale_pos_weight=spw, cat_features=idx, random_seed=SEED,
                               verbose=0, allow_writing_files=False, early_stopping_rounds=40, thread_count=4)
        m.fit(a.iloc[i2], ytr.iloc[i2], eval_set=(a.iloc[ies], ytr.iloc[ies]))
        oof[j] = m.predict_proba(a.iloc[j])[:, 1]
    i2, ies = train_test_split(np.arange(len(ytr)), test_size=0.15, random_state=SEED + 99, stratify=ytr)
    m = CatBoostClassifier(**params, scale_pos_weight=spw, cat_features=idx, random_seed=SEED,
                           verbose=0, allow_writing_files=False, early_stopping_rounds=40, thread_count=4)
    m.fit(a.iloc[i2], ytr.iloc[i2], eval_set=(a.iloc[ies], ytr.iloc[ies]))
    return oof, m.predict_proba(b)[:, 1]


# preprocess general once for fast HPO objective (validated honestly afterwards)
_df = load_panel("general")
from schema import partition_columns
_, _nc, _cc, _t = partition_columns(_df)
_X = _df[_nc + _cc].reset_index(drop=True); _y = _df[_t].astype(int).reset_index(drop=True)
_Xl, _ = _drop_leaky(_X, _y, _X.copy())
_Xpp, _ = fit_keep_nan(_Xl)
_spw = (_y == 0).sum() / max((_y == 1).sum(), 1)


def hpo_rf(n_trials):
    def obj(t):
        p = dict(n_estimators=t.suggest_int("n_estimators", 300, 1200, step=100),
                 max_features=t.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
                 max_depth=t.suggest_categorical("max_depth", [None, 8, 12, 20]),
                 min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 8),
                 min_samples_split=t.suggest_int("min_samples_split", 2, 12))
        o, _ = _rf_oof_test(_Xpp, _y, _Xpp, _spw, p, nf=3)
        return roc_auc_score(_y, o)
    s = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    s.optimize(obj, n_trials=n_trials); return s.best_params, s.best_value


def hpo_cat(n_trials):
    def obj(t):
        p = dict(iterations=t.suggest_int("iterations", 400, 1500, step=100),
                 depth=t.suggest_int("depth", 4, 8),
                 learning_rate=t.suggest_float("learning_rate", 0.01, 0.1, log=True),
                 l2_leaf_reg=t.suggest_float("l2_leaf_reg", 1.0, 10.0),
                 random_strength=t.suggest_float("random_strength", 0.0, 2.0),
                 bagging_temperature=t.suggest_float("bagging_temperature", 0.0, 1.0))
        o, _ = _cat_oof_test(_Xpp, _y, _Xpp, _spw, p, nf=3)
        return roc_auc_score(_y, o)
    s = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    s.optimize(obj, n_trials=n_trials); return s.best_params, s.best_value


RF_DEF = dict(n_estimators=600, max_features="sqrt", max_depth=None, min_samples_leaf=1, min_samples_split=2)
CAT_DEF = dict(iterations=1000, depth=6, learning_rate=0.05, l2_leaf_reg=3.0)


def make_blend(rf_p, cat_p):
    def _fn(X_tr, y_tr, X_te):
        Xtr, Xte = _drop_leaky(X_tr, y_tr, X_te)
        Xtr_pp, ppf = fit_keep_nan(Xtr); Xte_pp = apply_keep_nan(Xte, ppf)
        spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        params = get_model_params(len(X_tr), spw)
        L_t, L_o = lgb_oof_and_test(Xtr_pp, y_tr, Xte_pp, params, n_seeds=2, n_folds=5)
        R_o, R_t = _rf_oof_test(Xtr_pp, y_tr, Xte_pp, spw, rf_p)
        C_o, C_t = _cat_oof_test(Xtr_pp, y_tr, Xte_pp, spw, cat_p)
        def cal(o, t): pf = fit_platt(o, y_tr.values); return apply_platt(o, pf), apply_platt(t, pf)
        Lo, Lt = cal(L_o, L_t); Ro, Rt = cal(R_o, R_t); Co, Ct = cal(C_o, C_t)
        return (Lt + Rt + Ct) / 3, (Lo + Ro + Co) / 3
    return _fn


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--trials", type=int, default=25); a = ap.parse_args()
    t0 = time.time()
    rf_best, rf_auc = hpo_rf(a.trials)
    print(f"[RF]  best OOF AUC={rf_auc:.4f}  default OOF AUC="
          f"{roc_auc_score(_y, _rf_oof_test(_Xpp,_y,_Xpp,_spw,RF_DEF,nf=3)[0]):.4f}  {rf_best}", flush=True)
    cat_best, cat_auc = hpo_cat(a.trials)
    print(f"[CAT] best OOF AUC={cat_auc:.4f}  default OOF AUC="
          f"{roc_auc_score(_y, _cat_oof_test(_Xpp,_y,_Xpp,_spw,CAT_DEF,nf=3)[0]):.4f}  {cat_best}", flush=True)
    print(f"  (HPO {time.time()-t0:.0f}s)\n  honest blend validation (F1* at prior, paired):", flush=True)
    df = load_panel("general")
    ref = cv_evaluate("general", make_blend(RF_DEF, CAT_DEF), label="default", n_repeats=5, df=df)
    print(f"  default-members equal3 F1*={ref.summary()['f1_mean']:.4f}", flush=True)
    tuned = cv_evaluate("general", make_blend(rf_best, cat_best), label="tuned", n_repeats=5, df=df)
    c = compare(ref, tuned)
    print(f"  tuned-members   equal3 F1*={c['b_f1']:.4f}  Δ={c['delta_f1']:+.4f} ({c['sigma']:+.2f}σ)  {c['verdict']}", flush=True)
