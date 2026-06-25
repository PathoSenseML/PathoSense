"""FULL Optuna parameter optimization for the general model (the data-rich own-model vein; small panels
are transfer-scorer-locked and scorer-HPO is already a tested reject). Two phases + honest gate:
  Phase 1: Optuna on the general LGBM params (objective = 5-fold OOF AUC), N trials.
  Phase 2: Optuna on the general XGBoost member params (objective = 5-fold OOF AUC), N trials — the XGB
           member was added with untuned defaults, so this is the freshest target.
  Phase 3: HONEST GATE — confirm the tuned 4-way blend (tuned-LGBM + RF + CatBoost + tuned-XGB) vs the
           current deployed blend on HELD-OUT seeds (not the search seed), F1* @ prior + AUC, paired.
Deploy only if the tuned blend robustly beats current (non-negative both held-out seeds).
Run: .venv/bin/python -u analysis/exp_general_hpo_full.py [n_trials]
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from config import SEED
from utils import TEST_PRIORS
from schema import partition_columns
from calibration import fit_platt, apply_platt
from harness import load_panel, _drop_leaky, lgb_oof_and_test, pick_threshold_at_prior, tpr_fpr, f1_at_prior
from preprocess_v2 import fit_keep_nan, apply_keep_nan
from general_blend import _rf_repr, _cat_repr, _cat_model
from run_pipeline import get_model_params
from meeting_lib import native_nan_oof

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
df = load_panel("general"); _, NUM, CAT, TGT = partition_columns(df)
Xa = df[NUM + CAT].reset_index(drop=True); ya = df[TGT].astype(int).reset_index(drop=True)
SPW = (ya == 0).sum() / max((ya == 1).sum(), 1)
LGB_BASE = {**get_model_params(2000, SPW), "scale_pos_weight": 1.0, "bagging_freq": 1, "bagging_fraction": 0.8}
XGB_BASE = dict(n_estimators=400, learning_rate=0.03, max_depth=5, subsample=0.8,
                colsample_bytree=0.8, reg_lambda=1.5, eval_metric="logloss", n_jobs=4, random_state=SEED)

# preprocess once on full data for the SEARCH objective only (honest gate re-does per fold)
_Xl, _ppf = fit_keep_nan(_drop_leaky(Xa, ya, Xa)[0])
_CATF = [c for c in _Xl.columns if str(_Xl[c].dtype) == "category"]
_Xo, _ = _rf_repr(_Xl, _CATF, fit=True)


def lgbm_obj(trial):
    p = {**LGB_BASE,
         "num_leaves": trial.suggest_int("num_leaves", 16, 80),
         "max_depth": trial.suggest_int("max_depth", 4, 10),
         "min_child_samples": trial.suggest_int("min_child_samples", 5, 80),
         "learning_rate": trial.suggest_float("learning_rate", 0.008, 0.05, log=True),
         "n_estimators": trial.suggest_int("n_estimators", 600, 3500, step=100),
         "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
         "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
         "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 4.0),
         "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.3)}
    return roc_auc_score(ya.values, native_nan_oof(Xa, ya, Xa, params=p, n_seeds=1)[1])


def _xgb_oof(params):
    oof = np.zeros(len(ya)); skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for a, b in skf.split(_Xo, ya):
        m = XGBClassifier(**params); m.fit(_Xo.iloc[a], ya.iloc[a])
        oof[b] = m.predict_proba(_Xo.iloc[b])[:, 1]
    return roc_auc_score(ya.values, oof)


def xgb_obj(trial):
    p = {**XGB_BASE,
         "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
         "max_depth": trial.suggest_int("max_depth", 3, 9),
         "subsample": trial.suggest_float("subsample", 0.5, 1.0),
         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
         "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
         "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
         "min_child_weight": trial.suggest_int("min_child_weight", 1, 10)}
    return _xgb_oof(p)


def blend_fold(Xtr, ytr, Xte, lgb_p, xgb_kw):
    Xtr, Xte = _drop_leaky(Xtr, ytr, Xte)
    Xtr_pp, ppf = fit_keep_nan(Xtr); Xte_pp = apply_keep_nan(Xte, ppf)
    for c in Xtr_pp.columns:
        if str(Xtr_pp[c].dtype) == "category":
            Xte_pp[c] = pd.Categorical(Xte_pp[c].astype(object), categories=Xtr_pp[c].cat.categories)
    cf = [c for c in Xtr_pp.columns if str(Xtr_pp[c].dtype) == "category"]
    Xo_tr, enc = _rf_repr(Xtr_pp, cf, fit=True); Xo_te, _ = _rf_repr(Xte_pp, cf, enc=enc, fit=False)
    Xc_tr, Xc_te = _cat_repr(Xtr_pp, cf), _cat_repr(Xte_pp, cf); ci = [Xc_tr.columns.get_loc(c) for c in cf]
    n = len(ytr); lt, lo = lgb_oof_and_test(Xtr_pp, ytr, Xte_pp, lgb_p, n_seeds=3, n_folds=5)
    lpl = fit_platt(lo, ytr.values); lo, lt = apply_platt(lo, lpl), apply_platt(lt, lpl)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED); rf_o = np.zeros(n); xg_o = np.zeros(n); cb_o = np.zeros(n)
    spw = (ytr == 0).sum()/max((ytr == 1).sum(), 1)
    for k, (a, b) in enumerate(skf.split(Xo_tr, ytr)):
        rf = RandomForestClassifier(n_estimators=600, max_features="sqrt", n_jobs=4, class_weight="balanced",
                                    random_state=SEED).fit(Xo_tr.iloc[a], ytr.iloc[a]); rf_o[b] = rf.predict_proba(Xo_tr.iloc[b])[:, 1]
        xg = XGBClassifier(**xgb_kw).fit(Xo_tr.iloc[a], ytr.iloc[a]); xg_o[b] = xg.predict_proba(Xo_tr.iloc[b])[:, 1]
        i2, ies = train_test_split(a, test_size=0.15, random_state=SEED+k, stratify=ytr.iloc[a])
        cb = _cat_model(ci, spw); cb.fit(Xc_tr.iloc[i2], ytr.iloc[i2], eval_set=(Xc_tr.iloc[ies], ytr.iloc[ies])); cb_o[b] = cb.predict_proba(Xc_tr.iloc[b])[:, 1]
    rpl, xpl, cpl = fit_platt(rf_o, ytr.values), fit_platt(xg_o, ytr.values), fit_platt(cb_o, ytr.values)
    rf_f = RandomForestClassifier(n_estimators=600, max_features="sqrt", n_jobs=4, class_weight="balanced", random_state=SEED).fit(Xo_tr, ytr)
    xg_f = XGBClassifier(**xgb_kw).fit(Xo_tr, ytr)
    i2, ies = train_test_split(np.arange(n), test_size=0.15, random_state=SEED+99, stratify=ytr)
    cb_f = _cat_model(ci, spw); cb_f.fit(Xc_tr.iloc[i2], ytr.iloc[i2], eval_set=(Xc_tr.iloc[ies], ytr.iloc[ies]))
    oof = (lo + apply_platt(rf_o, rpl) + apply_platt(cb_o, cpl) + apply_platt(xg_o, xpl)) / 4.0
    test = (lt + apply_platt(rf_f.predict_proba(Xo_te)[:, 1], rpl) + apply_platt(cb_f.predict_proba(Xc_te)[:, 1], cpl)
            + apply_platt(xg_f.predict_proba(Xo_te)[:, 1], xpl)) / 4.0
    return oof, test


if __name__ == "__main__":
    nt = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    prior = TEST_PRIORS["general"]
    base_lgb_auc = roc_auc_score(ya.values, native_nan_oof(Xa, ya, Xa, params=LGB_BASE, n_seeds=1)[1])
    base_xgb_auc = _xgb_oof(XGB_BASE)
    print(f"=== FULL HPO general (LGB base OOF AUC={base_lgb_auc:.4f}, XGB base OOF AUC={base_xgb_auc:.4f}) ===", flush=True)
    s1 = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)); s1.optimize(lgbm_obj, n_trials=nt)
    s2 = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)); s2.optimize(xgb_obj, n_trials=nt)
    print(f"  LGBM search best OOF AUC={s1.best_value:.4f} (Δ{s1.best_value-base_lgb_auc:+.4f})  {s1.best_params}", flush=True)
    print(f"  XGB  search best OOF AUC={s2.best_value:.4f} (Δ{s2.best_value-base_xgb_auc:+.4f})  {s2.best_params}", flush=True)
    lgb_tuned = {**LGB_BASE, **s1.best_params}; xgb_tuned = {**XGB_BASE, **s2.best_params}
    print("\n  HONEST GATE — tuned 4-way blend vs current, held-out seeds:", flush=True)
    for seed in (SEED + 1000, SEED + 2000):
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=seed)
        cur_f, cur_a, tun_f, tun_a = [], [], [], []
        for tr, te in rskf.split(Xa, ya):
            tr, te = np.asarray(tr), np.asarray(te)
            Xtr, Xte = Xa.iloc[tr].reset_index(drop=True), Xa.iloc[te].reset_index(drop=True)
            ytr, yte = ya.iloc[tr].reset_index(drop=True), ya.iloc[te].values
            for lgbp, xgbp, fl, al in ((LGB_BASE, XGB_BASE, cur_f, cur_a), (lgb_tuned, xgb_tuned, tun_f, tun_a)):
                oof, test = blend_fold(Xtr, ytr, Xte, lgbp, xgbp)
                t = pick_threshold_at_prior(ytr.values, oof, prior)
                fl.append(f1_at_prior(*tpr_fpr(yte, (test >= t).astype(int)), prior)); al.append(roc_auc_score(yte, test))
        cf, ca, tf, ta = map(np.array, (cur_f, cur_a, tun_f, tun_a)); nf = len(cf)
        df1 = tf - cf; da = ta - ca
        sig = df1.mean()/(df1.std(ddof=1)/np.sqrt(nf)) if df1.std() > 0 else 0
        siga = da.mean()/(da.std(ddof=1)/np.sqrt(nf)) if da.std() > 0 else 0
        print(f"    seed{seed} ({nf}f): current F1*={cf.mean():.4f} AUC={ca.mean():.4f} | tuned F1*={tf.mean():.4f} "
              f"(Δ{df1.mean():+.4f},{sig:+.2f}σ) AUC={ta.mean():.4f} (Δ{da.mean():+.4f},{siga:+.2f}σ)", flush=True)
    import json
    json.dump({"lgbm": s1.best_params, "xgb": s2.best_params}, open(os.path.join(OUT, "general_hpo_full_best.json"), "w"), indent=2)
    print("\nsaved -> general_hpo_full_best.json", flush=True)
