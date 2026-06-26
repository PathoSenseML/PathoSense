"""JOINT pipeline optimization (vs the greedy deployed config). Co-optimize, as ONE object via Optuna:
impute strategy x missingness indicators x feature selection x model family x hyperparameters — instead of
the greedy 'fix preprocessing, then model, then tune' path. Does the joint optimum beat the greedy deployed
config (~AUC 0.864 / F1* 0.54)? Non-trees excluded (cap at 0.81). Threshold stays F1-opt-at-prior (optimal).

Honesty: Optuna optimizes INNER 4-fold CV-AUC (one seed). The winner is then evaluated on the FULL honest
harness with a DIFFERENT seed (fresh folds) + paired vs the deployed base — so the reported delta is not the
(upward-biased) search score. Run: .venv/bin/python -u analysis/exp_joint_opt.py
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "analysis"); sys.path.insert(0, "pipelines")
import warnings; warnings.filterwarnings("ignore")
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier
import harness
from harness import cv_evaluate, load_panel, compare, _drop_leaky, fast_mrmr
from schema import partition_columns
from meeting_lib import native_nan_oof
from preprocess_v2 import fit_keep_nan, apply_keep_nan
from general_blend import _rf_repr, _cat_repr, _cats
from run_pipeline import get_model_params
from config import SEED

df = load_panel("general"); _, NUM, CAT, TGT = partition_columns(df)
X_all = df[NUM + CAT].reset_index(drop=True); Y = df[TGT].astype(int).reset_index(drop=True)
DEP_P = {**get_model_params(2000, (Y == 0).sum() / max(Y.sum(), 1)),
         "scale_pos_weight": 1, "bagging_freq": 1, "bagging_fraction": 0.8}


def build(Xtr, ytr, Xte, cfg):
    """Per-fold pipeline preprocessing: native-NaN base, then optional impute/miss-ind/feature-select,
    then model-appropriate categorical repr."""
    _, ppf = fit_keep_nan(Xtr); Ptr = apply_keep_nan(Xtr, ppf); Pte = apply_keep_nan(Xte, ppf)
    catf = [c for c in Ptr.columns if isinstance(Ptr[c].dtype, pd.CategoricalDtype)]
    num = [c for c in Ptr.columns if c not in catf]
    if cfg["impute"] != "native":
        if cfg["miss_ind"]:
            sp = [c for c in num if Ptr[c].isna().mean() > 0.05]
            for c in sp:
                Ptr[f"MI_{c}"] = Ptr[c].isna().astype(float); Pte[f"MI_{c}"] = Pte[c].isna().astype(float)
        fill = (pd.Series(0.0, index=num) if cfg["impute"] == "zero" else Ptr[num].median())
        Ptr[num] = Ptr[num].fillna(fill); Pte[num] = Pte[num].fillna(fill)
    if cfg["fselect"] != "all":
        k = int(cfg["fselect"][4:])
        Ptr, _, Pte, _ = fast_mrmr(Ptr, ytr, Pte, Pte, max_features=k)
        catf = [c for c in catf if c in Ptr.columns]
    if cfg["model"] in ("rf", "xgb"):
        Atr, enc = _rf_repr(Ptr, catf, fit=True); Ate, _ = _rf_repr(Pte, catf, enc=enc, fit=False)
        return Atr, Ate, None
    return Ptr, Pte, catf


def make(cfg, seed):
    m = cfg["model"]
    if m == "lgbm":
        return lgb.LGBMClassifier(n_estimators=cfg["n_est"], learning_rate=cfg["lr"], num_leaves=cfg["leaves"],
            min_child_samples=cfg["mcs"], reg_lambda=cfg["l2"], subsample=cfg["sub"], bagging_freq=1,
            colsample_bytree=cfg["col"], scale_pos_weight=cfg["spw"], n_jobs=2, verbose=-1, random_state=seed)
    if m == "catboost":
        return CatBoostClassifier(iterations=cfg["n_est"], depth=cfg["depth"], learning_rate=cfg["lr"],
            l2_leaf_reg=cfg["l2"], scale_pos_weight=cfg["spw"], random_seed=seed, verbose=0,
            allow_writing_files=False, thread_count=2)
    if m == "rf":
        return RandomForestClassifier(n_estimators=cfg["n_est"], max_depth=cfg["depth"], max_features=cfg["mf"],
            min_samples_leaf=cfg["leaf"], class_weight="balanced", n_jobs=2, random_state=seed)
    return xgb.XGBClassifier(n_estimators=cfg["n_est"], max_depth=cfg["depth"], learning_rate=cfg["lr"],
        reg_lambda=cfg["l2"], subsample=cfg["sub"], colsample_bytree=cfg["col"], scale_pos_weight=cfg["spw"],
        eval_metric="logloss", n_jobs=2, random_state=seed, tree_method="hist")


def sample_cfg(t):
    spw = float((Y == 0).sum() / max(Y.sum(), 1))
    cfg = {"impute": t.suggest_categorical("impute", ["native", "zero", "median"]),
           "miss_ind": t.suggest_categorical("miss_ind", [True, False]),
           "fselect": t.suggest_categorical("fselect", ["all", "mrmr120", "mrmr200", "mrmr300"]),
           "model": t.suggest_categorical("model", ["lgbm", "catboost", "rf", "xgb"]),
           "spw": t.suggest_categorical("spw", [1.0, spw])}
    m = cfg["model"]
    if m == "lgbm":
        cfg.update(n_est=t.suggest_int("l_n", 400, 1500), lr=t.suggest_float("l_lr", 0.01, 0.1, log=True),
                   leaves=t.suggest_int("l_lv", 15, 127), mcs=t.suggest_int("l_mcs", 5, 60),
                   l2=t.suggest_float("l_l2", 0, 5), sub=t.suggest_float("l_sub", 0.6, 1.0),
                   col=t.suggest_float("l_col", 0.6, 1.0))
    elif m == "catboost":
        cfg.update(n_est=t.suggest_int("c_n", 500, 1500), depth=t.suggest_int("c_d", 4, 8),
                   lr=t.suggest_float("c_lr", 0.01, 0.1, log=True), l2=t.suggest_float("c_l2", 1, 10))
    elif m == "rf":
        cfg.update(n_est=t.suggest_int("r_n", 400, 1200), depth=t.suggest_int("r_d", 6, 22),
                   mf=t.suggest_float("r_mf", 0.2, 0.6), leaf=t.suggest_int("r_leaf", 2, 20))
    else:
        cfg.update(n_est=t.suggest_int("x_n", 300, 1000), depth=t.suggest_int("x_d", 3, 8),
                   lr=t.suggest_float("x_lr", 0.01, 0.1, log=True), l2=t.suggest_float("x_l2", 0, 5),
                   sub=t.suggest_float("x_sub", 0.6, 1.0), col=t.suggest_float("x_col", 0.6, 1.0))
    return cfg


def cv_auc(cfg):
    Xd, _ = _drop_leaky(X_all, Y, X_all)
    aucs = []
    for a, b in StratifiedKFold(4, shuffle=True, random_state=SEED).split(Xd, Y):
        Atr, Ate, catf = build(Xd.iloc[a].reset_index(drop=True), Y.iloc[a].reset_index(drop=True),
                               Xd.iloc[b].reset_index(drop=True), cfg)
        m = make(cfg, SEED)
        if cfg["model"] == "lgbm":
            m.fit(Atr, Y.iloc[a], categorical_feature=catf or "auto")
        elif cfg["model"] == "catboost":
            m.fit(Atr, Y.iloc[a], cat_features=[Atr.columns.get_loc(c) for c in (catf or [])])
        else:
            m.fit(Atr, Y.iloc[a])
        aucs.append(roc_auc_score(Y.iloc[b], m.predict_proba(Ate)[:, 1]))
    return float(np.mean(aucs))


def winner_pipeline(cfg):
    def fn(X_tr, y_tr, X_te):
        Xtr, Xte = _drop_leaky(X_tr.reset_index(drop=True), y_tr.reset_index(drop=True), X_te.reset_index(drop=True))
        y = y_tr.reset_index(drop=True).astype(int)
        oof = np.zeros(len(y)); test = []
        for s in range(3):
            skf = StratifiedKFold(5, shuffle=True, random_state=SEED + s); o = np.zeros(len(y))
            for a, b in skf.split(Xtr, y):
                Atr, Ab, catf = build(Xtr.iloc[a].reset_index(drop=True), y.iloc[a].reset_index(drop=True),
                                      Xtr.iloc[b].reset_index(drop=True), cfg)
                m = make(cfg, SEED + s)
                if cfg["model"] == "lgbm": m.fit(Atr, y.iloc[a], categorical_feature=catf or "auto")
                elif cfg["model"] == "catboost": m.fit(Atr, y.iloc[a], cat_features=[Atr.columns.get_loc(c) for c in (catf or [])])
                else: m.fit(Atr, y.iloc[a])
                o[b] = m.predict_proba(Ab)[:, 1]
            Atr, Ate, catf = build(Xtr, y, Xte, cfg); mf = make(cfg, SEED + s)
            if cfg["model"] == "lgbm": mf.fit(Atr, y, categorical_feature=catf or "auto")
            elif cfg["model"] == "catboost": mf.fit(Atr, y, cat_features=[Atr.columns.get_loc(c) for c in (catf or [])])
            else: mf.fit(Atr, y)
            test.append(mf.predict_proba(Ate)[:, 1]); oof += o / 3
        return np.mean(test, 0), oof
    return fn


def main():
    N = 100
    print(f"=== JOINT pipeline optimization: Optuna {N} trials (inner 4-fold CV-AUC) ===", flush=True)
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    done = [0]
    def obj(t):
        cfg = sample_cfg(t)
        try: v = cv_auc(cfg)
        except Exception as e: v = 0.5
        done[0] += 1
        if done[0] % 10 == 0: print(f"  trial {done[0]}/{N}  best CV-AUC so far={st.best_value:.4f}", flush=True)
        return v
    st.optimize(obj, n_trials=N, show_progress_bar=False)
    bp = st.best_params
    # reconstruct cfg from best_params
    spw = float((Y == 0).sum() / max(Y.sum(), 1))
    cfg = {"impute": bp["impute"], "miss_ind": bp["miss_ind"], "fselect": bp["fselect"],
           "model": bp["model"], "spw": bp["spw"]}
    pref = {"lgbm": "l_", "catboost": "c_", "rf": "r_", "xgb": "x_"}[bp["model"]]
    keymap = {"lgbm": dict(l_n="n_est", l_lr="lr", l_lv="leaves", l_mcs="mcs", l_l2="l2", l_sub="sub", l_col="col"),
              "catboost": dict(c_n="n_est", c_d="depth", c_lr="lr", c_l2="l2"),
              "rf": dict(r_n="n_est", r_d="depth", r_mf="mf", r_leaf="leaf"),
              "xgb": dict(x_n="n_est", x_d="depth", x_lr="lr", x_l2="l2", x_sub="sub", x_col="col")}[bp["model"]]
    for k, v in bp.items():
        if k in keymap: cfg[keymap[k]] = v
    print(f"\nBEST joint config (CV-AUC {st.best_value:.4f}): {cfg}\n", flush=True)
    print("=== HONEST eval on FRESH folds (different seed), paired vs deployed ===", flush=True)
    nrep = 4
    base = cv_evaluate("general", lambda a, b, c: native_nan_oof(a, b, c, params=DEP_P),
                       label="deployed_lgbm", n_repeats=nrep, df=df, n_jobs=6)
    win = cv_evaluate("general", winner_pipeline(cfg), label="joint_winner", n_repeats=nrep, df=df, n_jobs=4)
    c = compare(base, win)
    print(f"  deployed LGBM base : F1*={base.f1.mean():.4f} AUC={base.auc.mean():.4f}", flush=True)
    print(f"  joint winner       : F1*={win.f1.mean():.4f} AUC={win.auc.mean():.4f} "
          f"ΔF1={c['delta_f1']:+.4f} ({c['sigma']:+.2f}σ) ΔAUC={win.auc.mean()-base.auc.mean():+.4f} [{c['verdict']}]", flush=True)
    print(f"\n  (deployed BLEND bar ~AUC 0.864 / F1* ~0.54 — joint winner is a single model)", flush=True)
    pd.DataFrame([{"best_cv_auc": round(st.best_value, 4), **cfg,
                   "honest_auc": round(win.auc.mean(), 4), "honest_f1": round(win.f1.mean(), 4),
                   "d_f1_vs_base": c["delta_f1"], "sigma": c["sigma"]}]).to_csv("analysis/output/joint_opt.csv", index=False)
    print("saved -> analysis/output/joint_opt.csv", flush=True)


if __name__ == "__main__":
    main()
