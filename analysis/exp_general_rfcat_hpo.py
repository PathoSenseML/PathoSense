"""Overnight: Optuna HPO of the general blend's last-untuned members — RandomForest and CatBoost — on
5-fold OOF AUC (cheap single-model search). LGBM/XGB HPO already gave tiny AUC + sign-unstable F1; this
documents RF/CatBoost. If either OOF AUC gain is large (>0.004) gate it in the blend; else inert (the
established AUC-ceiling pattern says tiny AUC won't convert to F1).
Run: .venv/bin/python -u analysis/exp_general_rfcat_hpo.py [--trials N]
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from config import SEED
from schema import partition_columns
from harness import load_panel, _drop_leaky
from preprocess_v2 import fit_keep_nan
from general_blend import _rf_repr, _cat_repr

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
df = load_panel("general"); _, NUM, CAT, TGT = partition_columns(df)
Xa = df[NUM + CAT].reset_index(drop=True); ya = df[TGT].astype(int).reset_index(drop=True)
SPW = (ya == 0).sum() / max((ya == 1).sum(), 1)
_Xl, _ = fit_keep_nan(_drop_leaky(Xa, ya, Xa)[0])
_CATF = [c for c in _Xl.columns if str(_Xl[c].dtype) == "category"]
_Xo, _ = _rf_repr(_Xl, _CATF, fit=True)
_Xc = _cat_repr(_Xl, _CATF); _CIDX = [_Xc.columns.get_loc(c) for c in _CATF]
SKF = StratifiedKFold(5, shuffle=True, random_state=SEED)


def rf_auc(p):
    oof = np.zeros(len(ya))
    for a, b in SKF.split(_Xo, ya):
        m = RandomForestClassifier(**p, n_jobs=4, class_weight="balanced", random_state=SEED).fit(_Xo.iloc[a], ya.iloc[a])
        oof[b] = m.predict_proba(_Xo.iloc[b])[:, 1]
    return roc_auc_score(ya.values, oof)


def cat_auc(p):
    oof = np.zeros(len(ya))
    for k, (a, b) in enumerate(SKF.split(_Xc, ya)):
        i2, ies = train_test_split(a, test_size=0.15, random_state=SEED + k, stratify=ya.iloc[a])
        m = CatBoostClassifier(**p, scale_pos_weight=SPW, cat_features=_CIDX, random_seed=SEED,
                               verbose=0, allow_writing_files=False, early_stopping_rounds=40, thread_count=4)
        m.fit(_Xc.iloc[i2], ya.iloc[i2], eval_set=(_Xc.iloc[ies], ya.iloc[ies]))
        oof[b] = m.predict_proba(_Xc.iloc[b])[:, 1]
    return roc_auc_score(ya.values, oof)


if __name__ == "__main__":
    nt = int(sys.argv[sys.argv.index("--trials") + 1]) if "--trials" in sys.argv else 30
    base_rf = rf_auc(dict(n_estimators=600, max_features="sqrt"))
    base_cat = cat_auc(dict(iterations=1000, depth=6, learning_rate=0.05))
    print(f"=== general RF/CatBoost OOF-AUC HPO ({nt} trials); base RF AUC={base_rf:.4f} CatBoost AUC={base_cat:.4f} ===", flush=True)

    def rf_obj(t):
        return rf_auc(dict(n_estimators=t.suggest_int("n_estimators", 300, 1200, step=100),
                           max_features=t.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
                           max_depth=t.suggest_int("max_depth", 6, 30),
                           min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 20)))

    def cat_obj(t):
        return cat_auc(dict(iterations=t.suggest_int("iterations", 400, 1500, step=100),
                            depth=t.suggest_int("depth", 4, 8),
                            learning_rate=t.suggest_float("learning_rate", 0.02, 0.1, log=True),
                            l2_leaf_reg=t.suggest_float("l2_leaf_reg", 1.0, 10.0)))

    t0 = time.time()
    s_rf = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)); s_rf.optimize(rf_obj, n_trials=nt)
    s_cat = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)); s_cat.optimize(cat_obj, n_trials=nt)
    print(f"  RF  best OOF AUC={s_rf.best_value:.4f} (Δ{s_rf.best_value-base_rf:+.4f})  {s_rf.best_params}", flush=True)
    print(f"  Cat best OOF AUC={s_cat.best_value:.4f} (Δ{s_cat.best_value-base_cat:+.4f})  {s_cat.best_params}", flush=True)
    print(f"  ({time.time()-t0:.0f}s)  verdict: RF Δ={s_rf.best_value-base_rf:+.4f}, Cat Δ={s_cat.best_value-base_cat:+.4f} "
          f"-> {'GATE-WORTHY' if max(s_rf.best_value-base_rf, s_cat.best_value-base_cat) > 0.004 else 'tiny AUC, inert (no blend-F1 gain expected)'}", flush=True)
    import json
    json.dump({"rf": s_rf.best_params, "cat": s_cat.best_params}, open(os.path.join(OUT, "general_rfcat_hpo_best.json"), "w"), indent=2)
    print("saved -> general_rfcat_hpo_best.json", flush=True)
