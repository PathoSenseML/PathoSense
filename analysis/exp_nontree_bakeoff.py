"""Model bake-off + non-tree embedding + the log-scaling fix — three asks, one harness.

(1) BAKE-OFF: every model class as a panel own-model, honest CV F1* at the test prior.
(2) NON-TREE EMBEDDING: sklearn MLP on scaled features, and an MLP on a PCA embedding of the
    334-col AL population vector (a learned-ish low-dim profile trees can't form).
(3) LOG-SCALING FIX (Almira's win): log is RANK-invariant for trees, but allele frequencies span
    many orders of magnitude, so log10 genuinely linearizes them for SCALE-SENSITIVE models
    (LogReg/SVM/MLP/KNN). We test raw-scale vs log10-AF for those — the correct place log helps.

Preprocessing is per model family (each model at its best):
  native   LGBM, CatBoost           — native categoricals + native NaN
  num_nan  XGBoost, HistGB          — label-encoded cats, NaN kept (both split on missing)
  num_fill RandomForest, ExtraTrees — label-encoded cats, NaN -> -999
  scaled   LogReg, SVM, KNN, MLP, GaussianNB — label-enc + median-impute + (log10 AF) + standardize
All share the same fit_keep_nan numeric base (pop-freq rank, conservation clip) + train-only fit.
Run: .venv/bin/python -u analysis/exp_nontree_bakeoff.py [panel ...]
"""
import os, sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split

import harness
from config import SEED
from harness import cv_evaluate, load_panel, compare, _drop_leaky, lgb_oof_and_test
from preprocess_v2 import fit_keep_nan, apply_keep_nan
from calibration import fit_platt, apply_platt
from run_pipeline import get_model_params

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


# ── preprocessing helpers (all fit on train) ─────────────────────────────────
def _base(X_tr, y_tr, X_te):
    Xtr, Xte = _drop_leaky(X_tr.reset_index(drop=True), y_tr.reset_index(drop=True), X_te.reset_index(drop=True))
    Xtr_pp, ppf = fit_keep_nan(Xtr); Xte_pp = apply_keep_nan(Xte, ppf)
    return Xtr_pp, Xte_pp


def _cats(X):
    return [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]


def _enc(Xtr, Xte, fill=None):
    Xtr, Xte = Xtr.copy(), Xte.copy()
    for c in _cats(Xtr):
        cl = list(Xtr[c].cat.categories)
        tr = Xtr[c].cat.codes.to_numpy().astype(float)
        te = pd.Categorical(Xte[c].astype(object), categories=cl).codes.astype(float)
        Xtr[c] = np.where(tr < 0, np.nan, tr)
        Xte[c] = np.where(te < 0, np.nan, te)
    if fill is not None:
        Xtr = Xtr.fillna(fill); Xte = Xte.fillna(fill)
    return Xtr.astype(float), Xte.astype(float)


def _scaled(Xtr, Xte, log_af=False):
    Xtr, Xte = _enc(Xtr, Xte, fill=None)
    if log_af:
        # log10 the population-frequency AL columns (multi-order-of-magnitude) -> linearized for
        # scale-sensitive models. fit AF-col set on TRAIN. NaN preserved, imputed after (geo-median).
        al = [c for c in Xtr.columns if str(c).startswith("AL_")]
        af = [c for c in al if ((Xtr[c] > 0) & (Xtr[c] <= 1)).any() and Xtr[c].mean(skipna=True) < 0.1]
        for c in af:
            Xtr[c] = np.log10(Xtr[c].clip(lower=0) + 1e-9)
            Xte[c] = np.log10(Xte[c].clip(lower=0) + 1e-9)
    med = Xtr.median()
    Xtr = Xtr.fillna(med).fillna(0.0); Xte = Xte.fillna(med).fillna(0.0)
    sc = StandardScaler().fit(Xtr.values)
    return (pd.DataFrame(sc.transform(Xtr.values), columns=Xtr.columns),
            pd.DataFrame(sc.transform(Xte.values), columns=Xte.columns))


def _oof(Xtr, ytr, Xte, make, nf):
    oof = np.zeros(len(ytr)); skf = StratifiedKFold(nf, shuffle=True, random_state=SEED)
    for a, b in skf.split(Xtr, ytr):
        m = make(); m.fit(Xtr.iloc[a], ytr.iloc[a]); oof[b] = m.predict_proba(Xtr.iloc[b])[:, 1]
    m = make(); m.fit(Xtr, ytr)
    return apply_cal(oof, ytr, m.predict_proba(Xte)[:, 1])


def apply_cal(oof, ytr, test):
    pf = fit_platt(oof, ytr.values); return apply_platt(test, pf), apply_platt(oof, pf)


def _nf(n):
    return 3 if n < 200 else 5


# ── model pipeline_fns ───────────────────────────────────────────────────────
def fn_lgbm(X_tr, y_tr, X_te):
    Xtr, Xte = _base(X_tr, y_tr, X_te)
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    tp, oof = lgb_oof_and_test(Xtr, y_tr.reset_index(drop=True), Xte,
                               get_model_params(len(X_tr), spw), n_seeds=2, n_folds=_nf(len(X_tr)))
    return apply_cal(oof, y_tr.reset_index(drop=True), tp)


def fn_catboost(X_tr, y_tr, X_te):
    y_tr = y_tr.reset_index(drop=True)
    Xtr, Xte = _base(X_tr, y_tr, X_te)
    cats = _cats(Xtr)
    Xtr2, Xte2 = Xtr.copy(), Xte.copy()
    for c in cats:
        Xtr2[c] = Xtr[c].astype(object).where(Xtr[c].notna(), "NA").astype(str)
        Xte2[c] = Xte[c].astype(object).where(Xte[c].notna(), "NA").astype(str)
    idx = [Xtr2.columns.get_loc(c) for c in cats]
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    def make():
        return CatBoostClassifier(iterations=800, depth=6, learning_rate=0.05, scale_pos_weight=spw,
                                  cat_features=idx, random_seed=SEED, verbose=0,
                                  allow_writing_files=False, thread_count=4)
    return _oof(Xtr2, y_tr, Xte2, make, _nf(len(X_tr)))


def _tree_fn(kind):
    def fn(X_tr, y_tr, X_te):
        y_tr = y_tr.reset_index(drop=True)
        Xtr, Xte = _base(X_tr, y_tr, X_te)
        spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        if kind in ("xgb", "histgb"):
            Xtr2, Xte2 = _enc(Xtr, Xte, fill=None)                  # NaN kept
            if kind == "xgb":
                make = lambda: xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=4,
                    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, scale_pos_weight=spw,
                    tree_method="hist", eval_metric="logloss", n_jobs=4, random_state=SEED)
            else:
                make = lambda: HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                    max_depth=None, l2_regularization=1.0, class_weight="balanced", random_state=SEED)
        else:
            Xtr2, Xte2 = _enc(Xtr, Xte, fill=-999.0)
            if kind == "rf":
                make = lambda: RandomForestClassifier(n_estimators=500, max_features="sqrt",
                    class_weight="balanced", n_jobs=4, random_state=SEED)
            else:
                make = lambda: ExtraTreesClassifier(n_estimators=500, max_features="sqrt",
                    class_weight="balanced", n_jobs=4, random_state=SEED)
        return _oof(Xtr2, y_tr, Xte2, make, _nf(len(X_tr)))
    return fn


def _scaled_fn(kind, log_af=False, al_embed=0):
    def fn(X_tr, y_tr, X_te):
        y_tr = y_tr.reset_index(drop=True)
        Xtr, Xte = _base(X_tr, y_tr, X_te)
        Xtr2, Xte2 = _scaled(Xtr, Xte, log_af=log_af)
        if al_embed:                                                # PCA-embed the AL block
            al = [c for c in Xtr2.columns if str(c).startswith("AL_")]
            non = [c for c in Xtr2.columns if not str(c).startswith("AL_")]
            k = min(al_embed, len(al))
            pca = PCA(n_components=k, random_state=SEED).fit(Xtr2[al].values)
            etr = pd.DataFrame(pca.transform(Xtr2[al].values), columns=[f"ALe{i}" for i in range(k)])
            ete = pd.DataFrame(pca.transform(Xte2[al].values), columns=[f"ALe{i}" for i in range(k)])
            Xtr2 = pd.concat([Xtr2[non].reset_index(drop=True), etr], axis=1)
            Xte2 = pd.concat([Xte2[non].reset_index(drop=True), ete], axis=1)
        cw = "balanced"
        if kind == "logreg":
            make = lambda: LogisticRegression(max_iter=2000, C=0.5, class_weight=cw)
        elif kind == "svm":
            make = lambda: SVC(C=1.0, kernel="rbf", probability=True, class_weight=cw, random_state=SEED)
        elif kind == "knn":
            make = lambda: KNeighborsClassifier(n_neighbors=15)
        elif kind == "gnb":
            make = lambda: GaussianNB()
        else:  # mlp
            make = lambda: MLPClassifier(hidden_layer_sizes=(128, 32), alpha=1e-3, max_iter=300,
                                         early_stopping=True, random_state=SEED)
        return _oof(Xtr2, y_tr, Xte2, make, _nf(len(X_tr)))
    return fn


MODELS = {
    "lgbm": fn_lgbm, "catboost": fn_catboost,
    "xgb": _tree_fn("xgb"), "histgb": _tree_fn("histgb"),
    "rf": _tree_fn("rf"), "extratrees": _tree_fn("et"),
    "logreg": _scaled_fn("logreg"), "svm": _scaled_fn("svm"),
    "knn": _scaled_fn("knn"), "gnb": _scaled_fn("gnb"),
    "mlp": _scaled_fn("mlp"),
    # asks (2)/(3): embedding + log
    "mlp_alembed": _scaled_fn("mlp", al_embed=16),
    "logreg_log": _scaled_fn("logreg", log_af=True),
    "svm_log": _scaled_fn("svm", log_af=True),
    "mlp_log": _scaled_fn("mlp", log_af=True),
}


def run(panel, n_repeats):
    df = load_panel(panel)
    rows = []
    for name, fn in MODELS.items():
        t0 = time.time()
        try:
            r = cv_evaluate(panel, fn, label=name, n_repeats=n_repeats, df=df, n_jobs=4)
            s = r.summary()
            rows.append({"panel": panel, "model": name, "f1": s["f1_mean"], "auc": s["auc_mean"]})
            print(f"  [{panel}] {name:14s} F1*={s['f1_mean']:.4f}  AUC={s['auc_mean']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"  [{panel}] {name:14s} FAILED: {e}", flush=True)
    rows.sort(key=lambda r: -r["f1"])
    print(f"  [{panel}] >>> best: {rows[0]['model']} F1*={rows[0]['f1']:.4f}", flush=True)
    return rows


if __name__ == "__main__":
    panels = sys.argv[1:] or ["cftr", "pah", "cancer", "general"]
    print("=== Model bake-off + embedding + log-scaling — honest CV F1* at test prior ===", flush=True)
    allrows = []
    for pn in panels:
        print(f"\n[{pn}]", flush=True)
        allrows += run(pn, 3 if pn == "general" else 6)
    pd.DataFrame(allrows).to_csv(os.path.join(OUT, "nontree_bakeoff.csv"), index=False)
    print(f"\nsaved -> {os.path.join(OUT, 'nontree_bakeoff.csv')}", flush=True)
