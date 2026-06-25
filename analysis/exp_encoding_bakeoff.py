"""Comprehensive model bake-off where EACH model gets the encoding it needs:
  native-categorical  : LightGBM, CatBoost                          (optimal category partitions)
  ordinal/label       : XGBoost, RandomForest, ExtraTrees, GradientBoosting, AdaBoost, DecisionTree
                        (trees split fine on integer codes)
  ONE-HOT + scaled    : LogReg, SGD, SVM, KNN, MLP, GaussianNB, LDA, QDA
                        (linear/distance/NN/discriminant — nominal cats MUST be one-hot, not ordinal;
                        ordinal imposes a false order these models take literally)

This fixes the earlier bake-off (which gave the scale-sensitive models ordinal encoding) and adds the
untested model classes. Honest CV F1* at the test prior, each on the same fit_keep_nan numeric base.
Run: .venv/bin/python -u analysis/exp_encoding_bakeoff.py [panel ...]
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")

import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split

from config import SEED
from harness import cv_evaluate, load_panel, _drop_leaky, lgb_oof_and_test
from preprocess_v2 import fit_keep_nan, apply_keep_nan
from calibration import fit_platt, apply_platt
from run_pipeline import get_model_params

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def _base(X_tr, y_tr, X_te):
    Xtr, Xte = _drop_leaky(X_tr.reset_index(drop=True), y_tr.reset_index(drop=True), X_te.reset_index(drop=True))
    Xtr_pp, ppf = fit_keep_nan(Xtr); return Xtr_pp, apply_keep_nan(Xte, ppf)


def _cats(X): return [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]


def _ordinal(Xtr, Xte, fill):
    Xtr, Xte = Xtr.copy(), Xte.copy()
    for c in _cats(Xtr):
        cl = list(Xtr[c].cat.categories)
        tr = Xtr[c].cat.codes.to_numpy().astype(float)
        te = pd.Categorical(Xte[c].astype(object), categories=cl).codes.astype(float)
        Xtr[c] = np.where(tr < 0, np.nan, tr); Xte[c] = np.where(te < 0, np.nan, te)
    return Xtr.astype(float).fillna(fill), Xte.astype(float).fillna(fill)


def _onehot_scaled(Xtr, Xte):
    """one-hot the categoricals (train cats), median-impute + standardize the numerics."""
    cats = _cats(Xtr)
    num_tr, num_te = Xtr.drop(columns=cats), Xte.drop(columns=cats)
    med = num_tr.median()
    num_tr = num_tr.fillna(med).fillna(0.0); num_te = num_te.fillna(med).fillna(0.0)
    sc = StandardScaler().fit(num_tr.values)
    num_tr = pd.DataFrame(sc.transform(num_tr.values), columns=num_tr.columns)
    num_te = pd.DataFrame(sc.transform(num_te.values), columns=num_te.columns)
    dtr = pd.get_dummies(Xtr[cats].astype(object), dummy_na=True)
    dte = pd.get_dummies(Xte[cats].astype(object), dummy_na=True).reindex(columns=dtr.columns, fill_value=0)
    return (pd.concat([num_tr, dtr.reset_index(drop=True).astype(float)], axis=1),
            pd.concat([num_te, dte.reset_index(drop=True).astype(float)], axis=1))


def _oof(Xtr, ytr, Xte, make, nf):
    oof = np.zeros(len(ytr)); skf = StratifiedKFold(nf, shuffle=True, random_state=SEED)
    for a, b in skf.split(Xtr, ytr):
        m = make(); m.fit(Xtr.iloc[a], ytr.iloc[a]); oof[b] = m.predict_proba(Xtr.iloc[b])[:, 1]
    m = make(); m.fit(Xtr, ytr); test = m.predict_proba(Xte)[:, 1]
    pf = fit_platt(oof, ytr.values); return apply_platt(test, pf), apply_platt(oof, pf)


def _nf(n): return 3 if n < 200 else 5


def make_fn(name):
    def fn(X_tr, y_tr, X_te):
        y_tr = y_tr.reset_index(drop=True); Xtr, Xte = _base(X_tr, y_tr, X_te)
        spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1); nf = _nf(len(X_tr))
        if name == "lgbm":
            tp, oof = lgb_oof_and_test(Xtr, y_tr, Xte, get_model_params(len(X_tr), spw), n_seeds=2, n_folds=nf)
            pf = fit_platt(oof, y_tr.values); return apply_platt(tp, pf), apply_platt(oof, pf)
        if name == "catboost":
            cats = _cats(Xtr); Xtr2, Xte2 = Xtr.copy(), Xte.copy()
            for c in cats:
                Xtr2[c] = Xtr[c].astype(object).where(Xtr[c].notna(), "NA").astype(str)
                Xte2[c] = Xte[c].astype(object).where(Xte[c].notna(), "NA").astype(str)
            idx = [Xtr2.columns.get_loc(c) for c in cats]
            return _oof(Xtr2, y_tr, Xte2, lambda: CatBoostClassifier(iterations=600, depth=6, learning_rate=0.05,
                scale_pos_weight=spw, cat_features=idx, random_seed=SEED, verbose=0,
                allow_writing_files=False, thread_count=4), nf)
        # ordinal-encoded trees
        if name in ("xgb", "rf", "extratrees", "gradboost", "adaboost", "dtree"):
            fill = -999.0 if name in ("rf", "extratrees", "dtree", "gradboost", "adaboost") else np.nan
            Xtr2, Xte2 = _ordinal(Xtr, Xte, fill)
            mk = {"xgb": lambda: xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=4,
                      subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw, tree_method="hist",
                      eval_metric="logloss", n_jobs=4, random_state=SEED),
                  "rf": lambda: RandomForestClassifier(n_estimators=500, max_features="sqrt",
                      class_weight="balanced", n_jobs=4, random_state=SEED),
                  "extratrees": lambda: ExtraTreesClassifier(n_estimators=500, max_features="sqrt",
                      class_weight="balanced", n_jobs=4, random_state=SEED),
                  "gradboost": lambda: GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                      max_depth=3, random_state=SEED),
                  "adaboost": lambda: AdaBoostClassifier(n_estimators=300, learning_rate=0.5, random_state=SEED),
                  "dtree": lambda: DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=SEED)}[name]
            return _oof(Xtr2, y_tr, Xte2, mk, nf)
        # one-hot + scaled (linear / distance / NN / discriminant)
        Xtr2, Xte2 = _onehot_scaled(Xtr, Xte)
        mk = {"logreg": lambda: LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced"),
              "sgd": lambda: SGDClassifier(loss="log_loss", alpha=1e-3, class_weight="balanced", random_state=SEED),
              "svm": lambda: SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced", random_state=SEED),
              "knn": lambda: KNeighborsClassifier(n_neighbors=15),
              "mlp": lambda: MLPClassifier(hidden_layer_sizes=(128, 32), alpha=1e-3, max_iter=300,
                      early_stopping=True, random_state=SEED),
              "gnb": lambda: GaussianNB(),
              "lda": lambda: LinearDiscriminantAnalysis(),
              "qda": lambda: QuadraticDiscriminantAnalysis(reg_param=0.3)}[name]
        return _oof(Xtr2, y_tr, Xte2, mk, nf)
    return fn


ENC = {"lgbm": "native", "catboost": "native", "xgb": "ordinal", "rf": "ordinal", "extratrees": "ordinal",
       "gradboost": "ordinal", "adaboost": "ordinal", "dtree": "ordinal", "logreg": "onehot", "sgd": "onehot",
       "svm": "onehot", "knn": "onehot", "mlp": "onehot", "gnb": "onehot", "lda": "onehot", "qda": "onehot"}


def run(panel, nr):
    df = load_panel(panel); rows = []
    for name in ENC:
        t0 = time.time()
        try:
            s = cv_evaluate(panel, make_fn(name), label=name, n_repeats=nr, df=df, n_jobs=4).summary()
            rows.append({"panel": panel, "model": name, "enc": ENC[name], "f1": s["f1_mean"], "auc": s["auc_mean"]})
            print(f"  [{panel}] {name:10s} ({ENC[name]:7s}) F1*={s['f1_mean']:.4f} AUC={s['auc_mean']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"  [{panel}] {name:10s} FAILED: {str(e)[:60]}", flush=True)
    rows.sort(key=lambda r: -r["f1"]); print(f"  [{panel}] >>> best: {rows[0]['model']} ({rows[0]['enc']}) {rows[0]['f1']:.4f}", flush=True)
    return rows


if __name__ == "__main__":
    panels = sys.argv[1:] or ["cftr", "pah", "cancer", "general"]
    print("=== Comprehensive bake-off, encoding-per-model — honest CV F1* ===", flush=True)
    allrows = []
    for pn in panels:
        print(f"\n[{pn}]", flush=True); allrows += run(pn, 3 if pn == "general" else 6)
    pd.DataFrame(allrows).to_csv(os.path.join(OUT, "encoding_bakeoff.csv"), index=False)
    print(f"\nsaved -> {os.path.join(OUT, 'encoding_bakeoff.csv')}", flush=True)
