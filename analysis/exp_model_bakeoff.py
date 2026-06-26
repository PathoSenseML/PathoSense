"""Exhaustive MODEL BAKE-OFF on the general panel — every model family, honest harness
(AUC + analytic F1* at pi=0.143, PAIRED vs the deployed native-NaN LightGBM base).

Apples-to-apples: all candidates run on the same per-fold dense matrix (median-impute on train + scale;
scaling is a no-op for trees). The deployed native-NaN LightGBM is the reference to beat. Slow / high-dim-
sensitive models (SVM, kNN, QDA, GB, MLP, TabNet) get a mutual-info top-k subset. Each model is wrapped in
internal 5-fold OOF + full-fit test. Failures are caught per-model so one bad import can't kill the run.

Run: .venv/bin/python -u analysis/exp_model_bakeoff.py
"""
import os, sys, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier, GradientBoostingClassifier)
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostClassifier

from config import SEED
from harness import cv_evaluate, load_panel, compare, _drop_leaky
from schema import partition_columns
from meeting_lib import native_nan_oof
from run_pipeline import get_model_params
from exp_push_general import densify        # impute(train)+scale(+MI top-k)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

df0 = load_panel("general"); _, NUM, CAT, TGT = partition_columns(df0)
yf = df0[TGT].astype(int)
P = {**get_model_params(2000, (yf == 0).sum() / max(yf.sum(), 1)),
     "scale_pos_weight": 1, "bagging_freq": 1, "bagging_fraction": 0.8}
SPW = (yf == 0).sum() / max(yf.sum(), 1)


def base_native(a, b, c):                    # deployed reference: native-NaN LightGBM
    return native_nan_oof(a, b, c, params=P)


def sk_oof(factory, k=0):
    """Generic: dense impute+scale(+MI top-k) -> internal 5-fold OOF + full-fit test."""
    def fn(X_tr, y_tr, X_te):
        Xtr, Xte = _drop_leaky(X_tr.reset_index(drop=True), y_tr.reset_index(drop=True), X_te.reset_index(drop=True))
        from preprocess_v2 import fit_keep_nan, apply_keep_nan
        _, ppf = fit_keep_nan(Xtr); Xtr = apply_keep_nan(Xtr, ppf); Xte = apply_keep_nan(Xte, ppf)
        y = y_tr.reset_index(drop=True).astype(int).values
        A, B = densify(Xtr, Xte, k=k, ytr=y)
        oof = np.zeros(len(y))
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
        for a, b in skf.split(A, y):
            m = factory(); m.fit(A[a], y[a]); oof[b] = m.predict_proba(A[b])[:, 1]
        m = factory(); m.fit(A, y); test = m.predict_proba(B)[:, 1]
        return test, oof
    return fn


# ── model registry (factory, top-k features) ──
MODELS = {
    # boosting / trees
    "lightgbm_dense":  (lambda: lgb.LGBMClassifier(**{**P, "n_jobs": 1, "verbose": -1}), 0),
    "xgboost":         (lambda: xgb.XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.03,
                        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, scale_pos_weight=1,
                        eval_metric="logloss", n_jobs=1, tree_method="hist"), 0),
    "catboost":        (lambda: CatBoostClassifier(iterations=600, depth=6, learning_rate=0.03,
                        l2_leaf_reg=3, verbose=0, thread_count=1, random_seed=SEED), 0),
    "random_forest":   (lambda: RandomForestClassifier(n_estimators=600, max_features=0.3, min_samples_leaf=2,
                        class_weight="balanced_subsample", n_jobs=1, random_state=SEED), 0),
    "extra_trees":     (lambda: ExtraTreesClassifier(n_estimators=600, max_features=0.3, min_samples_leaf=2,
                        class_weight="balanced_subsample", n_jobs=1, random_state=SEED), 0),
    "hist_gb":         (lambda: HistGradientBoostingClassifier(max_iter=600, learning_rate=0.03,
                        l2_regularization=1.0, max_leaf_nodes=31, random_state=SEED), 0),
    "sk_gradient_boost": (lambda: GradientBoostingClassifier(n_estimators=300, max_depth=3,
                        learning_rate=0.05, subsample=0.8, random_state=SEED), 150),
    # linear / probabilistic
    "logreg_l2":       (lambda: LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced"), 0),
    "logreg_l1":       (lambda: LogisticRegression(penalty="l1", solver="saga", C=0.5, max_iter=3000,
                        class_weight="balanced"), 0),
    "logreg_enet":     (lambda: LogisticRegression(penalty="elasticnet", l1_ratio=0.5, solver="saga", C=0.5,
                        max_iter=3000, class_weight="balanced"), 0),
    "sgd_log":         (lambda: SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=2000,
                        class_weight="balanced", random_state=SEED), 0),
    "lda":             (lambda: LinearDiscriminantAnalysis(), 0),
    "qda":             (lambda: QuadraticDiscriminantAnalysis(reg_param=0.3), 80),
    "gaussian_nb":     (lambda: GaussianNB(), 0),
    # kernel / distance
    "svm_rbf":         (lambda: SVC(C=2.0, gamma="scale", probability=True, class_weight="balanced",
                        random_state=SEED), 120),
    "svm_linear":      (lambda: SVC(kernel="linear", C=0.5, probability=True, class_weight="balanced",
                        random_state=SEED), 120),
    "knn":             (lambda: KNeighborsClassifier(n_neighbors=25, weights="distance"), 80),
    # neural
    "mlp_2layer":      (lambda: MLPClassifier((128, 64), alpha=1e-3, max_iter=400, early_stopping=True,
                        random_state=SEED), 0),
    "mlp_3layer":      (lambda: MLPClassifier((256, 128, 64), alpha=1e-3, max_iter=400, early_stopping=True,
                        random_state=SEED), 0),
}

# optional extra families (added if installed)
try:
    from interpret.glassbox import ExplainableBoostingClassifier
    MODELS["ebm"] = (lambda: ExplainableBoostingClassifier(random_state=SEED, n_jobs=1), 0)
except Exception as e:
    print(f"  (EBM unavailable: {str(e)[:50]})", flush=True)
try:
    from ngboost import NGBClassifier
    from ngboost.distns import Bernoulli
    MODELS["ngboost"] = (lambda: NGBClassifier(Dist=Bernoulli, n_estimators=400, learning_rate=0.02,
                        verbose=False, random_state=SEED), 120)
except Exception as e:
    print(f"  (NGBoost unavailable: {str(e)[:50]})", flush=True)
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    class _TabNet:
        def __init__(self): self.m = TabNetClassifier(verbose=0, device_name=dev, seed=SEED)
        def fit(self, X, y): self.m.fit(X.astype("float32"), y, max_epochs=60, patience=12,
                                        batch_size=256, eval_set=[]); return self
        def predict_proba(self, X): return self.m.predict_proba(X.astype("float32"))
    MODELS["tabnet"] = (lambda: _TabNet(), 100)
except Exception as e:
    print(f"  (TabNet unavailable: {str(e)[:50]})", flush=True)


def main():
    nrep = 3
    print(f"=== MODEL BAKE-OFF, general panel ({len(MODELS)} models + base) ===", flush=True)
    base = cv_evaluate("general", base_native, label="base", n_repeats=nrep, df=df0, n_jobs=6)
    print(f"  {'LightGBM_nativeNaN(base)':26s} F1*={base.f1.mean():.4f} AUC={base.auc.mean():.4f}", flush=True)
    rows = [{"model": "LightGBM_nativeNaN(base)", "f1": round(base.f1.mean(), 4),
             "auc": round(base.auc.mean(), 4), "d_auc": 0.0, "sigma_f1": 0.0}]
    for name, (factory, k) in MODELS.items():
        t0 = time.time()
        try:
            nj = 1 if name in ("svm_rbf", "svm_linear", "ngboost", "tabnet", "ebm") else 6
            r = cv_evaluate("general", sk_oof(factory, k=k), label=name, n_repeats=nrep, df=df0, n_jobs=nj)
            c = compare(base, r)
            print(f"  {name:26s} F1*={r.f1.mean():.4f} AUC={r.auc.mean():.4f} "
                  f"ΔAUC={r.auc.mean()-base.auc.mean():+.4f} ΔF1={c['delta_f1']:+.4f} ({c['sigma']:+.2f}σ) "
                  f"[{time.time()-t0:.0f}s]", flush=True)
            rows.append({"model": name, "f1": round(r.f1.mean(), 4), "auc": round(r.auc.mean(), 4),
                         "d_auc": round(r.auc.mean()-base.auc.mean(), 4), "sigma_f1": c["sigma"]})
        except Exception as e:
            print(f"  {name:26s} FAILED: {type(e).__name__}: {str(e)[:70]}", flush=True)
            rows.append({"model": name, "f1": np.nan, "auc": np.nan, "d_auc": np.nan, "sigma_f1": np.nan})
    out = pd.DataFrame(rows).sort_values("auc", ascending=False, na_position="last")
    out.to_csv(os.path.join(OUT, "model_bakeoff.csv"), index=False)
    print("\n=== LEADERBOARD (by AUC) ===", flush=True)
    print(out.to_string(index=False), flush=True)
    print("\nsaved -> model_bakeoff.csv", flush=True)


if __name__ == "__main__":
    main()
