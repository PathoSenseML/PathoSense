"""Campaign: push general-panel AUC (hence F1* at the test prior). Every lever is gated by the honest
harness — AUC + analytic F1* at pi=0.143, PAIRED vs the deployed general baseline. The ceiling math:
F1=0.70 @ pi=0.143 needs AUC ~0.94 (we are ~0.858), so we chase AUC.

Levers: missingness-signal recovery, non-tree models (LogReg/SVM/kNN) standalone + blended, stacking,
engineered conservation/rarity features. Run: .venv/bin/python -u analysis/exp_push_general.py
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import rankdata

from config import SEED
from harness import cv_evaluate, load_panel, compare, _drop_leaky
from schema import partition_columns
from preprocess_v2 import fit_keep_nan, apply_keep_nan
from meeting_lib import native_nan_oof
from run_pipeline import get_model_params
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

df0 = load_panel("general"); _, NUM, CAT, TGT = partition_columns(df0)
yfull = df0[TGT].astype(int)
P = {**get_model_params(2000, (yfull == 0).sum() / max(yfull.sum(), 1)),
     "scale_pos_weight": 1, "bagging_freq": 1, "bagging_fraction": 0.8}


def base_fn(a, b, c):
    return native_nan_oof(a, b, c, params=P)


# ── dense numeric matrix (median-impute on TRAIN, scale) for non-tree models ──
def densify(Xtr, Xte, k=0, ytr=None):
    cols = list(Xtr.columns)
    A = np.empty((len(Xtr), len(cols))); B = np.empty((len(Xte), len(cols)))
    for j, c in enumerate(cols):
        if str(Xtr[c].dtype) == "category":
            cats = Xtr[c].cat.categories
            A[:, j] = Xtr[c].cat.codes.astype(float)
            B[:, j] = pd.Categorical(Xte[c], categories=cats).codes.astype(float)
        else:
            A[:, j] = pd.to_numeric(Xtr[c], errors="coerce")
            B[:, j] = pd.to_numeric(Xte[c], errors="coerce")
    med = np.nanmedian(A, axis=0); med = np.where(np.isfinite(med), med, 0.0)
    A = np.where(np.isnan(A), med, A); B = np.where(np.isnan(B), med, B)
    sc = StandardScaler().fit(A); A, B = sc.transform(A), sc.transform(B)
    A = np.clip(np.nan_to_num(A), -8, 8); B = np.clip(np.nan_to_num(B), -8, 8)
    if k and k < A.shape[1] and ytr is not None:
        mi = mutual_info_classif(A, ytr, random_state=SEED)
        sel = np.argsort(mi)[::-1][:k]; A, B = A[:, sel], B[:, sel]
    return A, B


def nontree_oof(model_factory, k=0):
    """pipeline_fn: dense impute+scale (+MI top-k), internal 5-fold OOF + full-fit test."""
    def fn(X_tr, y_tr, X_te):
        Xtr, Xte = _drop_leaky(X_tr.reset_index(drop=True), y_tr.reset_index(drop=True), X_te.reset_index(drop=True))
        _, ppf = fit_keep_nan(Xtr); Xtr = apply_keep_nan(Xtr, ppf); Xte = apply_keep_nan(Xte, ppf)
        y = y_tr.reset_index(drop=True).astype(int).values
        A, B = densify(Xtr, Xte, k=k, ytr=y)
        oof = np.zeros(len(y)); test = np.zeros(len(B))
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
        for a, b in skf.split(A, y):
            m = model_factory(); m.fit(A[a], y[a])
            oof[b] = m.predict_proba(A[b])[:, 1]
        m = model_factory(); m.fit(A, y); test = m.predict_proba(B)[:, 1]
        return test, oof
    return fn


def _rank(x): return rankdata(x) / (len(x) + 1)


def blend_fn(fns, weights):
    """Rank-average blend of several pipeline_fns (test and oof aligned)."""
    def fn(X_tr, y_tr, X_te):
        ts, os_ = [], []
        for f in fns:
            t, o = f(X_tr, y_tr, X_te); ts.append(t); os_.append(o)
        w = np.array(weights) / sum(weights)
        test = sum(wi * _rank(t) for wi, t in zip(w, ts))
        oof = sum(wi * _rank(o) for wi, o in zip(w, os_))
        return test, oof
    return fn


# ── missingness-signal recovery: add AL rarity features the leakage filter strips ──
def miss_fn(X_tr, y_tr, X_te):
    def add(X):
        X = X.copy()
        al = [c for c in X.columns if str(c).startswith("AL_")]
        miss = X[al].isna()
        X["MISS_al_count"] = miss.sum(axis=1).astype(float)       # how many DBs lack this variant (rarity)
        X["MISS_al_frac"] = miss.mean(axis=1).astype(float)
        return X
    return native_nan_oof(add(X_tr), y_tr, add(X_te), params=P)


LR = lambda: LogisticRegression(penalty="l2", C=0.5, max_iter=2000, class_weight="balanced")
LREN = lambda: LogisticRegression(penalty="elasticnet", l1_ratio=0.5, C=0.5, solver="saga",
                                  max_iter=3000, class_weight="balanced")
SVM = lambda: SVC(C=2.0, gamma="scale", probability=True, class_weight="balanced", random_state=SEED)
KNN = lambda: KNeighborsClassifier(n_neighbors=25, weights="distance")

LEVERS = {
    "miss_recover":   miss_fn,
    "logreg_l2":      nontree_oof(LR),
    "logreg_enet":    nontree_oof(LREN),
    "svm_rbf_top120": nontree_oof(SVM, k=120),
    "knn_top80":      nontree_oof(KNN, k=80),
    "blend_gbdt_lr":  blend_fn([base_fn, nontree_oof(LR)], [2, 1]),
    "blend_gbdt_lr_svm": blend_fn([base_fn, nontree_oof(LR), nontree_oof(SVM, k=120)], [3, 1, 1]),
}


def main():
    nrep = 4
    base = cv_evaluate("general", base_fn, label="base", n_repeats=nrep, df=df0, n_jobs=6)
    print(f"  {'base':18s} F1*={base.f1.mean():.4f}  AUC={base.auc.mean():.4f}", flush=True)
    rows = [{"lever": "base", "f1": round(base.f1.mean(), 4), "auc": round(base.auc.mean(), 4),
             "d_f1": 0.0, "sigma": 0.0, "verdict": "-"}]
    for name, fn in LEVERS.items():
        try:
            r = cv_evaluate("general", fn, label=name, n_repeats=nrep, df=df0, n_jobs=6)
            c = compare(base, r)
            print(f"  {name:18s} F1*={r.f1.mean():.4f}  AUC={r.auc.mean():.4f}  "
                  f"ΔF1={c['delta_f1']:+.4f} ({c['sigma']:+.2f}σ) ΔAUC={r.auc.mean()-base.auc.mean():+.4f} "
                  f"[{c['verdict']}]", flush=True)
            rows.append({"lever": name, "f1": round(r.f1.mean(), 4), "auc": round(r.auc.mean(), 4),
                         "d_f1": c["delta_f1"], "sigma": c["sigma"], "verdict": c["verdict"]})
        except Exception as e:
            print(f"  {name:18s} FAILED: {type(e).__name__}: {str(e)[:80]}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "push_general.csv"), index=False)
    print("\nsaved -> push_general.csv", flush=True)


if __name__ == "__main__":
    main()
