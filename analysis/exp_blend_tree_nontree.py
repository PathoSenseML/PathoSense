"""Blend TREES + NON-TREES. Pool of 8: 4 deployed-tuned trees (LightGBM, RF, CatBoost, XGBoost, native-NaN)
+ 4 diverse non-trees on their FAIR preprocessing (quantile + one-hot + missingness): LDA (best non-tree),
LogReg (linear), SVM-RBF (kernel), kNN (distance). Non-trees are individually weaker but decorrelated from
trees → may help a blend via variance reduction. Evaluate ALL 255 non-empty subsets as equal-weight blends,
nested (threshold on train-OOF, score held-out), paired vs the deployed 4-tree blend. F1* @ 0.143 + AUC.
Run: .venv/bin/python -u analysis/exp_blend_tree_nontree.py
"""
import sys, itertools, numpy as np, pandas as pd
sys.path.insert(0, "analysis"); sys.path.insert(0, "pipelines")
import warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
import lightgbm as lgb
from harness import load_panel, _drop_leaky, f1_at_prior, tpr_fpr, pick_threshold_at_prior
from schema import partition_columns
from preprocess_v2 import fit_keep_nan, apply_keep_nan
from calibration import fit_platt, apply_platt
from general_blend import _rf_general, _cat_general, _xgb_model, _rf_repr, _cat_repr, _cats
from run_pipeline import get_model_params
from config import SEED

PRIOR = 0.143
POOL = ["lgbm", "rf", "cat", "xgb", "lda", "logreg", "svm", "knn"]
DEPLOYED = ("lgbm", "rf", "cat", "xgb")
df = load_panel("general"); _, NUM, CAT, TGT = partition_columns(df)
X_all = df[NUM + CAT].reset_index(drop=True); Y = df[TGT].astype(int).reset_index(drop=True)


def _oof_test(mk, Atr, Ate, y, cat_eval=False):
    oof = np.zeros(len(y))
    for a, b in StratifiedKFold(3, shuffle=True, random_state=SEED).split(Atr, y):
        m = mk()
        if cat_eval:
            i2, ies = train_test_split(a, test_size=0.15, random_state=SEED, stratify=y[a] if isinstance(y, np.ndarray) else y.iloc[a])
            yy = y[i2] if isinstance(y, np.ndarray) else y.iloc[i2]
            ye = y[ies] if isinstance(y, np.ndarray) else y.iloc[ies]
            Xi = Atr.iloc[i2] if hasattr(Atr, "iloc") else Atr[i2]
            Xe = Atr.iloc[ies] if hasattr(Atr, "iloc") else Atr[ies]
            m.fit(Xi, yy, eval_set=(Xe, ye))
        else:
            m.fit(Atr.iloc[a] if hasattr(Atr, "iloc") else Atr[a], y.iloc[a] if hasattr(y, "iloc") else y[a])
        Xb = Atr.iloc[b] if hasattr(Atr, "iloc") else Atr[b]
        oof[b] = m.predict_proba(Xb)[:, 1]
    mf = mk()
    if cat_eval:
        i2, ies = train_test_split(np.arange(len(y)), test_size=0.15, random_state=SEED + 9,
                                   stratify=y if isinstance(y, np.ndarray) else y.values)
        m_fit_X = Atr.iloc[i2] if hasattr(Atr, "iloc") else Atr[i2]
        mf.fit(m_fit_X, (y[i2] if isinstance(y, np.ndarray) else y.iloc[i2]),
               eval_set=((Atr.iloc[ies] if hasattr(Atr, "iloc") else Atr[ies]),
                         (y[ies] if isinstance(y, np.ndarray) else y.iloc[ies])))
    else:
        mf.fit(Atr, y)
    test = mf.predict_proba(Ate)[:, 1]
    yv = y.values if hasattr(y, "values") else y
    pf = fit_platt(oof, yv)
    return apply_platt(test, pf), apply_platt(oof, pf)


def _nontree_prep(Xtr, ytr, k):
    num = [c for c in Xtr.columns if str(Xtr[c].dtype) not in ("category", "object")]
    cat = [c for c in Xtr.columns if str(Xtr[c].dtype) in ("category", "object")]
    A = Xtr[num].apply(pd.to_numeric, errors="coerce"); med = A.median()
    miss = [c for c in num if A[c].isna().mean() > 0.05]
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=min(1000, len(Xtr)), random_state=SEED).fit(A.fillna(med))
    ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False)
    if cat: ohe.fit(Xtr[cat].astype(str))
    def build(X):
        N = X[num].apply(pd.to_numeric, errors="coerce")
        Q = qt.transform(N.fillna(med))
        M = N[miss].isna().astype(float).values if miss else np.empty((len(X), 0))
        C = ohe.transform(X[cat].astype(str)) if cat else np.empty((len(X), 0))
        return np.clip(np.nan_to_num(np.hstack([Q, M, C])), -8, 8)
    Z = build(Xtr); sel = np.arange(Z.shape[1])
    if k and k < Z.shape[1]:
        sel = np.argsort(mutual_info_classif(Z, ytr, random_state=SEED))[::-1][:k]
    return Z[:, sel], (lambda X: build(X)[:, sel])


def build_all(Xtr, ytr, Xte):
    Xtr, Xte = _drop_leaky(Xtr.reset_index(drop=True), ytr.reset_index(drop=True), Xte.reset_index(drop=True))
    y = ytr.reset_index(drop=True).astype(int); spw = (y == 0).sum() / max(y.sum(), 1)
    # trees on native-NaN
    _, ppf = fit_keep_nan(Xtr); Ptr = apply_keep_nan(Xtr, ppf); Pte = apply_keep_nan(Xte, ppf)
    cats = _cats(Ptr)
    Xo_tr, enc = _rf_repr(Ptr, cats, fit=True); Xo_te, _ = _rf_repr(Pte, cats, enc=enc, fit=False)
    Xc_tr = _cat_repr(Ptr, cats); Xc_te = _cat_repr(Pte, cats); ci = [Xc_tr.columns.get_loc(c) for c in cats]
    P = {**get_model_params(2000, spw), "scale_pos_weight": 1, "bagging_freq": 1, "bagging_fraction": 0.8, "n_jobs": 2, "verbose": -1}
    test, oof = {}, {}
    test["lgbm"], oof["lgbm"] = _oof_test(lambda: lgb.LGBMClassifier(**P), Ptr, Pte, y)
    test["rf"], oof["rf"] = _oof_test(_rf_general, Xo_tr, Xo_te, y)
    test["xgb"], oof["xgb"] = _oof_test(_xgb_model, Xo_tr, Xo_te, y)
    test["cat"], oof["cat"] = _oof_test(lambda: _cat_general(ci, spw), Xc_tr, Xc_te, y, cat_eval=True)
    # non-trees on fair prep
    for name, mk, k in [
        ("lda", lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage=0.15), 300),
        ("logreg", lambda: LogisticRegression(penalty="l1", solver="saga", C=0.5, class_weight="balanced", max_iter=4000), 0),
        ("svm", lambda: SVC(C=3.0, gamma=0.01, probability=True, class_weight="balanced", random_state=SEED), 120),
        ("knn", lambda: KNeighborsClassifier(n_neighbors=40, weights="distance"), 60),
    ]:
        Z, ap = _nontree_prep(Xtr, y, k); Zte = ap(Xte)
        test[name], oof[name] = _oof_test(mk, pd.DataFrame(Z), pd.DataFrame(Zte), y)
    return (np.column_stack([test[k] for k in POOL]), np.column_stack([oof[k] for k in POOL]))


SUBSETS = [c for r in range(1, len(POOL) + 1) for c in itertools.combinations(range(len(POOL)), r)]


def _score(idx, oM, tM, ytr, yte):
    bo = oM[:, idx].mean(1); bt = tM[:, idx].mean(1)
    thr = pick_threshold_at_prior(ytr, bo, PRIOR); tpr, fpr = tpr_fpr(yte, (bt >= thr).astype(int))
    return f1_at_prior(tpr, fpr, PRIOR), float(roc_auc_score(yte, bt))


def one_fold(tr, te):
    tM, oM = build_all(X_all.iloc[tr], Y.iloc[tr], X_all.iloc[te])
    yv = Y.iloc[tr].values; yt = Y.iloc[te].values
    return {s: _score(list(s), oM, tM, yv, yt) for s in SUBSETS}


def main():
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED).split(X_all, Y))
    print(f"{len(SUBSETS)} subsets x {len(splits)} folds (trees+non-trees)...", flush=True)
    out = Parallel(n_jobs=5, backend="loky")(delayed(one_fold)(tr, te) for tr, te in splits)
    dep = tuple(POOL.index(m) for m in DEPLOYED); dep_f1 = np.array([r[dep][0] for r in out])
    rows = []
    for s in SUBSETS:
        f1 = np.array([r[s][0] for r in out]); auc = np.array([r[s][1] for r in out])
        d = f1 - dep_f1; se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        has_nt = any(POOL[i] in ("lda", "logreg", "svm", "knn") for i in s)
        rows.append({"members": "+".join(POOL[i] for i in s), "k": len(s), "has_nontree": has_nt,
                     "f1": round(f1.mean(), 4), "auc": round(auc.mean(), 4),
                     "d_f1_vs_deployed": round(d.mean(), 4), "sigma": round(d.mean() / se if se > 0 else 0.0, 2)})
    R = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    R.to_csv("analysis/output/blend_tree_nontree.csv", index=False)
    dn = "+".join(DEPLOYED)
    print(f"\ndeployed 4-tree ({dn}): F1*={dep_f1.mean():.4f}\n=== TOP 15 by F1* ===")
    print(R.head(15).to_string(index=False), flush=True)
    best_nt = R[R.has_nontree].iloc[0]
    print(f"\ndeployed-4tree rank: {R.index[R.members == dn][0] + 1}/{len(R)}")
    print(f"best subset WITH a non-tree: {best_nt['members']}  F1*={best_nt['f1']}  "
          f"Δ={best_nt['d_f1_vs_deployed']:+.4f} ({best_nt['sigma']:+.2f}σ)")
    print("\nsaved -> analysis/output/blend_tree_nontree.csv", flush=True)


if __name__ == "__main__":
    main()
