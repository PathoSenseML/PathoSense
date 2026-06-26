"""Give the teacher's preprocessing ideas their BEST shot (not one config + give up).
General panel, LightGBM, honest 10-fold, AUC + F1* @ 0.143, paired vs deployed native-NaN.

Sections:
  1. Imputation DONE RIGHT — keep missingness indicators (preserve the signal) + impute, k-sweep.
  2. corr-drop threshold SWEEP (0.85..0.99) to find the optimum vs deployed 0.95.
  3. UMAP/PCA DONE RIGHT — supervised UMAP + AUGMENTATION (append components to full features),
     n_components / n_neighbors sweep.

Run: .venv/bin/python -u analysis/exp_teacher_fair.py
"""
import sys, warnings
sys.path.insert(0, "analysis"); sys.path.insert(0, "pipelines")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.decomposition import PCA

from config import SEED
from schema import normalize_sentinels, partition_columns, drop_all_nan_rows
from utils import detect_leaky_columns, KNOWN_LEAKY, TEST_PRIORS, PANELS
from predict import _engineer, LGBM_PARAM_OVERRIDE
from recipe import PANEL_RECIPE
from preprocess_v2 import fit_keep_nan, apply_keep_nan
from run_pipeline import get_model_params
from harness import f1_at_prior, tpr_fpr
from exp_teacher_preprocessing import _split_nc, _f1star

PANEL = "general"; PRIOR = TEST_PRIORS[PANEL]


def prep_native(Xtr, Xte, ytr=None):
    a, ppf = fit_keep_nan(Xtr); b = apply_keep_nan(Xte, ppf).reindex(columns=a.columns)
    _, cat = _split_nc(a)
    return a.reset_index(drop=True), b.reset_index(drop=True), cat


def _impute_ind(Xtr, Xte, imputer):
    num, cat = _split_nc(Xtr)
    nun = Xtr[num].nunique(dropna=True); kn = [c for c in num if nun[c] > 1]
    miss = Xtr[kn].isna().mean(); mind = miss[(miss > .05) & (miss < .95)].index.tolist()
    A = pd.DataFrame(imputer.fit_transform(Xtr[kn]), columns=kn)
    B = pd.DataFrame(imputer.transform(Xte[kn]), columns=kn)
    for c in mind:                                   # KEEP the missingness signal
        A[f"{c}__miss"] = Xtr[c].isna().astype(int).values
        B[f"{c}__miss"] = Xte[c].isna().astype(int).values
    A = pd.concat([A, Xtr[cat].reset_index(drop=True)], axis=1)
    B = pd.concat([B, Xte[cat].reset_index(drop=True)], axis=1)
    return A, B, cat


def make_impute_ind(imputer):
    return lambda Xtr, Xte, ytr=None: _impute_ind(Xtr, Xte, imputer)


def make_corr(thr):
    def fn(Xtr, Xte, ytr=None):
        a, b, cat = prep_native(Xtr, Xte)
        num = [c for c in a.columns if c not in cat]
        if len(num) > 1:
            corr = a[num].corr().abs(); up = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))
            drop = [c for c in up.columns if (up[c] > thr).any()]
            a = a.drop(columns=drop); b = b.drop(columns=drop)
        return a, b, [c for c in cat if c in a.columns]
    return fn


def _reduce_matrix(Xtr, Xte):
    num, cat = _split_nc(Xtr)
    nun = Xtr[num].nunique(dropna=True); kn = [c for c in num if nun[c] > 1]
    imp = SimpleImputer(strategy="median")
    M = imp.fit_transform(Xtr[kn]); Mt = imp.transform(Xte[kn])
    sc = StandardScaler(); M = sc.fit_transform(M); Mt = sc.transform(Mt)
    if cat:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        Ac = enc.fit_transform(Xtr[cat].astype(object).where(Xtr[cat].notna(), "NA"))
        Bc = enc.transform(Xte[cat].astype(object).where(Xte[cat].notna(), "NA"))
        M = np.hstack([M, Ac]); Mt = np.hstack([Mt, Bc])
    return M, Mt


def make_augment(kind, n_comp, n_neighbors=15):
    """Append reduced components to the FULL native-NaN feature set (augmentation, not replacement)."""
    def fn(Xtr, Xte, ytr=None):
        a, b, cat = prep_native(Xtr, Xte)
        M, Mt = _reduce_matrix(Xtr, Xte)
        if kind == "pca":
            r = PCA(n_components=n_comp, random_state=SEED); C = r.fit_transform(M); Ct = r.transform(Mt)
        else:
            import umap
            sup = (kind == "sumap")
            r = umap.UMAP(n_components=n_comp, n_neighbors=n_neighbors, min_dist=0.1,
                          random_state=SEED, target_metric="categorical")
            C = r.fit_transform(M, y=ytr.values if sup else None); Ct = r.transform(Mt)
        for i in range(n_comp):
            a[f"{kind}{i}"] = C[:, i]; b[f"{kind}{i}"] = Ct[:, i]
        return a, b, cat
    return fn


def evaluate(prep_fn, X, y, params, n_rep=2):
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_rep, random_state=SEED)
    aucs, f1s = [], []
    for tr, te in outer.split(X, y):
        Xtr = X.iloc[tr].reset_index(drop=True); Xte = X.iloc[te].reset_index(drop=True)
        ytr = y.iloc[tr].reset_index(drop=True); yte = y.iloc[te].values
        a, b, cat = prep_fn(Xtr, Xte, ytr)
        m = lgb.LGBMClassifier(**params); m.fit(a, ytr, categorical_feature=cat or "auto")
        p = m.predict_proba(b)[:, 1]
        aucs.append(roc_auc_score(yte, p)); f1s.append(_f1star(yte, p, PRIOR))
    return np.array(aucs), np.array(f1s)


def main():
    df = pd.read_csv(PANELS[PANEL]); df = normalize_sentinels(df); df = drop_all_nan_rows(df)
    de = _engineer(df.copy(), PANEL_RECIPE[PANEL])
    _, num_cols, cat_cols, tgt = partition_columns(de)
    y = de[tgt].reset_index(drop=True); X = de[num_cols + cat_cols].reset_index(drop=True)
    leaky = detect_leaky_columns(X[num_cols], y, auc_threshold=0.99, verbose=False)
    X = X.drop(columns=sorted(set(leaky) | (KNOWN_LEAKY & set(X.columns))))
    params = {**get_model_params(len(X), 1.0), **LGBM_PARAM_OVERRIDE[PANEL], "verbose": -1}

    bA, bF = evaluate(prep_native, X, y, params)
    print(f"=== FAIR teacher-prep effort (general, vs native-NaN baseline AUC={bA.mean():.4f}) ===")

    def show(name, au, f1):
        dA = au - bA; dF = f1 - bF
        sA = dA.std(ddof=1)/np.sqrt(len(dA)); sF = dF.std(ddof=1)/np.sqrt(len(dF))
        tag = "  <<< WIN" if (dA.mean()/sA if sA > 0 else 0) >= 2 and dF.mean() > 0 else ""
        print(f"  {name:24s} AUC={au.mean():.4f} (Δ{dA.mean():+.4f} {dA.mean()/sA if sA>0 else 0:+.1f}σ)  "
              f"F1*={f1.mean():.4f} (Δ{dF.mean():+.4f} {dF.mean()/sF if sF>0 else 0:+.1f}σ){tag}", flush=True)

    print("--- 1. imputation + MISSINGNESS INDICATORS (preserve signal) ---")
    for nm, imp in [("median+ind", SimpleImputer(strategy="median")),
                    ("knn3+ind", KNNImputer(n_neighbors=3)),
                    ("knn5+ind", KNNImputer(n_neighbors=5)),
                    ("knn10+ind", KNNImputer(n_neighbors=10)),
                    ("mice+ind", IterativeImputer(max_iter=3, random_state=SEED, sample_posterior=False))]:
        au, f1 = evaluate(make_impute_ind(imp), X, y, params); show(nm, au, f1)

    print("--- 2. corr-drop threshold sweep (deployed=0.95) ---")
    for thr in [0.85, 0.90, 0.92, 0.95, 0.97, 0.99]:
        au, f1 = evaluate(make_corr(thr), X, y, params); show(f"corr>{thr}", au, f1)

    print("--- 3. UMAP/PCA AUGMENTATION (append to full features) + supervised ---")
    for kind, nc in [("pca", 20), ("pca", 50), ("umap", 15), ("umap", 30),
                     ("sumap", 10), ("sumap", 30)]:
        au, f1 = evaluate(make_augment(kind, nc), X, y, params); show(f"{kind}-{nc} augment", au, f1)
    print("DONE")


if __name__ == "__main__":
    main()
