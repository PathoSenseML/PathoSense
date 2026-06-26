"""Empirically test the teacher's 10-step preprocessing checklist on the GENERAL panel.
Honest RepeatedStratifiedKFold, LightGBM (deployed general params), AUC + F1* at prior 0.143,
paired vs the deployed native-NaN baseline. Isolates each preprocessing choice on the LGBM.

Variants:
  A native-NaN (DEPLOYED)         baseline
  B median impute                 control for C/D/F/G (step 2 baseline)
  C KNN imputer (k=5)             teacher step 2
  D Iterative imputer / MICE      teacher step 2
  E native-NaN + corr>0.90 drop   teacher step 5 (deployed is 0.95)
  F median + Yeo-Johnson          teacher step 4 (power transform -> show inert for trees)
  G median + RobustScaler         teacher step 3 (show inert for trees)
  H median + UMAP(30 comp)        teacher step 9 (nonlinear reduction)

(Steps 1=const-drop, 6=MI selection, 7/8=importance selection, 10=L1-LogReg are covered
 elsewhere — see the point-by-point map; this script targets the not-yet-run methods.)

Run: .venv/bin/python -u analysis/exp_teacher_preprocessing.py
"""
import sys, warnings, time
sys.path.insert(0, "analysis"); sys.path.insert(0, "pipelines")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import PowerTransformer, RobustScaler, OrdinalEncoder

from config import SEED
from schema import normalize_sentinels, partition_columns, drop_all_nan_rows, to_categorical
from utils import detect_leaky_columns, KNOWN_LEAKY, TEST_PRIORS, PANELS
from predict import _engineer, LGBM_PARAM_OVERRIDE
from recipe import PANEL_RECIPE
from preprocess_v2 import fit_keep_nan, apply_keep_nan
from run_pipeline import get_model_params
from harness import f1_at_prior, tpr_fpr

PANEL = "general"; PRIOR = TEST_PRIORS[PANEL]


def _split_nc(X):
    cat = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
    return [c for c in X.columns if c not in cat], cat


def prep_native(Xtr, Xte):
    a, ppf = fit_keep_nan(Xtr)
    b = apply_keep_nan(Xte, ppf).reindex(columns=a.columns)
    _, cat = _split_nc(a)
    return a, b, cat


def _impute_variant(Xtr, Xte, imputer):
    num, cat = _split_nc(Xtr)
    nun = Xtr[num].nunique(dropna=True)
    kn = [c for c in num if nun[c] > 1]                      # drop constants (step 1)
    A = pd.DataFrame(imputer.fit_transform(Xtr[kn]), columns=kn)
    B = pd.DataFrame(imputer.transform(Xte[kn]), columns=kn)
    A = pd.concat([A, Xtr[cat].reset_index(drop=True)], axis=1)
    B = pd.concat([B, Xte[cat].reset_index(drop=True)], axis=1)
    return A, B, cat


def prep_median(Xtr, Xte): return _impute_variant(Xtr, Xte, SimpleImputer(strategy="median"))
def prep_knn(Xtr, Xte):    return _impute_variant(Xtr, Xte, KNNImputer(n_neighbors=5))
def prep_mice(Xtr, Xte):
    return _impute_variant(Xtr, Xte, IterativeImputer(max_iter=3, random_state=SEED, sample_posterior=False))


def prep_corr090(Xtr, Xte):
    a, b, cat = prep_native(Xtr, Xte)
    num = [c for c in a.columns if c not in cat]
    if len(num) > 1:
        corr = a[num].corr().abs()
        up = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))
        drop = [c for c in up.columns if (up[c] > 0.90).any()]
        a = a.drop(columns=drop); b = b.drop(columns=drop)
    return a, b, [c for c in cat if c in a.columns]


def _median_then(Xtr, Xte, transformer):
    A, B, cat = prep_median(Xtr, Xte)
    num = [c for c in A.columns if c not in cat]
    A2 = A.copy(); B2 = B.copy()
    A2[num] = transformer.fit_transform(A[num]); B2[num] = transformer.transform(B[num])
    return A2, B2, cat


def prep_yeo(Xtr, Xte):    return _median_then(Xtr, Xte, PowerTransformer(method="yeo-johnson"))
def prep_robust(Xtr, Xte): return _median_then(Xtr, Xte, RobustScaler())


def prep_umap(Xtr, Xte, n_comp=30):
    import umap
    num, cat = _split_nc(Xtr)
    nun = Xtr[num].nunique(dropna=True); kn = [c for c in num if nun[c] > 1]
    imp = SimpleImputer(strategy="median")
    A = imp.fit_transform(Xtr[kn]); B = imp.transform(Xte[kn])
    if cat:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        Ac = enc.fit_transform(Xtr[cat].astype(object).where(Xtr[cat].notna(), "NA"))
        Bc = enc.transform(Xte[cat].astype(object).where(Xte[cat].notna(), "NA"))
        A = np.hstack([A, Ac]); B = np.hstack([B, Bc])
    um = umap.UMAP(n_components=n_comp, random_state=SEED, n_neighbors=15, min_dist=0.1)
    A2 = um.fit_transform(A); B2 = um.transform(B)
    cols = [f"u{i}" for i in range(n_comp)]
    return pd.DataFrame(A2, columns=cols), pd.DataFrame(B2, columns=cols), []


def _f1star(y, p, prior):
    qs = np.quantile(p, np.arange(.02, .98, .01))
    return max(f1_at_prior(*tpr_fpr(y, (p >= q).astype(int)), prior) for q in qs)


def evaluate(name, prep_fn, X, y, spw, params, n_rep=2):
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_rep, random_state=SEED)
    aucs, f1s = [], []; t0 = time.time()
    for tr, te in outer.split(X, y):
        Xtr = X.iloc[tr].reset_index(drop=True); Xte = X.iloc[te].reset_index(drop=True)
        ytr = y.iloc[tr].reset_index(drop=True); yte = y.iloc[te].values
        try:
            a, b, cat = prep_fn(Xtr, Xte)
        except Exception as e:
            print(f"  {name:18s} FAILED: {e}"); return None, None
        m = lgb.LGBMClassifier(**params)
        m.fit(a, ytr, categorical_feature=cat or "auto")
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
    spw = 1.0
    params = {**get_model_params(len(X), spw), **LGBM_PARAM_OVERRIDE[PANEL], "verbose": -1}
    print(f"=== GENERAL preprocessing bake-off (prior {PRIOR:.3f}, LGBM, paired vs native-NaN) ===")
    variants = [("A native-NaN(dep)", prep_native), ("B median", prep_median),
                ("C KNN-impute", prep_knn), ("D MICE-impute", prep_mice),
                ("E corr>0.90 drop", prep_corr090), ("F Yeo-Johnson", prep_yeo),
                ("G RobustScaler", prep_robust), ("H UMAP-30", prep_umap)]
    base_auc = base_f1 = None
    for name, fn in variants:
        au, f1 = evaluate(name, fn, X, y, spw, params)
        if au is None:
            continue
        if base_auc is None:
            base_auc, base_f1 = au, f1
            print(f"  {name:18s} AUC={au.mean():.4f}  F1*={f1.mean():.4f}  [baseline]", flush=True)
        else:
            dA = au - base_auc; dF = f1 - base_f1
            sA = dA.std(ddof=1)/np.sqrt(len(dA)); sF = dF.std(ddof=1)/np.sqrt(len(dF))
            print(f"  {name:18s} AUC={au.mean():.4f} (Δ{dA.mean():+.4f} {dA.mean()/sA if sA>0 else 0:+.1f}σ)  "
                  f"F1*={f1.mean():.4f} (Δ{dF.mean():+.4f} {dF.mean()/sF if sF>0 else 0:+.1f}σ)", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
