"""ALL blend COMBINATIONS for the general panel. Pool of 7 candidate members (deployed-tuned, calibrated):
LightGBM, RandomForest, CatBoost, XGBoost (the deployed 4) + HistGB, ExtraTrees (weaker trees) + LogReg
(best non-tree, decorrelation long-shot). Evaluate EVERY non-empty subset as an equal-weight blend,
honestly (nested: threshold on train-OOF, score held-out), paired vs the deployed 4-way {lgbm,rf,cat,xgb}.
F1* at the test prior 0.143 + AUC. Run: .venv/bin/python -u analysis/exp_blend_combos.py
"""
import sys, itertools, numpy as np, pandas as pd
sys.path.insert(0, "analysis"); sys.path.insert(0, "pipelines")
import warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
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
POOL = ["lgbm", "rf", "cat", "xgb", "hgb", "et", "logreg"]
DEPLOYED = ("lgbm", "rf", "cat", "xgb")
df = load_panel("general"); _, NUM, CAT, TGT = partition_columns(df)
X_all = df[NUM + CAT].reset_index(drop=True); Y = df[TGT].astype(int).reset_index(drop=True)


def _oof_test(mk, Atr, Ate, y, cat_eval=False):
    oof = np.zeros(len(y))
    for a, b in StratifiedKFold(3, shuffle=True, random_state=SEED).split(Atr, y):
        m = mk()
        if cat_eval:
            i2, ies = train_test_split(a, test_size=0.15, random_state=SEED, stratify=y.iloc[a])
            m.fit(Atr.iloc[i2], y.iloc[i2], eval_set=(Atr.iloc[ies], y.iloc[ies]))
        else:
            m.fit(Atr.iloc[a], y.iloc[a])
        oof[b] = m.predict_proba(Atr.iloc[b])[:, 1]
    mf = mk()
    if cat_eval:
        i2, ies = train_test_split(np.arange(len(y)), test_size=0.15, random_state=SEED + 9, stratify=y)
        mf.fit(Atr.iloc[i2], y.iloc[i2], eval_set=(Atr.iloc[ies], y.iloc[ies]))
    else:
        mf.fit(Atr, y)
    pf = fit_platt(oof, y.values)
    return apply_platt(mf.predict_proba(Ate)[:, 1], pf), apply_platt(oof, pf)


def _dense(Ptr, Pte):
    cols = list(Ptr.columns); cmap = {}
    def num(X):
        A = np.empty((len(X), len(cols)))
        for j, c in enumerate(cols):
            s = X[c]
            if str(s.dtype) in ("category", "object"):
                cmap.setdefault(c, {v: i for i, v in enumerate(Ptr[c].astype("object").value_counts().index)})
                A[:, j] = X[c].astype("object").map(cmap[c]).astype(float)
            else:
                A[:, j] = pd.to_numeric(s, errors="coerce")
        return A
    A = num(Ptr); med = np.nanmedian(A, 0); med = np.where(np.isfinite(med), med, 0.0)
    sc = StandardScaler().fit(np.where(np.isnan(A), med, A))
    def tf(X):
        B = num(X); B = np.where(np.isnan(B), med, B); return np.clip(np.nan_to_num(sc.transform(B)), -8, 8)
    return tf(Ptr), tf(Pte)


def build_members(Xtr, ytr, Xte):
    Xtr, Xte = _drop_leaky(Xtr.reset_index(drop=True), ytr.reset_index(drop=True), Xte.reset_index(drop=True))
    _, ppf = fit_keep_nan(Xtr); Ptr = apply_keep_nan(Xtr, ppf); Pte = apply_keep_nan(Xte, ppf)
    y = ytr.reset_index(drop=True).astype(int); spw = (y == 0).sum() / max(y.sum(), 1)
    cats = _cats(Ptr)
    Xo_tr, enc = _rf_repr(Ptr, cats, fit=True); Xo_te, _ = _rf_repr(Pte, cats, enc=enc, fit=False)
    Xc_tr = _cat_repr(Ptr, cats); Xc_te = _cat_repr(Pte, cats); ci = [Xc_tr.columns.get_loc(c) for c in cats]
    Dtr, Dte = _dense(Ptr, Pte); Dtr = pd.DataFrame(Dtr); Dte = pd.DataFrame(Dte)
    P = {**get_model_params(2000, spw), "scale_pos_weight": 1, "bagging_freq": 1,
         "bagging_fraction": 0.8, "n_jobs": 2, "verbose": -1}
    mks = {
        "lgbm": (lambda: lgb.LGBMClassifier(**P), Ptr, Pte, False),
        "rf":   (_rf_general, Xo_tr, Xo_te, False),
        "xgb":  (_xgb_model, Xo_tr, Xo_te, False),
        "cat":  (lambda: _cat_general(ci, spw), Xc_tr, Xc_te, True),
        "hgb":  (lambda: HistGradientBoostingClassifier(max_iter=600, learning_rate=0.03,
                 l2_regularization=2.0, max_leaf_nodes=31, random_state=SEED), Xo_tr, Xo_te, False),
        "et":   (lambda: ExtraTreesClassifier(n_estimators=800, max_features=0.3, min_samples_leaf=4,
                 max_depth=14, class_weight="balanced", n_jobs=2, random_state=SEED), Xo_tr, Xo_te, False),
        "logreg": (lambda: LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced"), Dtr, Dte, False),
    }
    test, oof = {}, {}
    for k in POOL:
        mk, At, Ae, ce = mks[k]
        test[k], oof[k] = _oof_test(mk, At, Ae, y, cat_eval=ce)
    return (np.column_stack([test[k] for k in POOL]), np.column_stack([oof[k] for k in POOL]))


def _score(idx, oof_M, test_M, ytr, yte):
    bo = oof_M[:, idx].mean(1); bt = test_M[:, idx].mean(1)
    thr = pick_threshold_at_prior(ytr, bo, PRIOR)
    tpr, fpr = tpr_fpr(yte, (bt >= thr).astype(int))
    return f1_at_prior(tpr, fpr, PRIOR), float(roc_auc_score(yte, bt))


SUBSETS = [c for r in range(1, len(POOL) + 1) for c in itertools.combinations(range(len(POOL)), r)]


def one_fold(tr, te):
    Xtr, ytr, Xte, yte = X_all.iloc[tr], Y.iloc[tr], X_all.iloc[te], Y.iloc[te]
    test_M, oof_M = build_members(Xtr, ytr, Xte)
    yv = ytr.reset_index(drop=True).values; yt = yte.values
    return {s: _score(list(s), oof_M, test_M, yv, yt) for s in SUBSETS}


def main():
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED).split(X_all, Y))
    print(f"{len(SUBSETS)} subsets x {len(splits)} folds (nested)...", flush=True)
    out = Parallel(n_jobs=5, backend="loky")(delayed(one_fold)(tr, te) for tr, te in splits)
    dep = tuple(POOL.index(m) for m in DEPLOYED)
    dep_f1 = np.array([r[dep][0] for r in out])
    rows = []
    for s in SUBSETS:
        f1 = np.array([r[s][0] for r in out]); auc = np.array([r[s][1] for r in out])
        d = f1 - dep_f1; se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        rows.append({"members": "+".join(POOL[i] for i in s), "k": len(s),
                     "f1": round(f1.mean(), 4), "auc": round(auc.mean(), 4),
                     "d_f1_vs_deployed": round(d.mean(), 4), "sigma": round(d.mean() / se if se > 0 else 0.0, 2)})
    R = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    R.to_csv("analysis/output/blend_combos.csv", index=False)
    dep_name = "+".join(DEPLOYED)
    print(f"\ndeployed 4-way ({dep_name}): F1*={dep_f1.mean():.4f}\n")
    print("=== TOP 15 subsets by F1* ===")
    print(R.head(15).to_string(index=False), flush=True)
    print(f"\ndeployed rank: {R.index[R['members'] == dep_name][0] + 1} / {len(R)}")
    best = R.iloc[0]
    print(f"best subset: {best['members']}  F1*={best['f1']}  Δ vs deployed={best['d_f1_vs_deployed']:+.4f} ({best['sigma']:+.2f}σ)")
    print("\nsaved -> analysis/output/blend_combos.csv", flush=True)


if __name__ == "__main__":
    main()
