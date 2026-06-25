"""General blend: add a NON-tree 5th member (linear on one-hot) for max error decorrelation from the 4
tree members. Diversity (XGBoost) was the last win; a linear model's errors are fundamentally different.
Test +LogReg and +SVM (one-hot + standardized, calibrated) as a 5th member of the deployed 4-way blend
(LGBM_spw1bag+RF+CatBoost+XGB). Paired Δ vs current 4-way, 2 seeds, F1* + AUC.
Run: .venv/bin/python -u analysis/exp_general_linear.py
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from config import SEED
from utils import TEST_PRIORS
from schema import partition_columns
from calibration import fit_platt, apply_platt
from harness import load_panel, _drop_leaky, lgb_oof_and_test, pick_threshold_at_prior, tpr_fpr, f1_at_prior
from preprocess_v2 import fit_keep_nan, apply_keep_nan
from general_blend import _rf_repr, _cat_repr, _cat_model, _xgb_model
from run_pipeline import get_model_params

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def _linear_pipe(cat_feats, num_feats, kind):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_feats),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), cat_feats)])
    if kind == "logreg":
        clf = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")
    else:
        clf = SVC(C=1.0, kernel="rbf", class_weight="balanced", probability=False)
    return Pipeline([("pre", pre), ("clf", clf)])


def _lin_proba(pipe, X, kind):
    if kind == "logreg":
        return pipe.predict_proba(X)[:, 1]
    s = pipe.decision_function(X)                       # SVC → sigmoid-squash decision
    return 1.0 / (1.0 + np.exp(-s))


def components(Xtr_pp, Xte_pp, y_tr, cat_feats, num_feats, spw, lgb_params):
    out = {}
    tp, oof = lgb_oof_and_test(Xtr_pp, y_tr, Xte_pp, lgb_params, n_seeds=3, n_folds=5)
    out["lgbm"] = (oof, tp)
    X_ord_tr, enc = _rf_repr(Xtr_pp, cat_feats, fit=True); X_ord_te, _ = _rf_repr(Xte_pp, cat_feats, enc=enc, fit=False)
    X_cat_tr, X_cat_te = _cat_repr(Xtr_pp, cat_feats), _cat_repr(Xte_pp, cat_feats)
    cat_idx = [X_cat_tr.columns.get_loc(c) for c in cat_feats]; n = len(y_tr)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    def sk(make, Xtr, Xte, proba):
        oo = np.zeros(n)
        for a, b in skf.split(Xtr, y_tr):
            m = make(); m.fit(Xtr.iloc[a], y_tr.iloc[a]); oo[b] = proba(m, Xtr.iloc[b])
        mf = make(); mf.fit(Xtr, y_tr); return oo, proba(mf, Xte)
    pr = lambda m, X: m.predict_proba(X)[:, 1]
    out["rf"] = sk(lambda: RandomForestClassifier(n_estimators=600, max_features="sqrt", n_jobs=4,
                                                  class_weight="balanced", random_state=SEED), X_ord_tr, X_ord_te, pr)
    out["xgb"] = sk(lambda: _xgb_model(), X_ord_tr, X_ord_te, pr)
    for kind in ("logreg", "svm"):
        out[kind] = sk(lambda k=kind: _linear_pipe(cat_feats, num_feats, k), Xtr_pp, Xte_pp,
                       lambda m, X, k=kind: _lin_proba(m, X, k))
    cb_oof = np.zeros(n)
    for k, (a, b) in enumerate(skf.split(X_cat_tr, y_tr)):
        i2, ies = train_test_split(a, test_size=0.15, random_state=SEED + k, stratify=y_tr.iloc[a])
        cb = _cat_model(cat_idx, spw); cb.fit(X_cat_tr.iloc[i2], y_tr.iloc[i2], eval_set=(X_cat_tr.iloc[ies], y_tr.iloc[ies]))
        cb_oof[b] = cb.predict_proba(X_cat_tr.iloc[b])[:, 1]
    i2, ies = train_test_split(np.arange(n), test_size=0.15, random_state=SEED + 99, stratify=y_tr)
    cbf = _cat_model(cat_idx, spw); cbf.fit(X_cat_tr.iloc[i2], y_tr.iloc[i2], eval_set=(X_cat_tr.iloc[ies], y_tr.iloc[ies]))
    out["cat"] = (cb_oof, cbf.predict_proba(X_cat_te)[:, 1])
    for k, (oo, tt) in out.items():
        pl = fit_platt(oo, y_tr.values); out[k] = (apply_platt(oo, pl), apply_platt(tt, pl))
    return out


BLENDS = {"current4": ["lgbm", "rf", "cat", "xgb"],
          "+logreg": ["lgbm", "rf", "cat", "xgb", "logreg"],
          "+svm": ["lgbm", "rf", "cat", "xgb", "svm"]}


def main():
    prior = TEST_PRIORS["general"]
    df = load_panel("general"); _, num, cat, tgt = partition_columns(df)
    X = df[num + cat].reset_index(drop=True); y = df[tgt].astype(int).reset_index(drop=True)
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    lgb_params = {**get_model_params(2000, spw), "scale_pos_weight": 1.0, "bagging_freq": 1, "bagging_fraction": 0.8}
    print("=== general +linear 5th member (vs deployed 4-way) ===", flush=True)
    rows = []
    for seed in (SEED, SEED + 1000):
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=seed)
        accF = {k: [] for k in BLENDS}; accA = {k: [] for k in BLENDS}
        for tr, te in rskf.split(X, y):
            tr, te = np.asarray(tr), np.asarray(te)
            Xtr, Xte = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True)
            ytr, yte = y.iloc[tr].reset_index(drop=True), y.iloc[te].reset_index(drop=True)
            Xtr, Xte = _drop_leaky(Xtr, ytr, Xte)
            Xtr_pp, ppf = fit_keep_nan(Xtr); Xte_pp = apply_keep_nan(Xte, ppf)
            for c in Xtr_pp.columns:
                if str(Xtr_pp[c].dtype) == "category":
                    Xte_pp[c] = pd.Categorical(Xte_pp[c].astype(object), categories=Xtr_pp[c].cat.categories)
            cat_feats = [c for c in Xtr_pp.columns if str(Xtr_pp[c].dtype) == "category"]
            num_feats = [c for c in Xtr_pp.columns if c not in cat_feats]
            comp = components(Xtr_pp, Xte_pp, ytr, cat_feats, num_feats, spw, lgb_params)
            for name, mem in BLENDS.items():
                btr = np.mean([comp[m][0] for m in mem], axis=0); bte = np.mean([comp[m][1] for m in mem], axis=0)
                t = pick_threshold_at_prior(ytr.values, btr, prior)
                accF[name].append(f1_at_prior(*tpr_fpr(yte.values, (bte >= t).astype(int)), prior))
                accA[name].append(roc_auc_score(yte.values, bte))
        base = np.array(accF["current4"]); baseA = np.array(accA["current4"]); nf = len(base)
        print(f"  seed{seed} ({nf} folds): current4 F1*={base.mean():.4f} AUC={baseA.mean():.4f}", flush=True)
        for k in ("+logreg", "+svm"):
            f = np.array(accF[k]); a = np.array(accA[k]); d = f - base; da = a - baseA
            sig = d.mean()/(d.std(ddof=1)/np.sqrt(nf)) if d.std() > 0 else 0
            siga = da.mean()/(da.std(ddof=1)/np.sqrt(nf)) if da.std() > 0 else 0
            print(f"    {k:8s} F1*={f.mean():.4f} Δ={d.mean():+.4f} ({sig:+.2f}σ) | AUC Δ={da.mean():+.4f} ({siga:+.2f}σ)", flush=True)
            rows.append({"seed": seed, "blend": k, "f1": f.mean(), "df1": d.mean(), "sig_f1": sig, "dauc": da.mean(), "sig_auc": siga})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "general_linear.csv"), index=False)
    print(f"\nsaved -> general_linear.csv", flush=True)


if __name__ == "__main__":
    main()
