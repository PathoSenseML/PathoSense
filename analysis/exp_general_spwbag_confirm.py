"""Confirm spw1_bag (scale_pos_weight=1 + bagging_freq=1/bagging_fraction=0.8) in the FULL deployed
general blend (LGBM+RF+CatBoost), not just LGBM-only. Per fold: keep_nan preprocess; LGBM computed
TWICE (default params vs spw1_bag), RF+CatBoost shared (so paired delta isolates the LGBM-param change);
blend equal 1/3; F1-opt-at-prior threshold; analytic F1* + AUC. Paired, 2 seeds. Deploy if non-negative.
Run: .venv/bin/python -u analysis/exp_general_spwbag_confirm.py
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from config import SEED
from utils import TEST_PRIORS
from schema import partition_columns
from calibration import fit_platt, apply_platt
from harness import load_panel, _drop_leaky, lgb_oof_and_test, pick_threshold_at_prior, tpr_fpr, f1_at_prior
from preprocess_v2 import fit_keep_nan, apply_keep_nan
from general_blend import _rf_repr, _cat_repr, _cat_model
from run_pipeline import get_model_params

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def rf_cat_components(Xtr_pp, Xte_pp, y_tr, cat_feats, spw):
    X_ord_tr, enc = _rf_repr(Xtr_pp, cat_feats, fit=True)
    X_ord_te, _ = _rf_repr(Xte_pp, cat_feats, enc=enc, fit=False)
    X_cat_tr, X_cat_te = _cat_repr(Xtr_pp, cat_feats), _cat_repr(Xte_pp, cat_feats)
    cat_idx = [X_cat_tr.columns.get_loc(c) for c in cat_feats]
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rf_oof = np.zeros(len(y_tr)); cat_oof = np.zeros(len(y_tr))
    for k, (a, b) in enumerate(skf.split(Xtr_pp, y_tr)):
        rf = RandomForestClassifier(n_estimators=400, max_features="sqrt", n_jobs=4,
                                    class_weight="balanced", random_state=SEED)
        rf.fit(X_ord_tr.iloc[a], y_tr.iloc[a]); rf_oof[b] = rf.predict_proba(X_ord_tr.iloc[b])[:, 1]
        i2, ies = train_test_split(a, test_size=0.15, random_state=SEED + k, stratify=y_tr.iloc[a])
        cb = _cat_model(cat_idx, spw); cb.fit(X_cat_tr.iloc[i2], y_tr.iloc[i2],
                                              eval_set=(X_cat_tr.iloc[ies], y_tr.iloc[ies]))
        cat_oof[b] = cb.predict_proba(X_cat_tr.iloc[b])[:, 1]
    rpl, cpl = fit_platt(rf_oof, y_tr.values), fit_platt(cat_oof, y_tr.values)
    rf_full = RandomForestClassifier(n_estimators=400, max_features="sqrt", n_jobs=4,
                                     class_weight="balanced", random_state=SEED).fit(X_ord_tr, y_tr)
    i2, ies = train_test_split(np.arange(len(y_tr)), test_size=0.15, random_state=SEED + 99, stratify=y_tr)
    cb_full = _cat_model(cat_idx, spw); cb_full.fit(X_cat_tr.iloc[i2], y_tr.iloc[i2],
                                                    eval_set=(X_cat_tr.iloc[ies], y_tr.iloc[ies]))
    rf_c = (apply_platt(rf_oof, rpl), apply_platt(rf_full.predict_proba(X_ord_te)[:, 1], rpl))
    cat_c = (apply_platt(cat_oof, cpl), apply_platt(cb_full.predict_proba(X_cat_te)[:, 1], cpl))
    return rf_c, cat_c


def lgb_component(Xtr_pp, Xte_pp, y_tr, params):
    tp, oof = lgb_oof_and_test(Xtr_pp, y_tr, Xte_pp, params, n_seeds=3, n_folds=5)
    pl = fit_platt(oof, y_tr.values)
    return apply_platt(oof, pl), apply_platt(tp, pl)


if __name__ == "__main__":
    prior = TEST_PRIORS["general"]
    df = load_panel("general"); _, num, cat, tgt = partition_columns(df)
    X = df[num + cat].reset_index(drop=True); y = df[tgt].astype(int).reset_index(drop=True)
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    P_DEF = get_model_params(2000, spw)
    P_SB = {**P_DEF, "scale_pos_weight": 1.0, "bagging_freq": 1, "bagging_fraction": 0.8}
    print(f"=== general blend: spw1_bag LGBM vs default LGBM (RF/cat shared) ===", flush=True)
    rows = []
    for seed in (SEED, SEED + 1000):
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=seed)
        f_def, f_sb, a_def, a_sb = [], [], [], []
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
            rf_c, cat_c = rf_cat_components(Xtr_pp, Xte_pp, ytr, cat_feats, spw)
            for params, fl, al in ((P_DEF, f_def, a_def), (P_SB, f_sb, a_sb)):
                lo, lt = lgb_component(Xtr_pp, Xte_pp, ytr, params)
                b_tr = (lo + rf_c[0] + cat_c[0]) / 3.0
                b_te = (lt + rf_c[1] + cat_c[1]) / 3.0
                t = pick_threshold_at_prior(ytr.values, b_tr, prior)
                fl.append(f1_at_prior(*tpr_fpr(yte.values, (b_te >= t).astype(int)), prior))
                al.append(roc_auc_score(yte.values, b_te))
        f_def, f_sb = np.array(f_def), np.array(f_sb); a_def, a_sb = np.array(a_def), np.array(a_sb)
        df1 = f_sb - f_def; se = df1.std(ddof=1)/np.sqrt(len(df1)); sig = df1.mean()/se if se > 0 else 0
        da = a_sb - a_def; sea = da.std(ddof=1)/np.sqrt(len(da)); siga = da.mean()/sea if sea > 0 else 0
        print(f"  seed{seed} ({len(f_def)} folds): blend F1* {f_def.mean():.4f}->{f_sb.mean():.4f} "
              f"Δ={df1.mean():+.4f} ({sig:+.2f}σ) | AUC {a_def.mean():.4f}->{a_sb.mean():.4f} "
              f"Δ={da.mean():+.4f} ({siga:+.2f}σ)", flush=True)
        rows.append({"seed": seed, "f1_def": f_def.mean(), "f1_sb": f_sb.mean(), "df1": df1.mean(),
                     "sig_f1": sig, "auc_def": a_def.mean(), "auc_sb": a_sb.mean(), "dauc": da.mean(), "sig_auc": siga})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "general_spwbag_confirm.csv"), index=False)
    print(f"\nsaved -> general_spwbag_confirm.csv", flush=True)
