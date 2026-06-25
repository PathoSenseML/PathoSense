"""Gate the tuned RF (+0.0198 OOF AUC) + tuned CatBoost (+0.0046) in the deployed general 4-way blend vs
current, held-out seeds, F1* @ prior + AUC, paired. Deploy if robustly non-negative on F1*.
Run: .venv/bin/python -u analysis/exp_general_rfcat_gate.py
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from config import SEED
from utils import TEST_PRIORS
from schema import partition_columns
from calibration import fit_platt, apply_platt
from harness import load_panel, _drop_leaky, lgb_oof_and_test, pick_threshold_at_prior, tpr_fpr, f1_at_prior
from preprocess_v2 import fit_keep_nan, apply_keep_nan
from general_blend import _rf_repr, _cat_repr, _xgb_model
from run_pipeline import get_model_params

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
best = json.load(open(os.path.join(OUT, "general_rfcat_hpo_best.json")))
RF_TUNED, CAT_TUNED = best["rf"], best["cat"]
RF_CUR = dict(n_estimators=600, max_features="sqrt")
CAT_CUR = dict(iterations=1000, depth=6, learning_rate=0.05)


def blend(Xtr, ytr, Xte, lgb_p, rf_kw, cat_kw):
    Xtr, Xte = _drop_leaky(Xtr, ytr, Xte)
    Xtr_pp, ppf = fit_keep_nan(Xtr); Xte_pp = apply_keep_nan(Xte, ppf)
    for c in Xtr_pp.columns:
        if str(Xtr_pp[c].dtype) == "category":
            Xte_pp[c] = pd.Categorical(Xte_pp[c].astype(object), categories=Xtr_pp[c].cat.categories)
    cf = [c for c in Xtr_pp.columns if str(Xtr_pp[c].dtype) == "category"]
    Xo_tr, enc = _rf_repr(Xtr_pp, cf, fit=True); Xo_te, _ = _rf_repr(Xte_pp, cf, enc=enc, fit=False)
    Xc_tr, Xc_te = _cat_repr(Xtr_pp, cf), _cat_repr(Xte_pp, cf); ci = [Xc_tr.columns.get_loc(c) for c in cf]
    n = len(ytr); spw = (ytr == 0).sum()/max((ytr == 1).sum(), 1)
    lt, lo = lgb_oof_and_test(Xtr_pp, ytr, Xte_pp, lgb_p, n_seeds=3, n_folds=5)
    lpl = fit_platt(lo, ytr.values); lo, lt = apply_platt(lo, lpl), apply_platt(lt, lpl)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED); rf_o = np.zeros(n); xg_o = np.zeros(n); cb_o = np.zeros(n)
    def mk_cat():
        return CatBoostClassifier(**cat_kw, scale_pos_weight=spw, cat_features=ci, random_seed=SEED,
                                  verbose=0, allow_writing_files=False, early_stopping_rounds=40, thread_count=4)
    for k, (a, b) in enumerate(skf.split(Xo_tr, ytr)):
        rf = RandomForestClassifier(**rf_kw, n_jobs=4, class_weight="balanced", random_state=SEED).fit(Xo_tr.iloc[a], ytr.iloc[a]); rf_o[b] = rf.predict_proba(Xo_tr.iloc[b])[:, 1]
        xg = _xgb_model().fit(Xo_tr.iloc[a], ytr.iloc[a]); xg_o[b] = xg.predict_proba(Xo_tr.iloc[b])[:, 1]
        i2, ies = train_test_split(a, test_size=0.15, random_state=SEED+k, stratify=ytr.iloc[a])
        cb = mk_cat(); cb.fit(Xc_tr.iloc[i2], ytr.iloc[i2], eval_set=(Xc_tr.iloc[ies], ytr.iloc[ies])); cb_o[b] = cb.predict_proba(Xc_tr.iloc[b])[:, 1]
    rpl, xpl, cpl = fit_platt(rf_o, ytr.values), fit_platt(xg_o, ytr.values), fit_platt(cb_o, ytr.values)
    rf_f = RandomForestClassifier(**rf_kw, n_jobs=4, class_weight="balanced", random_state=SEED).fit(Xo_tr, ytr)
    xg_f = _xgb_model().fit(Xo_tr, ytr)
    i2, ies = train_test_split(np.arange(n), test_size=0.15, random_state=SEED+99, stratify=ytr)
    cb_f = mk_cat(); cb_f.fit(Xc_tr.iloc[i2], ytr.iloc[i2], eval_set=(Xc_tr.iloc[ies], ytr.iloc[ies]))
    oof = (lo + apply_platt(rf_o, rpl) + apply_platt(cb_o, cpl) + apply_platt(xg_o, xpl)) / 4.0
    test = (lt + apply_platt(rf_f.predict_proba(Xo_te)[:, 1], rpl) + apply_platt(cb_f.predict_proba(Xc_te)[:, 1], cpl)
            + apply_platt(xg_f.predict_proba(Xo_te)[:, 1], xpl)) / 4.0
    return oof, test


def main():
    prior = TEST_PRIORS["general"]
    df = load_panel("general"); _, num, cat, tgt = partition_columns(df)
    X = df[num + cat].reset_index(drop=True); y = df[tgt].astype(int).reset_index(drop=True)
    spw = (y == 0).sum()/max((y == 1).sum(), 1)
    lgbp = {**get_model_params(2000, spw), "scale_pos_weight": 1.0, "bagging_freq": 1, "bagging_fraction": 0.8}
    print("=== general RF+CatBoost tuned vs current in 4-way blend (held-out seeds) ===", flush=True)
    rows = []
    for seed in (SEED + 1000, SEED + 2000):
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=seed)
        cf_, ca_, tf_, ta_ = [], [], [], []
        for tr, te in rskf.split(X, y):
            tr, te = np.asarray(tr), np.asarray(te)
            Xtr, Xte = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True)
            ytr, yte = y.iloc[tr].reset_index(drop=True), y.iloc[te].values
            for rfk, catk, fl, al in ((RF_CUR, CAT_CUR, cf_, ca_), (RF_TUNED, CAT_TUNED, tf_, ta_)):
                oof, test = blend(Xtr, ytr, Xte, lgbp, rfk, catk)
                t = pick_threshold_at_prior(ytr.values, oof, prior)
                fl.append(f1_at_prior(*tpr_fpr(yte, (test >= t).astype(int)), prior)); al.append(roc_auc_score(yte, test))
        cfa, caa, tfa, taa = map(np.array, (cf_, ca_, tf_, ta_)); nf = len(cfa)
        d = tfa - cfa; da = taa - caa
        sig = d.mean()/(d.std(ddof=1)/np.sqrt(nf)) if d.std() > 0 else 0
        siga = da.mean()/(da.std(ddof=1)/np.sqrt(nf)) if da.std() > 0 else 0
        print(f"  seed{seed} ({nf}f): current F1*={cfa.mean():.4f} AUC={caa.mean():.4f} | tuned F1*={tfa.mean():.4f} "
              f"(Δ{d.mean():+.4f},{sig:+.2f}σ) AUC={taa.mean():.4f} (Δ{da.mean():+.4f},{siga:+.2f}σ)", flush=True)
        rows.append({"seed": seed, "cur_f1": cfa.mean(), "tuned_f1": tfa.mean(), "df1": d.mean(), "sig_f1": sig,
                     "cur_auc": caa.mean(), "tuned_auc": taa.mean(), "dauc": da.mean(), "sig_auc": siga})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "general_rfcat_gate.csv"), index=False)
    print("\nsaved -> general_rfcat_gate.csv", flush=True)


if __name__ == "__main__":
    main()
