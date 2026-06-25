"""Gate the feature-masking wins in the DEPLOYED general 4-way blend (LGBM_spw1bag + tuned RF + tuned
CatBoost + XGB): base vs redun>0.99 (label-free, drop 274 near-duplicates) vs imp-bottom50% (per-fold
LGBM-gain prune). Held-out seeds, F1* + AUC, paired. Deploy whichever is robustly non-negative.
Run: .venv/bin/python -u analysis/exp_mask_blend_gate.py
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from config import SEED
from utils import TEST_PRIORS
from schema import partition_columns
from calibration import fit_platt, apply_platt
from harness import load_panel, _drop_leaky, lgb_oof_and_test, pick_threshold_at_prior, tpr_fpr, f1_at_prior
from preprocess_v2 import fit_keep_nan, apply_keep_nan
from general_blend import _rf_repr, _cat_repr, _xgb_model, _rf_general, _cat_general
from run_pipeline import get_model_params
from exp_mask_features import redundancy_droplist

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
REDUN99 = set(redundancy_droplist(0.99))     # label-free, computed once


def prune_cols(Xtr_pp, ytr, mode):
    if mode == "base":
        return []
    if mode == "redun99":
        return [c for c in Xtr_pp.columns if c in REDUN99]
    if mode == "imp50":
        catf = [c for c in Xtr_pp.columns if str(Xtr_pp[c].dtype) == "category"]
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, n_jobs=-1,
                               verbose=-1, random_state=SEED).fit(Xtr_pp, ytr, categorical_feature=catf or "auto")
        imp = pd.Series(m.feature_importances_, index=Xtr_pp.columns)
        return list(imp.sort_values().index[:int(0.5 * len(imp))])
    return []


def blend(Xtr, ytr, Xte, lgbp, mode):
    Xtr, Xte = _drop_leaky(Xtr, ytr, Xte)
    Xtr_pp, ppf = fit_keep_nan(Xtr); Xte_pp = apply_keep_nan(Xte, ppf)
    for c in Xtr_pp.columns:
        if str(Xtr_pp[c].dtype) == "category":
            Xte_pp[c] = pd.Categorical(Xte_pp[c].astype(object), categories=Xtr_pp[c].cat.categories)
    drop = [c for c in prune_cols(Xtr_pp, ytr, mode) if c in Xtr_pp.columns]
    Xtr_pp = Xtr_pp.drop(columns=drop); Xte_pp = Xte_pp.drop(columns=[c for c in drop if c in Xte_pp.columns])
    cf = [c for c in Xtr_pp.columns if str(Xtr_pp[c].dtype) == "category"]
    Xo_tr, enc = _rf_repr(Xtr_pp, cf, fit=True); Xo_te, _ = _rf_repr(Xte_pp, cf, enc=enc, fit=False)
    Xc_tr, Xc_te = _cat_repr(Xtr_pp, cf), _cat_repr(Xte_pp, cf); ci = [Xc_tr.columns.get_loc(c) for c in cf]
    n = len(ytr); spw = (ytr == 0).sum()/max((ytr == 1).sum(), 1)
    lt, lo = lgb_oof_and_test(Xtr_pp, ytr, Xte_pp, lgbp, n_seeds=3, n_folds=5)
    lpl = fit_platt(lo, ytr.values); lo, lt = apply_platt(lo, lpl), apply_platt(lt, lpl)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED); rf_o = np.zeros(n); xg_o = np.zeros(n); cb_o = np.zeros(n)
    for k, (a, b) in enumerate(skf.split(Xo_tr, ytr)):
        rf = _rf_general().fit(Xo_tr.iloc[a], ytr.iloc[a]); rf_o[b] = rf.predict_proba(Xo_tr.iloc[b])[:, 1]
        xg = _xgb_model().fit(Xo_tr.iloc[a], ytr.iloc[a]); xg_o[b] = xg.predict_proba(Xo_tr.iloc[b])[:, 1]
        i2, ies = train_test_split(a, test_size=0.15, random_state=SEED+k, stratify=ytr.iloc[a])
        cb = _cat_general(ci, spw); cb.fit(Xc_tr.iloc[i2], ytr.iloc[i2], eval_set=(Xc_tr.iloc[ies], ytr.iloc[ies])); cb_o[b] = cb.predict_proba(Xc_tr.iloc[b])[:, 1]
    rpl, xpl, cpl = fit_platt(rf_o, ytr.values), fit_platt(xg_o, ytr.values), fit_platt(cb_o, ytr.values)
    rf_f = _rf_general().fit(Xo_tr, ytr); xg_f = _xgb_model().fit(Xo_tr, ytr)
    i2, ies = train_test_split(np.arange(n), test_size=0.15, random_state=SEED+99, stratify=ytr)
    cb_f = _cat_general(ci, spw); cb_f.fit(Xc_tr.iloc[i2], ytr.iloc[i2], eval_set=(Xc_tr.iloc[ies], ytr.iloc[ies]))
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
    print(f"=== feature-masking in deployed 4-way blend (redun99 drops {len(REDUN99)}) ===", flush=True)
    rows = []
    for seed in (SEED + 1000, SEED + 2000):
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=seed)
        acc = {m: {"f1": [], "auc": []} for m in ("base", "redun99", "imp50")}
        for tr, te in rskf.split(X, y):
            tr, te = np.asarray(tr), np.asarray(te)
            Xtr, Xte = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True)
            ytr, yte = y.iloc[tr].reset_index(drop=True), y.iloc[te].values
            for mode in ("base", "redun99", "imp50"):
                oof, test = blend(Xtr, ytr, Xte, lgbp, mode)
                t = pick_threshold_at_prior(ytr.values, oof, prior)
                acc[mode]["f1"].append(f1_at_prior(*tpr_fpr(yte, (test >= t).astype(int)), prior))
                acc[mode]["auc"].append(roc_auc_score(yte, test))
        bf = np.array(acc["base"]["f1"]); ba = np.array(acc["base"]["auc"]); nf = len(bf)
        print(f"  seed{seed} ({nf}f): base F1*={bf.mean():.4f} AUC={ba.mean():.4f}", flush=True)
        for mode in ("redun99", "imp50"):
            f = np.array(acc[mode]["f1"]); a = np.array(acc[mode]["auc"]); d = f - bf; da = a - ba
            sig = d.mean()/(d.std(ddof=1)/np.sqrt(nf)) if d.std() > 0 else 0
            print(f"    {mode:8s} F1*={f.mean():.4f} Δ={d.mean():+.4f} ({sig:+.2f}σ) | AUC Δ={da.mean():+.4f}", flush=True)
            rows.append({"seed": seed, "mode": mode, "f1": f.mean(), "df1": d.mean(), "sigma": sig, "dauc": da.mean()})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "mask_blend_gate.csv"), index=False)
    print(f"\nsaved -> mask_blend_gate.csv", flush=True)


if __name__ == "__main__":
    main()
