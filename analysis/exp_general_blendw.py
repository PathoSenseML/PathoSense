"""General own-model blend (LGBM+RF+CatBoost) is fixed at equal 1/3 (general_blend.py). Untested
whether component reweighting helps — general has DOUBLE value (panel model + transfer source).
Mirror of the cancer scorer test, but on general's OWN deployed blend.

Per outer fold: keep_nan preprocess (all features) → LGBM 3-seed ensemble OOF+test, RF + CatBoost
inner-OOF+test, Platt-calibrate each. Then evaluate weightings {equal, rf-up, cat-up, lgb-up} by
F1-opt-at-prior threshold on the blended train-OOF, analytic F1* on test. Paired Δ vs equal.
Same components per fold → fair paired deltas (LGBM n_seeds doesn't bias the comparison).
Deploy bar (new): non-negative across both seeds.
Run: .venv/bin/python -u analysis/exp_general_blendw.py
"""
import os, sys, time
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
from harness import (load_panel, _drop_leaky, lgb_oof_and_test, pick_threshold_at_prior,
                     tpr_fpr, f1_at_prior)
from preprocess_v2 import fit_keep_nan, apply_keep_nan
from general_blend import _rf_repr, _cat_repr, _cat_model
from run_pipeline import get_model_params

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
WEIGHTINGS = {"equal": (1/3, 1/3, 1/3), "rf_up": (0.25, 0.50, 0.25),
              "cat_up": (0.25, 0.25, 0.50), "lgb_up": (0.50, 0.25, 0.25)}


def components(X_tr, y_tr, X_te):
    Xtr, Xte = _drop_leaky(X_tr, y_tr, X_te)
    Xtr_pp, ppf = fit_keep_nan(Xtr); Xte_pp = apply_keep_nan(Xte, ppf)
    for c in Xtr_pp.columns:
        if str(Xtr_pp[c].dtype) == "category":
            Xte_pp[c] = pd.Categorical(Xte_pp[c].astype(object), categories=Xtr_pp[c].cat.categories)
    cat_feats = [c for c in Xtr_pp.columns if str(Xtr_pp[c].dtype) == "category"]
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    params = get_model_params(len(Xtr_pp), spw)
    lgb_test, lgb_oof = lgb_oof_and_test(Xtr_pp, y_tr, Xte_pp, params, n_seeds=3, n_folds=5)
    lpl = fit_platt(lgb_oof, y_tr.values)
    lgb_oof_c, lgb_test_c = apply_platt(lgb_oof, lpl), apply_platt(lgb_test, lpl)
    # RF + CatBoost inner-OOF (train) + full-train model (test)
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
        cb = _cat_model(cat_idx, spw)
        cb.fit(X_cat_tr.iloc[i2], y_tr.iloc[i2], eval_set=(X_cat_tr.iloc[ies], y_tr.iloc[ies]))
        cat_oof[b] = cb.predict_proba(X_cat_tr.iloc[b])[:, 1]
    rpl, cpl = fit_platt(rf_oof, y_tr.values), fit_platt(cat_oof, y_tr.values)
    rf_full = RandomForestClassifier(n_estimators=400, max_features="sqrt", n_jobs=4,
                                     class_weight="balanced", random_state=SEED).fit(X_ord_tr, y_tr)
    i2, ies = train_test_split(np.arange(len(y_tr)), test_size=0.15, random_state=SEED + 99, stratify=y_tr)
    cb_full = _cat_model(cat_idx, spw)
    cb_full.fit(X_cat_tr.iloc[i2], y_tr.iloc[i2], eval_set=(X_cat_tr.iloc[ies], y_tr.iloc[ies]))
    rf_oof_c = apply_platt(rf_oof, rpl); rf_test_c = apply_platt(rf_full.predict_proba(X_ord_te)[:, 1], rpl)
    cat_oof_c = apply_platt(cat_oof, cpl); cat_test_c = apply_platt(cb_full.predict_proba(X_cat_te)[:, 1], cpl)
    return (lgb_oof_c, rf_oof_c, cat_oof_c), (lgb_test_c, rf_test_c, cat_test_c)


if __name__ == "__main__":
    prior = TEST_PRIORS["general"]
    df = load_panel("general"); _, num, cat, tgt = partition_columns(df)
    X = df[num + cat].reset_index(drop=True); y = df[tgt].astype(int).reset_index(drop=True)
    print("=== general own-blend component reweighting (deployed config) ===", flush=True)
    allrows = []
    for bs in (SEED, SEED + 1000):
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=bs)
        acc = {wn: [] for wn in WEIGHTINGS}
        t0 = time.time()
        for tr, te in rskf.split(X, y):
            tr, te = np.asarray(tr), np.asarray(te)
            (lo, ro, co), (lt, rt, ct) = components(
                X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True),
                X.iloc[te].reset_index(drop=True))
            ytr, yte = y.iloc[tr].values, y.iloc[te].values
            for wn, w in WEIGHTINGS.items():
                b_tr = w[0]*lo + w[1]*ro + w[2]*co
                b_te = w[0]*lt + w[1]*rt + w[2]*ct
                t = pick_threshold_at_prior(ytr, b_tr, prior)
                acc[wn].append(f1_at_prior(*tpr_fpr(yte, (b_te >= t).astype(int)), prior))
        base = np.array(acc["equal"]); nf = len(base)
        print(f"  seed{bs}: equal F1*={base.mean():.4f} ({nf} folds, {time.time()-t0:.0f}s)", flush=True)
        for wn in WEIGHTINGS:
            if wn == "equal":
                continue
            a = np.array(acc[wn]); d = a - base
            se = d.std(ddof=1)/np.sqrt(nf); sig = d.mean()/se if se > 0 else 0.0
            print(f"    {wn:7s} F1*={a.mean():.4f}  Δ={d.mean():+.4f} ({sig:+.2f}σ)", flush=True)
            allrows.append({"seed": bs, "weighting": wn, "f1": a.mean(), "delta": d.mean(), "sigma": sig})
    pd.DataFrame(allrows).to_csv(os.path.join(OUT, "general_blendw.csv"), index=False)
    print(f"\nsaved -> general_blendw.csv", flush=True)
