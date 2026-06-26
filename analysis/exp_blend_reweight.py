"""Honest blend-weight optimization for the general 4-member blend (LightGBM, RandomForest, CatBoost,
XGBoost). NESTED: weights chosen per outer fold on the TRAIN out-of-fold preds only, applied to the
held-out fold. Compares weightings (equal / cat-heavy / rf-heavy / catrf / data-optimized for F1* and
for AUC) — paired vs the deployed EQUAL-weight blend. Members = deployed-tuned params, native-NaN, Platt.
Run: .venv/bin/python -u analysis/exp_blend_reweight.py
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "analysis"); sys.path.insert(0, "pipelines")
import warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, train_test_split
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
ORDER = ["lgbm", "rf", "cat", "xgb"]
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
    test = mf.predict_proba(Ate)[:, 1]
    pf = fit_platt(oof, y.values)
    return apply_platt(test, pf), apply_platt(oof, pf)


def build_members(Xtr, ytr, Xte):
    Xtr, Xte = _drop_leaky(Xtr.reset_index(drop=True), ytr.reset_index(drop=True), Xte.reset_index(drop=True))
    _, ppf = fit_keep_nan(Xtr); Ptr = apply_keep_nan(Xtr, ppf); Pte = apply_keep_nan(Xte, ppf)
    y = ytr.reset_index(drop=True).astype(int); spw = (y == 0).sum() / max(y.sum(), 1)
    cats = _cats(Ptr)
    Xo_tr, enc = _rf_repr(Ptr, cats, fit=True); Xo_te, _ = _rf_repr(Pte, cats, enc=enc, fit=False)
    Xc_tr = _cat_repr(Ptr, cats); Xc_te = _cat_repr(Pte, cats); ci = [Xc_tr.columns.get_loc(c) for c in cats]
    P = {**get_model_params(2000, spw), "scale_pos_weight": 1, "bagging_freq": 1,
         "bagging_fraction": 0.8, "n_jobs": 2, "verbose": -1}
    test, oof = {}, {}
    test["lgbm"], oof["lgbm"] = _oof_test(lambda: lgb.LGBMClassifier(**P), Ptr, Pte, y)
    test["rf"], oof["rf"] = _oof_test(_rf_general, Xo_tr, Xo_te, y)
    test["xgb"], oof["xgb"] = _oof_test(_xgb_model, Xo_tr, Xo_te, y)
    test["cat"], oof["cat"] = _oof_test(lambda: _cat_general(ci, spw), Xc_tr, Xc_te, y, cat_eval=True)
    return (np.column_stack([test[k] for k in ORDER]), np.column_stack([oof[k] for k in ORDER]))


def f1_of(w, M, y):
    b = M @ w; thr = pick_threshold_at_prior(y, b, PRIOR)
    tpr, fpr = tpr_fpr(y, (b >= thr).astype(int)); return f1_at_prior(tpr, fpr, PRIOR)


def opt_weights(oof_M, y, objective):
    rng = np.random.RandomState(SEED)
    cands = list(np.eye(4)) + [np.ones(4) / 4] + [rng.dirichlet(np.ones(4) * a) for a in (0.5, 1, 2) for _ in range(200)]
    best_w, best = np.ones(4) / 4, -1.0
    for w in cands:
        b = oof_M @ w
        s = roc_auc_score(y, b) if objective == "auc" else f1_of(w, oof_M, pd.Series(y).values)
        if s > best: best, best_w = s, w
    return best_w


STRATS = {
    "equal":     lambda oM, y: np.ones(4) / 4,
    "cat_heavy": lambda oM, y: np.array([1, 1, 3, 1]) / 6.0,        # lgbm,rf,cat,xgb
    "rf_heavy":  lambda oM, y: np.array([1, 3, 1, 1]) / 6.0,
    "catrf":     lambda oM, y: np.array([1, 3, 3, 1]) / 8.0,
    "opt_f1":    lambda oM, y: opt_weights(oM, y, "f1"),
    "opt_auc":   lambda oM, y: opt_weights(oM, y, "auc"),
}


def one_fold(tr, te):
    Xtr, ytr, Xte, yte = X_all.iloc[tr], Y.iloc[tr], X_all.iloc[te], Y.iloc[te]
    test_M, oof_M = build_members(Xtr, ytr, Xte)
    yv = ytr.reset_index(drop=True).values
    res = {}
    for name, strat in STRATS.items():
        w = strat(oof_M, yv)
        thr = pick_threshold_at_prior(yv, oof_M @ w, PRIOR)
        bt = test_M @ w
        tpr, fpr = tpr_fpr(yte.values, (bt >= thr).astype(int))
        res[name] = (f1_at_prior(tpr, fpr, PRIOR), float(roc_auc_score(yte, bt)), tuple(np.round(w, 3)))
    return res


def main():
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED).split(X_all, Y))
    print(f"running {len(splits)} folds (nested weight opt on train-OOF)...", flush=True)
    out = Parallel(n_jobs=5, backend="loky")(delayed(one_fold)(tr, te) for tr, te in splits)
    agg = {n: {"f1": [], "auc": [], "w": []} for n in STRATS}
    for r in out:
        for n, (f1, auc, w) in r.items():
            agg[n]["f1"].append(f1); agg[n]["auc"].append(auc); agg[n]["w"].append(w)
    eq_f1 = np.array(agg["equal"]["f1"])
    print(f"\n{'strategy':12s} {'F1*':>8s} {'AUC':>8s} {'ΔF1 vs equal':>20s}  mean w (lgbm,rf,cat,xgb)")
    rows = []
    for n in STRATS:
        f1 = np.array(agg[n]["f1"]); auc = np.array(agg[n]["auc"])
        d = f1 - eq_f1; se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        sig = d.mean() / se if se > 0 else 0.0
        wm = np.mean([list(w) for w in agg[n]["w"]], axis=0)
        ds = "—" if n == "equal" else f"{d.mean():+.4f} ({sig:+.2f}σ)"
        print(f"{n:12s} {f1.mean():8.4f} {auc.mean():8.4f} {ds:>20s}  ({','.join(f'{x:.2f}' for x in wm)})", flush=True)
        rows.append({"strategy": n, "f1": round(f1.mean(), 4), "auc": round(auc.mean(), 4),
                     "d_f1": round(d.mean(), 4), "sigma": round(sig, 2),
                     "w_lgbm": round(wm[0], 3), "w_rf": round(wm[1], 3), "w_cat": round(wm[2], 3), "w_xgb": round(wm[3], 3)})
    pd.DataFrame(rows).to_csv("analysis/output/blend_reweight.csv", index=False)
    print("\nsaved -> analysis/output/blend_reweight.csv", flush=True)


if __name__ == "__main__":
    main()
