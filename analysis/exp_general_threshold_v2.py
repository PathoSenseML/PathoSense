"""Re-check general's deployed threshold on the ACTUAL deployed model (spw1_bag LGBM + 4-way
LGBM+RF+CatBoost+XGB blend) — the earlier 0.82 was validated on the old LGBM-only general. Build the
deployed 4-way blend OOF+test per fold, then: deployed bootstrap selector pick + test F1*, and a fixed
threshold grid. 2 seeds. If a fixed threshold robustly beats the selector AND agrees with its per-fold
center, redeploy it.
Run: .venv/bin/python -u analysis/exp_general_threshold_v2.py
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from config import SEED
from utils import TEST_PRIORS, optimize_threshold_f1_balanced
from schema import partition_columns
from calibration import fit_platt, apply_platt
from harness import load_panel, _drop_leaky, lgb_oof_and_test, pick_threshold_at_prior, tpr_fpr, f1_at_prior
from preprocess_v2 import fit_keep_nan, apply_keep_nan
from general_blend import _rf_repr, _cat_repr, _cat_model, _xgb_model
from run_pipeline import get_model_params

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
GRID = [0.78, 0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86]


def blend_fold(Xtr, ytr, Xte, spw, lgb_params):
    Xtr, Xte = _drop_leaky(Xtr, ytr, Xte)
    Xtr_pp, ppf = fit_keep_nan(Xtr); Xte_pp = apply_keep_nan(Xte, ppf)
    for c in Xtr_pp.columns:
        if str(Xtr_pp[c].dtype) == "category":
            Xte_pp[c] = pd.Categorical(Xte_pp[c].astype(object), categories=Xtr_pp[c].cat.categories)
    cat_feats = [c for c in Xtr_pp.columns if str(Xtr_pp[c].dtype) == "category"]
    X_ord_tr, enc = _rf_repr(Xtr_pp, cat_feats, fit=True)
    X_ord_te, _ = _rf_repr(Xte_pp, cat_feats, enc=enc, fit=False)
    X_cat_tr, X_cat_te = _cat_repr(Xtr_pp, cat_feats), _cat_repr(Xte_pp, cat_feats)
    cat_idx = [X_cat_tr.columns.get_loc(c) for c in cat_feats]
    n = len(ytr)
    lt, lo = lgb_oof_and_test(Xtr_pp, ytr, Xte_pp, lgb_params, n_seeds=3, n_folds=5)
    lpl = fit_platt(lo, ytr.values); lo, lt = apply_platt(lo, lpl), apply_platt(lt, lpl)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rf_oof = np.zeros(n); xg_oof = np.zeros(n); cb_oof = np.zeros(n)
    for k, (a, b) in enumerate(skf.split(X_ord_tr, ytr)):
        rf = RandomForestClassifier(n_estimators=600, max_features="sqrt", n_jobs=4,
                                    class_weight="balanced", random_state=SEED).fit(X_ord_tr.iloc[a], ytr.iloc[a])
        rf_oof[b] = rf.predict_proba(X_ord_tr.iloc[b])[:, 1]
        xg = _xgb_model().fit(X_ord_tr.iloc[a], ytr.iloc[a]); xg_oof[b] = xg.predict_proba(X_ord_tr.iloc[b])[:, 1]
        i2, ies = train_test_split(a, test_size=0.15, random_state=SEED + k, stratify=ytr.iloc[a])
        cb = _cat_model(cat_idx, spw); cb.fit(X_cat_tr.iloc[i2], ytr.iloc[i2], eval_set=(X_cat_tr.iloc[ies], ytr.iloc[ies]))
        cb_oof[b] = cb.predict_proba(X_cat_tr.iloc[b])[:, 1]
    rpl, xpl, cpl = fit_platt(rf_oof, ytr.values), fit_platt(xg_oof, ytr.values), fit_platt(cb_oof, ytr.values)
    rf_f = RandomForestClassifier(n_estimators=600, max_features="sqrt", n_jobs=4,
                                  class_weight="balanced", random_state=SEED).fit(X_ord_tr, ytr)
    xg_f = _xgb_model().fit(X_ord_tr, ytr)
    i2, ies = train_test_split(np.arange(n), test_size=0.15, random_state=SEED + 99, stratify=ytr)
    cb_f = _cat_model(cat_idx, spw); cb_f.fit(X_cat_tr.iloc[i2], ytr.iloc[i2], eval_set=(X_cat_tr.iloc[ies], ytr.iloc[ies]))
    oof = (lo + apply_platt(rf_oof, rpl) + apply_platt(cb_oof, cpl) + apply_platt(xg_oof, xpl)) / 4.0
    test = (lt + apply_platt(rf_f.predict_proba(X_ord_te)[:, 1], rpl)
            + apply_platt(cb_f.predict_proba(X_cat_te)[:, 1], cpl)
            + apply_platt(xg_f.predict_proba(X_ord_te)[:, 1], xpl)) / 4.0
    return oof, test


def main():
    prior = TEST_PRIORS["general"]
    df = load_panel("general"); _, num, cat, tgt = partition_columns(df)
    X = df[num + cat].reset_index(drop=True); y = df[tgt].astype(int).reset_index(drop=True)
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    lgb_params = {**get_model_params(2000, spw), "scale_pos_weight": 1.0, "bagging_freq": 1, "bagging_fraction": 0.8}
    print("=== general threshold on DEPLOYED 4-way blend (spw1_bag + xgb) ===", flush=True)
    rows = []
    for seed in (SEED, SEED + 1000):
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=seed)
        f_sel, tsel = [], []; fixed = {t: [] for t in GRID}
        for tr, te in rskf.split(X, y):
            tr, te = np.asarray(tr), np.asarray(te)
            oof, test = blend_fold(X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True),
                                   X.iloc[te].reset_index(drop=True), spw, lgb_params)
            ytr, yte = y.iloc[tr].values, y.iloc[te].values
            td, _, _ = optimize_threshold_f1_balanced(ytr, oof, target_prior=prior, n_bootstrap=200)
            tsel.append(td)
            f_sel.append(f1_at_prior(*tpr_fpr(yte, (test >= td).astype(int)), prior))
            for t in GRID:
                fixed[t].append(f1_at_prior(*tpr_fpr(yte, (test >= t).astype(int)), prior))
        f_sel = np.array(f_sel); nf = len(f_sel)
        best_t = max(GRID, key=lambda t: np.mean(fixed[t]))
        print(f"  seed{seed} ({nf} folds): selector F1*={f_sel.mean():.4f} (picks ~{np.mean(tsel):.3f}); "
              f"best fixed t={best_t} F1*={np.mean(fixed[best_t]):.4f}", flush=True)
        for t in GRID:
            a = np.array(fixed[t]); d = a - f_sel
            se = d.std(ddof=1)/np.sqrt(nf); sig = d.mean()/se if se > 0 else 0
            mark = " *" if t == best_t else (" <-deployed" if t == 0.82 else "")
            print(f"      t={t:.2f}  F1*={a.mean():.4f}  Δ={d.mean():+.4f} ({sig:+.2f}σ){mark}", flush=True)
            rows.append({"seed": seed, "threshold": t, "f1": a.mean(), "delta": d.mean(), "sigma": sig})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "general_threshold_v2.csv"), index=False)
    print(f"\nsaved -> general_threshold_v2.csv", flush=True)


if __name__ == "__main__":
    main()
