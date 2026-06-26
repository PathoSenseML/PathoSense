"""Try to push general F1* above 0.55 — operating-point shaping (NOT ranking; AUC is ceilinged ~0.86).

The threshold recal showed +0.0048 F1* at fixed AUC, so the headroom is at the π=0.143 operating
point. Levers, all on the deployed 4-model general blend, honest per-fold RECALIBRATED threshold,
3 seeds, ≥2σ seed-stable bar:
  A. scale_pos_weight sweep (all members) — reshapes precision/recall near the operating point.
  B. blend weights optimized for F1-at-prior on train-OOF (nested) vs deployed equal-weight.
  C. oracle: equal-weight blend at the per-fold SWEPT-best threshold = upper bound of pure
     threshold/operating-point optimization (test-peeked -> optimistic ceiling, for context).

Run: .venv/bin/python -u analysis/exp_general_push_055.py
"""
import sys, warnings
sys.path.insert(0, "analysis"); sys.path.insert(0, "pipelines")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

from config import SEED
from schema import normalize_sentinels, partition_columns, drop_all_nan_rows
from utils import detect_leaky_columns, KNOWN_LEAKY, TEST_PRIORS, PANELS
from predict import _engineer, _fit_own_preprocess, _apply_own_preprocess, _lgbm_redun_keep
from recipe import PANEL_RECIPE
from general_blend import _cats, _rf_repr, _cat_repr, _rf_general, _cat_general, _xgb_model
from models import get_model_params
from calibration import fit_platt, apply_platt
from harness import f1_at_prior, tpr_fpr, pick_threshold_at_prior

PANEL = "general"; PRIOR = TEST_PRIORS[PANEL]
SPW_SWEEP = [1.0, 2.0, 3.0, 4.5, 6.0]           # 1.0 = deployed; find the peak (was monotonic to 2.2)
LGBM_BASE = {"scale_pos_weight": 1.0, "bagging_freq": 1, "bagging_fraction": 0.8}


def _oof_test(kind, Xtr, ytr, Xte, cf, ci, enc, spw, seed, nfold=3):
    """Calibrated (train-OOF, test) for one member at a given scale_pos_weight."""
    def fitpred(Xa, ya, Xb):
        if kind == "lgbm":
            p = {**get_model_params(len(ya), spw), **LGBM_BASE, "scale_pos_weight": spw,
                 "random_state": seed, "verbose": -1}
            m = lgb.LGBMClassifier(**p); m.fit(Xa, ya, categorical_feature=cf or "auto")
            return m.predict_proba(Xb)[:, 1]
        if kind == "rf":
            m = RandomForestClassifier(n_estimators=1100, max_features=0.3, max_depth=9,
                                       min_samples_leaf=10, class_weight={0: 1.0, 1: spw},
                                       n_jobs=4, random_state=seed)
            Xa2, _ = _rf_repr(Xa, cf, enc=enc); Xb2, _ = _rf_repr(Xb, cf, enc=enc)
            m.fit(Xa2, ya); return m.predict_proba(Xb2)[:, 1]
        if kind == "cat":
            m = CatBoostClassifier(iterations=1300, depth=5, learning_rate=0.0268, l2_leaf_reg=2.65,
                                   scale_pos_weight=spw, cat_features=ci, random_seed=seed,
                                   verbose=0, allow_writing_files=False, thread_count=4)
            m.fit(_cat_repr(Xa, cf), ya); return m.predict_proba(_cat_repr(Xb, cf))[:, 1]
        if kind == "xgb":
            m = XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=5, subsample=0.8,
                              colsample_bytree=0.8, reg_lambda=1.5, scale_pos_weight=spw,
                              eval_metric="logloss", n_jobs=4, random_state=seed)
            Xa2, _ = _rf_repr(Xa, cf, enc=enc); Xb2, _ = _rf_repr(Xb, cf, enc=enc)
            m.fit(Xa2, ya); return m.predict_proba(Xb2)[:, 1]
    oof = np.zeros(len(ytr)); skf = StratifiedKFold(nfold, shuffle=True, random_state=seed)
    for a, b in skf.split(Xtr, ytr):
        oof[b] = fitpred(Xtr.iloc[a], ytr.iloc[a], Xtr.iloc[b])
    pl = fit_platt(oof, ytr.values)
    return apply_platt(oof, pl), apply_platt(fitpred(Xtr, ytr, Xte), pl)


def _f1_recal(ytr, btr, yte, bte):
    thr = pick_threshold_at_prior(ytr, btr, PRIOR)
    return f1_at_prior(*tpr_fpr(yte, (bte >= thr).astype(int)), PRIOR)


def _f1_swept(yte, bte):
    qs = np.quantile(bte, np.arange(.02, .98, .005))
    return max(f1_at_prior(*tpr_fpr(yte, (bte >= q).astype(int)), PRIOR) for q in qs)


def main():
    df = pd.read_csv(PANELS[PANEL]); df = normalize_sentinels(df); df = drop_all_nan_rows(df)
    de = _engineer(df.copy(), PANEL_RECIPE[PANEL]); _, num, cat, tgt = partition_columns(de)
    y = de[tgt].astype(int).reset_index(drop=True); X = de[num + cat].reset_index(drop=True)
    leaky = detect_leaky_columns(X[num], y, auc_threshold=0.99, verbose=False)
    X = X.drop(columns=sorted(set(leaky) | (KNOWN_LEAKY & set(X.columns))))

    rows = {f"spw={s}": [] for s in SPW_SWEEP}
    rows["f1wt_blend"] = []; rows["oracle_swept"] = []; aucs = []
    for seed in (SEED, SEED + 1000, SEED + 2000):
        outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=seed)
        for tr, te in outer.split(X, y):
            Xtr0, ytr = X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True)
            Xte0, yte = X.iloc[te].reset_index(drop=True), y.iloc[te].reset_index(drop=True)
            Xtr, prep = _fit_own_preprocess(Xtr0, ytr, PANEL_RECIPE[PANEL])
            Xte = _apply_own_preprocess(Xte0, prep).reindex(columns=Xtr.columns)
            cf = _cats(Xtr); ci = [Xtr.columns.get_loc(c) for c in cf]
            _, enc = _rf_repr(Xtr, cf, fit=True)
            keep = _lgbm_redun_keep(Xtr); Xtr_l, Xte_l = Xtr[keep], Xte[keep]
            yv_tr, yv_te = ytr.values, yte.values
            # build members per spw (lgbm on pruned cols; others full)
            for s in SPW_SWEEP:
                lo, le = _oof_test("lgbm", Xtr_l, ytr, Xte_l, cf, ci, enc, s, seed)
                ro, re = _oof_test("rf", Xtr, ytr, Xte, cf, ci, enc, s, seed)
                co, ce = _oof_test("cat", Xtr, ytr, Xte, cf, ci, enc, s, seed)
                xo, xe = _oof_test("xgb", Xtr, ytr, Xte, cf, ci, enc, s, seed)
                btr = (lo + ro + co + xo) / 4; bte = (le + re + ce + xe) / 4
                rows[f"spw={s}"].append(_f1_recal(yv_tr, btr, yv_te, bte))
                if s == 1.0:
                    aucs.append(roc_auc_score(yv_te, bte))
                    rows["oracle_swept"].append(_f1_swept(yv_te, bte))
                    # F1-at-prior optimized blend weights (nested grid on train-OOF)
                    members_tr = [lo, ro, co, xo]; members_te = [le, re, ce, xe]
                    best_w, best_f = None, -1
                    grid = [0.1, 0.25, 0.4, 0.55, 0.7]
                    for wl in grid:
                        for wr in grid:
                            for wc in grid:
                                wx = 1 - wl - wr - wc
                                if wx < 0.05: continue
                                wb = wl*members_tr[0]+wr*members_tr[1]+wc*members_tr[2]+wx*members_tr[3]
                                thr = pick_threshold_at_prior(yv_tr, wb, PRIOR)
                                f = f1_at_prior(*tpr_fpr(yv_tr, (wb >= thr).astype(int)), PRIOR)
                                if f > best_f: best_f, best_w = f, (wl, wr, wc, wx)
                    wl, wr, wc, wx = best_w
                    wbte = wl*members_te[0]+wr*members_te[1]+wc*members_te[2]+wx*members_te[3]
                    rows["f1wt_blend"].append(_f1_recal(yv_tr,
                        wl*members_tr[0]+wr*members_tr[1]+wc*members_tr[2]+wx*members_tr[3], yv_te, wbte))

    base = np.array(rows["spw=1.0"])
    print(f"=== general push>0.55 (blend AUC={np.mean(aucs):.4f}, prior {PRIOR}, recalibrated thr, {len(base)} folds) ===")
    print(f"  DEPLOYED (spw=1.0, equal-weight)  F1*={base.mean():.4f}")
    for k in [f"spw={s}" for s in SPW_SWEEP if s != 1.0] + ["f1wt_blend", "oracle_swept"]:
        a = np.array(rows[k]); d = a - base
        se = d.std(ddof=1)/np.sqrt(len(d)) if len(d) > 1 else 0; sig = d.mean()/se if se > 0 else 0
        over = "  >0.55!" if a.mean() > 0.55 else ""
        print(f"  {k:16s} F1*={a.mean():.4f}  Δ={d.mean():+.4f} ({sig:+.2f}σ){over}")
    print("DONE")


if __name__ == "__main__":
    main()
