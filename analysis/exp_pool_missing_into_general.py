"""PROPER measurement: add ONLY the subpanel-only ("missing") variants to the GENERAL panel's
training and measure general's own deployed-blend F1*/AUC. Deploy decision = net-positive AND
sign-stable across seeds (NOT a >=2sigma gate — zero-cost, never-negative is the bar).

Why missing-only: 66.5% of subpanel variants already live in general; adding those just upweights
existing rows (a confound). The clean test is the 289 variants general has NEVER seen (128 P / 161 B
-> benign-enriched, and the test prior is benign-dominant, so this is the plausibly-useful direction).
Because they are subpanel-only they can never appear in a general test fold -> leakage-free by
construction; it is a CONSTANT +289-row augmentation on every fold.

Model = the actual DEPLOYED general 4-way blend (LGBM spw1_bag + tuned RF + tuned CatBoost + XGB),
native-NaN, ALL features, each Platt-calibrated, equal average. Inner models forced SINGLE-THREAD so
the threaded fold fan-out cannot deadlock (the multi-thread x multi-thread nesting zombied the first
attempt). Threshold picked on the GENERAL train rows' OOF only.

ARM A: general-train only.  ARM B: general-train + 289 missing variants.  Same pure-general test fold.
4 seeds x 5x5 = 100 paired folds. CSV saved after each seed; per-seed + pooled + trajectory printed.

Run: .venv/bin/python -u analysis/exp_pool_missing_into_general.py
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from joblib import Parallel, delayed

from config import SEED
from schema import partition_columns, to_categorical, CATEGORICAL_COLS
from utils import TEST_PRIORS
from run_pipeline import get_model_params
from calibration import fit_platt, apply_platt
import harness
from harness import (load_panel, _drop_leaky, lgb_oof_and_test, pick_threshold_at_prior,
                     tpr_fpr, f1_at_prior)
from preprocess_v2 import fit_keep_nan, apply_keep_nan
from general_blend import _rf_repr, _cat_repr

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
CSV = os.path.join(OUT, "pool_missing_into_general.csv")
PRIOR = TEST_PRIORS["general"]
SEEDS = [SEED, SEED + 1000, SEED + 2000, SEED + 3000]
harness._LGB_THREADS = 1          # single-thread LGBM (folds parallelize across the joblib pool)


# deployed general blend factories, forced single-thread (results are seed-deterministic; thread
# count does not change the column set / splits, and any tiny float noise cancels in the paired delta)
def _rf():
    return RandomForestClassifier(n_estimators=1100, max_features=0.3, max_depth=9,
                                  min_samples_leaf=10, n_jobs=1, class_weight="balanced",
                                  random_state=SEED)

def _xgb():
    return XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=5, subsample=0.8,
                         colsample_bytree=0.8, reg_lambda=1.5, eval_metric="logloss",
                         n_jobs=1, random_state=SEED)

def _cat(cat_idx, spw):
    return CatBoostClassifier(iterations=1300, depth=5, learning_rate=0.0268, l2_leaf_reg=2.65,
                              scale_pos_weight=spw, cat_features=cat_idx, random_seed=SEED,
                              verbose=0, allow_writing_files=False, early_stopping_rounds=40,
                              thread_count=1)


def blend_fold(Xtr, ytr, Xte):
    """Deployed general 4-way blend (single-thread). Returns (test_probs, oof), both equal-avg calibrated."""
    Xtr, Xte = _drop_leaky(Xtr, ytr, Xte)
    Xtr_pp, ppf = fit_keep_nan(Xtr); Xte_pp = apply_keep_nan(Xte, ppf)
    cat_feats = [c for c in Xtr_pp.columns if str(Xtr_pp[c].dtype) == "category"]
    for c in cat_feats:
        Xte_pp[c] = pd.Categorical(Xte_pp[c].astype(object), categories=Xtr_pp[c].cat.categories)
    n = len(ytr); spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    lgb_params = {**get_model_params(n, spw), "scale_pos_weight": 1.0, "bagging_freq": 1, "bagging_fraction": 0.8}

    tp_l, oof_l = lgb_oof_and_test(Xtr_pp, ytr, Xte_pp, lgb_params, n_seeds=3, n_folds=5)
    X_ord_tr, enc = _rf_repr(Xtr_pp, cat_feats, fit=True)
    X_ord_te, _ = _rf_repr(Xte_pp, cat_feats, enc=enc, fit=False)
    X_cat_tr, X_cat_te = _cat_repr(Xtr_pp, cat_feats), _cat_repr(Xte_pp, cat_feats)
    cat_idx = [X_cat_tr.columns.get_loc(c) for c in cat_feats]

    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rf_oof = np.zeros(n); xgb_oof = np.zeros(n); cat_oof = np.zeros(n)
    for k, (a, b) in enumerate(skf.split(Xtr_pp, ytr)):
        rf = _rf(); rf.fit(X_ord_tr.iloc[a], ytr.iloc[a]); rf_oof[b] = rf.predict_proba(X_ord_tr.iloc[b])[:, 1]
        xg = _xgb(); xg.fit(X_ord_tr.iloc[a], ytr.iloc[a]); xgb_oof[b] = xg.predict_proba(X_ord_tr.iloc[b])[:, 1]
        i2, ies = train_test_split(a, test_size=0.15, random_state=SEED + k, stratify=ytr.iloc[a])
        cb = _cat(cat_idx, spw); cb.fit(X_cat_tr.iloc[i2], ytr.iloc[i2], eval_set=(X_cat_tr.iloc[ies], ytr.iloc[ies]))
        cat_oof[b] = cb.predict_proba(X_cat_tr.iloc[b])[:, 1]
    rf_f = _rf().fit(X_ord_tr, ytr); xgb_f = _xgb().fit(X_ord_tr, ytr)
    i2, ies = train_test_split(np.arange(n), test_size=0.15, random_state=SEED + 99, stratify=ytr)
    cat_f = _cat(cat_idx, spw); cat_f.fit(X_cat_tr.iloc[i2], ytr.iloc[i2], eval_set=(X_cat_tr.iloc[ies], ytr.iloc[ies]))
    rf_te = rf_f.predict_proba(X_ord_te)[:, 1]; xgb_te = xgb_f.predict_proba(X_ord_te)[:, 1]
    cat_te = cat_f.predict_proba(X_cat_te)[:, 1]

    mo, mt = [], []
    for oof, te in [(oof_l, tp_l), (rf_oof, rf_te), (cat_oof, cat_te), (xgb_oof, xgb_te)]:
        pl = fit_platt(oof, ytr.values); mo.append(apply_platt(oof, pl)); mt.append(apply_platt(te, pl))
    return np.mean(mt, axis=0), np.mean(mo, axis=0)


def one_fold(Xg, yg, Xm, ym, tr, te):
    Xtr_g = Xg.iloc[tr].reset_index(drop=True); ytr_g = yg.iloc[tr].reset_index(drop=True)
    Xte = Xg.iloc[te].reset_index(drop=True); yte = yg.iloc[te].reset_index(drop=True)
    # ARM A
    teA, oofA = blend_fold(Xtr_g, ytr_g, Xte)
    tA = pick_threshold_at_prior(ytr_g.values, oofA, PRIOR)
    f1A = f1_at_prior(*tpr_fpr(yte.values, (teA >= tA).astype(int)), PRIOR); aucA = roc_auc_score(yte.values, teA)
    # ARM B  (general train + constant missing rows; threshold on general rows only)
    Xtr_b = to_categorical(pd.concat([Xtr_g, Xm], ignore_index=True))
    ytr_b = pd.concat([ytr_g, ym], ignore_index=True)
    is_gen = np.concatenate([np.ones(len(ytr_g), bool), np.zeros(len(ym), bool)])
    teB, oofB = blend_fold(Xtr_b, ytr_b, Xte)
    tB = pick_threshold_at_prior(ytr_b.values[is_gen], oofB[is_gen], PRIOR)
    f1B = f1_at_prior(*tpr_fpr(yte.values, (teB >= tB).astype(int)), PRIOR); aucB = roc_auc_score(yte.values, teB)
    return f1A, aucA, f1B, aucB


def main():
    df_gen = load_panel("general", acmg=False)
    _, num, cat, tgt = partition_columns(df_gen)
    feat_cols = num + cat
    catcols = [c for c in CATEGORICAL_COLS if c in feat_cols]
    vidg = set(df_gen["Variant_ID"])
    Xg = df_gen[feat_cols].reset_index(drop=True); yg = df_gen[tgt].astype(int).reset_index(drop=True)

    pool = pd.concat([load_panel(p, acmg=False) for p in ("cancer", "pah", "cftr")], ignore_index=True)
    pool = pool.drop_duplicates("Variant_ID")
    miss = pool[~pool["Variant_ID"].isin(vidg)].reset_index(drop=True)

    def obj(X):
        X = X.copy()
        for c in catcols:
            if c in X.columns:
                X[c] = X[c].astype(object)
        return X
    # arm A / test frame: general with category dtype (load_panel already cast it). Missing block as
    # OBJECT so the arm-B concat upcasts to object and to_categorical then UNIONS categories cleanly.
    Xg_catA = Xg                                    # already category dtype from load_panel
    Xm = obj(miss.reindex(columns=feat_cols).reset_index(drop=True))
    ym = miss[tgt].astype(int).reset_index(drop=True)

    print(f"general n={len(Xg)} (P={int(yg.sum())}/B={int((yg==0).sum())})  "
          f"missing-only added={len(Xm)} (P={int(ym.sum())}/B={int((ym==0).sum())})  "
          f"prior={PRIOR:.3f}  seeds={len(SEEDS)}x25=100 folds", flush=True)

    rows = []
    for si, seed in enumerate(SEEDS):
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
        splits = list(rskf.split(Xg_catA, yg)); t0 = time.time()
        # arm A uses category-dtype general; arm B unions with object missing inside one_fold
        out = Parallel(n_jobs=14, backend="threading")(
            delayed(one_fold)(Xg_catA, yg, Xm, ym, tr, te) for tr, te in splits)
        for fi, (f1A, aucA, f1B, aucB) in enumerate(out):
            rows.append({"seed": seed, "fold": fi, "f1_A": f1A, "f1_B": f1B, "auc_A": aucA, "auc_B": aucB})
        pd.DataFrame(rows).to_csv(CSV, index=False)
        df = pd.DataFrame(rows)
        s = df[df.seed == seed]; dF = (s.f1_B - s.f1_A).values; dA = (s.auc_B - s.auc_A).values; nf = len(dF)
        sigF = dF.mean() / (dF.std(ddof=1) / np.sqrt(nf)); sigA = dA.mean() / (dA.std(ddof=1) / np.sqrt(nf))
        # pooled-so-far trajectory
        dFa = (df.f1_B - df.f1_A).values; dAa = (df.auc_B - df.auc_A).values; nfa = len(dFa)
        sigFa = dFa.mean() / (dFa.std(ddof=1) / np.sqrt(nfa)); sigAa = dAa.mean() / (dAa.std(ddof=1) / np.sqrt(nfa))
        print(f"seed {seed} ({nf}f, {time.time()-t0:.0f}s): A_F1={s.f1_A.mean():.4f} B_F1={s.f1_B.mean():.4f} "
              f"dF1={dF.mean():+.4f}({sigF:+.2f}s) dAUC={dA.mean():+.4f}({sigA:+.2f}s)  |  "
              f"POOLED-{nfa}f dF1={dFa.mean():+.4f}({sigFa:+.2f}s) dAUC={dAa.mean():+.4f}({sigAa:+.2f}s)", flush=True)

    df = pd.DataFrame(rows)
    dF = (df.f1_B - df.f1_A).values; dA = (df.auc_B - df.auc_A).values; nf = len(dF)
    sigF = dF.mean() / (dF.std(ddof=1) / np.sqrt(nf)); sigA = dA.mean() / (dA.std(ddof=1) / np.sqrt(nf))
    n_seed_pos = sum((df[df.seed == s].f1_B.mean() - df[df.seed == s].f1_A.mean()) > 0 for s in SEEDS)
    print(f"\n=== FINAL ({nf} folds, {len(SEEDS)} seeds) ===", flush=True)
    print(f"general-only F1={df.f1_A.mean():.4f}  +missing F1={df.f1_B.mean():.4f}", flush=True)
    print(f"dF1={dF.mean():+.4f} ({sigF:+.2f}s)  dAUC={dA.mean():+.4f} ({sigA:+.2f}s)  "
          f"seeds_positive={n_seed_pos}/{len(SEEDS)}  B_wins_F1={int((df.f1_B>df.f1_A).sum())}/{nf}", flush=True)
    stable = (dF.mean() > 0) and (n_seed_pos == len(SEEDS))
    print(f"DECISION: {'DEPLOY (net-positive + every seed positive)' if stable else 'do not deploy (sign-unstable or non-positive)'}", flush=True)
    print(f"saved -> {os.path.basename(CSV)}", flush=True)


if __name__ == "__main__":
    main()
