"""DIRECT TEST (teammate question): does adding the cancer/pah/cftr subpanel rows to the
GENERAL panel's OWN training improve general's own F1*?

The prior "pooled transfer source" experiment (analysis/exp_transfer_pooled.py) measured the
effect on the SMALL-panel targets via the transfer scorer (hurt all three). It did NOT log a
direct F1* number for the GENERAL panel itself ("AUC barely moved"). This script measures that
single number honestly.

Design — paired RepeatedStratifiedKFold on the GENERAL panel (test fold is ALWAYS pure general):
  ARM A (baseline) : train the deployed general 4-way blend on general-train only.
  ARM B (pooled)   : train the SAME blend on general-train + cancer+pah+cftr rows, EXCLUDING any
                     subpanel row whose Variant_ID is in the held-out general test fold (the
                     leakage control transfer.py uses). Threshold is picked on the GENERAL train
                     rows' OOF only (gives pooling its best shot — isolates "do the extra rows
                     improve the model" from "do they corrupt threshold calibration").
Both arms scored on the identical general test fold, analytic F1* at the general test prior (0.143).

Blend = deployed general: LGBM(spw1_bag) + tuned RF + tuned CatBoost + XGBoost, equal-avg, each
Platt-calibrated. (REDUN_PRUNE_LGBM omitted — second-order +0.0018, applies to both arms equally.)

Run: .venv/bin/python -u analysis/exp_pool_subpanels_into_general.py
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

from config import SEED
from schema import partition_columns, to_categorical, CATEGORICAL_COLS
from calibration import fit_platt, apply_platt
from utils import TEST_PRIORS
from run_pipeline import get_model_params
import harness
from harness import (load_panel, _drop_leaky, lgb_oof_and_test, pick_threshold_at_prior,
                     tpr_fpr, f1_at_prior)
from preprocess_v2 import fit_keep_nan, apply_keep_nan
from general_blend import _rf_repr, _cat_repr, _rf_general, _cat_general, _xgb_model

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
PRIOR = TEST_PRIORS["general"]
harness._LGB_THREADS = 4          # folds run 4-way parallel; cap LGBM threads to match RF/cat/xgb


def blend_fold(Xtr, ytr, Xte, spw, lgb_params):
    """Deployed general 4-way blend. Returns (test_probs, train_oof) both calibrated/equal-avg."""
    Xtr, Xte = _drop_leaky(Xtr, ytr, Xte)
    Xtr_pp, ppf = fit_keep_nan(Xtr); Xte_pp = apply_keep_nan(Xte, ppf)
    cat_feats = [c for c in Xtr_pp.columns if str(Xtr_pp[c].dtype) == "category"]
    for c in cat_feats:                                   # align test categories to TRAIN's
        Xte_pp[c] = pd.Categorical(Xte_pp[c].astype(object), categories=Xtr_pp[c].cat.categories)
    n = len(ytr)

    tp_l, oof_l = lgb_oof_and_test(Xtr_pp, ytr, Xte_pp, lgb_params, n_seeds=3, n_folds=5)

    X_ord_tr, enc = _rf_repr(Xtr_pp, cat_feats, fit=True)
    X_ord_te, _ = _rf_repr(Xte_pp, cat_feats, enc=enc, fit=False)
    X_cat_tr, X_cat_te = _cat_repr(Xtr_pp, cat_feats), _cat_repr(Xte_pp, cat_feats)
    cat_idx = [X_cat_tr.columns.get_loc(c) for c in cat_feats]

    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rf_oof = np.zeros(n); xgb_oof = np.zeros(n); cat_oof = np.zeros(n)
    for k, (a, b) in enumerate(skf.split(Xtr_pp, ytr)):
        rf = _rf_general(); rf.fit(X_ord_tr.iloc[a], ytr.iloc[a])
        rf_oof[b] = rf.predict_proba(X_ord_tr.iloc[b])[:, 1]
        xg = _xgb_model(); xg.fit(X_ord_tr.iloc[a], ytr.iloc[a])
        xgb_oof[b] = xg.predict_proba(X_ord_tr.iloc[b])[:, 1]
        i2, ies = train_test_split(a, test_size=0.15, random_state=SEED + k, stratify=ytr.iloc[a])
        cb = _cat_general(cat_idx, spw)
        cb.fit(X_cat_tr.iloc[i2], ytr.iloc[i2], eval_set=(X_cat_tr.iloc[ies], ytr.iloc[ies]))
        cat_oof[b] = cb.predict_proba(X_cat_tr.iloc[b])[:, 1]

    rf_f = _rf_general().fit(X_ord_tr, ytr); xgb_f = _xgb_model().fit(X_ord_tr, ytr)
    i2, ies = train_test_split(np.arange(n), test_size=0.15, random_state=SEED + 99, stratify=ytr)
    cat_f = _cat_general(cat_idx, spw)
    cat_f.fit(X_cat_tr.iloc[i2], ytr.iloc[i2], eval_set=(X_cat_tr.iloc[ies], ytr.iloc[ies]))
    rf_te = rf_f.predict_proba(X_ord_te)[:, 1]; xgb_te = xgb_f.predict_proba(X_ord_te)[:, 1]
    cat_te = cat_f.predict_proba(X_cat_te)[:, 1]

    members_oof, members_te = [], []
    for oof, te in [(oof_l, tp_l), (rf_oof, rf_te), (cat_oof, cat_te), (xgb_oof, xgb_te)]:
        pl = fit_platt(oof, ytr.values)
        members_oof.append(apply_platt(oof, pl)); members_te.append(apply_platt(te, pl))
    return np.mean(members_te, axis=0), np.mean(members_oof, axis=0)


def _lgb_params(ytr):
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    return spw, {**get_model_params(len(ytr), spw), "scale_pos_weight": 1.0,
                 "bagging_freq": 1, "bagging_fraction": 0.8}


def one_fold(Xg, yg, vid_gen, Xp, yp, vid_pool, catcols, tr, te):
    Xtr_g = Xg.iloc[tr].reset_index(drop=True); ytr_g = yg.iloc[tr].reset_index(drop=True)
    Xte = Xg.iloc[te].reset_index(drop=True); yte = yg.iloc[te].reset_index(drop=True)

    # ARM A — general only
    spwA, lpA = _lgb_params(ytr_g)
    teA, oofA = blend_fold(Xtr_g, ytr_g, Xte, spwA, lpA)
    tA = pick_threshold_at_prior(ytr_g.values, oofA, PRIOR)
    f1A = f1_at_prior(*tpr_fpr(yte.values, (teA >= tA).astype(int)), PRIOR)
    aucA = roc_auc_score(yte.values, teA)

    # ARM B — general + subpanels (exclude test-fold variant ids from the pool)
    test_vids = set(vid_gen[te])
    keep = ~np.isin(vid_pool, list(test_vids))
    Xp_k = Xp[keep].reset_index(drop=True); yp_k = yp[keep].reset_index(drop=True)

    def obj(X):                                            # cat->object so union-concat is clean
        X = X.copy()
        for c in catcols:
            if c in X.columns:
                X[c] = X[c].astype(object)
        return X
    Xtr_b = to_categorical(pd.concat([obj(Xtr_g), obj(Xp_k)], ignore_index=True))
    ytr_b = pd.concat([ytr_g, yp_k], ignore_index=True)
    is_gen = np.concatenate([np.ones(len(ytr_g), bool), np.zeros(len(yp_k), bool)])
    spwB, lpB = _lgb_params(ytr_b)
    teB, oofB = blend_fold(Xtr_b, ytr_b, Xte, spwB, lpB)
    tB = pick_threshold_at_prior(ytr_b.values[is_gen], oofB[is_gen], PRIOR)   # threshold on general rows
    f1B = f1_at_prior(*tpr_fpr(yte.values, (teB >= tB).astype(int)), PRIOR)
    aucB = roc_auc_score(yte.values, teB)
    return f1A, aucA, f1B, aucB, len(Xp_k)


def main():
    df_gen = load_panel("general", acmg=False)
    _, num, cat, tgt = partition_columns(df_gen)
    feat_cols = num + cat
    catcols = [c for c in CATEGORICAL_COLS if c in feat_cols]
    vid_gen = df_gen["Variant_ID"].values
    Xg = df_gen[feat_cols].reset_index(drop=True); yg = df_gen[tgt].astype(int).reset_index(drop=True)

    pools = [load_panel(p, acmg=False) for p in ("cancer", "pah", "cftr")]
    pool = pd.concat(pools, ignore_index=True)
    vid_pool = pool["Variant_ID"].values
    Xp = pool.reindex(columns=feat_cols).reset_index(drop=True)
    yp = pool[tgt].astype(int).reset_index(drop=True)

    shared = np.isin(vid_pool, vid_gen).mean()
    print(f"general n={len(Xg)} (P={int(yg.sum())}/B={int((yg==0).sum())})  "
          f"pool n={len(Xp)} (P={int(yp.sum())}/B={int((yp==0).sum())})  "
          f"pool∩general Variant_ID share={shared:.1%}  prior={PRIOR:.3f}", flush=True)

    rows = []
    for seed in (SEED, SEED + 1000):
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
        splits = list(rskf.split(Xg, yg))
        t0 = time.time()
        out = Parallel(n_jobs=4, backend="threading")(
            delayed(one_fold)(Xg, yg, vid_gen, Xp, yp, vid_pool, catcols, tr, te)
            for tr, te in splits)
        f1A, aucA, f1B, aucB, npk = (np.array(x) for x in zip(*out))
        dF = f1B - f1A; dA = aucB - aucA; nf = len(dF)
        seF = dF.std(ddof=1) / np.sqrt(nf); sigF = dF.mean() / seF if seF > 0 else 0
        seA = dA.std(ddof=1) / np.sqrt(nf); sigA = dA.mean() / seA if seA > 0 else 0
        print(f"\nseed {seed} ({nf} folds, {time.time()-t0:.0f}s, pool kept ~{int(npk.mean())}/fold):", flush=True)
        print(f"  ARM A general-only : F1*={f1A.mean():.4f}  AUC={aucA.mean():.4f}", flush=True)
        print(f"  ARM B +subpanels   : F1*={f1B.mean():.4f}  AUC={aucB.mean():.4f}", flush=True)
        print(f"  paired Δ (B−A)     : F1* {dF.mean():+.4f} ({sigF:+.2f}σ)   AUC {dA.mean():+.4f} ({sigA:+.2f}σ)", flush=True)
        for i in range(nf):
            rows.append({"seed": seed, "fold": i, "f1_A": f1A[i], "f1_B": f1B[i],
                         "auc_A": aucA[i], "auc_B": aucB[i]})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "pool_subpanels_into_general.csv"), index=False)
    dF = (df.f1_B - df.f1_A).values; dA = (df.auc_B - df.auc_A).values; nf = len(dF)
    sigF = dF.mean() / (dF.std(ddof=1) / np.sqrt(nf)); sigA = dA.mean() / (dA.std(ddof=1) / np.sqrt(nf))
    print(f"\n=== POOLED over both seeds ({nf} folds) ===", flush=True)
    print(f"  general-only F1*={df.f1_A.mean():.4f}  +subpanels F1*={df.f1_B.mean():.4f}", flush=True)
    print(f"  Δ F1* {dF.mean():+.4f} ({sigF:+.2f}σ)   Δ AUC {dA.mean():+.4f} ({sigA:+.2f}σ)", flush=True)
    verdict = "REJECT (negative)" if dF.mean() <= 0 else (
        "DEPLOY" if sigF >= 2 else "MARGINAL" if sigF >= 1 else "NOISE")
    print(f"  VERDICT: {verdict}", flush=True)
    print("saved -> output/pool_subpanels_into_general.csv", flush=True)


if __name__ == "__main__":
    main()
