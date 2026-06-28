"""FAST/ROBUST rerun of the pool-subpanels-into-general test (the threaded 4-way-blend version
deadlocked into a zombie). Same question, same leakage control, but:
  * SEQUENTIAL folds (no joblib) -> no nested-threadpool deadlock; LGBM uses all cores per fold.
  * Model = deployed general LGBM backbone (spw1_bag params, native-NaN, ALL features, multi-seed).
    The pooling EFFECT SIGN is driven by the data (subpanel AL_ covariate shift + label-prior), not
    the model family, so the LGBM backbone is a faithful proxy for the 4-way blend's direction.
  * Incremental CSV save after EVERY fold -> nothing lost if interrupted.
  * ASCII-only prints.

ARM A: general-train only.  ARM B: general-train + cancer+pah+cftr (subpanel rows whose Variant_ID
is in the held-out general test fold are EXCLUDED). Threshold picked on general train rows only.
Both scored on the identical pure-general test fold, analytic F1* at the general prior (0.143).

Run: .venv/bin/python -u analysis/exp_pool_subpanels_fast.py
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score

from config import SEED
from schema import partition_columns, to_categorical, CATEGORICAL_COLS
from utils import TEST_PRIORS
from run_pipeline import get_model_params
import harness
from harness import (load_panel, _drop_leaky, lgb_oof_and_test, pick_threshold_at_prior,
                     tpr_fpr, f1_at_prior)
from preprocess_v2 import fit_keep_nan, apply_keep_nan

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
CSV = os.path.join(OUT, "pool_subpanels_into_general.csv")
PRIOR = TEST_PRIORS["general"]
N_SEEDS_LGB = 2          # internal LGBM seeds (paired delta cancels most of this)
harness._LGB_THREADS = -1  # sequential folds -> let LGBM use all cores; no nested threadpools


def lgb_fold(Xtr, ytr, Xte):
    """Deployed general LGBM backbone: native-NaN all-features, spw1_bag. Returns (test_probs, oof)."""
    Xtr, Xte = _drop_leaky(Xtr, ytr, Xte)
    Xtr_pp, ppf = fit_keep_nan(Xtr); Xte_pp = apply_keep_nan(Xte, ppf)
    for c in Xtr_pp.columns:
        if str(Xtr_pp[c].dtype) == "category":
            Xte_pp[c] = pd.Categorical(Xte_pp[c].astype(object), categories=Xtr_pp[c].cat.categories)
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    params = {**get_model_params(len(ytr), spw), "scale_pos_weight": 1.0,
              "bagging_freq": 1, "bagging_fraction": 0.8}
    return lgb_oof_and_test(Xtr_pp, ytr, Xte_pp, params, n_seeds=N_SEEDS_LGB, n_folds=5)


def main():
    df_gen = load_panel("general", acmg=False)
    _, num, cat, tgt = partition_columns(df_gen)
    feat_cols = num + cat
    catcols = [c for c in CATEGORICAL_COLS if c in feat_cols]
    vid_gen = df_gen["Variant_ID"].values
    Xg = df_gen[feat_cols].reset_index(drop=True); yg = df_gen[tgt].astype(int).reset_index(drop=True)

    pool = pd.concat([load_panel(p, acmg=False) for p in ("cancer", "pah", "cftr")], ignore_index=True)
    vid_pool = pool["Variant_ID"].values
    Xp = pool.reindex(columns=feat_cols).reset_index(drop=True)
    yp = pool[tgt].astype(int).reset_index(drop=True)

    def obj(X):
        X = X.copy()
        for c in catcols:
            if c in X.columns:
                X[c] = X[c].astype(object)
        return X

    share = float(np.isin(vid_pool, vid_gen).mean())
    print(f"general n={len(Xg)} (P={int(yg.sum())}/B={int((yg==0).sum())})  "
          f"pool n={len(Xp)} (P={int(yp.sum())}/B={int((yp==0).sum())})  "
          f"pool-in-general share={share:.1%}  prior={PRIOR:.3f}  lgb_seeds={N_SEEDS_LGB}", flush=True)

    rows = []
    for seed in (SEED, SEED + 1000):
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
        splits = list(rskf.split(Xg, yg)); t0 = time.time()
        for fi, (tr, te) in enumerate(splits):
            Xtr_g = Xg.iloc[tr].reset_index(drop=True); ytr_g = yg.iloc[tr].reset_index(drop=True)
            Xte = Xg.iloc[te].reset_index(drop=True); yte = yg.iloc[te].reset_index(drop=True)

            teA, oofA = lgb_fold(Xtr_g, ytr_g, Xte)
            tA = pick_threshold_at_prior(ytr_g.values, oofA, PRIOR)
            f1A = f1_at_prior(*tpr_fpr(yte.values, (teA >= tA).astype(int)), PRIOR)
            aucA = roc_auc_score(yte.values, teA)

            keep = ~np.isin(vid_pool, list(set(vid_gen[te])))
            Xp_k = Xp[keep].reset_index(drop=True); yp_k = yp[keep].reset_index(drop=True)
            Xtr_b = to_categorical(pd.concat([obj(Xtr_g), obj(Xp_k)], ignore_index=True))
            ytr_b = pd.concat([ytr_g, yp_k], ignore_index=True)
            is_gen = np.concatenate([np.ones(len(ytr_g), bool), np.zeros(len(yp_k), bool)])
            teB, oofB = lgb_fold(Xtr_b, ytr_b, Xte)
            tB = pick_threshold_at_prior(ytr_b.values[is_gen], oofB[is_gen], PRIOR)
            f1B = f1_at_prior(*tpr_fpr(yte.values, (teB >= tB).astype(int)), PRIOR)
            aucB = roc_auc_score(yte.values, teB)

            rows.append({"seed": seed, "fold": fi, "f1_A": f1A, "f1_B": f1B, "auc_A": aucA, "auc_B": aucB})
            pd.DataFrame(rows).to_csv(CSV, index=False)        # incremental save
            if (fi + 1) % 5 == 0:
                d = pd.DataFrame(rows); d = d[d.seed == seed]
                print(f"  seed {seed} fold {fi+1}/25  ({time.time()-t0:.0f}s)  "
                      f"running dF1={float((d.f1_B-d.f1_A).mean()):+.4f} "
                      f"dAUC={float((d.auc_B-d.auc_A).mean()):+.4f}", flush=True)

    df = pd.DataFrame(rows)
    for seed in (SEED, SEED + 1000):
        s = df[df.seed == seed]; dF = (s.f1_B - s.f1_A).values; dA = (s.auc_B - s.auc_A).values; nf = len(dF)
        sigF = dF.mean() / (dF.std(ddof=1) / np.sqrt(nf)); sigA = dA.mean() / (dA.std(ddof=1) / np.sqrt(nf))
        print(f"seed {seed} ({nf}f): A_F1={s.f1_A.mean():.4f} B_F1={s.f1_B.mean():.4f} "
              f"dF1={dF.mean():+.4f} ({sigF:+.2f}s) | A_AUC={s.auc_A.mean():.4f} "
              f"B_AUC={s.auc_B.mean():.4f} dAUC={dA.mean():+.4f} ({sigA:+.2f}s)", flush=True)
    dF = (df.f1_B - df.f1_A).values; dA = (df.auc_B - df.auc_A).values; nf = len(dF)
    sigF = dF.mean() / (dF.std(ddof=1) / np.sqrt(nf)); sigA = dA.mean() / (dA.std(ddof=1) / np.sqrt(nf))
    print(f"POOLED ({nf}f): general-only F1={df.f1_A.mean():.4f}  +subpanels F1={df.f1_B.mean():.4f}", flush=True)
    print(f"POOLED dF1={dF.mean():+.4f} ({sigF:+.2f}s)  dAUC={dA.mean():+.4f} ({sigA:+.2f}s)  "
          f"B_wins_F1={int((df.f1_B>df.f1_A).sum())}/{nf} B_wins_AUC={int((df.auc_B>df.auc_A).sum())}/{nf}", flush=True)
    verdict = "REJECT (pooling hurts/neutral)" if dF.mean() <= 0 else (
        "HELPS (>=2s)" if sigF >= 2 else "marginal/noise")
    print(f"VERDICT: {verdict}   saved -> {os.path.basename(CSV)}", flush=True)


if __name__ == "__main__":
    main()
