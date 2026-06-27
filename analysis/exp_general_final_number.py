"""Deployment-faithful held-out CV of the FULL general pipeline to nail down the exact F1/AUC.
Replicates predict.py's general path per fold: 5-seed LGBM (redun-prune + spw1_bag) -> Platt ->
fit_general_blend (RF + CatBoost + XGB + optional TabPFN-v2.6) -> general_blend_probs on held-out,
evaluated at the deployed fixed threshold (THRESHOLD_OVERRIDE['general']=0.81), prior 0.143.
Reports BOTH the deployed 5-member (TabPFN on) and the 4-tree fallback. 3 seeds x 5 folds.

Run: TABPFN_TOKEN=<key> .venv/bin/python -u analysis/exp_general_final_number.py
"""
import sys, os, warnings
sys.path.insert(0, "analysis"); sys.path.insert(0, "pipelines")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score

from config import SEED
from schema import normalize_sentinels, partition_columns, drop_all_nan_rows
from utils import detect_leaky_columns, KNOWN_LEAKY, TEST_PRIORS, PANELS
from predict import (_engineer, _fit_own_preprocess, _apply_own_preprocess, _lgbm_redun_keep,
                     LGBM_PARAM_OVERRIDE, THRESHOLD_OVERRIDE)
from recipe import PANEL_RECIPE
from models import get_model_params, train_multi_seed_ensemble
from general_blend import fit_general_blend, general_blend_probs
from calibration import fit_platt, apply_platt
from harness import f1_at_prior, tpr_fpr, pick_threshold_at_prior

PANEL = "general"; PRIOR = TEST_PRIORS[PANEL]; THR = THRESHOLD_OVERRIDE[PANEL]   # 0.81


def _swept(y, p):
    qs = np.quantile(p, np.arange(.02, .98, .005))
    return max(f1_at_prior(*tpr_fpr(y, (p >= q).astype(int)), PRIOR) for q in qs)


def main():
    df = pd.read_csv(PANELS[PANEL]); df = normalize_sentinels(df); df = drop_all_nan_rows(df)
    cfg = PANEL_RECIPE[PANEL]
    de = _engineer(df.copy(), cfg); _, num, cat, tgt = partition_columns(de)
    y = de[tgt].astype(int).reset_index(drop=True); X = de[num + cat].reset_index(drop=True)
    leaky = detect_leaky_columns(X[num], y, 0.99, verbose=False)
    X = X.drop(columns=sorted(set(leaky) | (KNOWN_LEAKY & set(X.columns))))
    print(f"TabPFN active: {os.environ.get('TABPFN_TOKEN') is not None} | threshold={THR}, prior={PRIOR}")

    res5 = {"auc": [], "f1": [], "f1_recal": [], "f1_swept": []}
    res4 = {"auc": [], "f1": [], "f1_recal": [], "f1_swept": []}; tab_on = []; thr5_picks = []
    for si, seed in enumerate((SEED, SEED + 1000, SEED + 2000)):
        for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=seed).split(X, y):
            Xtr0, ytr = X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True)
            Xte0, yte = X.iloc[te].reset_index(drop=True), y.iloc[te].reset_index(drop=True)
            Xtr, prep = _fit_own_preprocess(Xtr0, ytr, cfg)
            Xte = _apply_own_preprocess(Xte0, prep).reindex(columns=Xtr.columns)
            spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
            params = {**get_model_params(len(Xtr), spw), **LGBM_PARAM_OVERRIDE[PANEL]}
            keep = _lgbm_redun_keep(Xtr)
            _, oof, test, _ = train_multi_seed_ensemble(Xtr[keep], ytr, Xte[keep], params, n_seeds=5, n_folds=5)
            pl = fit_platt(oof, ytr.values); lgb_oof = apply_platt(oof, pl); lgb_test = apply_platt(test, pl)
            art, blend_oof5 = fit_general_blend(Xtr, ytr, lgb_oof, spw, n_splits=5)
            tab_on.append(art.get("tabpfn") is not None)
            b5 = general_blend_probs(Xte, lgb_test, art)
            b4 = general_blend_probs(Xte, lgb_test, {**art, "tabpfn": None})
            yv = yte.values
            thr5 = pick_threshold_at_prior(ytr.values, blend_oof5, PRIOR); thr5_picks.append(thr5)
            res5["auc"].append(roc_auc_score(yv, b5)); res4["auc"].append(roc_auc_score(yv, b4))
            res5["f1"].append(f1_at_prior(*tpr_fpr(yv, (b5 >= THR).astype(int)), PRIOR))
            res4["f1"].append(f1_at_prior(*tpr_fpr(yv, (b4 >= THR).astype(int)), PRIOR))
            res5["f1_recal"].append(f1_at_prior(*tpr_fpr(yv, (b5 >= thr5).astype(int)), PRIOR))
            res5["f1_swept"].append(_swept(yv, b5)); res4["f1_swept"].append(_swept(yv, b4))
        print(f"  seed {seed} done ({sum(tab_on)}/{len(tab_on)} folds had TabPFN; mean thr5={np.mean(thr5_picks):.3f})", flush=True)

    def m(x): a = np.array(x); return a.mean(), a.std(ddof=1)/np.sqrt(len(a))
    print(f"\n=== EXACT general deployment-faithful held-out CV (prior {PRIOR}, n={len(res5['auc'])} folds) ===")
    a5 = m(res5["auc"]); a4 = m(res4["auc"])
    print(f"  5-member+TabPFN  AUC={a5[0]:.4f}±{a5[1]:.4f} | F1@0.81={m(res5['f1'])[0]:.4f}±{m(res5['f1'])[1]:.4f}"
          f" | F1@recal-thr={m(res5['f1_recal'])[0]:.4f}±{m(res5['f1_recal'])[1]:.4f}"
          f" | F1@swept={m(res5['f1_swept'])[0]:.4f}")
    print(f"  4-tree fallback  AUC={a4[0]:.4f}±{a4[1]:.4f} | F1@0.81={m(res4['f1'])[0]:.4f}±{m(res4['f1'])[1]:.4f}"
          f" | F1@swept={m(res4['f1_swept'])[0]:.4f}")
    print(f"  honest threshold for the 5-member (mean per-fold pick): {np.mean(thr5_picks):.3f}")
    dr = np.array(res5["f1_recal"]) - np.array(res4["f1"]); se = dr.std(ddof=1)/np.sqrt(len(dr))
    print(f"  TabPFN lift WITH recalibrated threshold vs 4-tree@0.81: ΔF1={dr.mean():+.4f} ({dr.mean()/se if se else 0:+.2f}σ)")
    print(f"  (TabPFN active in {sum(tab_on)}/{len(tab_on)} folds)")
    print("DONE")


if __name__ == "__main__":
    main()
