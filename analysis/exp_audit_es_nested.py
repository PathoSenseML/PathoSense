"""#6 keep/revert decision — NESTED held-out test of the early-stopping fix.

DEPLOYMENT FACT: for general & cancer the deployed threshold is a FIXED OVERRIDE
(predict.THRESHOLD_OVERRIDE = {general:0.82, cancer:0.81}), NOT selected from the OOF.
So #6 cannot change threshold selection. The deployed final model is also UNCHANGED by
#6. The ONLY deployed effect of #6 is the Platt calibration (fit on the #6-affected
OOF), evaluated at the fixed threshold. Held-out AUC is identical for both arms (same
final model, monotonic Platt) — confirmed below as a sanity check.

Protocol (deployment-faithful): outer RepeatedStratifiedKFold. Train ONE 3-seed final
ensemble per outer-train (shared by both arms). Build the OOF two ways (OLD=ES on OOF
fold, NEW=nested ES), fit Platt on each, apply to the held-out raw probs, threshold at
the FIXED deployed override, and measure F1 at the test prior. Paired NEW-vs-OLD.
(general also blends LGBM with RF/Cat/XGB at deploy, which further dilutes the Platt
effect — so this LGBM-only test is a conservative UPPER BOUND on #6's real impact.)

Run: .venv/bin/python -u analysis/exp_audit_es_nested.py
"""
import sys, warnings
sys.path.insert(0, "analysis"); sys.path.insert(0, "pipelines")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

from config import SEED
from schema import normalize_sentinels, partition_columns, drop_all_nan_rows
from utils import detect_leaky_columns, KNOWN_LEAKY, TEST_PRIORS, PANELS
from recipe import PANEL_RECIPE
from predict import (_engineer, _fit_own_preprocess, _lgbm_redun_keep, REDUN_PRUNE_LGBM,
                     THRESHOLD_OVERRIDE, LGBM_PARAM_OVERRIDE)
from run_pipeline import get_model_params
from calibration import fit_platt, apply_platt
from harness import f1_at_prior, tpr_fpr


def _load(panel):
    df = pd.read_csv(PANELS[panel]); df = normalize_sentinels(df); df = drop_all_nan_rows(df)
    cfg = PANEL_RECIPE[panel]
    de = _engineer(df.copy(), cfg)
    _, num_cols, cat_cols, tgt = partition_columns(de)
    y = de[tgt].reset_index(drop=True); X = de[num_cols + cat_cols].reset_index(drop=True)
    leaky = detect_leaky_columns(X[num_cols], y, auc_threshold=0.99, verbose=False)
    X = X.drop(columns=sorted(set(leaky) | (KNOWN_LEAKY & set(X.columns))))
    Xpp, _ = _fit_own_preprocess(X, y, cfg)
    if panel in REDUN_PRUNE_LGBM:
        Xpp = Xpp[_lgbm_redun_keep(Xpp)]
    return Xpp.reset_index(drop=True), y.reset_index(drop=True), cfg


def _build_oof(X, y, params, n_folds, seed_i, nested):
    patience = 50 if len(y) > 300 else 30
    p = {**params, "random_state": seed_i}
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed_i)
    oof = np.zeros(len(y))
    yv = y.values
    for tr, va in skf.split(X, y):
        if nested:
            in_tr, in_es = train_test_split(tr, test_size=0.15, random_state=seed_i,
                                            stratify=yv[tr])
        else:
            in_tr, in_es = tr, va
        m = lgb.LGBMClassifier(**p)
        m.fit(X.iloc[in_tr], y.iloc[in_tr], eval_set=[(X.iloc[in_es], y.iloc[in_es])],
              callbacks=[lgb.early_stopping(patience, verbose=False), lgb.log_evaluation(0)])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return oof


def _final_ensemble(X, y, params, seed_base, n_seeds=3):
    patience = 50 if len(y) > 300 else 30
    models = []
    for s in range(n_seeds):
        p = {**params, "random_state": seed_base + s}
        Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.1, random_state=seed_base + s, stratify=y)
        m = lgb.LGBMClassifier(**p)
        m.fit(Xt, yt, eval_set=[(Xe, ye)],
              callbacks=[lgb.early_stopping(patience, verbose=False), lgb.log_evaluation(0)])
        models.append(m)
    return models


def _score(models, X):
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def run(panel, n_repeats=3):
    X, y, cfg = _load(panel)
    prior = TEST_PRIORS[panel]
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    params = get_model_params(len(X), spw)
    if panel in LGBM_PARAM_OVERRIDE:
        params = {**params, **LGBM_PARAM_OVERRIDE[panel]}
    thr_fixed = THRESHOLD_OVERRIDE[panel]
    n_folds = 3 if len(X) < 200 else 5
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED)
    old_f1, new_f1, aucs = [], [], []
    for k, (otr, ote) in enumerate(outer.split(X, y)):
        Xtr, ytr = X.iloc[otr].reset_index(drop=True), y.iloc[otr].reset_index(drop=True)
        Xte, yte = X.iloc[ote].reset_index(drop=True), y.iloc[ote].reset_index(drop=True)
        final = _final_ensemble(Xtr, ytr, params, SEED + 1000 + k)   # SAME final model for both arms
        pte_raw = _score(final, Xte)
        aucs.append(roc_auc_score(yte.values, pte_raw))
        for nested, store in ((False, old_f1), (True, new_f1)):
            oof = _build_oof(Xtr, ytr, params, n_folds, SEED + k, nested)
            platt = fit_platt(oof, ytr.values)
            pte = apply_platt(pte_raw, platt)            # only the Platt differs between arms
            tpr, fpr = tpr_fpr(yte.values, (pte >= thr_fixed).astype(int))
            store.append(f1_at_prior(tpr, fpr, prior))
    old_f1, new_f1 = np.array(old_f1), np.array(new_f1)
    d = new_f1 - old_f1
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"  {panel:7s} held-out AUC={np.mean(aucs):.4f} (same both arms) | "
          f"F1*@{prior:.3f} t={thr_fixed}: OLD {old_f1.mean():.4f} -> NEW {new_f1.mean():.4f}  "
          f"Δ={d.mean():+.4f} ({d.mean()/se if se>0 else 0:+.2f}σ, n={len(d)} folds)", flush=True)


if __name__ == "__main__":
    print("=== #6 nested held-out test: does the honest-OOF threshold generalize better? ===")
    print("    (final model identical for both arms; only Platt+threshold differ)")
    for panel in ("general", "cancer"):
        run(panel)
    print("DONE")
