"""Per-panel OWN-model bake-off — apply the general-panel model pass (CatBoost / RF / XGB /
4-way blend) to each small panel's own model, and check it IN THE DEPLOYED CONTEXT.

Prior finding: own-model gains "wash out vs the transfer scorer".
This re-tests it rigorously. For each panel we honestly CV the own model (deployed median+mRMR
preprocessing, per-fold) across 5 candidates — lgbm (current), catboost, rf, xgb, blend4 — and
evaluate EACH blended with the OOS transfer scorer at w in {0,.25,.5,.75,1.0}
(probs = w·transfer + (1-w)·own; w=1 = pure transfer = current pah/cftr, w=.5 = current cancer).

Reports, per panel: standalone own AUC per candidate (is CatBoost/blend a better own model?),
and the deployed-context F1* surface (does any candidate × weight beat the CURRENT deployed
config?), paired sigma vs deployed.

Run: .venv/bin/python -u analysis/exp_own_model_bakeoff.py
"""
import sys, warnings
sys.path.insert(0, "analysis"); sys.path.insert(0, "pipelines")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score

from config import SEED
from schema import normalize_sentinels, partition_columns, drop_all_nan_rows
from utils import detect_leaky_columns, KNOWN_LEAKY, TEST_PRIORS, PANELS
from recipe import PANEL_RECIPE
from predict import _engineer, _fit_own_preprocess, _apply_own_preprocess
from transfer import fit_transfer_scorer, transfer_score, TRANSFER_PANELS
from run_pipeline import get_model_params
from general_blend import _cats, _rf_repr, _cat_repr
from calibration import fit_platt, apply_platt
from harness import f1_at_prior, tpr_fpr

WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]       # probs = w·transfer + (1-w)·own ; w=1 = pure transfer
KINDS = ["lgbm", "catboost", "rf", "xgb", "blend4"]


def _load(panel):
    df = pd.read_csv(PANELS[panel]); df = normalize_sentinels(df); df = drop_all_nan_rows(df)
    cfg = PANEL_RECIPE[panel]
    de = _engineer(df.copy(), cfg)
    _, num_cols, cat_cols, tgt = partition_columns(de)
    y = de[tgt].reset_index(drop=True)
    X = de[num_cols + cat_cols].reset_index(drop=True)
    leaky = detect_leaky_columns(X[num_cols], y, auc_threshold=0.99, verbose=False)
    X = X.drop(columns=sorted(set(leaky) | (KNOWN_LEAKY & set(X.columns))))
    # OOS transfer score (leakage-correct: target Variant_IDs excluded from the scorer)
    g = transfer_score(df.reset_index(drop=True), fit_transfer_scorer(panel))
    return df.reset_index(drop=True), X, y, cfg, np.asarray(g)


def _make(kind, n, spw, seed):
    if kind == "lgbm":
        return lgb.LGBMClassifier(**{**get_model_params(n, spw), "random_state": seed})
    if kind == "catboost":
        return CatBoostClassifier(iterations=500, depth=4, learning_rate=0.03, l2_leaf_reg=3.0,
                                  scale_pos_weight=spw, random_seed=seed, verbose=0,
                                  allow_writing_files=False, thread_count=4)
    if kind == "rf":
        return RandomForestClassifier(n_estimators=600, max_features="sqrt", min_samples_leaf=3,
                                      class_weight="balanced", n_jobs=4, random_state=seed)
    if kind == "xgb":
        return XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=4, subsample=0.8,
                             colsample_bytree=0.8, reg_lambda=1.5, scale_pos_weight=spw,
                             eval_metric="logloss", n_jobs=4, random_state=seed)


def _fit_pred(kind, Xtr, ytr, Xte, cat_feats, enc, cat_idx, spw, seed):
    m = _make(kind, len(ytr), spw, seed)
    if kind == "lgbm":
        m.fit(Xtr, ytr, categorical_feature=cat_feats or "auto")
        return m.predict_proba(Xte)[:, 1]
    if kind == "catboost":
        m.fit(_cat_repr(Xtr, cat_feats), ytr, cat_features=cat_idx)
        return m.predict_proba(_cat_repr(Xte, cat_feats))[:, 1]
    # rf / xgb on ordinal repr (NaN->-999), encoder fit on the training fold
    Xtr_o, _ = _rf_repr(Xtr, cat_feats, enc=enc)
    Xte_o, _ = _rf_repr(Xte, cat_feats, enc=enc)
    m.fit(Xtr_o, ytr)
    return m.predict_proba(Xte_o)[:, 1]


def _own_cal(kind, Xtr, ytr, Xte, cat_feats, enc, cat_idx, spw, seed):
    """Calibrated own test probs via inner 3-fold OOF Platt (honest)."""
    oof = np.zeros(len(ytr))
    skf = StratifiedKFold(3, shuffle=True, random_state=seed)
    for a, b in skf.split(Xtr, ytr):
        oof[b] = _fit_pred(kind, Xtr.iloc[a], ytr.iloc[a], Xtr.iloc[b],
                           cat_feats, enc, cat_idx, spw, seed)
    platt = fit_platt(oof, ytr.values)
    raw_te = _fit_pred(kind, Xtr, ytr, Xte, cat_feats, enc, cat_idx, spw, seed)
    return apply_platt(raw_te, platt), raw_te


def _f1star(y, p, prior):
    qs = np.quantile(p, np.arange(.02, .98, .01))
    return max(f1_at_prior(*tpr_fpr(y, (p >= q).astype(int)), prior) for q in qs)


def run(panel, n_repeats=2):
    df, X, y, cfg, g = _load(panel)
    prior = TEST_PRIORS[panel]; w_dep = TRANSFER_PANELS[panel]
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED)
    # per-fold records: f1[kind][w] list, plus standalone own AUC + transfer AUC
    f1 = {k: {w: [] for w in WEIGHTS} for k in KINDS}
    own_auc = {k: [] for k in KINDS}; tr_auc = []
    for k, (tr, te) in enumerate(outer.split(X, y)):
        Xtr0, ytr = X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True)
        Xte0, yte = X.iloc[te].reset_index(drop=True), y.iloc[te].reset_index(drop=True)
        # deployed own preprocessing fit on the train fold only
        Xtr, prep = _fit_own_preprocess(Xtr0, ytr, cfg)
        Xte = _apply_own_preprocess(Xte0, prep)
        Xte = Xte.reindex(columns=Xtr.columns)
        cat_feats = _cats(Xtr); cat_idx = [Xtr.columns.get_loc(c) for c in cat_feats]
        _, enc = _rf_repr(Xtr, cat_feats, fit=True) if cat_feats else (None, None)
        # transfer calibrated on this fold
        gp_tr, gp_te = g[tr], g[te]
        platt_g = fit_platt(gp_tr, ytr.values); g_cal = apply_platt(gp_te, platt_g)
        tr_auc.append(roc_auc_score(yte.values, gp_te))
        # base candidates
        cals = {}
        for kind in ["lgbm", "catboost", "rf", "xgb"]:
            cal, raw = _own_cal(kind, Xtr, ytr, Xte, cat_feats, enc, cat_idx, spw, SEED + k)
            cals[kind] = cal
            own_auc[kind].append(roc_auc_score(yte.values, raw))
        cals["blend4"] = np.mean([cals[c] for c in ["lgbm", "catboost", "rf", "xgb"]], axis=0)
        own_auc["blend4"].append(roc_auc_score(yte.values, cals["blend4"]))
        for kind in KINDS:
            for w in WEIGHTS:
                blend = w * g_cal + (1 - w) * cals[kind]
                f1[kind][w].append(_f1star(yte.values, blend, prior))

    print(f"\n=== {panel.upper()}  (prior {prior:.3f}, deployed w={w_dep}, transfer AUC {np.mean(tr_auc):.4f}) ===")
    print("  standalone OWN AUC:  " + "  ".join(f"{k}={np.mean(own_auc[k]):.4f}" for k in KINDS))
    # deployed baseline F1*: cancer = lgbm@0.5 ; pah/cftr = pure transfer (w=1, any kind identical)
    base_kind = "lgbm"; base = np.array(f1[base_kind][w_dep])
    print(f"  deployed baseline ({base_kind}@w={w_dep}) F1* = {base.mean():.4f}")
    print("  F1* surface (rows=own candidate, cols=w):")
    header = "         " + "".join(f"  w={w:<4}" for w in WEIGHTS)
    print(header)
    best = (None, None, -1, 0.0)
    for kind in KINDS:
        cells = []
        for w in WEIGHTS:
            v = np.mean(f1[kind][w]); cells.append(f"{v:.4f}")
        print(f"  {kind:8s}" + "".join(f"  {c}" for c in cells))
        for w in WEIGHTS:
            arr = np.array(f1[kind][w]); d = arr - base
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0
            sig = d.mean() / se if se > 0 else 0
            if arr.mean() > best[2]:
                best = (kind, w, arr.mean(), sig)
    bk, bw, bm, bsig = best
    print(f"  BEST: {bk}@w={bw} F1*={bm:.4f}  Δ vs deployed = {bm-base.mean():+.4f} ({bsig:+.2f}σ)")
    return panel, bk, bw, bm, base.mean(), bsig


if __name__ == "__main__":
    print("=== Per-panel OWN-model bake-off (deployed-context, honest CV) ===")
    rows = []
    for p in ("cancer", "pah", "cftr"):
        rows.append(run(p))
    print("\n=== SUMMARY ===")
    for panel, bk, bw, bm, base, sig in rows:
        verdict = "IMPROVE" if (bm - base) > 0 and sig >= 2 else "no gain (wash-out)"
        print(f"  {panel:7s} deployed={base:.4f}  best={bk}@w{bw}={bm:.4f}  Δ={bm-base:+.4f} ({sig:+.2f}σ)  -> {verdict}")
    print("DONE")
