"""Can better blend weights + thresholds convert TabPFN's AUC/oracle edge into honest F1?
Build a DEPLOYMENT-FAITHFUL member cache (exact deployed members: 5-seed LGBM redun-prune+spw1bag,
the general_blend RF/CatBoost/XGB, TabPFN-v2.6 — each with honest OOF + test), then test the full
grid of {weight scheme} x {threshold-selection method} for honest held-out F1 @ 0.143. Nested where
weights/thresholds are data-chosen (no test peeking). 3 seeds x 5 folds.

Run: TABPFN_TOKEN=<key> .venv/bin/python -u analysis/exp_general_capture.py
"""
import sys, os, warnings
sys.path.insert(0, "analysis"); sys.path.insert(0, "pipelines")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize

from config import SEED
from schema import normalize_sentinels, partition_columns, drop_all_nan_rows
from utils import detect_leaky_columns, KNOWN_LEAKY, TEST_PRIORS, PANELS, optimize_threshold_f1_balanced
from predict import _engineer, _fit_own_preprocess, _apply_own_preprocess, _lgbm_redun_keep, LGBM_PARAM_OVERRIDE
from recipe import PANEL_RECIPE
from models import get_model_params, train_multi_seed_ensemble
from general_blend import _cats, _rf_repr, _cat_repr, _rf_general, _cat_general, _xgb_model
from tabpfn_member import fit_tabpfn_member, tabpfn_member_probs, _encode
from calibration import fit_platt, apply_platt
from harness import f1_at_prior, tpr_fpr, pick_threshold_at_prior

PANEL = "general"; PRIOR = TEST_PRIORS[PANEL]
# NOT /tmp: that is a small tmpfs here, so the cache the canonical PDR table is built from
# vanished on reboot and general became unreproducible. Survives reboots now.
CACHE = os.environ.get("PATHOSENSE_CACHE",
                       os.path.expanduser("~/.cache/pathosense/general_faithful_cache.npz"))
NAMES = ["lgbm", "rf", "cat", "xgb", "tabpfn"]


def build():
    df = pd.read_csv(PANELS[PANEL]); df = normalize_sentinels(df); df = drop_all_nan_rows(df)
    cfg = PANEL_RECIPE[PANEL]
    de = _engineer(df.copy(), cfg); _, num, cat, tgt = partition_columns(de)
    y = de[tgt].astype(int).reset_index(drop=True); X = de[num + cat].reset_index(drop=True)
    leaky = detect_leaky_columns(X[num], y, 0.99, verbose=False)
    X = X.drop(columns=sorted(set(leaky) | (KNOWN_LEAKY & set(X.columns))))
    folds = []
    for seed in (SEED, SEED + 1000, SEED + 2000):
        for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=seed).split(X, y):
            Xtr0, ytr = X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True)
            Xte0, yte = X.iloc[te].reset_index(drop=True), y.iloc[te].reset_index(drop=True)
            Xpp, prep = _fit_own_preprocess(Xtr0, ytr, cfg)
            Xte = _apply_own_preprocess(Xte0, prep).reindex(columns=Xpp.columns)
            yv = ytr.values; cat_feats = _cats(Xpp)
            X_ord, enc = _rf_repr(Xpp, cat_feats, fit=True); Xte_ord, _ = _rf_repr(Xte, cat_feats, enc=enc)
            X_cat = _cat_repr(Xpp, cat_feats); Xte_cat = _cat_repr(Xte, cat_feats)
            cat_idx = [X_cat.columns.get_loc(c) for c in cat_feats]
            spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
            # LGBM (deployed 5-seed, redun-prune, spw1bag)
            params = {**get_model_params(len(Xpp), spw), **LGBM_PARAM_OVERRIDE[PANEL]}
            keep = _lgbm_redun_keep(Xpp)
            _, lgo, lgt, _ = train_multi_seed_ensemble(Xpp[keep], ytr, Xte[keep], params, n_seeds=5, n_folds=5)
            pl = fit_platt(lgo, yv); O = [apply_platt(lgo, pl)]; E = [apply_platt(lgt, pl)]
            # RF / XGB / CatBoost OOF (5-fold inner) + finals (mirror fit_general_blend)
            skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
            rf_o = np.zeros(len(yv)); xg_o = np.zeros(len(yv)); cb_o = np.zeros(len(yv))
            for k, (a, b) in enumerate(skf.split(Xpp, ytr)):
                rf = _rf_general().fit(X_ord.iloc[a], ytr.iloc[a]); rf_o[b] = rf.predict_proba(X_ord.iloc[b])[:, 1]
                xg = _xgb_model().fit(X_ord.iloc[a], ytr.iloc[a]); xg_o[b] = xg.predict_proba(X_ord.iloc[b])[:, 1]
                i2, ies = train_test_split(a, test_size=0.15, random_state=SEED + k, stratify=ytr.iloc[a])
                cb = _cat_general(cat_idx, spw); cb.fit(X_cat.iloc[i2], ytr.iloc[i2], eval_set=(X_cat.iloc[ies], ytr.iloc[ies]))
                cb_o[b] = cb.predict_proba(X_cat.iloc[b])[:, 1]
            i2, ies = train_test_split(np.arange(len(yv)), test_size=0.15, random_state=SEED + 99, stratify=ytr)
            cbf = _cat_general(cat_idx, spw); cbf.fit(X_cat.iloc[i2], ytr.iloc[i2], eval_set=(X_cat.iloc[ies], ytr.iloc[ies]))
            for oof, te in ((rf_o, _rf_general().fit(X_ord, ytr).predict_proba(Xte_ord)[:, 1]),
                            (cb_o, cbf.predict_proba(Xte_cat)[:, 1]),
                            (xg_o, _xgb_model().fit(X_ord, ytr).predict_proba(Xte_ord)[:, 1])):
                p = fit_platt(oof, yv); O.append(apply_platt(oof, p)); E.append(apply_platt(te, p))
            # TabPFN (optional 5th member; absent -> 4-tree cache, which is what the
            # canonical PDR table uses anyway: exp_pdr_canonical.py takes E[:4])
            if os.environ.get("TABPFN_TOKEN"):
                tab_art, tab_oof = fit_tabpfn_member(Xpp, ytr, cat_feats, n_splits=5)
                O.append(tab_oof); E.append(tabpfn_member_probs(Xte, tab_art))
            folds.append((np.array(O), np.array(E), yv, yte.values))
            print(f"  fold cached ({len(folds)})", flush=True)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, **{f"O{i}": f[0] for i, f in enumerate(folds)}, **{f"E{i}": f[1] for i, f in enumerate(folds)},
             **{f"ytr{i}": f[2] for i, f in enumerate(folds)}, **{f"yte{i}": f[3] for i, f in enumerate(folds)}, n=len(folds))
    return folds


def load():
    z = np.load(CACHE); n = int(z["n"])
    return [(z[f"O{i}"], z[f"E{i}"], z[f"ytr{i}"], z[f"yte{i}"]) for i in range(n)]


def f1_at(y, p, t): return f1_at_prior(*tpr_fpr(y, (p >= t).astype(int)), PRIOR)
def swept(y, p): return max(f1_at(y, p, q) for q in np.quantile(p, np.arange(.02, .98, .004)))


def thr_methods(ytr, otr):
    return {"prior": pick_threshold_at_prior(ytr, otr, PRIOR),
            "boot": optimize_threshold_f1_balanced(ytr, otr, target_prior=PRIOR, n_bootstrap=300)[0]}


def opt_w_f1(O, y):
    """Nested weights maximizing F1-at-prior on train (then its own threshold)."""
    k = O.shape[0]
    def neg(w):
        w = np.clip(w, 0, None); s = w.sum()
        if s <= 0: return 0
        p = (w / s) @ O; t = pick_threshold_at_prior(y, p, PRIOR); return -f1_at(y, p, t)
    r = minimize(neg, np.ones(k) / k, method="Nelder-Mead", options={"maxiter": 400, "xatol": 1e-3})
    w = np.clip(r.x, 0, None); return w / w.sum()


def main():
    if not os.environ.get("TABPFN_TOKEN"):
        print("set TABPFN_TOKEN"); return
    folds = load() if os.path.exists(CACHE) else build()
    print(f"\n=== Capture TabPFN's edge? weights x thresholds, deployment-faithful ({len(folds)} folds) ===")
    IDX4 = [0, 1, 2, 3]; IDX5 = [0, 1, 2, 3, 4]
    # configs: (label, member idx, weight-fn or None=equal, threshold method)
    def evalcfg(idx, wfn, tname):
        f1 = []
        for O, E, ytr, yte in folds:
            Ou, Eu = O[idx], E[idx]
            w = wfn(Ou, ytr) if wfn else np.ones(len(idx)) / len(idx)
            otr = w @ Ou; ote = w @ Eu
            t = thr_methods(ytr, otr)[tname]
            f1.append(f1_at(yte, ote, t))
        return np.array(f1)
    base = evalcfg(IDX4, None, "prior")   # 4-tree, prior threshold = current deployed-ish
    print(f"  4-tree equal @prior-thr  F1={base.mean():.4f}±{base.std(ddof=1)/np.sqrt(len(base)):.4f}  [baseline]")
    for label, idx, wfn, tn in [
        ("5-eq @prior", IDX5, None, "prior"), ("5-eq @boot", IDX5, None, "boot"),
        ("5-F1wts @prior", IDX5, opt_w_f1, "prior"), ("5-F1wts @boot", IDX5, opt_w_f1, "boot"),
        ("4-eq @boot", IDX4, None, "boot"), ("4-F1wts @prior", IDX4, opt_w_f1, "prior")]:
        f1 = evalcfg(idx, wfn, tn); d = f1 - base; se = d.std(ddof=1)/np.sqrt(len(d))
        win = "  <<<" if d.mean()/se >= 2 and d.mean() > 0 else ""
        print(f"  {label:18s} F1={f1.mean():.4f}±{f1.std(ddof=1)/np.sqrt(len(f1)):.4f}  Δ={d.mean():+.4f} ({d.mean()/se if se else 0:+.2f}σ){win}")
    # oracle ceilings
    o5 = np.mean([swept(yte, O[IDX5].mean(0) if False else E[IDX5].mean(0)) for O, E, _, yte in folds])
    o4 = np.mean([swept(yte, E[IDX4].mean(0)) for _, E, _, yte in folds])
    print(f"  oracle (test-peeked): 5-member={o5:.4f}  4-tree={o4:.4f}")
    print("DONE")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build()          # cache only (4-tree unless TABPFN_TOKEN is set)
    else:
        main()
