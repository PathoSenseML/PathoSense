"""PROPER full 5-way ensemble weight optimization for general with TabPFN v2.6.

Members: [lgbm, rf, cat, xgb, tabpfn(v2.6)]. The earlier 1-D sweep forced the 4 trees to equal weight.
Here: (a) several principled FIXED weight vectors evaluated honestly (held-out, multi-seed); (b) a
per-fold constrained optimizer (maximize train-OOF AUC over the simplex) — the upper bound + an
overfit check (4-tree reweighting overfit -1.65σ). Deploy weights MUST be fixed, so the winner is the
best FIXED scheme that holds up. AUC + honest F1* @ 0.143, recalibrated threshold, 3 seeds.

Run: TABPFN_TOKEN=<key> .venv/bin/python -u analysis/exp_tabpfn_weights.py
"""
import sys, os, warnings
sys.path.insert(0, "analysis"); sys.path.insert(0, "pipelines")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score

from config import SEED
from schema import normalize_sentinels, partition_columns, drop_all_nan_rows
from utils import detect_leaky_columns, KNOWN_LEAKY, TEST_PRIORS, PANELS
from predict import _engineer, _fit_own_preprocess, _apply_own_preprocess, _lgbm_redun_keep
from recipe import PANEL_RECIPE
from general_blend import _cats, _rf_repr
from harness import f1_at_prior, tpr_fpr, pick_threshold_at_prior
from exp_general_push_055 import _oof_test
from exp_tabpfn_general import _tabpfn_member

PANEL = "general"; PRIOR = TEST_PRIORS[PANEL]
M = ["lgbm", "rf", "cat", "xgb", "tabpfn"]
# principled FIXED schemes over [lgbm,rf,cat,xgb,tabpfn]
SCHEMES = {
    "4tree(dep)":  [.25, .25, .25, .25, 0.0],
    "equal5":      [.20, .20, .20, .20, .20],
    "tab@.25":     [.1875, .1875, .1875, .1875, .25],
    "tab@.33":     [.1675, .1675, .1675, .1675, .33],
    "cat+tab heavy": [.15, .15, .25, .15, .30],
    "drop-xgb":    [.25, .25, .25, 0.0, .25],
    "strong3(lgbm,cat,tab)": [.30, 0.0, .35, 0.0, .35],
}


def _hF1(ytr, otr, yte, ote):
    thr = pick_threshold_at_prior(ytr, otr, PRIOR)
    return f1_at_prior(*tpr_fpr(yte, (ote >= thr).astype(int)), PRIOR)


def _opt_weights(oof_mat, y):
    """Maximize train-OOF AUC over the simplex (SLSQP)."""
    n = oof_mat.shape[1]
    def neg_auc(w):
        w = np.clip(w, 0, None); s = w.sum()
        if s <= 0: return 0.0
        return -roc_auc_score(y, oof_mat @ (w / s))
    res = minimize(neg_auc, np.ones(n) / n, method="SLSQP",
                   bounds=[(0, 1)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"maxiter": 200, "ftol": 1e-7})
    w = np.clip(res.x, 0, None); return w / w.sum()


def main():
    if not os.environ.get("TABPFN_TOKEN"):
        print("set TABPFN_TOKEN"); return
    df = pd.read_csv(PANELS[PANEL]); df = normalize_sentinels(df); df = drop_all_nan_rows(df)
    de = _engineer(df.copy(), PANEL_RECIPE[PANEL]); _, num, cat, tgt = partition_columns(de)
    y = de[tgt].astype(int).reset_index(drop=True); X = de[num + cat].reset_index(drop=True)
    leaky = detect_leaky_columns(X[num], y, 0.99, verbose=False)
    X = X.drop(columns=sorted(set(leaky) | (KNOWN_LEAKY & set(X.columns))))

    keys = list(SCHEMES) + ["OPT(nested)", "OPT(mean-w fixed)"]
    rec = {k: {"auc": [], "fh": []} for k in keys}
    memb_auc = {m: [] for m in M}
    opt_ws = []
    for seed in (SEED, SEED + 1000, SEED + 2000):
        outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=seed)
        for tr, te in outer.split(X, y):
            Xtr0, ytr = X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True)
            Xte0, yte = X.iloc[te].reset_index(drop=True), y.iloc[te].reset_index(drop=True)
            Xtr, prep = _fit_own_preprocess(Xtr0, ytr, PANEL_RECIPE[PANEL])
            Xte = _apply_own_preprocess(Xte0, prep).reindex(columns=Xtr.columns)
            cf = _cats(Xtr); ci = [Xtr.columns.get_loc(c) for c in cf]
            _, enc = _rf_repr(Xtr, cf, fit=True); keep = _lgbm_redun_keep(Xtr)
            yv_tr, yv_te = ytr.values, yte.values
            o_list, e_list = [], []
            for kind, (Xa, Xb) in [("lgbm", (Xtr[keep], Xte[keep])), ("rf", (Xtr, Xte)),
                                   ("cat", (Xtr, Xte)), ("xgb", (Xtr, Xte))]:
                o, e = _oof_test(kind, Xa, ytr, Xb, cf, ci, enc, 1.0, seed); o_list.append(o); e_list.append(e)
            to, teo = _tabpfn_member(Xtr, ytr, Xte, cf, seed); o_list.append(to); e_list.append(teo)
            O = np.column_stack(o_list); E = np.column_stack(e_list)   # (n,5)
            for m, oi, ei in zip(M, o_list, e_list):
                memb_auc[m].append(roc_auc_score(yv_te, ei))
            for k, w in SCHEMES.items():
                w = np.array(w); bo = O @ w; be = E @ w
                rec[k]["auc"].append(roc_auc_score(yv_te, be)); rec[k]["fh"].append(_hF1(yv_tr, bo, yv_te, be))
            # per-fold optimized (nested)
            w = _opt_weights(O, yv_tr); opt_ws.append(w)
            bo = O @ w; be = E @ w
            rec["OPT(nested)"]["auc"].append(roc_auc_score(yv_te, be)); rec["OPT(nested)"]["fh"].append(_hF1(yv_tr, bo, yv_te, be))
    # evaluate the MEAN optimized weight as a fixed scheme (re-loop cheaply: recompute from stored? -> approximate by re-eval not stored; skip exact, report mean w)
    mw = np.mean(opt_ws, axis=0)
    print(f"=== PROPER 5-way ensemble weights (general, TabPFN v2.6, {len(rec['equal5']['auc'])} folds, 3 seeds) ===")
    print("  member standalone AUC: " + " ".join(f"{m}={np.mean(memb_auc[m]):.4f}" for m in M))
    base = np.array(rec["4tree(dep)"]["auc"])
    for k in keys[:-1]:
        a = np.array(rec[k]["auc"]); fh = np.array(rec[k]["fh"])
        dA = a - base; se = dA.std(ddof=1)/np.sqrt(len(dA)) if len(dA) > 1 else 0
        star = "  >0.55!" if fh.mean() > 0.55 else ""
        print(f"  {k:24s} AUC={a.mean():.4f} (Δ{dA.mean():+.4f} {dA.mean()/se if se>0 else 0:+.1f}σ)  honest-F1={fh.mean():.4f}{star}")
    print(f"  mean OPT weight vector [lgbm,rf,cat,xgb,tabpfn] = {np.round(mw,3).tolist()}")
    print("DONE")


if __name__ == "__main__":
    main()
