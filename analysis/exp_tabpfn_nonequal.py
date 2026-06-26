"""NON-EQUAL ensemble weighting, tested HONESTLY (naive weight-fitting overfits on small data).
Build the 6 members once per fold, CACHE their OOF/test preds (so future weight experiments are
instant), then evaluate non-equal strategies vs equal-weight:
  FIXED principled (no fit): auc-proportional, auc-softmax(τ).
  LEARNED + regularized (held-out): logistic stacking sweep C (incl. non-negative), NNLS,
    simplex AUC-max with L2-toward-equal penalty (λ sweep).
Honest 3-seed CV, AUC + F1* @ 0.143, recalibrated threshold, paired vs equal5. >=2σ seed-stable to win.

Run: TABPFN_TOKEN=<key> .venv/bin/python -u analysis/exp_tabpfn_nonequal.py
"""
import sys, os, warnings
sys.path.insert(0, "analysis"); sys.path.insert(0, "pipelines")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.optimize import minimize, nnls
from sklearn.linear_model import LogisticRegression
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
from exp_general_max_auc import _logreg_member

PANEL = "general"; PRIOR = TEST_PRIORS[PANEL]
NAMES = ["lgbm", "rf", "cat", "xgb", "tabpfn", "logreg"]
USE = [0, 1, 2, 3, 4]          # weight over the 5 strong members (logreg kept for reference, weight via strategies)
CACHE = "/tmp/general_members_cache.npz"


def _hF1(ytr, otr, yte, ote):
    thr = pick_threshold_at_prior(ytr, otr, PRIOR)
    return f1_at_prior(*tpr_fpr(yte, (ote >= thr).astype(int)), PRIOR)


def build_cache(X, y):
    folds = []
    for seed in (SEED, SEED + 1000, SEED + 2000):
        for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=seed).split(X, y):
            Xtr0, ytr = X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True)
            Xte0, yte = X.iloc[te].reset_index(drop=True), y.iloc[te].reset_index(drop=True)
            Xtr, prep = _fit_own_preprocess(Xtr0, ytr, PANEL_RECIPE[PANEL])
            Xte = _apply_own_preprocess(Xte0, prep).reindex(columns=Xtr.columns)
            cf = _cats(Xtr); ci = [Xtr.columns.get_loc(c) for c in cf]
            _, enc = _rf_repr(Xtr, cf, fit=True); keep = _lgbm_redun_keep(Xtr)
            O, E = [], []
            for kind, (Xa, Xb) in [("lgbm", (Xtr[keep], Xte[keep])), ("rf", (Xtr, Xte)),
                                   ("cat", (Xtr, Xte)), ("xgb", (Xtr, Xte))]:
                o, e = _oof_test(kind, Xa, ytr, Xb, cf, ci, enc, 1.0, seed); O.append(o); E.append(e)
            to, teo = _tabpfn_member(Xtr, ytr, Xte, cf, seed); O.append(to); E.append(teo)
            lo, le = _logreg_member(Xtr, ytr, Xte, cf, seed); O.append(lo); E.append(le)
            folds.append((np.array(O), np.array(E), ytr.values, yte.values))
            print("fold cached", flush=True)
    np.savez(CACHE, **{f"O{i}": f[0] for i, f in enumerate(folds)},
             **{f"E{i}": f[1] for i, f in enumerate(folds)},
             **{f"ytr{i}": f[2] for i, f in enumerate(folds)},
             **{f"yte{i}": f[3] for i, f in enumerate(folds)}, n=len(folds))
    return folds


def load_cache():
    z = np.load(CACHE); n = int(z["n"])
    return [(z[f"O{i}"], z[f"E{i}"], z[f"ytr{i}"], z[f"yte{i}"]) for i in range(n)]


def eval_w(folds, wfun):
    """wfun(O_train, ytr)->weights over USE members. Returns per-fold (auc, f1)."""
    A, F = [], []
    for O, E, ytr, yte in folds:
        Ou, Eu = O[USE], E[USE]
        w = wfun(Ou, ytr); w = np.clip(w, 0, None); w = w / w.sum() if w.sum() > 0 else np.ones(len(USE)) / len(USE)
        bo = w @ Ou; be = w @ Eu
        A.append(roc_auc_score(yte, be)); F.append(_hF1(ytr, bo, yte, be))
    return np.array(A), np.array(F)


def w_equal(O, y): return np.ones(len(USE)) / len(USE)
def w_aucprop(O, y):
    a = np.array([max(roc_auc_score(y, O[i]) - 0.5, 1e-3) for i in range(len(USE))]); return a / a.sum()
def w_aucsoft(tau):
    def f(O, y):
        a = np.array([roc_auc_score(y, O[i]) for i in range(len(USE))]); e = np.exp((a - a.max()) / tau); return e / e.sum()
    return f
def w_stack(C):
    def f(O, y):
        lr = LogisticRegression(C=C, max_iter=2000, penalty="l2"); lr.fit(O.T, y)
        return np.clip(lr.coef_[0], 0, None) + 1e-6
    return f
def w_nnls(O, y):
    w, _ = nnls(O.T, y.astype(float)); return w + 1e-6
def w_simplex_l2(lam):
    def f(O, y):
        k = len(USE); eq = np.ones(k) / k
        def neg(w):
            w = np.clip(w, 0, None); s = w.sum()
            if s <= 0: return 0
            return -roc_auc_score(y, (w / s) @ O) + lam * np.sum((w / s - eq) ** 2)
        r = minimize(neg, eq, method="SLSQP", bounds=[(0, 1)] * k,
                     constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], options={"maxiter": 150})
        return np.clip(r.x, 0, None)
    return f


def subset_analysis(folds):
    """All 63 equal-weight subsets over the 6 members (the 'every combination' question)."""
    import itertools
    subs = [s for r in range(1, 7) for s in itertools.combinations(range(6), r)]
    rec = {s: {"auc": [], "fh": []} for s in subs}
    for O, E, ytr, yte in folds:
        for s in subs:
            idx = list(s); bo = O[idx].mean(0); be = E[idx].mean(0)
            rec[s]["auc"].append(roc_auc_score(yte, be)); rec[s]["fh"].append(_hF1(ytr, bo, yte, be))
    dep = (0, 1, 2, 3); bA = np.array(rec[dep]["auc"]); bF = np.array(rec[dep]["fh"])
    rows = []
    for s in subs:
        a = np.array(rec[s]["auc"]); f = np.array(rec[s]["fh"]); dA = a - bA
        seA = dA.std(ddof=1)/np.sqrt(len(dA)) if len(dA) > 1 else 1
        rows.append({"members": "+".join(NAMES[i] for i in s), "k": len(s),
                     "auc": a.mean(), "f1": f.mean(), "dAUC": dA.mean(), "sigAUC": dA.mean()/seA if seA else 0})
    R = pd.DataFrame(rows); R.to_csv("analysis/output/tabpfn_allsubsets.csv", index=False)
    fmt = {c: (lambda v: f"{v:.4f}") for c in ["auc", "f1", "dAUC"]}
    print(f"\n=== ALL 63 equal-weight subsets vs deployed 4-tree (lgbm+rf+cat+xgb) ===")
    print("--- TOP 12 by AUC ---")
    print(R.sort_values("auc", ascending=False).head(12).to_string(index=False, formatters=fmt))
    print("--- TOP 8 by honest F1 ---")
    print(R.sort_values("f1", ascending=False).head(8).to_string(index=False, formatters=fmt))
    print("--- best <=4-model subsets by AUC ---")
    print(R[R.k <= 4].sort_values("auc", ascending=False).head(8).to_string(index=False, formatters=fmt))


def main():
    if not os.environ.get("TABPFN_TOKEN"):
        print("set TABPFN_TOKEN"); return
    df = pd.read_csv(PANELS[PANEL]); df = normalize_sentinels(df); df = drop_all_nan_rows(df)
    de = _engineer(df.copy(), PANEL_RECIPE[PANEL]); _, num, cat, tgt = partition_columns(de)
    y = de[tgt].astype(int).reset_index(drop=True); X = de[num + cat].reset_index(drop=True)
    leaky = detect_leaky_columns(X[num], y, 0.99, verbose=False)
    X = X.drop(columns=sorted(set(leaky) | (KNOWN_LEAKY & set(X.columns))))
    folds = load_cache() if os.path.exists(CACHE) else build_cache(X, y)

    strategies = {
        "equal5": w_equal, "auc-prop": w_aucprop,
        "auc-softmax τ=.03": w_aucsoft(0.03), "auc-softmax τ=.01": w_aucsoft(0.01),
        "stack C=0.1": w_stack(0.1), "stack C=1": w_stack(1.0), "stack C=10": w_stack(10.0),
        "nnls": w_nnls,
        "simplex λ=0": w_simplex_l2(0.0), "simplex λ=0.05": w_simplex_l2(0.05), "simplex λ=0.5": w_simplex_l2(0.5),
    }
    bA, bF = eval_w(folds, w_equal)
    print(f"\n=== NON-EQUAL weighting (general, 5 strong members, {len(folds)} folds) vs equal5 ===")
    print(f"  equal5  AUC={bA.mean():.4f}  honest-F1={bF.mean():.4f}  [baseline]")
    for name, wf in strategies.items():
        if name == "equal5":
            continue
        a, f = eval_w(folds, wf); dA = a - bA; dF = f - bF
        sA = dA.std(ddof=1)/np.sqrt(len(dA)); sF = dF.std(ddof=1)/np.sqrt(len(dF))
        win = "  <<< WIN" if (dA.mean()/sA if sA else 0) >= 2 and dF.mean() > 0 else ""
        print(f"  {name:18s} AUC={a.mean():.4f} (Δ{dA.mean():+.4f} {dA.mean()/sA if sA else 0:+.1f}σ)  "
              f"F1={f.mean():.4f} (Δ{dF.mean():+.4f} {dF.mean()/sF if sF else 0:+.1f}σ){win}")
    subset_analysis(folds)
    print("DONE")


if __name__ == "__main__":
    main()
