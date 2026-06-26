"""EXHAUSTIVE ensemble subset search WITH TabPFN-v2.6. Build all 6 candidate members once per fold
(lgbm, rf, cat, xgb, tabpfn-v2.6, logreg), then evaluate EVERY non-empty subset (63) at equal weight.
Answers: is a subset of <5 models better than the equal-5? Honest 3-seed CV, AUC + F1* @ 0.143,
recalibrated threshold. Paired sigma vs the deployed 4-tree blend.

Run: TABPFN_TOKEN=<key> .venv/bin/python -u analysis/exp_tabpfn_allsubsets.py
"""
import sys, os, warnings, itertools
sys.path.insert(0, "analysis"); sys.path.insert(0, "pipelines")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
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
DEPLOYED = (0, 1, 2, 3)   # lgbm+rf+cat+xgb


def _hF1(ytr, otr, yte, ote):
    thr = pick_threshold_at_prior(ytr, otr, PRIOR)
    return f1_at_prior(*tpr_fpr(yte, (ote >= thr).astype(int)), PRIOR)


def main():
    if not os.environ.get("TABPFN_TOKEN"):
        print("set TABPFN_TOKEN"); return
    df = pd.read_csv(PANELS[PANEL]); df = normalize_sentinels(df); df = drop_all_nan_rows(df)
    de = _engineer(df.copy(), PANEL_RECIPE[PANEL]); _, num, cat, tgt = partition_columns(de)
    y = de[tgt].astype(int).reset_index(drop=True); X = de[num + cat].reset_index(drop=True)
    leaky = detect_leaky_columns(X[num], y, 0.99, verbose=False)
    X = X.drop(columns=sorted(set(leaky) | (KNOWN_LEAKY & set(X.columns))))

    subsets = [s for r in range(1, 7) for s in itertools.combinations(range(6), r)]
    rec = {s: {"auc": [], "fh": []} for s in subsets}
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
            O, E = [], []
            for kind, (Xa, Xb) in [("lgbm", (Xtr[keep], Xte[keep])), ("rf", (Xtr, Xte)),
                                   ("cat", (Xtr, Xte)), ("xgb", (Xtr, Xte))]:
                o, e = _oof_test(kind, Xa, ytr, Xb, cf, ci, enc, 1.0, seed); O.append(o); E.append(e)
            to, teo = _tabpfn_member(Xtr, ytr, Xte, cf, seed); O.append(to); E.append(teo)
            lo, le = _logreg_member(Xtr, ytr, Xte, cf, seed); O.append(lo); E.append(le)
            O = np.array(O); E = np.array(E)   # (6, n)
            for s in subsets:
                idx = list(s); bo = O[idx].mean(0); be = E[idx].mean(0)
                rec[s]["auc"].append(roc_auc_score(yv_te, be)); rec[s]["fh"].append(_hF1(yv_tr, bo, yv_te, be))
        print("seed done", flush=True)

    base_auc = np.array(rec[DEPLOYED]["auc"]); base_fh = np.array(rec[DEPLOYED]["fh"])
    rows = []
    for s in subsets:
        a = np.array(rec[s]["auc"]); fh = np.array(rec[s]["fh"])
        dA = a - base_auc; seA = dA.std(ddof=1)/np.sqrt(len(dA)) if len(dA) > 1 else 1
        dF = fh - base_fh; seF = dF.std(ddof=1)/np.sqrt(len(dF)) if len(dF) > 1 else 1
        rows.append({"members": "+".join(NAMES[i] for i in s), "k": len(s), "auc": a.mean(),
                     "f1": fh.mean(), "dAUC": dA.mean(), "sigA": dA.mean()/seA if seA else 0,
                     "dF1": dF.mean(), "sigF": dF.mean()/seF if seF else 0})
    R = pd.DataFrame(rows)
    R.to_csv("analysis/output/tabpfn_allsubsets.csv", index=False)
    print(f"\n=== ALL {len(subsets)} subsets (general, TabPFN v2.6, 15 folds) — vs deployed 4-tree ===")
    print("--- TOP 15 by AUC ---")
    print(R.sort_values("auc", ascending=False).head(15).to_string(index=False,
          formatters={c: (lambda v: f"{v:.4f}") for c in ["auc", "f1", "dAUC", "dF1"]}))
    print("\n--- TOP 10 by honest F1 ---")
    print(R.sort_values("f1", ascending=False).head(10).to_string(index=False,
          formatters={c: (lambda v: f"{v:.4f}") for c in ["auc", "f1", "dAUC", "dF1"]}))
    print("\n--- best <5-model subsets (k<=4) by AUC ---")
    print(R[R.k <= 4].sort_values("auc", ascending=False).head(8).to_string(index=False,
          formatters={c: (lambda v: f"{v:.4f}") for c in ["auc", "f1", "dAUC", "dF1"]}))
    print("DONE")


if __name__ == "__main__":
    main()
