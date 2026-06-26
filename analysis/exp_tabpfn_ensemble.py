"""PROPER ensemble calculation for the 4-tree + TabPFN blend on general.

exp_tabpfn_general used naive equal-weight (1/5 each). This optimizes TabPFN's mixing weight:
  blend = (1-w)*mean(4 trees) + w*TabPFN
- FIXED-weight sweep w in {0,.1,.15,.2,.25,.33,.5} (no overfitting; w=.2 == equal 5-member).
- NESTED data-driven w (picked per fold on train-OOF, applied held-out) — checks if optimizing helps
  or overfits (the 4-tree blend reweighting overfit -1.65σ).
- logistic STACKING meta-learner on the 5 members.
Honest 3-seed CV, recalibrated threshold + oracle. AUC + F1* @ 0.143.

Run: TABPFN_TOKEN=<key> .venv/bin/python -u analysis/exp_tabpfn_ensemble.py
"""
import sys, os, warnings
sys.path.insert(0, "analysis"); sys.path.insert(0, "pipelines")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from config import SEED
from schema import normalize_sentinels, partition_columns, drop_all_nan_rows
from utils import detect_leaky_columns, KNOWN_LEAKY, TEST_PRIORS, PANELS
from predict import _engineer, _fit_own_preprocess, _apply_own_preprocess, _lgbm_redun_keep
from recipe import PANEL_RECIPE
from general_blend import _cats, _rf_repr
from harness import f1_at_prior, tpr_fpr, pick_threshold_at_prior
from exp_general_push_055 import _oof_test, _f1_swept
from exp_tabpfn_general import _tabpfn_member

PANEL = "general"; PRIOR = TEST_PRIORS[PANEL]
WGRID = [0.0, 0.10, 0.15, 0.20, 0.25, 0.33, 0.50]


def _metrics(ytr, btr, yte, bte):
    thr = pick_threshold_at_prior(ytr, btr, PRIOR)
    return (roc_auc_score(yte, bte), _f1_swept(yte, bte),
            f1_at_prior(*tpr_fpr(yte, (bte >= thr).astype(int)), PRIOR))


def main():
    if not os.environ.get("TABPFN_TOKEN"):
        print("ERROR: set TABPFN_TOKEN"); return
    df = pd.read_csv(PANELS[PANEL]); df = normalize_sentinels(df); df = drop_all_nan_rows(df)
    de = _engineer(df.copy(), PANEL_RECIPE[PANEL]); _, num, cat, tgt = partition_columns(de)
    y = de[tgt].astype(int).reset_index(drop=True); X = de[num + cat].reset_index(drop=True)
    leaky = detect_leaky_columns(X[num], y, auc_threshold=0.99, verbose=False)
    X = X.drop(columns=sorted(set(leaky) | (KNOWN_LEAKY & set(X.columns))))

    keys = [f"w={w}" for w in WGRID] + ["nested_w", "stack5"]
    rec = {k: {"auc": [], "fo": [], "fh": []} for k in keys}
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
            t_o, t_e = [], []
            for kind, (Xa, Xb) in [("lgbm", (Xtr[keep], Xte[keep])), ("rf", (Xtr, Xte)),
                                   ("cat", (Xtr, Xte)), ("xgb", (Xtr, Xte))]:
                o, e = _oof_test(kind, Xa, ytr, Xb, cf, ci, enc, 1.0, seed); t_o.append(o); t_e.append(e)
            tab_o, tab_e = _tabpfn_member(Xtr, ytr, Xte, cf, seed)
            tree_o = sum(t_o) / 4; tree_e = sum(t_e) / 4
            # fixed-weight sweep
            for w in WGRID:
                bo = (1 - w) * tree_o + w * tab_o; be = (1 - w) * tree_e + w * tab_e
                a, fo, fh = _metrics(yv_tr, bo, yv_te, be)
                rec[f"w={w}"]["auc"].append(a); rec[f"w={w}"]["fo"].append(fo); rec[f"w={w}"]["fh"].append(fh)
            # nested data-driven w (max train-OOF AUC)
            wn = max(WGRID, key=lambda w: roc_auc_score(yv_tr, (1 - w) * tree_o + w * tab_o))
            a, fo, fh = _metrics(yv_tr, (1-wn)*tree_o+wn*tab_o, yv_te, (1-wn)*tree_e+wn*tab_e)
            rec["nested_w"]["auc"].append(a); rec["nested_w"]["fo"].append(fo); rec["nested_w"]["fh"].append(fh)
            # logistic stacking on 5 members
            M_o = np.column_stack(t_o + [tab_o]); M_e = np.column_stack(t_e + [tab_e])
            sk = StratifiedKFold(3, shuffle=True, random_state=seed); soof = np.zeros(len(ytr))
            for a_, b_ in sk.split(M_o, ytr):
                lr = LogisticRegression(max_iter=1000); lr.fit(M_o[a_], ytr.iloc[a_]); soof[b_] = lr.predict_proba(M_o[b_])[:, 1]
            lr = LogisticRegression(max_iter=1000); lr.fit(M_o, ytr)
            a, fo, fh = _metrics(yv_tr, soof, yv_te, lr.predict_proba(M_e)[:, 1])
            rec["stack5"]["auc"].append(a); rec["stack5"]["fo"].append(fo); rec["stack5"]["fh"].append(fh)
        print("seed done", flush=True)

    base = np.array(rec["w=0.0"]["auc"])
    print(f"\n=== PROPER ensemble: TabPFN weight optimization ({len(base)} folds, 3 seeds) ===")
    print("  w=0 == 4-tree deployed; w=.2 == naive equal-5-member.  F1>0.55 needs AUC~0.866")
    for k in keys:
        a = np.array(rec[k]["auc"]); fo = np.array(rec[k]["fo"]); fh = np.array(rec[k]["fh"])
        dA = a - base; se = dA.std(ddof=1)/np.sqrt(len(dA)) if len(dA) > 1 else 0
        star = "  >0.55!" if fo.mean() > 0.55 else ""
        print(f"  {k:10s} AUC={a.mean():.4f} (Δ{dA.mean():+.4f} {dA.mean()/se if se>0 else 0:+.1f}σ)  "
              f"oracle={fo.mean():.4f}  honest={fh.mean():.4f}{star}")
    print("DONE")


if __name__ == "__main__":
    main()
