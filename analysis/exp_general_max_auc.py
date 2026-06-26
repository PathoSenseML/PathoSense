"""Throw EVERY available model at general's AUC — do the skipped AUC-positive levers (+logreg,
stacked meta-learner) stack onto the deployed 4-tree blend enough to lift the F1 ceiling past 0.55?

The oracle (test-peeked best threshold) F1 is the upper bound for ANY threshold/operating-point trick
at a given ranking. F1>0.55 @ π=0.143 needs AUC ~0.866. Deployed blend ~0.864. This builds the richest
honest blend (LGBM+RF+CatBoost+XGB + L2-LogReg + a logistic stacking meta-learner) and reports
blended AUC + oracle-F1 + honest-recalibrated-F1, 3 seeds.

Run: .venv/bin/python -u analysis/exp_general_max_auc.py
"""
import sys, warnings
sys.path.insert(0, "analysis"); sys.path.insert(0, "pipelines")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import QuantileTransformer, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

from config import SEED
from schema import normalize_sentinels, partition_columns, drop_all_nan_rows
from utils import detect_leaky_columns, KNOWN_LEAKY, TEST_PRIORS, PANELS
from predict import _engineer, _fit_own_preprocess, _apply_own_preprocess, _lgbm_redun_keep
from recipe import PANEL_RECIPE
from general_blend import _cats, _rf_repr
from calibration import fit_platt, apply_platt
from harness import f1_at_prior, tpr_fpr, pick_threshold_at_prior
from exp_general_push_055 import _oof_test, _f1_swept

PANEL = "general"; PRIOR = TEST_PRIORS[PANEL]


def _logreg_member(Xtr, ytr, Xte, cats, seed):
    num = [c for c in Xtr.columns if c not in cats]
    imp = SimpleImputer(strategy="median"); qt = QuantileTransformer(output_distribution="normal", random_state=seed)
    A = qt.fit_transform(imp.fit_transform(Xtr[num])); B = qt.transform(imp.transform(Xte[num]))
    A = pd.DataFrame(A, columns=num); B = pd.DataFrame(B, columns=num)
    if cats:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        Ac = enc.fit_transform(Xtr[cats].astype(object).where(Xtr[cats].notna(), "NA"))
        Bc = enc.transform(Xte[cats].astype(object).where(Xte[cats].notna(), "NA"))
        for i, c in enumerate(cats): A[f"c{c}"] = Ac[:, i]; B[f"c{c}"] = Bc[:, i]
    def fp(Xa, ya, Xb):
        m = LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000); m.fit(Xa, ya)
        return m.predict_proba(Xb)[:, 1]
    oof = np.zeros(len(ytr)); skf = StratifiedKFold(3, shuffle=True, random_state=seed)
    for a, b in skf.split(A, ytr): oof[b] = fp(A.iloc[a], ytr.iloc[a], A.iloc[b])
    pl = fit_platt(oof, ytr.values)
    return apply_platt(oof, pl), apply_platt(fp(A, ytr, B), pl)


def main():
    df = pd.read_csv(PANELS[PANEL]); df = normalize_sentinels(df); df = drop_all_nan_rows(df)
    de = _engineer(df.copy(), PANEL_RECIPE[PANEL]); _, num, cat, tgt = partition_columns(de)
    y = de[tgt].astype(int).reset_index(drop=True); X = de[num + cat].reset_index(drop=True)
    leaky = detect_leaky_columns(X[num], y, auc_threshold=0.99, verbose=False)
    X = X.drop(columns=sorted(set(leaky) | (KNOWN_LEAKY & set(X.columns))))

    res = {k: {"auc": [], "f1o": [], "f1h": []} for k in ["4tree(dep)", "+logreg", "+logreg+stack"]}
    for seed in (SEED, SEED + 1000, SEED + 2000):
        outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=seed)
        for tr, te in outer.split(X, y):
            Xtr0, ytr = X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True)
            Xte0, yte = X.iloc[te].reset_index(drop=True), y.iloc[te].reset_index(drop=True)
            Xtr, prep = _fit_own_preprocess(Xtr0, ytr, PANEL_RECIPE[PANEL])
            Xte = _apply_own_preprocess(Xte0, prep).reindex(columns=Xtr.columns)
            cf = _cats(Xtr); ci = [Xtr.columns.get_loc(c) for c in cf]
            _, enc = _rf_repr(Xtr, cf, fit=True)
            keep = _lgbm_redun_keep(Xtr); Xtr_l, Xte_l = Xtr[keep], Xte[keep]
            yv_tr, yv_te = ytr.values, yte.values
            lo, le = _oof_test("lgbm", Xtr_l, ytr, Xte_l, cf, ci, enc, 1.0, seed)
            ro, re = _oof_test("rf", Xtr, ytr, Xte, cf, ci, enc, 1.0, seed)
            co, ce = _oof_test("cat", Xtr, ytr, Xte, cf, ci, enc, 1.0, seed)
            xo, xe = _oof_test("xgb", Xtr, ytr, Xte, cf, ci, enc, 1.0, seed)
            go, ge = _logreg_member(Xtr, ytr, Xte, cf, seed)
            trees_o = [lo, ro, co, xo]; trees_e = [le, re, ce, xe]

            def record(key, btr, bte):
                thr = pick_threshold_at_prior(yv_tr, btr, PRIOR)
                res[key]["auc"].append(roc_auc_score(yv_te, bte))
                res[key]["f1o"].append(_f1_swept(yv_te, bte))
                res[key]["f1h"].append(f1_at_prior(*tpr_fpr(yv_te, (bte >= thr).astype(int)), PRIOR))

            record("4tree(dep)", sum(trees_o) / 4, sum(trees_e) / 4)
            record("+logreg", (sum(trees_o) + go) / 5, (sum(trees_e) + ge) / 5)
            # logistic stacking meta-learner on the 5 members' OOF
            Mtr = np.column_stack(trees_o + [go]); Mte = np.column_stack(trees_e + [ge])
            sk = StratifiedKFold(3, shuffle=True, random_state=seed); soof = np.zeros(len(ytr))
            for a, b in sk.split(Mtr, ytr):
                lr = LogisticRegression(max_iter=1000); lr.fit(Mtr[a], ytr.iloc[a]); soof[b] = lr.predict_proba(Mtr[b])[:, 1]
            lr = LogisticRegression(max_iter=1000); lr.fit(Mtr, ytr)
            record("+logreg+stack", soof, lr.predict_proba(Mte)[:, 1])

    print(f"=== general MAX-AUC (3 seeds, {len(res['4tree(dep)']['auc'])} folds, prior {PRIOR}) ===")
    print("  F1>0.55 needs AUC ~0.866; oracle = test-peeked best-threshold F1 (optimistic ceiling)")
    base = np.array(res["4tree(dep)"]["auc"])
    for k in res:
        a = np.array(res[k]["auc"]); fo = np.array(res[k]["f1o"]); fh = np.array(res[k]["f1h"])
        dA = a - base; se = dA.std(ddof=1)/np.sqrt(len(dA)) if len(dA) > 1 else 0
        print(f"  {k:16s} AUC={a.mean():.4f} (Δ{dA.mean():+.4f} {dA.mean()/se if se>0 else 0:+.1f}σ)  "
              f"oracle-F1={fo.mean():.4f}{'  >0.55!' if fo.mean()>0.55 else ''}  honest-F1={fh.mean():.4f}")
    print("DONE")


if __name__ == "__main__":
    main()
