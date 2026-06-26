"""Best shot at making the teacher's LINEAR models useful: add a properly-preprocessed non-tree
model (impute + QuantileTransformer) as a DECORRELATED member of the general tree blend. Even at
~0.81 standalone AUC, a decorrelated linear model could lift the blend (the way RF/CatBoost do).

General panel, honest 10-fold. Members calibrated via inner-OOF Platt. Tree blend = LGBM+RF+CatBoost.
Tests tree-blend vs tree-blend + {L1-LogReg, LDA, ElasticNet} at non-tree weights {0.1,0.2,0.3}.
AUC + F1* @ 0.143, paired vs tree-blend.

Run: .venv/bin/python -u analysis/exp_teacher_fair_blend.py
"""
import sys, warnings
sys.path.insert(0, "analysis"); sys.path.insert(0, "pipelines")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import QuantileTransformer, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score

from config import SEED
from schema import normalize_sentinels, partition_columns, drop_all_nan_rows
from utils import detect_leaky_columns, KNOWN_LEAKY, TEST_PRIORS, PANELS
from predict import _engineer, LGBM_PARAM_OVERRIDE
from recipe import PANEL_RECIPE
from preprocess_v2 import fit_keep_nan, apply_keep_nan
from run_pipeline import get_model_params
from general_blend import _cats, _rf_repr, _cat_repr
from calibration import fit_platt, apply_platt
from harness import f1_at_prior, tpr_fpr
from exp_teacher_preprocessing import _f1star

PANEL = "general"; PRIOR = TEST_PRIORS[PANEL]
WNT = [0.1, 0.2, 0.3]


def _nt_matrix(Xtr_pp, Xte_pp, cats):
    """Non-tree matrix: median-impute numerics + QuantileTransformer(normal) + ordinal cats."""
    num = [c for c in Xtr_pp.columns if c not in cats]
    imp = SimpleImputer(strategy="median"); qt = QuantileTransformer(output_distribution="normal", random_state=SEED)
    A = qt.fit_transform(imp.fit_transform(Xtr_pp[num])); B = qt.transform(imp.transform(Xte_pp[num]))
    A = pd.DataFrame(A, columns=num); B = pd.DataFrame(B, columns=num)
    if cats:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        A2 = enc.fit_transform(Xtr_pp[cats].astype(object).where(Xtr_pp[cats].notna(), "NA"))
        B2 = enc.transform(Xte_pp[cats].astype(object).where(Xte_pp[cats].notna(), "NA"))
        for i, c in enumerate(cats):
            A[f"c_{c}"] = A2[:, i]; B[f"c_{c}"] = B2[:, i]
    return A, B


def _fitpred(kind, Xa, ya, Xb, ctx):
    if kind == "lgbm":
        m = lgb.LGBMClassifier(**ctx["p"]); m.fit(Xa, ya, categorical_feature=ctx["cats"] or "auto")
        return m.predict_proba(Xb)[:, 1]
    if kind == "rf":
        m = RandomForestClassifier(n_estimators=600, max_features="sqrt", min_samples_leaf=2,
                                   class_weight="balanced", n_jobs=4, random_state=SEED)
        m.fit(Xa, ya); return m.predict_proba(Xb)[:, 1]
    if kind == "cat":
        m = CatBoostClassifier(iterations=800, depth=6, learning_rate=0.04, l2_leaf_reg=3,
                               scale_pos_weight=1.0, cat_features=ctx["cidx"], random_seed=SEED,
                               verbose=0, allow_writing_files=False, thread_count=4)
        m.fit(Xa, ya); return m.predict_proba(Xb)[:, 1]
    if kind == "l1":
        m = LogisticRegression(penalty="l1", solver="saga", C=0.1, class_weight="balanced", max_iter=2000)
        m.fit(Xa, ya); return m.predict_proba(Xb)[:, 1]
    if kind == "elastic":
        m = LogisticRegression(penalty="elasticnet", l1_ratio=0.5, solver="saga", C=0.3,
                               class_weight="balanced", max_iter=2000)
        m.fit(Xa, ya); return m.predict_proba(Xb)[:, 1]
    if kind == "lda":
        m = LinearDiscriminantAnalysis(); m.fit(Xa, ya); return m.predict_proba(Xb)[:, 1]


def _cal(kind, T, ytr, E, ctx, seed):
    oof = np.zeros(len(ytr)); skf = StratifiedKFold(3, shuffle=True, random_state=seed)
    for a, b in skf.split(T, ytr):
        oof[b] = _fitpred(kind, T.iloc[a], ytr.iloc[a], T.iloc[b], ctx)
    platt = fit_platt(oof, ytr.values)
    return apply_platt(_fitpred(kind, T, ytr, E, ctx), platt)


def main():
    df = pd.read_csv(PANELS[PANEL]); df = normalize_sentinels(df); df = drop_all_nan_rows(df)
    de = _engineer(df.copy(), PANEL_RECIPE[PANEL])
    _, num_cols, cat_cols, tgt = partition_columns(de)
    y = de[tgt].reset_index(drop=True); X = de[num_cols + cat_cols].reset_index(drop=True)
    leaky = detect_leaky_columns(X[num_cols], y, auc_threshold=0.99, verbose=False)
    X = X.drop(columns=sorted(set(leaky) | (KNOWN_LEAKY & set(X.columns))))
    params = {**get_model_params(len(X), 1.0), **LGBM_PARAM_OVERRIDE[PANEL], "verbose": -1}

    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
    rec = {k: {"auc": [], "f1": []} for k in
           ["tree", "+l1@.1", "+l1@.2", "+l1@.3", "+lda@.1", "+lda@.2", "+lda@.3",
            "+elastic@.2"]}
    nt_auc = {"l1": [], "lda": [], "elastic": []}
    for k, (tr, te) in enumerate(outer.split(X, y)):
        Xtr = X.iloc[tr].reset_index(drop=True); Xte = X.iloc[te].reset_index(drop=True)
        ytr = y.iloc[tr].reset_index(drop=True); yte = y.iloc[te].values
        Xtr_pp, ppf = fit_keep_nan(Xtr); Xte_pp = apply_keep_nan(Xte, ppf).reindex(columns=Xtr_pp.columns)
        cats = _cats(Xtr_pp)
        Trf, enc = _rf_repr(Xtr_pp, cats, fit=True); Erf, _ = _rf_repr(Xte_pp, cats, enc=enc)
        Tcat = _cat_repr(Xtr_pp, cats); Ecat = _cat_repr(Xte_pp, cats)
        cidx = [Tcat.columns.get_loc(c) for c in cats]
        Tnt, Ent = _nt_matrix(Xtr_pp, Xte_pp, cats)
        ctx = {"p": params, "cats": cats, "cidx": cidx}
        s = SEED + k
        lg = _cal("lgbm", Xtr_pp, ytr, Xte_pp, ctx, s)
        rf = _cal("rf", Trf, ytr, Erf, ctx, s)
        cb = _cal("cat", Tcat, ytr, Ecat, ctx, s)
        tree = (lg + rf + cb) / 3
        nts = {nm: _cal(nm, Tnt, ytr, Ent, ctx, s) for nm in ["l1", "lda", "elastic"]}
        for nm in nts:
            nt_auc[nm].append(roc_auc_score(yte, nts[nm]))
        def rF1(p): return _f1star(yte, p, PRIOR)
        rec["tree"]["auc"].append(roc_auc_score(yte, tree)); rec["tree"]["f1"].append(rF1(tree))
        for nm, w in [("l1", .1), ("l1", .2), ("l1", .3), ("lda", .1), ("lda", .2), ("lda", .3), ("elastic", .2)]:
            blend = (1 - w) * tree + w * nts[nm]
            key = f"+{nm}@{w:.0f}".replace("@0", "@.") if False else f"+{nm}@.{int(w*10)}"
            rec[key]["auc"].append(roc_auc_score(yte, blend)); rec[key]["f1"].append(rF1(blend))

    bA = np.array(rec["tree"]["auc"]); bF = np.array(rec["tree"]["f1"])
    print("=== FAIR non-tree-as-blend-member (general, vs LGBM+RF+CatBoost tree blend) ===")
    print(f"  standalone non-tree AUC: " + " ".join(f"{k}={np.mean(v):.4f}" for k, v in nt_auc.items()))
    print(f"  tree blend  AUC={bA.mean():.4f}  F1*={bF.mean():.4f}  [baseline]")
    for key in rec:
        if key == "tree":
            continue
        au = np.array(rec[key]["auc"]); f1 = np.array(rec[key]["f1"])
        dA = au - bA; dF = f1 - bF
        sA = dA.std(ddof=1)/np.sqrt(len(dA)); sF = dF.std(ddof=1)/np.sqrt(len(dF))
        tag = "  <<< WIN" if (dA.mean()/sA if sA > 0 else 0) >= 2 and dF.mean() > 0 else ""
        print(f"  {key:12s} AUC={au.mean():.4f} (Δ{dA.mean():+.4f} {dA.mean()/sA if sA>0 else 0:+.1f}σ)  "
              f"F1*={f1.mean():.4f} (Δ{dF.mean():+.4f} {dF.mean()/sF if sF>0 else 0:+.1f}σ){tag}", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
