"""General blend EXPANSION: the deployed blend is (LGBM_spw1bag + RF + CatBoost)/3 equal-average. Test
whether adding diverse learners (XGBoost / ExtraTrees / HistGB) and/or a STACKED logistic meta-learner
beats it. Compute 6 calibrated component OOF+test per fold (shared preprocessing + inner CV), then
evaluate blend subsets + stack. Paired Δ vs current 3-way. Metric: F1* at prior + AUC. 2 seeds.
Run: .venv/bin/python -u analysis/exp_general_blend_expand.py
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from config import SEED
from utils import TEST_PRIORS
from schema import partition_columns
from calibration import fit_platt, apply_platt
from harness import load_panel, _drop_leaky, lgb_oof_and_test, pick_threshold_at_prior, tpr_fpr, f1_at_prior
from preprocess_v2 import fit_keep_nan, apply_keep_nan
from general_blend import _rf_repr, _cat_repr, _cat_model
from run_pipeline import get_model_params

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def components(Xtr_pp, Xte_pp, y_tr, cat_feats, spw, lgb_params):
    """Return {model: (oof_cal, test_cal)} for 6 learners, calibrated."""
    X_ord_tr, enc = _rf_repr(Xtr_pp, cat_feats, fit=True)
    X_ord_te, _ = _rf_repr(Xte_pp, cat_feats, enc=enc, fit=False)
    X_cat_tr, X_cat_te = _cat_repr(Xtr_pp, cat_feats), _cat_repr(Xte_pp, cat_feats)
    cat_idx = [X_cat_tr.columns.get_loc(c) for c in cat_feats]
    n = len(y_tr); out = {}

    # LGBM (native cats) — deployed spw1_bag params
    tp, oof = lgb_oof_and_test(Xtr_pp, y_tr, Xte_pp, lgb_params, n_seeds=3, n_folds=5)
    out["lgbm"] = (oof, tp)

    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    def sk_oof(make, Xtr, Xte, fitkw=None):
        oo = np.zeros(n)
        for a, b in skf.split(Xtr, y_tr):
            m = make(); m.fit(Xtr.iloc[a], y_tr.iloc[a], **(fitkw or {}))
            oo[b] = m.predict_proba(Xtr.iloc[b])[:, 1]
        mf = make(); mf.fit(Xtr, y_tr, **(fitkw or {}))
        return oo, mf.predict_proba(Xte)[:, 1]
    out["rf"] = sk_oof(lambda: RandomForestClassifier(n_estimators=300, max_features="sqrt", n_jobs=4,
                                                      class_weight="balanced", random_state=SEED), X_ord_tr, X_ord_te)
    out["et"] = sk_oof(lambda: ExtraTreesClassifier(n_estimators=400, max_features="sqrt", n_jobs=4,
                                                    class_weight="balanced", random_state=SEED), X_ord_tr, X_ord_te)
    out["xgb"] = sk_oof(lambda: XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=5,
                                              subsample=0.8, colsample_bytree=0.8, reg_lambda=1.5,
                                              eval_metric="logloss", n_jobs=4, random_state=SEED), X_ord_tr, X_ord_te)
    out["hgb"] = sk_oof(lambda: HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                                               l2_regularization=1.0, random_state=SEED,
                                                               class_weight="balanced"), X_ord_tr, X_ord_te)
    # CatBoost (cat repr)
    cb_oof = np.zeros(n)
    for k, (a, b) in enumerate(skf.split(X_cat_tr, y_tr)):
        i2, ies = train_test_split(a, test_size=0.15, random_state=SEED + k, stratify=y_tr.iloc[a])
        cb = _cat_model(cat_idx, spw); cb.fit(X_cat_tr.iloc[i2], y_tr.iloc[i2], eval_set=(X_cat_tr.iloc[ies], y_tr.iloc[ies]))
        cb_oof[b] = cb.predict_proba(X_cat_tr.iloc[b])[:, 1]
    i2, ies = train_test_split(np.arange(n), test_size=0.15, random_state=SEED + 99, stratify=y_tr)
    cbf = _cat_model(cat_idx, spw); cbf.fit(X_cat_tr.iloc[i2], y_tr.iloc[i2], eval_set=(X_cat_tr.iloc[ies], y_tr.iloc[ies]))
    out["cat"] = (cb_oof, cbf.predict_proba(X_cat_te)[:, 1])

    # calibrate each
    for k, (oo, tt) in out.items():
        pl = fit_platt(oo, y_tr.values); out[k] = (apply_platt(oo, pl), apply_platt(tt, pl))
    return out


BLENDS = {
    "current(l+r+c)": ["lgbm", "rf", "cat"],
    "+xgb":           ["lgbm", "rf", "cat", "xgb"],
    "+et":            ["lgbm", "rf", "cat", "et"],
    "+hgb":           ["lgbm", "rf", "cat", "hgb"],
    "+xgb+et":        ["lgbm", "rf", "cat", "xgb", "et"],
    "all6":           ["lgbm", "rf", "cat", "xgb", "et", "hgb"],
}


def main():
    prior = TEST_PRIORS["general"]
    df = load_panel("general"); _, num, cat, tgt = partition_columns(df)
    X = df[num + cat].reset_index(drop=True); y = df[tgt].astype(int).reset_index(drop=True)
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    lgb_params = {**get_model_params(2000, spw), "scale_pos_weight": 1.0, "bagging_freq": 1, "bagging_fraction": 0.8}
    print("=== general blend expansion (vs deployed lgbm_spw1bag+rf+cat) ===", flush=True)
    rows = []
    keys = list(BLENDS) + ["stack_logit"]
    for seed in (SEED, SEED + 1000):
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=seed)
        accF = {k: [] for k in keys}; accA = {k: [] for k in keys}
        for tr, te in rskf.split(X, y):
            tr, te = np.asarray(tr), np.asarray(te)
            Xtr, Xte = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True)
            ytr, yte = y.iloc[tr].reset_index(drop=True), y.iloc[te].reset_index(drop=True)
            Xtr, Xte = _drop_leaky(Xtr, ytr, Xte)
            Xtr_pp, ppf = fit_keep_nan(Xtr); Xte_pp = apply_keep_nan(Xte, ppf)
            for c in Xtr_pp.columns:
                if str(Xtr_pp[c].dtype) == "category":
                    Xte_pp[c] = pd.Categorical(Xte_pp[c].astype(object), categories=Xtr_pp[c].cat.categories)
            cat_feats = [c for c in Xtr_pp.columns if str(Xtr_pp[c].dtype) == "category"]
            comp = components(Xtr_pp, Xte_pp, ytr, cat_feats, spw, lgb_params)
            def score(btr, bte):
                t = pick_threshold_at_prior(ytr.values, btr, prior)
                return f1_at_prior(*tpr_fpr(yte.values, (bte >= t).astype(int)), prior), roc_auc_score(yte.values, bte)
            for name, mem in BLENDS.items():
                btr = np.mean([comp[m][0] for m in mem], axis=0); bte = np.mean([comp[m][1] for m in mem], axis=0)
                f, a = score(btr, bte); accF[name].append(f); accA[name].append(a)
            # stacked logistic meta-learner over all 6 OOF probs
            order = ["lgbm", "rf", "cat", "xgb", "et", "hgb"]
            Ztr = np.column_stack([comp[m][0] for m in order]); Zte = np.column_stack([comp[m][1] for m in order])
            lr = LogisticRegression(max_iter=1000, C=1.0).fit(Ztr, ytr.values)
            f, a = score(lr.predict_proba(Ztr)[:, 1], lr.predict_proba(Zte)[:, 1])
            accF["stack_logit"].append(f); accA["stack_logit"].append(a)
        base = np.array(accF["current(l+r+c)"]); baseA = np.array(accA["current(l+r+c)"]); nf = len(base)
        print(f"  seed{seed} ({nf} folds): current F1*={base.mean():.4f} AUC={baseA.mean():.4f}", flush=True)
        for k in keys:
            if k == "current(l+r+c)": continue
            f = np.array(accF[k]); a = np.array(accA[k]); d = f - base; da = a - baseA
            se = d.std(ddof=1)/np.sqrt(nf); sig = d.mean()/se if se > 0 else 0
            sea = da.std(ddof=1)/np.sqrt(nf); siga = da.mean()/sea if sea > 0 else 0
            print(f"    {k:14s} F1*={f.mean():.4f} Δ={d.mean():+.4f} ({sig:+.2f}σ) | AUC Δ={da.mean():+.4f} ({siga:+.2f}σ)", flush=True)
            rows.append({"seed": seed, "blend": k, "f1": f.mean(), "df1": d.mean(), "sig_f1": sig,
                         "auc": a.mean(), "dauc": da.mean(), "sig_auc": siga})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "general_blend_expand.csv"), index=False)
    print(f"\nsaved -> general_blend_expand.csv", flush=True)


if __name__ == "__main__":
    main()
