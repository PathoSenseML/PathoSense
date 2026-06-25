"""Cancer = 0.5·transfer + 0.5·OWN. The own model (median+mRMR single LGBM) is the one component never
improved this session. Test, in the deployed cancer config (0.5 blend, scorer=rfup, threshold 0.81):
  bag      : activate bagging on the own LGBM (subsample=0.8 is inert without bagging_freq)
  blend    : own = (LGBM + RF + XGB)/3 on the median+mRMR matrix (decorrelation, like the general blend)
  bag+blend: both
Transfer g is the fixed deployed scorer. Per fold: own variant OOF+test, blend 0.5·g+0.5·own, fixed
threshold 0.81, analytic F1*. Paired vs deployed own, 2 seeds.
Run: .venv/bin/python -u analysis/exp_cancer_own.py
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from config import SEED
from utils import TEST_PRIORS
from calibration import fit_platt, apply_platt
from harness import _drop_leaky, lgb_oof_and_test, tpr_fpr, f1_at_prior
from run_pipeline import preprocess_data, get_model_params
from feature_engineering import fit_score_agreement, apply_score_agreement
from harness import fast_mrmr
from general_blend import _rf_repr, _xgb_model
from transfer import fit_transfer_scorer, transfer_score
from exp_transfer import _load, load_target_engineered, target_deployed_fn

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
THR = 0.81  # deployed cancer threshold


def own_preprocess(X_tr, y_tr, X_te):
    Xtr, Xte = _drop_leaky(X_tr, y_tr, X_te)
    Xtr_pp, _, Xte_pp, _ = preprocess_data(Xtr.copy(), Xtr.copy(), Xte.copy(), 0.95)
    saf = fit_score_agreement(Xtr_pp, y_tr, top_k=7)
    Xtr_pp = apply_score_agreement(Xtr_pp, saf); Xte_pp = apply_score_agreement(Xte_pp, saf)
    Xtr_pp, _, Xte_pp, _ = fast_mrmr(Xtr_pp, y_tr, Xte_pp, Xte_pp)
    return Xtr_pp, Xte_pp


def own_variant(Xtr_pp, Xte_pp, y_tr, kind):
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    params = get_model_params(len(y_tr), spw)
    if kind in ("bag", "bag_blend"):
        params = {**params, "bagging_freq": 1, "bagging_fraction": 0.8}
    tp, oof = lgb_oof_and_test(Xtr_pp, y_tr, Xte_pp, params, n_seeds=5, n_folds=5)
    pf = fit_platt(oof, y_tr.values); lo, lt = apply_platt(oof, pf), apply_platt(tp, pf)
    if kind in ("blend", "bag_blend"):
        cat_feats = [c for c in Xtr_pp.columns if str(Xtr_pp[c].dtype) == "category"]
        Xo_tr, enc = _rf_repr(Xtr_pp, cat_feats, fit=True)
        Xo_te, _ = _rf_repr(Xte_pp, cat_feats, enc=enc, fit=False)
        n = len(y_tr); rf_o = np.zeros(n); xg_o = np.zeros(n)
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
        for a, b in skf.split(Xo_tr, y_tr):
            rf = RandomForestClassifier(n_estimators=400, max_features="sqrt", n_jobs=4,
                                        class_weight="balanced", random_state=SEED).fit(Xo_tr.iloc[a], y_tr.iloc[a])
            rf_o[b] = rf.predict_proba(Xo_tr.iloc[b])[:, 1]
            xg = _xgb_model().fit(Xo_tr.iloc[a], y_tr.iloc[a]); xg_o[b] = xg.predict_proba(Xo_tr.iloc[b])[:, 1]
        rpl, xpl = fit_platt(rf_o, y_tr.values), fit_platt(xg_o, y_tr.values)
        rf_f = RandomForestClassifier(n_estimators=400, max_features="sqrt", n_jobs=4,
                                      class_weight="balanced", random_state=SEED).fit(Xo_tr, y_tr)
        xg_f = _xgb_model().fit(Xo_tr, y_tr)
        rf_t = apply_platt(rf_f.predict_proba(Xo_te)[:, 1], rpl); xg_t = apply_platt(xg_f.predict_proba(Xo_te)[:, 1], xpl)
        lo = (lo + apply_platt(rf_o, rpl) + apply_platt(xg_o, xpl)) / 3.0
        lt = (lt + rf_t + xg_t) / 3.0
    return lt, lo


KINDS = ["base", "bag", "blend", "bag_blend"]


def main():
    prior = TEST_PRIORS["cancer"]
    df = _load("cancer"); y = df["Label"].astype(int).values
    g = transfer_score(df, fit_transfer_scorer("cancer"))
    Xt_own, yt = load_target_engineered("cancer"); assert (yt.values == y).all()
    print("=== cancer own-model improvement (deployed 0.5 blend, scorer=rfup, t=0.81) ===", flush=True)
    rows = []
    for seed in (SEED, SEED + 1000):
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=8, random_state=seed)
        acc = {k: [] for k in KINDS}
        for tr, te in rskf.split(Xt_own, yt):
            tr, te = np.asarray(tr), np.asarray(te)
            ytr, yte = yt.iloc[tr].reset_index(drop=True), yt.iloc[te].reset_index(drop=True)
            Xtr_pp, Xte_pp = own_preprocess(Xt_own.iloc[tr].reset_index(drop=True), ytr,
                                            Xt_own.iloc[te].reset_index(drop=True))
            pl = fit_platt(g[tr], ytr.values); gtr, gte = apply_platt(g[tr], pl), apply_platt(g[te], pl)
            for k in KINDS:
                lt, lo = own_variant(Xtr_pp, Xte_pp, ytr, k)
                b_te = 0.5 * gte + 0.5 * lt
                acc[k].append(f1_at_prior(*tpr_fpr(yte.values, (b_te >= THR).astype(int)), prior))
        base = np.array(acc["base"]); nf = len(base)
        print(f"  seed{seed} ({nf} folds): base F1*={base.mean():.4f}", flush=True)
        for k in KINDS[1:]:
            a = np.array(acc[k]); d = a - base
            se = d.std(ddof=1)/np.sqrt(nf); sig = d.mean()/se if se > 0 else 0
            print(f"    {k:9s} F1*={a.mean():.4f}  Δ={d.mean():+.4f} ({sig:+.2f}σ)", flush=True)
            rows.append({"seed": seed, "kind": k, "f1": a.mean(), "delta": d.mean(), "sigma": sig})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "cancer_own.csv"), index=False)
    print(f"\nsaved -> cancer_own.csv", flush=True)


if __name__ == "__main__":
    main()
