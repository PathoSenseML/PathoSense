"""Add XGBoost as a 4th member of the TRANSFER-scorer blend (the deployed prediction for cftr/cancer).
XGBoost helped the general OWN blend (+0.005); does it help the general-trained transfer scorer?
Build a 4-way scorer (TRANSFER_ADD_XGB), extract its 4 components on the target, sweep blend weights vs
the current deployed 3-way (cftr equal, cancer rfup), in the deployed config (cftr w=1.0; cancer 0.5+own),
honest CV, 2 seeds, paired ΔF1*. Deploy a 4-way weighting if robustly non-negative.
Run: .venv/bin/python -u analysis/exp_transfer_xgb.py [panel ...]
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from config import SEED
from utils import TEST_PRIORS
from schema import to_categorical
from feature_engineering import engineer_aa_features
from calibration import fit_platt, apply_platt
from harness import pick_threshold_at_prior, tpr_fpr, f1_at_prior
import transfer
from transfer import fit_transfer_scorer, transfer_score, TRANSFER_PANELS, TRANSFER_BLEND_WEIGHTS
from preprocess_v2 import apply_keep_nan, apply_al
from general_blend import _rf_repr, _cat_repr
from exp_transfer import _load, load_target_engineered, target_deployed_fn

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def four_components(df_raw, art):
    """Project target rows → (lgb_p, rf_p, cat_p, xgb_p) from a 4-way scorer art."""
    df = engineer_aa_features(to_categorical(df_raw.copy()))
    X = df.reindex(columns=art["raw_cols"])
    Xs = apply_keep_nan(X, art["ppf"])
    if art["alf"] is not None:
        Xs = pd.concat([Xs.reset_index(drop=True), apply_al(X, art["alf"]).reset_index(drop=True)], axis=1)
        Xs = Xs.loc[:, ~Xs.columns.duplicated()]
    for col in art["feats"]:
        if col not in Xs.columns:
            Xs[col] = np.nan
    Xs = Xs[art["feats"]]
    for col, cats in art["cat_cats"].items():
        Xs[col] = pd.Categorical(Xs[col].astype(object), categories=cats)
    mdls = art.get("models") or [art["model"]]
    lgb_p = np.mean([mm.predict_proba(Xs)[:, 1] for mm in mdls], axis=0)
    bl = art["blend"]
    X_ord, _ = _rf_repr(Xs, bl["cat_feats"], enc=bl["encoder"], fit=False)
    rf_p = bl["rf_model"].predict_proba(X_ord)[:, 1]
    cat_p = bl["cat_model"].predict_proba(_cat_repr(Xs, bl["cat_feats"]))[:, 1]
    xgb_p = bl["xgb_model"].predict_proba(X_ord)[:, 1]
    return lgb_p, rf_p, cat_p, xgb_p


WEIGHT_SETS = {
    "equal4":  (0.25, 0.25, 0.25, 0.25),
    "rfup4":   (0.20, 0.40, 0.20, 0.20),
    "rfup4b":  (0.25, 0.40, 0.20, 0.15),
}


def evaluate(panel):
    prior = TEST_PRIORS[panel]; w_t = TRANSFER_PANELS[panel]
    df = _load(panel); y = df["Label"].astype(int).values
    g_cur = transfer_score(df, fit_transfer_scorer(panel))          # current deployed 3-way
    transfer.TRANSFER_ADD_XGB = {panel}
    art4 = fit_transfer_scorer(panel)
    transfer.TRANSFER_ADD_XGB = set()
    L, R, C, Xg = four_components(df, art4)
    print(f"[{panel}] component AUC: lgb={roc_auc_score(y,L):.4f} rf={roc_auc_score(y,R):.4f} "
          f"cat={roc_auc_score(y,C):.4f} xgb={roc_auc_score(y,Xg):.4f} | cur3way={roc_auc_score(y,g_cur):.4f} "
          f"equal4={roc_auc_score(y,(L+R+C+Xg)/4):.4f}", flush=True)
    need_own = w_t < 1.0
    if need_own:
        Xt_own, yt = load_target_engineered(panel); fn = target_deployed_fn(panel)
        assert (yt.values == y).all()
    cands = {"cur3way": g_cur, **{k: (w[0]*L+w[1]*R+w[2]*C+w[3]*Xg) for k, w in WEIGHT_SETS.items()}}
    rows = []
    for seed in (SEED, SEED + 1000):
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=12 if not need_own else 8, random_state=seed)
        acc = {k: [] for k in cands}
        for tr, te in rskf.split(np.zeros(len(y)), y):
            tr, te = np.asarray(tr), np.asarray(te)
            ytr, yte = y[tr], y[te]
            if need_own:
                e_te, e_tr = fn(Xt_own.iloc[tr].reset_index(drop=True), pd.Series(ytr),
                                Xt_own.iloc[te].reset_index(drop=True))
            for k, g in cands.items():
                pl = fit_platt(g[tr], ytr); gtr, gte = apply_platt(g[tr], pl), apply_platt(g[te], pl)
                if need_own:
                    b_tr = w_t*gtr + (1-w_t)*e_tr; b_te = w_t*gte + (1-w_t)*e_te
                else:
                    b_tr, b_te = gtr, gte
                t = pick_threshold_at_prior(ytr, b_tr, prior)
                acc[k].append(f1_at_prior(*tpr_fpr(yte, (b_te >= t).astype(int)), prior))
        base = np.array(acc["cur3way"]); nf = len(base)
        print(f"  seed{seed} ({nf} folds): cur3way F1*={base.mean():.4f}", flush=True)
        for k in WEIGHT_SETS:
            a = np.array(acc[k]); d = a - base
            se = d.std(ddof=1)/np.sqrt(nf); sig = d.mean()/se if se > 0 else 0
            print(f"    {k:8s} F1*={a.mean():.4f}  Δ={d.mean():+.4f} ({sig:+.2f}σ)", flush=True)
            rows.append({"panel": panel, "seed": seed, "weights": k, "f1": a.mean(), "delta": d.mean(), "sigma": sig})
    return rows


if __name__ == "__main__":
    panels = sys.argv[1:] or ["cftr", "cancer"]
    print("=== XGBoost as 4th transfer-scorer member (vs current 3-way) ===", flush=True)
    rows = []
    for pn in panels:
        t0 = time.time(); rows += evaluate(pn); print(f"  ({pn} {time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "transfer_xgb.csv"), index=False)
    print(f"\nsaved -> transfer_xgb.csv", flush=True)
