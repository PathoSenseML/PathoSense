"""Untested mechanical lever: the transfer scorer averages its 3 components with FIXED equal
weights, g = (lgb + rf + cat) / 3. Never optimized. Test whether convex weights tuned on the
target's OWN training labels beat 1/3.

The scorer is trained ONCE on general MINUS all target Variant_IDs, so its 3 component prob vectors
on the target rows are FIXED (no refit per fold) and leakage-safe (target-train labels only choose
the weights). cftr = pure transfer (w=1.0) → a scorer-weight win is DIRECTLY deployable; cancer =
0.5·transfer+0.5·own → test the scorer F1* first (deployable into the blend if it holds).

Per fold (RepeatedStratifiedKFold): pick convex weights over a simplex grid (step 0.1) maximizing
train-fold AUC (robust smooth objective; threshold handles F1), apply to the held-out fold, Platt +
F1-optimal-at-prior threshold, score analytic F1* at the test prior. Paired vs equal weights. High
fold counts (cftr is 111 rows = the small-sample false-positive zone; demand >=2σ AND sanity).
Run: .venv/bin/python -u analysis/exp_transfer_blendw.py [panel ...]
"""
import os, sys, time, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from config import SEED
from utils import TEST_PRIORS, PANELS
from schema import normalize_sentinels, drop_all_nan_rows, to_categorical, partition_columns
from calibration import fit_platt, apply_platt
from harness import pick_threshold_at_prior, tpr_fpr, f1_at_prior
from preprocess_v2 import apply_keep_nan, apply_al
from general_blend import _rf_repr, _cat_repr
from transfer import fit_transfer_scorer

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# convex simplex grid, step 0.1 (66 weight triples) — coarse on purpose (less overfit than continuous)
GRID = [(a/10, b/10, (10-a-b)/10) for a in range(11) for b in range(11-a)]


def transfer_components(df_raw, art):
    """The 3 prob vectors the transfer scorer averages: (lgb, rf, cat). Pure replay of transfer_score
    minus the final mean. Returns equal-weight g too for sanity (== transfer_score)."""
    df = to_categorical(df_raw.copy())
    from feature_engineering import engineer_aa_features
    df = engineer_aa_features(df)
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
    X_cat = _cat_repr(Xs, bl["cat_feats"])
    cat_p = bl["cat_model"].predict_proba(X_cat)[:, 1]
    return lgb_p, rf_p, cat_p


def best_weights(comps_tr, ytr):
    L, R, C = comps_tr
    best, bw = -1, (1/3, 1/3, 1/3)
    for w in GRID:
        g = w[0]*L + w[1]*R + w[2]*C
        try:
            a = roc_auc_score(ytr, g)
        except Exception:
            continue
        if a > best:
            best, bw = a, w
    return bw


def evaluate(panel, nr):
    df = drop_all_nan_rows(normalize_sentinels(pd.read_csv(PANELS[panel])), verbose=False).reset_index(drop=True)
    _, num, cat, tgt = partition_columns(to_categorical(df))
    y = df[tgt].astype(int).values
    prior = TEST_PRIORS[panel]
    art = fit_transfer_scorer(panel)
    if art.get("blend") is None:
        print(f"[{panel}] scorer has no blend (single-LGBM) — skip"); return []
    L, R, C = transfer_components(df, art)
    comps = np.vstack([L, R, C])
    print(f"[{panel}] scorer AUC: lgb={roc_auc_score(y,L):.4f} rf={roc_auc_score(y,R):.4f} "
          f"cat={roc_auc_score(y,C):.4f} equal={roc_auc_score(y,(L+R+C)/3):.4f}", flush=True)

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=nr, random_state=SEED)
    f1_eq, f1_op, wsel = [], [], []
    for tr, te in rskf.split(comps.T, y):
        ytr, yte = y[tr], y[te]
        # equal weights
        g = (L + R + C) / 3.0
        pe = fit_platt(g[tr], ytr); gtr, gte = apply_platt(g[tr], pe), apply_platt(g[te], pe)
        t = pick_threshold_at_prior(ytr, gtr, prior)
        f1_eq.append(f1_at_prior(*tpr_fpr(yte, (gte >= t).astype(int)), prior))
        # optimized weights (tuned on train fold only)
        w = best_weights((L[tr], R[tr], C[tr]), ytr); wsel.append(w)
        go = w[0]*L + w[1]*R + w[2]*C
        po = fit_platt(go[tr], ytr); otr, ote = apply_platt(go[tr], po), apply_platt(go[te], po)
        to = pick_threshold_at_prior(ytr, otr, prior)
        f1_op.append(f1_at_prior(*tpr_fpr(yte, (ote >= to).astype(int)), prior))
    f1_eq, f1_op = np.array(f1_eq), np.array(f1_op)
    d = f1_op - f1_eq
    se = d.std(ddof=1) / np.sqrt(len(d)); sig = d.mean()/se if se > 0 else 0.0
    wmean = np.mean(wsel, axis=0)
    vd = "DEPLOY" if d.mean() > 0 and sig >= 2 else "marginal" if d.mean() > 0 and sig >= 1 else "no gain"
    print(f"  [{panel}] equal F1*={f1_eq.mean():.4f}  opt F1*={f1_op.mean():.4f}  "
          f"Δ={d.mean():+.4f} ({sig:+.2f}σ)  {vd}  | mean w(lgb,rf,cat)=({wmean[0]:.2f},{wmean[1]:.2f},{wmean[2]:.2f})",
          flush=True)
    return [{"panel": panel, "equal": f1_eq.mean(), "opt": f1_op.mean(), "delta": d.mean(),
             "sigma": sig, "w_lgb": wmean[0], "w_rf": wmean[1], "w_cat": wmean[2]}]


if __name__ == "__main__":
    panels = sys.argv[1:] or ["cftr", "cancer"]
    print("=== Transfer-scorer internal 3-way blend weights: optimized vs equal 1/3 ===", flush=True)
    rows = []
    for pn in panels:
        t0 = time.time(); rows += evaluate(pn, 20 if pn == "cftr" else 15)
        print(f"  ({pn} {time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "transfer_blendw.csv"), index=False)
    print(f"\nsaved -> transfer_blendw.csv", flush=True)
