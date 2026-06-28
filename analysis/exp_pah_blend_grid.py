"""PAH transfer-architecture x own-model blend GRID.

The deployed PAH scorer is a SINGLE LightGBM trained on general (pure transfer, w=1.0) — it is
EXCLUDED from the heterogeneous LGBM+RF+CatBoost blend that cftr/cancer use. User: test ALL general-
ensemble model combinations x ALL blend weights w against PAH's own model, to see if single-LGBM/w=1.0
is really optimal.

AXIS A (transfer architecture): all 15 non-empty subsets of the 4 general-trained members
  {lgb, rf, cat, xgb} (each trained on general MINUS all pah Variant_IDs = leakage-correct), equal-weight
  prob-average within the subset (the deployed blend convention).
AXIS B (blend weight w): transfer share w in {1.0..0.0}; (1-w) on PAH's own deployed-recipe model
  (median+mRMR+ACMG multi-seed LGBM+Platt, target_deployed_fn).

Baseline cell = (arch={lgb}, w=1.0) = the DEPLOYED pah scorer. All deltas paired vs it (identical folds).
Member prob vectors are FIXED across folds (scorer trained once, pah excluded) — only the own model and
the per-fold Platt/threshold are refit per fold, so the 15x11 grid is cheap.

Honest: F1* analytic at the pah test prior (0.286); threshold picked on TRAIN-OOF; paired RSKF deltas.
SCREEN ONLY — any winner must clear a FRESH-seed pre-registered confirm (mode=confirm) at >=2sigma.

Run:  .venv/bin/python -u analysis/exp_pah_blend_grid.py screen
      .venv/bin/python -u analysis/exp_pah_blend_grid.py confirm "lgb+rf@0.9,rf+cat@1.0"
"""
import os, sys, time, itertools, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score

import transfer as T
from config import SEED
from utils import TEST_PRIORS
from schema import to_categorical
from feature_engineering import engineer_aa_features
from calibration import fit_platt, apply_platt
from harness import f1_at_prior, tpr_fpr, pick_threshold_at_prior
from preprocess_v2 import apply_keep_nan, apply_al
from general_blend import _rf_repr, _cat_repr
from exp_transfer import _load, load_target_engineered, target_deployed_fn

PANEL = "pah"; PRIOR = TEST_PRIORS[PANEL]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
MEMBERS = ["lgb", "rf", "cat", "xgb"]
W_GRID = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]


def build_members():
    """Fit the general->pah transfer scorer with ALL 4 members (force pah into the blend+xgb sets),
    excluding pah Variant_IDs. Return member raw-prob vectors on ALL pah rows (fixed across folds)."""
    T.TRANSFER_BLEND = set(T.TRANSFER_BLEND) | {PANEL}
    T.TRANSFER_ADD_XGB = set(T.TRANSFER_ADD_XGB) | {PANEL}
    art = T.fit_transfer_scorer(PANEL)
    df = _load(PANEL)
    # project pah rows into the scorer space (mirror transfer_score), then each member's prob
    d = engineer_aa_features(to_categorical(df.copy()))
    X = d.reindex(columns=art["raw_cols"]); Xs = apply_keep_nan(X, art["ppf"])
    if art["alf"] is not None:
        Xs = pd.concat([Xs.reset_index(drop=True), apply_al(X, art["alf"]).reset_index(drop=True)], axis=1)
        Xs = Xs.loc[:, ~Xs.columns.duplicated()]
    for c in art["feats"]:
        if c not in Xs.columns:
            Xs[c] = np.nan
    Xs = Xs[art["feats"]]
    for c, cats in art["cat_cats"].items():
        Xs[c] = pd.Categorical(Xs[c].astype(object), categories=cats)
    bl = art["blend"]
    X_ord, _ = _rf_repr(Xs, bl["cat_feats"], enc=bl["encoder"], fit=False)
    X_cat = _cat_repr(Xs, bl["cat_feats"])
    mdls = art.get("models") or [art["model"]]
    P = {
        "lgb": np.mean([m.predict_proba(Xs)[:, 1] for m in mdls], axis=0),
        "rf":  bl["rf_model"].predict_proba(X_ord)[:, 1],
        "cat": bl["cat_model"].predict_proba(X_cat)[:, 1],
        "xgb": bl["xgb_model"].predict_proba(X_ord)[:, 1],
    }
    _, y = load_target_engineered(PANEL)
    for m in MEMBERS:
        print(f"  member {m:4s} OOS AUC on pah = {roc_auc_score(y, P[m]):.4f}")
    return P, df, y


def arch_subsets():
    out = []
    for r in range(1, len(MEMBERS) + 1):
        out += ["+".join(c) for c in itertools.combinations(MEMBERS, r)]
    return out                                   # 15 subsets, e.g. 'lgb', 'lgb+rf', ...


def arch_prob(P, name, idx=slice(None)):
    members = name.split("+")
    return np.mean([P[m][idx] for m in members], axis=0)


def eval_grid(P, df, y, archs, weights, seeds, n_repeats):
    """Per (arch,w): list of paired per-fold F1*. Baseline cell = ('lgb', 1.0)."""
    fn = target_deployed_fn(PANEL)
    Xt, _ = load_target_engineered(PANEL)        # invariant across folds — engineer once
    acc = {(a, w): [] for a in archs for w in weights}
    n_folds_total = 0
    for sd in seeds:
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=sd)
        for tr, te in rskf.split(np.zeros(len(y)), y):
            tr, te = np.asarray(tr), np.asarray(te)
            ytr, yte = y.iloc[tr].reset_index(drop=True), y.iloc[te].reset_index(drop=True)
            # own model (deployed recipe), calibrated OOF/test
            e_te, e_tr = fn(Xt.iloc[tr].reset_index(drop=True), ytr, Xt.iloc[te].reset_index(drop=True))
            # per-architecture transfer prob (raw avg -> Platt on train fold)
            for a in archs:
                g_raw_tr = arch_prob(P, a, tr); g_raw_te = arch_prob(P, a, te)
                pl = fit_platt(g_raw_tr, ytr.values)
                g_tr, g_te = apply_platt(g_raw_tr, pl), apply_platt(g_raw_te, pl)
                for w in weights:
                    b_tr = w * g_tr + (1 - w) * e_tr
                    b_te = w * g_te + (1 - w) * e_te
                    t = pick_threshold_at_prior(ytr.values, b_tr, PRIOR)
                    acc[(a, w)].append(f1_at_prior(*tpr_fpr(yte.values, (b_te >= t).astype(int)), PRIOR))
            n_folds_total += 1
    return acc, n_folds_total


def _stats(acc, base_key, nf):
    base = np.array(acc[base_key])
    rows = []
    for k, v in acc.items():
        v = np.array(v); d = v - base; se = d.std(ddof=1) / np.sqrt(nf) if nf > 1 else 0
        rows.append({"arch": k[0], "w": k[1], "f1": v.mean(), "delta": d.mean(),
                     "sigma": (d.mean() / se if se > 0 else 0.0)})
    return pd.DataFrame(rows)


def screen():
    t0 = time.time()
    P, df, y = build_members()
    archs = arch_subsets()
    seeds = [SEED]                                # discovery seed family
    acc, nf = eval_grid(P, df, y, archs, W_GRID, seeds, n_repeats=20)
    base_key = ("lgb", 1.0)
    R = _stats(acc, base_key, nf)
    base_f1 = R[(R.arch == "lgb") & (R.w == 1.0)].f1.iloc[0]
    print(f"\nfolds={nf}  baseline (single-LGBM transfer, w=1.0) F1*={base_f1:.4f}  ({time.time()-t0:.0f}s)")

    print("\n== AXIS A: pure transfer (w=1.0), which general-ensemble architecture? ==")
    a1 = R[R.w == 1.0].sort_values("f1", ascending=False)
    for _, r in a1.iterrows():
        tag = "  <- DEPLOYED" if r.arch == "lgb" else ""
        print(f"  {r.arch:18s} F1*={r.f1:.4f}  Δ={r.delta:+.4f} ({r.sigma:+.2f}σ){tag}")

    print("\n== best blend weight per architecture (across w) ==")
    for a in archs:
        sub = R[R.arch == a]; best = sub.loc[sub.f1.idxmax()]
        print(f"  {a:18s} best w={best.w:<4} F1*={best.f1:.4f}  Δ={best.delta:+.4f} ({best.sigma:+.2f}σ)")

    print("\n== TOP-12 cells (arch,w) by F1*, paired vs deployed ==")
    top = R.sort_values("f1", ascending=False).head(12)
    for _, r in top.iterrows():
        flag = "DEPLOY?" if (r.delta > 0 and r.sigma >= 2) else ""
        print(f"  {r.arch:18s} w={r.w:<4} F1*={r.f1:.4f}  Δ={r.delta:+.4f} ({r.sigma:+.2f}σ) {flag}")

    R.sort_values("f1", ascending=False).to_csv(os.path.join(OUT, "pah_blend_grid.csv"), index=False)
    print(f"\n[saved] analysis/output/pah_blend_grid.csv  (full {len(R)}-cell landscape)")
    cand = R[(R.delta > 0) & (R.sigma >= 2)].sort_values("f1", ascending=False)
    if len(cand):
        print(f"\n>>> {len(cand)} cell(s) pass screen (Δ>0 & >=2σ). Pre-register top-2 for fresh-seed confirm:")
        for _, r in cand.head(2).iterrows():
            print(f"      {r.arch}@{r.w}")
    else:
        print("\n>>> NO cell beats the deployed single-LGBM w=1.0 at >=2σ. Current setup is optimal.")


def confirm(cells):
    """cells = 'arch@w,arch@w'. FRESH seeds, n_repeats=30, deploy bar = Δ>0 all seeds AND mean σ>=2."""
    t0 = time.time(); P, df, y = build_members()
    parsed = [(c.split("@")[0], float(c.split("@")[1])) for c in cells.split(",") if c.strip()]
    archs = sorted({"lgb"} | {a for a, _ in parsed})
    weights = sorted({1.0} | {w for _, w in parsed})
    seeds = [SEED + 41041, SEED + 44044, SEED + 47047]
    per_seed = {p: [] for p in parsed}
    for sd in seeds:
        acc, nf = eval_grid(P, df, y, archs, weights, [sd], n_repeats=30)
        base = np.array(acc[("lgb", 1.0)])
        print(f"\n=== seed {sd} (folds={nf}) baseline F1*={base.mean():.4f} ===")
        for (a, w) in parsed:
            v = np.array(acc[(a, w)]); d = v - base; se = d.std(ddof=1) / np.sqrt(nf)
            sg = d.mean() / se if se > 0 else 0; per_seed[(a, w)].append((d.mean(), sg))
            print(f"  {a:14s}@{w:<4} F1*={v.mean():.4f}  Δ={d.mean():+.4f} ({sg:+.2f}σ)")
    print("\n===== cross-seed verdict (3 fresh seeds) =====")
    for (a, w), lst in per_seed.items():
        ds = [x[0] for x in lst]; sgs = [x[1] for x in lst]
        allpos = all(x > 0 for x in ds); verdict = "DEPLOY" if (allpos and np.mean(sgs) >= 2) else ("watch" if allpos else "reject")
        print(f"  {a:14s}@{w:<4} mean Δ={np.mean(ds):+.4f}  mean σ={np.mean(sgs):+.2f}  all_pos={allpos} -> {verdict}")
    print(f"\n[done] {time.time()-t0:.0f}s")


def archconfirm():
    """FRESH-seed confirmation of AXIS A only (pure transfer w=1.0, no own model — cheap). Robustly
    answers 'is the single-LGBM transfer the best general-ensemble architecture for PAH?'"""
    t0 = time.time(); P, df, y = build_members()
    archs = arch_subsets()
    from calibration import fit_platt, apply_platt
    seeds = [SEED + 41041, SEED + 44044, SEED + 47047]
    per = {a: [] for a in archs}
    for sd in seeds:
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=30, random_state=sd)
        acc = {a: [] for a in archs}
        for tr, te in rskf.split(np.zeros(len(y)), y):
            tr, te = np.asarray(tr), np.asarray(te)
            ytr, yte = y.iloc[tr].reset_index(drop=True), y.iloc[te].reset_index(drop=True)
            for a in archs:
                pl = fit_platt(arch_prob(P, a, tr), ytr.values)
                g_tr = apply_platt(arch_prob(P, a, tr), pl); g_te = apply_platt(arch_prob(P, a, te), pl)
                t = pick_threshold_at_prior(ytr.values, g_tr, PRIOR)
                acc[a].append(f1_at_prior(*tpr_fpr(yte.values, (g_te >= t).astype(int)), PRIOR))
        base = np.array(acc["lgb"]); nf = len(base)
        for a in archs:
            v = np.array(acc[a]); d = v - base; se = d.std(ddof=1) / np.sqrt(nf)
            per[a].append((v.mean(), d.mean(), d.mean() / se if se > 0 else 0))
    print(f"\n== AXIS A fresh-seed confirm (3 seeds x 150 folds, pure transfer w=1.0) ==  ({time.time()-t0:.0f}s)")
    print(f"  {'arch':18s} {'meanF1*':>8s} {'meanΔ':>8s} {'meanσ':>7s}  vs single-LGBM")
    order = sorted(archs, key=lambda a: -np.mean([x[0] for x in per[a]]))
    for a in order:
        f1 = np.mean([x[0] for x in per[a]]); dd = np.mean([x[1] for x in per[a]]); sg = np.mean([x[2] for x in per[a]])
        tag = "  <- DEPLOYED (best)" if a == "lgb" else ""
        print(f"  {a:18s} {f1:8.4f} {dd:+8.4f} {sg:+7.2f}{tag}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "screen"
    if mode == "confirm":
        confirm(sys.argv[2])
    elif mode == "archconfirm":
        archconfirm()
    else:
        screen()
