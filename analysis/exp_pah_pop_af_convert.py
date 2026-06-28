"""Can the steelman owner-AF AUC gain (+0.009, +4.5sigma, robust 2 seeds) convert to a DEPLOYABLE
F1* gain at the PAH test prior (0.286)? The fixed convex blend at w=0.85 barely moved F1* (+0.001).
The logistic STACK captured the AUC gain (it can use a tiny optimal weight on the pop model), so test:
  (a) fine fixed-weight grid w in {.90,.92,.94,.96,.98} — small nudges of the owner-AF model
  (b) logistic-stack-as-deployment: g + pop combined by a 2-feature logistic, thresholded at prior
  (c) a 'soft-add' form: g + beta*(pop-0.5) for small beta on the RANK, threshold at prior
All pre-registered, paired vs deployed (w=1.0), 3 fresh seeds, nested-CV gate. Deploy bar: a config
must beat deployed by >=2sigma F1* AND be positive on ALL 3 seeds (small-panel false-positive guard).

Run: .venv/bin/python -u analysis/exp_pah_pop_af_convert.py
"""
import os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

from config import SEED
from utils import TEST_PRIORS, PANELS
from schema import normalize_sentinels, drop_all_nan_rows, to_categorical, partition_columns
from feature_engineering import engineer_aa_features, engineer_acmg_features
from calibration import fit_platt, apply_platt
from harness import f1_at_prior, tpr_fpr, pick_threshold_at_prior
from models import get_model_params
from transfer import fit_transfer_scorer, transfer_score
from exp_pah_pop_af_steelman import fit_owner, make_steelman

PANEL = "pah"; PRIOR = TEST_PRIORS[PANEL]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
WFINE = [0.90, 0.92, 0.94, 0.96, 0.98]


def _lgbm(n, spw, seed):
    p = {**get_model_params(n, spw), "n_estimators": 300, "random_state": seed, "n_jobs": 1, "verbose": -1}
    return lgb.LGBMClassifier(**p)


def _oof_test(Xtr, ytr, Xte, spw, seed, nf=4):
    oof = np.zeros(len(ytr)); skf = StratifiedKFold(nf, shuffle=True, random_state=seed)
    for a, b in skf.split(Xtr, ytr):
        m = _lgbm(len(a), spw, seed); m.fit(Xtr.iloc[a], ytr.iloc[a]); oof[b] = m.predict_proba(Xtr.iloc[b])[:, 1]
    pl = fit_platt(oof, ytr.values); mf = _lgbm(len(ytr), spw, seed); mf.fit(Xtr, ytr)
    return apply_platt(oof, pl), apply_platt(mf.predict_proba(Xte)[:, 1], pl)


def _f1(y, p, t): return f1_at_prior(*tpr_fpr(y, (p >= t).astype(int)), PRIOR)


def run(seed):
    df = pd.read_csv(PANELS[PANEL]); df = normalize_sentinels(df); df = drop_all_nan_rows(df, verbose=False)
    df = df.reset_index(drop=True)
    de = engineer_acmg_features(engineer_aa_features(to_categorical(df.copy())))
    _, num, cat, tgt = partition_columns(de); y = de[tgt].reset_index(drop=True).astype(int)
    g_all = np.asarray(transfer_score(df, fit_transfer_scorer(PANEL)))
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=seed)
    keys = ["dep"] + [f"w{int(w*100)}" for w in WFINE] + ["stack", "nested_fine"]
    F1 = {k: [] for k in keys}
    for tr, te in rskf.split(de, y):
        tr, te = np.asarray(tr), np.asarray(te)
        ytr, yte = y.iloc[tr].reset_index(drop=True), y.iloc[te].reset_index(drop=True)
        spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1); s = seed + len(F1["dep"])
        plg = fit_platt(g_all[tr], ytr.values)
        g_tr, g_te = apply_platt(g_all[tr], plg), apply_platt(g_all[te], plg)
        f1o, f2o = fit_owner(df.iloc[tr], "CAT_1"), fit_owner(df.iloc[tr], "CAT_2")
        S = make_steelman(df, f1o, f2o)
        o_tr, o_te = _oof_test(S.iloc[tr].reset_index(drop=True), ytr, S.iloc[te].reset_index(drop=True), spw, s)
        t = pick_threshold_at_prior(ytr.values, g_tr, PRIOR); F1["dep"].append(_f1(yte.values, g_te, t))
        best_w, best_tr = 1.0, -1
        for w in WFINE:
            btr, bte = w * g_tr + (1 - w) * o_tr, w * g_te + (1 - w) * o_te
            tt = pick_threshold_at_prior(ytr.values, btr, PRIOR)
            F1[f"w{int(w*100)}"].append(_f1(yte.values, bte, tt))
            f1tr = _f1(ytr.values, btr, tt)
            if f1tr > best_tr: best_tr, best_w = f1tr, w
        # logistic stack deployment (the form that captured the AUC gain)
        lr = LogisticRegression(max_iter=1000).fit(np.column_stack([g_tr, o_tr]), ytr.values)
        st_tr = lr.predict_proba(np.column_stack([g_tr, o_tr]))[:, 1]
        st_te = lr.predict_proba(np.column_stack([g_te, o_te]))[:, 1]
        tt = pick_threshold_at_prior(ytr.values, st_tr, PRIOR); F1["stack"].append(_f1(yte.values, st_te, tt))
        # nested-w over the fine grid (w chosen per fold on train-OOF F1*)
        bn_tr, bn_te = best_w * g_tr + (1 - best_w) * o_tr, best_w * g_te + (1 - best_w) * o_te
        tt = pick_threshold_at_prior(ytr.values, bn_tr, PRIOR); F1["nested_fine"].append(_f1(yte.values, bn_te, tt))
    nf = len(F1["dep"]); dep = np.array(F1["dep"])
    res = {}
    for k in keys:
        if k == "dep": continue
        v = np.array(F1[k]); d = v - dep; se = d.std(ddof=1) / np.sqrt(nf)
        res[k] = (v.mean(), d.mean(), d.mean() / se if se > 0 else 0.0)
    print(f"\n=== seed {seed} (folds={nf}) dep F1*={dep.mean():.4f} ===")
    for k in keys[1:]:
        m, dm, sg = res[k]; print(f"  {k:12s} F1*={m:.4f}  dF1={dm:+.4f} ({sg:+.2f}s)")
    return {k: res[k] for k in keys[1:]}, dep.mean()


if __name__ == "__main__":
    t0 = time.time(); seeds = [SEED + 21021, SEED + 24024, SEED + 27027]
    allres = {}
    for sd in seeds:
        r, _ = run(sd)
        for k, v in r.items():
            allres.setdefault(k, []).append(v)
    print(f"\n===== cross-seed summary (3 seeds) — deploy bar: dF1>0 ALL seeds AND mean sigma>=2 =====")
    rows = []
    for k, lst in allres.items():
        dfs = [x[1] for x in lst]; sgs = [x[2] for x in lst]
        allpos = all(d > 0 for d in dfs); meandf = np.mean(dfs); meansg = np.mean(sgs)
        verdict = "DEPLOY" if (allpos and meansg >= 2) else ("watch" if allpos else "reject")
        rows.append((k, meandf, meansg, allpos, verdict))
        print(f"  {k:12s} mean dF1={meandf:+.4f}  mean sigma={meansg:+.2f}  all_pos={allpos}  -> {verdict}")
    pd.DataFrame(rows, columns=["config", "mean_dF1", "mean_sigma", "all_pos", "verdict"]).to_csv(
        os.path.join(OUT, "pah_pop_af_convert.csv"), index=False)
    print(f"\n[done] {time.time()-t0:.0f}s  saved pah_pop_af_convert.csv")
