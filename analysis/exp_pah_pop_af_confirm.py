"""High-fidelity, LEAK-SAFE confirmation of the PAH owner-AF blend win (w96/w98).

Pre-registered (NO grid search here — only the two weights that passed convert): w in {0.96, 0.98}.
Two pop-model feature sets:
  full  = steelman owner-AF + presence/missingness flags (the convert winner; possibly leak-tainted)
  vals  = VALUES-ONLY owner-AF (no presence flags, absent->NaN) = leak-safe (audit's clean variant)
The deployable claim requires the VALS variant to hold (>=2sigma, all seeds positive). If only `full`
holds, the win is the AF-missingness leak and must be REJECTED for finals (novel variants).

FRESH seeds (disjoint from steelman/convert), n_repeats=30 -> 150 folds/seed. Paired vs deployed w=1.0.
Run: .venv/bin/python -u analysis/exp_pah_pop_af_confirm.py
"""
import os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
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
from exp_pah_pop_af_audit import make_values_only

PANEL = "pah"; PRIOR = TEST_PRIORS[PANEL]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
WEIGHTS = [0.96, 0.98]


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
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=30, random_state=seed)
    keys = ["dep"] + [f"{fs}_w{int(w*100)}" for fs in ("full", "vals") for w in WEIGHTS]
    F1 = {k: [] for k in keys}
    for tr, te in rskf.split(de, y):
        tr, te = np.asarray(tr), np.asarray(te)
        ytr, yte = y.iloc[tr].reset_index(drop=True), y.iloc[te].reset_index(drop=True)
        spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1); s = seed + len(F1["dep"])
        plg = fit_platt(g_all[tr], ytr.values); g_tr, g_te = apply_platt(g_all[tr], plg), apply_platt(g_all[te], plg)
        t = pick_threshold_at_prior(ytr.values, g_tr, PRIOR); F1["dep"].append(_f1(yte.values, g_te, t))
        f1o, f2o = fit_owner(df.iloc[tr], "CAT_1"), fit_owner(df.iloc[tr], "CAT_2")
        for fs, mk in [("full", make_steelman), ("vals", make_values_only)]:
            S = mk(df, f1o, f2o)
            o_tr, o_te = _oof_test(S.iloc[tr].reset_index(drop=True), ytr, S.iloc[te].reset_index(drop=True), spw, s)
            for w in WEIGHTS:
                btr, bte = w * g_tr + (1 - w) * o_tr, w * g_te + (1 - w) * o_te
                tt = pick_threshold_at_prior(ytr.values, btr, PRIOR)
                F1[f"{fs}_w{int(w*100)}"].append(_f1(yte.values, bte, tt))
    nf = len(F1["dep"]); dep = np.array(F1["dep"]); out = {}
    print(f"\n=== seed {seed} (folds={nf}) dep F1*={dep.mean():.4f} ===")
    for k in keys[1:]:
        v = np.array(F1[k]); d = v - dep; se = d.std(ddof=1) / np.sqrt(nf); sg = d.mean()/se if se>0 else 0
        out[k] = (d.mean(), sg); print(f"  {k:10s} F1*={v.mean():.4f}  dF1={d.mean():+.4f} ({sg:+.2f}s)")
    return out


if __name__ == "__main__":
    t0 = time.time(); seeds = [SEED + 41041, SEED + 44044, SEED + 47047]
    agg = {}
    for sd in seeds:
        for k, v in run(sd).items():
            agg.setdefault(k, []).append(v)
    print("\n===== cross-seed (3 fresh seeds) — DEPLOYABLE iff vals_* is all-pos AND mean sigma>=2 =====")
    rows = []
    for k, lst in agg.items():
        dfs = [x[0] for x in lst]; sgs = [x[1] for x in lst]
        allpos = all(d > 0 for d in dfs); md = np.mean(dfs); ms = np.mean(sgs)
        leaksafe = k.startswith("vals")
        verdict = ("DEPLOY" if (allpos and ms >= 2 and leaksafe) else
                   ("leak-only" if (allpos and ms >= 2) else ("watch" if allpos else "reject")))
        rows.append((k, md, ms, allpos, verdict)); print(f"  {k:10s} mean dF1={md:+.4f} mean s={ms:+.2f} all_pos={allpos} -> {verdict}")
    pd.DataFrame(rows, columns=["config","mean_dF1","mean_sigma","all_pos","verdict"]).to_csv(
        os.path.join(OUT, "pah_pop_af_confirm.csv"), index=False)
    print(f"\n[done] {time.time()-t0:.0f}s  saved pah_pop_af_confirm.csv")
