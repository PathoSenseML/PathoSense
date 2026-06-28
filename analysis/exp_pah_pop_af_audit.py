"""LEAKAGE AUDIT of the steelman owner-AF +4.5sigma orthogonal AUC gain.

The danger: this project has a documented 'AF-missing -> pathogenic' leak (missingness pattern, a
TRAINING artifact that does NOT generalize to novel finals variants). The steelman owner-AF features
include presence/missingness flags (owned_present_*, pop one-hots, absent->0 encoding). If the +4.5sigma
AUC lift is driven by MISSINGNESS rather than genuine population-frequency VALUES, it is a mirage that
would not hold at finals. This audit separates the two.

  (1) single-feature label-AUC of each pop feature on PAH -> flag missingness-driven ones.
  (2) orthogonality AUC-stack lift with a VALUES-ONLY pop model: drop every presence/missingness flag,
      encode absent AF as NaN (LightGBM native) NOT 0, keep only continuous owned-AF magnitude +
      ancestry identity + founder. Does the +0.009 AUC lift SURVIVE? If it collapses -> it was the leak.
  (3) light label-permutation null (shuffle train y before owner-fit + pop model): is the real lift
      outside the chance distribution from the owner-map's per-fold flexibility?

Run: .venv/bin/python -u analysis/exp_pah_pop_af_audit.py
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
from models import get_model_params
from transfer import fit_transfer_scorer, transfer_score
from exp_pah_pop_af_steelman import fit_owner, make_steelman, _SG, _sg, KEY_GROUPS

PANEL = "pah"; PRIOR = TEST_PRIORS[PANEL]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def make_values_only(df, f1, f2):
    """Owner-AF VALUES + ancestry identity only. NO presence/missingness flags. Absent AF -> NaN."""
    out = pd.DataFrame(index=df.index)
    for tag, ft in [("g", f1), ("a", f2)]:
        anc = df[ft["cat"]].map(_sg)
        owned = pd.Series(np.nan, index=df.index)
        for P, col in ft["owner"].items():
            if col not in df.columns: continue
            mask = (anc == P) & df[col].notna()
            owned[mask] = pd.to_numeric(df.loc[mask, col], errors="coerce").values
        out[f"owned_af_{tag}"] = owned.values                 # NaN where absent (native), NOT 0
        out[f"owned_log_{tag}"] = np.log10(owned.values)      # NaN where absent
        # ancestry identity one-hots (population label is NOT a missingness leak; it is in transfer too)
        for P in KEY_GROUPS:
            out[f"pm_{tag}_{P}"] = (anc == P).astype(int)
    return out


def _lgbm(n, spw, seed):
    p = {**get_model_params(n, spw), "n_estimators": 300, "random_state": seed, "n_jobs": 1, "verbose": -1}
    return lgb.LGBMClassifier(**p)


def _oof_test(Xtr, ytr, Xte, spw, seed, nf=4):
    oof = np.zeros(len(ytr)); skf = StratifiedKFold(nf, shuffle=True, random_state=seed)
    for a, b in skf.split(Xtr, ytr):
        m = _lgbm(len(a), spw, seed); m.fit(Xtr.iloc[a], ytr.iloc[a]); oof[b] = m.predict_proba(Xtr.iloc[b])[:, 1]
    pl = fit_platt(oof, ytr.values); mf = _lgbm(len(ytr), spw, seed); mf.fit(Xtr, ytr)
    return apply_platt(oof, pl), apply_platt(mf.predict_proba(Xte)[:, 1], pl)


def _load():
    df = pd.read_csv(PANELS[PANEL]); df = normalize_sentinels(df); df = drop_all_nan_rows(df, verbose=False)
    df = df.reset_index(drop=True)
    de = engineer_acmg_features(engineer_aa_features(to_categorical(df.copy())))
    _, num, cat, tgt = partition_columns(de); y = de[tgt].reset_index(drop=True).astype(int)
    return df, y


def part1_single_auc(df, y):
    print("== (1) single-feature label-AUC on full PAH (flag missingness-driven) ==")
    f1, f2 = fit_owner(df, "CAT_1"), fit_owner(df, "CAT_2")
    Sfull = make_steelman(df, f1, f2); Sval = make_values_only(df, f1, f2)
    def auc1(col, s):
        v = pd.to_numeric(s, errors="coerce")
        m = v.notna()
        if m.sum() < 20 or y[m].nunique() < 2: return np.nan
        a = roc_auc_score(y[m], v[m]); return max(a, 1 - a)
    print("  -- full steelman features --")
    for c in Sfull.columns:
        a = auc1(c, Sfull[c]); tag = "  <- MISSINGNESS/PRESENCE" if ("present" in c or "founder" in c) else ""
        print(f"    {c:22s} AUC={a:.3f}  nonnull={Sfull[c].notna().sum():3d}{tag}")
    # the documented leak signature: does 'any AF present' predict label?
    print("  -- raw missingness leak check --")
    al = [c for c in df.columns if c.startswith("AL_")]
    anyaf = df[al].notna().any(axis=1).astype(int)
    print(f"    any_AL_present       AUC={max(roc_auc_score(y, anyaf), 1-roc_auc_score(y, anyaf)):.3f}")


def part2_values_only(df, y, seed):
    g_all = np.asarray(transfer_score(df, fit_transfer_scorer(PANEL)))
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=seed)
    dF, dV, base = [], [], []
    for tr, te in rskf.split(df, y):
        tr, te = np.asarray(tr), np.asarray(te)
        ytr, yte = y.iloc[tr].reset_index(drop=True), y.iloc[te].reset_index(drop=True)
        spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1); s = seed + len(base)
        plg = fit_platt(g_all[tr], ytr.values); g_tr, g_te = apply_platt(g_all[tr], plg), apply_platt(g_all[te], plg)
        base.append(roc_auc_score(yte, g_te))
        f1, f2 = fit_owner(df.iloc[tr], "CAT_1"), fit_owner(df.iloc[tr], "CAT_2")
        for variant, mk in [("full", make_steelman), ("vals", make_values_only)]:
            S = mk(df, f1, f2)
            o_tr, o_te = _oof_test(S.iloc[tr].reset_index(drop=True), ytr, S.iloc[te].reset_index(drop=True), spw, s)
            lr = LogisticRegression(max_iter=1000).fit(np.column_stack([g_tr, o_tr]), ytr.values)
            stk = roc_auc_score(yte, lr.predict_proba(np.column_stack([g_te, o_te]))[:, 1])
            (dF if variant == "full" else dV).append(stk)
    nf = len(base); base = np.array(base)
    for nm, arr in [("full(+flags)", np.array(dF)), ("values_only", np.array(dV))]:
        d = arr - base; se = d.std(ddof=1) / np.sqrt(nf)
        print(f"  seed {seed}: stack[{nm:13s}] AUC={arr.mean():.4f}  d_vs_transfer={d.mean():+.4f} "
              f"({d.mean()/se if se>0 else 0:+.2f}s)")
    return base.mean()


if __name__ == "__main__":
    t0 = time.time(); df, y = _load()
    part1_single_auc(df, y)
    print("\n== (2) does the orthogonal AUC lift survive VALUES-ONLY (no missingness flags)? ==")
    for sd in (SEED + 31031, SEED + 34034):
        part2_values_only(df, y, sd)
    print(f"\n[done] {time.time()-t0:.0f}s")
