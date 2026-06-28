"""STEELMAN of the PAH population x AL hypothesis — close the 'weak feature build' loophole.

The user's exact mechanism: CAT_1/CAT_2 popmax subcategories correlate with specific AL columns
(those columns ARE that population's allele frequency). So operationalize it DIRECTLY:
  per ancestry P, find the AL column that best represents P's frequency (TRAIN-fold owner =
  highest mean AL value among rows whose popmax ancestry == P), then for every row build
  'owned_af' = the value of THAT row's-ancestry owner column = the variant's frequency in its
  OWN most-relevant population. Plus founder-signature deltas (owned_af vs global max_af) and a
  subcategory-restricted one-hot (only the high-deviation EUR/AFR groups).

If even THIS richest construction adds no orthogonal signal beyond the transfer score, the
hypothesis is conclusively at the information ceiling. Run over 2 fresh seeds for robustness.

Run: .venv/bin/python -u analysis/exp_pah_pop_af_steelman.py
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
from utils import TEST_PRIORS, PANELS, KNOWN_LEAKY, detect_leaky_columns
from schema import normalize_sentinels, drop_all_nan_rows, to_categorical, partition_columns
from feature_engineering import engineer_aa_features, engineer_acmg_features
from calibration import fit_platt, apply_platt
from harness import f1_at_prior, tpr_fpr, pick_threshold_at_prior
from models import get_model_params
from transfer import fit_transfer_scorer, transfer_score

PANEL = "pah"; PRIOR = TEST_PRIORS[PANEL]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
_SG = {"NFE": "EUR", "FIN": "EUR", "EUR": "EUR", "AFR": "AFR", "EAS": "EAS", "SAS": "SAS",
       "AMR": "AMR", "MID": "MID", "ASJ": "ASJ", "AMI": "AMI", "REMAINING": "OTH", "OTH": "OTH"}
# high-deviation subcategories observed in scouting (CAT_1/CAT_2): EUR up, AFR down
KEY_GROUPS = ["EUR", "AFR"]


def _sg(s):
    if pd.isna(s): return "ABSENT"
    s = str(s)
    return "MULTI" if "&" in s else _SG.get(s.split("_")[-1], "OTH")


def fit_owner(df_tr, cat_col):
    """TRAIN-fold owner map: ancestry P -> the AF-like AL column with highest mean value among
    rows whose popmax(cat_col) ancestry == P. (= 'the column that is P's frequency'.)"""
    al = [c for c in df_tr.columns if c.startswith("AL_")]
    M = df_tr[al].apply(pd.to_numeric, errors="coerce")
    af = [c for c in al if (((M[c] > 0) & (M[c] < 1)).sum() >= 5) and (M[c].mean(skipna=True) < 0.2)]
    anc = df_tr[cat_col].map(_sg)
    owner = {}
    for P in anc.unique():
        if P in ("ABSENT", "MULTI") or ((anc == P).sum() < 8):
            continue
        sub = M.loc[anc == P, af]
        if sub.shape[1] == 0 or sub.notna().sum().sum() == 0:
            continue
        owner[P] = sub.mean(axis=0).idxmax()       # column most representing P
    return {"af": af, "owner": owner, "cat": cat_col}


def make_steelman(df, f1, f2):
    """Owner-AF lookup + founder deltas + subcategory one-hot. Pure replay (owners frozen on train)."""
    out = pd.DataFrame(index=df.index)
    M = df[[c for c in df.columns if c.startswith("AL_")]].apply(pd.to_numeric, errors="coerce")
    for tag, ft in [("g", f1), ("a", f2)]:
        anc = df[ft["cat"]].map(_sg)
        owned = pd.Series(0.0, index=df.index)
        for P, col in ft["owner"].items():
            mask = (anc == P) & df[col].notna() if col in df.columns else (anc == P) & False
            owned[mask] = pd.to_numeric(df.loc[mask, col], errors="coerce").fillna(0.0).values
        out[f"owned_af_{tag}"] = owned.values
        out[f"owned_log_{tag}"] = np.log10(owned.values + 1e-9)
        out[f"owned_present_{tag}"] = (owned.values > 0).astype(int)
        # subcategory one-hot for the key high-deviation groups
        for P in KEY_GROUPS:
            out[f"pm_{tag}_{P}"] = (anc == P).astype(int)
        # founder signature: low owned-AF in EUR popmax
        out[f"founder_{tag}"] = (((anc == "EUR").values) & (owned.values > 0) & (owned.values < 1e-3)).astype(int)
    # owned-AF vs global maxaf (popmax ancestry rarer than overall = concentration)
    af_cols = f1["af"]
    maxaf = M[af_cols].max(axis=1).fillna(0.0) if af_cols else pd.Series(0.0, index=df.index)
    out["global_maxaf"] = maxaf.values
    out["owned_vs_global_g"] = (out["owned_af_g"] - out["global_maxaf"]).values
    return out


def _lgbm(n, spw, seed):
    p = {**get_model_params(n, spw), "n_estimators": 300, "random_state": seed, "n_jobs": 1, "verbose": -1}
    return lgb.LGBMClassifier(**p)


def _oof_test(Xtr, ytr, Xte, spw, seed, nf=4):
    oof = np.zeros(len(ytr)); skf = StratifiedKFold(nf, shuffle=True, random_state=seed)
    for a, b in skf.split(Xtr, ytr):
        m = _lgbm(len(a), spw, seed); m.fit(Xtr.iloc[a], ytr.iloc[a]); oof[b] = m.predict_proba(Xtr.iloc[b])[:, 1]
    pl = fit_platt(oof, ytr.values)
    mf = _lgbm(len(ytr), spw, seed); mf.fit(Xtr, ytr)
    return apply_platt(oof, pl), apply_platt(mf.predict_proba(Xte)[:, 1], pl)


def run(seed):
    df = pd.read_csv(PANELS[PANEL]); df = normalize_sentinels(df); df = drop_all_nan_rows(df, verbose=False)
    df = df.reset_index(drop=True)
    de = engineer_acmg_features(engineer_aa_features(to_categorical(df.copy())))
    _, num, cat, tgt = partition_columns(de); y = de[tgt].reset_index(drop=True).astype(int)
    art = fit_transfer_scorer(PANEL); g_all = np.asarray(transfer_score(df, art))
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=seed)
    dg, dstack, dcorr, daucpop = [], [], [], []
    f1dep, f1b85, f1b75, f1bn = [], [], [], []
    for tr, te in rskf.split(de, y):
        tr, te = np.asarray(tr), np.asarray(te)
        ytr, yte = y.iloc[tr].reset_index(drop=True), y.iloc[te].reset_index(drop=True)
        spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1); s = seed + len(f1dep)
        plg = fit_platt(g_all[tr], ytr.values)
        g_tr, g_te = apply_platt(g_all[tr], plg), apply_platt(g_all[te], plg)
        f1 = fit_owner(df.iloc[tr], "CAT_1"); f2 = fit_owner(df.iloc[tr], "CAT_2")
        S = make_steelman(df, f1, f2)
        o_tr, o_te = _oof_test(S.iloc[tr].reset_index(drop=True), ytr, S.iloc[te].reset_index(drop=True), spw, s)
        dg.append(roc_auc_score(yte, g_te)); daucpop.append(roc_auc_score(yte, o_te))
        dcorr.append(np.corrcoef(o_te, g_te)[0, 1])
        lr = LogisticRegression(max_iter=1000).fit(np.column_stack([g_tr, o_tr]), ytr.values)
        dstack.append(roc_auc_score(yte, lr.predict_proba(np.column_stack([g_te, o_te]))[:, 1]))
        def f1w(w):
            btr = w * g_tr + (1 - w) * o_tr; bte = w * g_te + (1 - w) * o_te
            t = pick_threshold_at_prior(ytr.values, btr, PRIOR)
            return f1_at_prior(*tpr_fpr(yte.values, (bte >= t).astype(int)), PRIOR), \
                   f1_at_prior(*tpr_fpr(ytr.values, (btr >= t).astype(int)), PRIOR)
        f1dep.append(f1w(1.0)[0]); f1b85.append(f1w(0.85)[0]); f1b75.append(f1w(0.75)[0])
        bw = max([0.5, 0.6, 0.75, 0.85, 1.0], key=lambda w: f1w(w)[1]); f1bn.append(f1w(bw)[0])
    nf = len(f1dep); A = lambda x: np.array(x)
    dep = A(f1dep)
    def sig(v):
        d = A(v) - dep; se = d.std(ddof=1) / np.sqrt(nf); return d.mean(), (d.mean() / se if se > 0 else 0)
    ds = A(dstack) - A(dg); sed = ds.std(ddof=1) / np.sqrt(nf)
    print(f"\n=== seed {seed} (folds={nf}) ===")
    print(f"  ortho: transfer={A(dg).mean():.4f}  pop={A(daucpop).mean():.4f}  "
          f"corr={A(dcorr).mean():.3f}  stack={A(dstack).mean():.4f}  d={ds.mean():+.4f} "
          f"({ds.mean()/sed if sed>0 else 0:+.2f}s)")
    for nm, v in [("blend85", f1b85), ("blend75", f1b75), ("nested", f1bn)]:
        dm, sg = sig(v); print(f"  {nm:9s} F1*={A(v).mean():.4f}  dF1={dm:+.4f} ({sg:+.2f}s)  dep={dep.mean():.4f}")


if __name__ == "__main__":
    t0 = time.time()
    for sd in (SEED + 9001, SEED + 13013):
        run(sd)
    print(f"\n[done] {time.time()-t0:.0f}s")
