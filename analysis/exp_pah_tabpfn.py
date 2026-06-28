"""TabPFN for PAH — does an in-context foundation model beat GBDTs where PAH is data-starved?

PAH = 367 rows / 60 benign = TabPFN's small-data sweet spot (the regime GBDTs struggle in). The grid
(exp_pah_blend_grid) showed every tree own-model / heterogeneous transfer blend LOSES to the single-LGBM
transfer at w=1.0. TabPFN is qualitatively different (in-context Bayesian inference), so test:

  ROLE 1 (own model, the promising one): TabPFN trained on PAH's OWN features, blended with the deployed
          single-LGBM transfer at w in {0.9..0.0}. Two feature sets: full native-NaN, and mRMR-40.
  ROLE 2 (transfer member): TabPFN trained on GENERAL (minus pah Variant_IDs, capped 3000 for GPU mem),
          scoring pah — pure (w=1.0) and averaged with the LGBM transfer.
  control: LGBM own-model blend (known to hurt).

Honest: analytic F1* at the pah test prior (0.286); per-fold OOF threshold; paired RSKF vs deployed
(single-LGBM transfer, w=1.0). SCREEN (50 folds); any winner -> fresh-seed confirm before deploy.
GPU: TabPFN-v2.6 on the cached ckpt (RTX 4060). Run: .venv/bin/python -u analysis/exp_pah_tabpfn.py
"""
import os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
warnings.filterwarnings("ignore")
# enable offline TabPFN (cached ckpt + token file)
_tok = os.path.expanduser("~/.cache/tabpfn/auth_token")
if os.path.exists(_tok) and not os.environ.get("TABPFN_TOKEN"):
    os.environ["TABPFN_TOKEN"] = open(_tok).read().strip()
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score

import tabpfn_member as TM
from config import SEED
from utils import TEST_PRIORS, PANELS, KNOWN_LEAKY, detect_leaky_columns
from schema import normalize_sentinels, drop_all_nan_rows, to_categorical, partition_columns
from feature_engineering import engineer_aa_features, engineer_acmg_features
from calibration import fit_platt, apply_platt
from harness import f1_at_prior, tpr_fpr, pick_threshold_at_prior
from preprocess_v2 import fast_mrmr
from transfer import fit_transfer_scorer, transfer_score
from exp_transfer import load_target_engineered, target_deployed_fn

PANEL = "pah"; PRIOR = TEST_PRIORS[PANEL]
TM.TABPFN_MODEL = os.path.expanduser("~/.cache/tabpfn/tabpfn-v2.6-classifier-v2.6_default.ckpt")
TM.TABPFN_NEST = 8                                   # speed (default 16); plenty for a screen
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def _engineer(df):
    de = engineer_acmg_features(engineer_aa_features(to_categorical(df.copy())))
    _, num, cat, tgt = partition_columns(de)
    X = de[num + cat].reset_index(drop=True); y = de[tgt].reset_index(drop=True).astype(int)
    leaky = detect_leaky_columns(X[num], y, 0.99, verbose=False)
    X = X.drop(columns=sorted(set(leaky) | (KNOWN_LEAKY & set(X.columns))))
    return X, y


def _tabpfn_oof_test(Xtr, ytr, Xte, cat_feats, seed, nf=4):
    """Calibrated (train-OOF, test) TabPFN probs. In-context fit per inner fold + final fit on all train."""
    num = [c for c in Xtr.columns if c not in cat_feats]
    Atr, enc, cidx = TM._encode(Xtr, cat_feats, num)
    Ate, _, _ = TM._encode(Xte, cat_feats, num, enc=enc)
    yv = ytr.values.astype(int); oof = np.zeros(len(yv))
    skf = StratifiedKFold(nf, shuffle=True, random_state=seed)
    for a, b in skf.split(Atr, yv):
        m = TM._clf(cidx, seed=seed); m.fit(Atr[a], yv[a]); oof[b] = m.predict_proba(Atr[b])[:, 1]
    pl = fit_platt(oof, yv)
    mf = TM._clf(cidx, seed=seed); mf.fit(Atr, yv); te = mf.predict_proba(Ate)[:, 1]
    return apply_platt(oof, pl), apply_platt(te, pl)


def build_tabpfn_transfer(pah_df, Xpah_cols, cat_feats):
    """ROLE 2: TabPFN trained on GENERAL minus pah Variant_IDs (cap 3000), scoring pah rows. Fixed."""
    g = pd.read_csv(PANELS["general"]); g = drop_all_nan_rows(normalize_sentinels(g), verbose=False)
    tids = set(pah_df["Variant_ID"]); g = g[~g["Variant_ID"].isin(tids)].reset_index(drop=True)
    Xg, yg = _engineer(g)
    cols = [c for c in Xpah_cols if c in Xg.columns]            # align to pah feature space
    Xg = Xg[cols]
    if len(Xg) > 3000:                                          # GPU-mem cap for the in-context set
        idx = StratifiedKFold(int(np.ceil(len(Xg)/3000)), shuffle=True, random_state=SEED)
        keep, _ = next(iter([(tr, te) for tr, te in idx.split(Xg, yg)]))
        Xg, yg = Xg.iloc[keep].reset_index(drop=True), yg.iloc[keep].reset_index(drop=True)
    num = [c for c in cols if c not in cat_feats]
    A, enc, cidx = TM._encode(Xg, cat_feats, num)
    m = TM._clf(cidx); m.fit(A, yg.values.astype(int))
    return {"m": m, "enc": enc, "cidx": cidx, "cols": cols, "num": num, "cat": cat_feats, "n": len(Xg)}


def tabpfn_transfer_score(Xpah, art):
    X = Xpah.reindex(columns=art["cols"])
    A, _, _ = TM._encode(X, art["cat"], art["num"], enc=art["enc"])
    return art["m"].predict_proba(A)[:, 1]


def _f1(y, p, t): return f1_at_prior(*tpr_fpr(y, (p >= t).astype(int)), PRIOR)
WS = [0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.0]


def main():
    t0 = time.time()
    df = pd.read_csv(PANELS[PANEL]); df = drop_all_nan_rows(normalize_sentinels(df), verbose=False).reset_index(drop=True)
    Xfull, y = _engineer(df)
    cat_feats = [c for c in Xfull.columns if isinstance(Xfull[c].dtype, pd.CategoricalDtype)]
    # deployed single-LGBM transfer (fixed OOS), + LGBM own model fn
    g_all = np.asarray(transfer_score(df, fit_transfer_scorer(PANEL)))
    Xt_own, _ = load_target_engineered(PANEL); lgb_fn = target_deployed_fn(PANEL)
    print(f"PAH n={len(df)} ({int(y.sum())}P/{int((y==0).sum())}B)  transfer LGBM OOS AUC={roc_auc_score(y,g_all):.4f}")
    # ROLE 2 transfer member (fixed)
    tpt_art = build_tabpfn_transfer(df, list(Xfull.columns), cat_feats)
    t_all = tabpfn_transfer_score(Xfull, tpt_art)
    print(f"TabPFN transfer (trained on {tpt_art['n']} general rows) OOS AUC on pah = {roc_auc_score(y,t_all):.4f}")
    # mRMR-40 column set for the reduced TabPFN own (fit per fold inside the loop)

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    keys = ["dep"] + [f"tpfull_w{int(w*100)}" for w in WS] + [f"tpmrmr_w{int(w*100)}" for w in WS] \
           + [f"lgbown_w{int(w*100)}" for w in WS] + ["tptransfer", "tptransfer+lgb", "tpown_auc", "tpownM_auc"]
    F = {k: [] for k in keys}
    nf = 0
    for tr, te in rskf.split(Xfull, y):
        tr, te = np.asarray(tr), np.asarray(te)
        ytr, yte = y.iloc[tr].reset_index(drop=True), y.iloc[te].reset_index(drop=True)
        seed = SEED + nf
        # transfer (Platt per fold)
        pl = fit_platt(g_all[tr], ytr.values); g_tr, g_te = apply_platt(g_all[tr], pl), apply_platt(g_all[te], pl)
        plt2 = fit_platt(t_all[tr], ytr.values); t_tr, t_te = apply_platt(t_all[tr], plt2), apply_platt(t_all[te], plt2)
        # TabPFN own (full features)
        Xtr_f, Xte_f = Xfull.iloc[tr].reset_index(drop=True), Xfull.iloc[te].reset_index(drop=True)
        tpf_tr, tpf_te = _tabpfn_oof_test(Xtr_f, ytr, Xte_f, cat_feats, seed)
        # TabPFN own (mRMR-40 on this fold)
        Xm_tr, _, Xm_te, _ = fast_mrmr(Xtr_f, ytr, Xte_f, Xte_f, max_features=40)
        catm = [c for c in Xm_tr.columns if c in cat_feats]
        tpm_tr, tpm_te = _tabpfn_oof_test(Xm_tr, ytr, Xm_te, catm, seed)
        # LGBM own (deployed recipe)
        e_te, e_tr = lgb_fn(Xt_own.iloc[tr].reset_index(drop=True), ytr, Xt_own.iloc[te].reset_index(drop=True))
        # metrics
        td = pick_threshold_at_prior(ytr.values, g_tr, PRIOR); F["dep"].append(_f1(yte.values, g_te, td))
        F["tpown_auc"].append(roc_auc_score(yte, tpf_te)); F["tpownM_auc"].append(roc_auc_score(yte, tpm_te))
        for w in WS:
            for tag, own_tr, own_te in [("tpfull", tpf_tr, tpf_te), ("tpmrmr", tpm_tr, tpm_te), ("lgbown", e_tr, e_te)]:
                b_tr = w * g_tr + (1 - w) * own_tr; b_te = w * g_te + (1 - w) * own_te
                t = pick_threshold_at_prior(ytr.values, b_tr, PRIOR)
                F[f"{tag}_w{int(w*100)}"].append(_f1(yte.values, b_te, t))
        tt = pick_threshold_at_prior(ytr.values, t_tr, PRIOR); F["tptransfer"].append(_f1(yte.values, t_te, tt))
        gtb_tr, gtb_te = 0.5*g_tr+0.5*t_tr, 0.5*g_te+0.5*t_te
        tt = pick_threshold_at_prior(ytr.values, gtb_tr, PRIOR); F["tptransfer+lgb"].append(_f1(yte.values, gtb_te, tt))
        nf += 1
    dep = np.array(F["dep"])
    print(f"\nfolds={nf}  dep (single-LGBM transfer w=1.0) F1*={dep.mean():.4f}  ({time.time()-t0:.0f}s)")
    print(f"  [aux] TabPFN-own OOS AUC: full={np.mean(F['tpown_auc']):.4f}  mRMR40={np.mean(F['tpownM_auc']):.4f}  (LGBM transfer={roc_auc_score(y,g_all):.4f})")
    rows = []
    for k in keys:
        if k in ("dep", "tpown_auc", "tpownM_auc"): continue
        v = np.array(F[k]); d = v - dep; se = d.std(ddof=1)/np.sqrt(nf); sg = d.mean()/se if se>0 else 0
        rows.append((k, v.mean(), d.mean(), sg))
    rows.sort(key=lambda r: -r[1])
    print(f"\n  {'config':16s} {'F1*':>7s} {'dF1':>8s} {'sigma':>7s}")
    for k, m, dm, sg in rows:
        flag = "  DEPLOY?" if (dm > 0 and sg >= 2) else ""
        print(f"  {k:16s} {m:7.4f} {dm:+8.4f} {sg:+7.2f}{flag}")
    pd.DataFrame(rows, columns=["config","f1","delta","sigma"]).to_csv(os.path.join(OUT,"pah_tabpfn.csv"), index=False)
    best = max(rows, key=lambda r: r[1])
    print(f"\n[saved] pah_tabpfn.csv   best non-dep cell: {best[0]} F1*={best[1]:.4f} ({best[3]:+.2f}σ)")
    print("[done]" if best[2] <= 0 or best[3] < 2 else "[CANDIDATE — confirm on fresh seeds]")


if __name__ == "__main__":
    main()
