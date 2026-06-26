"""AUDIT LEAD #2: the per-member LGBM redundancy prune (drop >0.99-correlated cols from the LGBM only)
gave the general own-model +0.0018. The TRANSFER scorer's LGBM never got it — its cftr/pah/cancer LGBM
carry the redundant AL_ clusters. Replicate the transfer LGBM (general MINUS target Variant_IDs, native-NaN,
deployed params) WITH vs WITHOUT the redundant-cluster prune, score the held-out target panel. AUC (clean)
+ max-F1*, 5 seeds, paired. Isolates the prune's effect on the transfer LGBM.
Run: .venv/bin/python -u analysis/exp_transfer_lgbm_prune.py
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "analysis"); sys.path.insert(0, "pipelines")
import warnings; warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from transfer import _load_raw
from schema import to_categorical, partition_columns
from feature_engineering import engineer_aa_features
from preprocess_v2 import fit_keep_nan, apply_keep_nan
from utils import detect_leaky_columns, KNOWN_LEAKY, TEST_PRIORS
from run_pipeline import get_model_params
from harness import f1_at_prior, tpr_fpr
from config import SEED


def redun_keep(Xpp, thr=0.99):
    """Columns to KEEP after dropping >thr-correlated clusters (mirror predict._lgbm_redun_keep idea)."""
    num = Xpp.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return list(Xpp.columns)
    C = num.corr().abs().values
    cols = list(num.columns); drop = set()
    for i in range(len(cols)):
        if cols[i] in drop:
            continue
        for j in range(i + 1, len(cols)):
            if cols[j] not in drop and C[i, j] > thr:
                drop.add(cols[j])
    return [c for c in Xpp.columns if c not in drop]


def transfer_lgbm(panel, prune, seed_off):
    tids = set(_load_raw(panel)["Variant_ID"])
    dg = _load_raw("general")[lambda d: ~d["Variant_ID"].isin(tids)].reset_index(drop=True)
    dg = engineer_aa_features(to_categorical(dg)); _, n, c, t = partition_columns(dg)
    yg = dg[t].astype(int).reset_index(drop=True); Xg = dg[n + c].reset_index(drop=True)
    leaky = sorted(set(detect_leaky_columns(Xg, yg, 0.99)) | (KNOWN_LEAKY & set(Xg.columns)))
    Xg = Xg.drop(columns=leaky, errors="ignore")
    Xg_pp, ppf = fit_keep_nan(Xg)
    keep = redun_keep(Xg_pp) if prune else list(Xg_pp.columns)
    Xg_pp = Xg_pp[keep]
    cat_feats = [cc for cc in Xg_pp.columns if isinstance(Xg_pp[cc].dtype, pd.CategoricalDtype)]
    spw = (yg == 0).sum() / max((yg == 1).sum(), 1)
    params = {**get_model_params(len(Xg_pp), spw), "verbose": -1}
    i2, ies = train_test_split(np.arange(len(yg)), test_size=0.15, random_state=SEED + 7 + seed_off, stratify=yg)
    m = lgb.LGBMClassifier(**params, random_state=SEED + seed_off)
    m.fit(Xg_pp.iloc[i2], yg.iloc[i2], eval_set=[(Xg_pp.iloc[ies], yg.iloc[ies])],
          callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
          categorical_feature=cat_feats or "auto")
    # score target (held out via Variant_ID exclusion)
    raw = _load_raw(panel); rdf = engineer_aa_features(to_categorical(raw))
    Xt = rdf.reindex(columns=Xg.columns)
    Xt_pp = apply_keep_nan(Xt, ppf)[keep]
    for cc in cat_feats:
        Xt_pp[cc] = pd.Categorical(Xt_pp[cc], categories=Xg_pp[cc].cat.categories)
    probs = m.predict_proba(Xt_pp)[:, 1]
    y = raw[t].astype(int).values
    return float(roc_auc_score(y, probs)), probs, y, len(keep)


def main():
    for panel in ("pah", "cftr", "cancer"):
        prior = TEST_PRIORS[panel]; nkeep = None
        cur, prn = [], []
        for s in range(5):
            a0, p0, y, ncols = transfer_lgbm(panel, False, s)
            a1, p1, _, nk = transfer_lgbm(panel, True, s)
            f0 = max(f1_at_prior(*tpr_fpr(y, (p0 >= q)), prior) for q in np.quantile(p0, np.arange(.02, .98, .01)))
            f1v = max(f1_at_prior(*tpr_fpr(y, (p1 >= q)), prior) for q in np.quantile(p1, np.arange(.02, .98, .01)))
            cur.append((a0, f0)); prn.append((a1, f1v)); nkeep = (ncols, nk)
        ca = np.array([x[0] for x in cur]); pa = np.array([x[0] for x in prn])
        cf = np.array([x[1] for x in cur]); pf = np.array([x[1] for x in prn])
        dA = pa - ca; dF = pf - cf
        seA = dA.std(ddof=1) / np.sqrt(5); seF = dF.std(ddof=1) / np.sqrt(5)
        print(f"[{panel}] LGBM cols {nkeep[0]}->{nkeep[1]} (pruned)  | "
              f"AUC {ca.mean():.4f}->{pa.mean():.4f} Δ={dA.mean():+.4f} ({dA.mean()/seA if seA>0 else 0:+.2f}σ)  | "
              f"F1* {cf.mean():.4f}->{pf.mean():.4f} Δ={dF.mean():+.4f} ({dF.mean()/seF if seF>0 else 0:+.2f}σ)", flush=True)


if __name__ == "__main__":
    main()
