"""Research-backed domain feature engineering for GENERAL (the weak link + transfer
source). From the variant-pathogenicity literature:
  - conservation x rarity (ACMG PP3+PM2): conserved AND rare => pathogenic; conserved
    AND common => benign. ClinPred's documented win was adding allele frequency.
  - cross-track conservation consensus / depth pattern: a variant conserved across
    primate(EK_8)/vertebrate(EK_7)/mammalian(EK_9) depths is more confidently
    deleterious; shallow-only conservation is weaker (PHACT-style evolutionary context).
Trees can learn interactions in principle, so we TEST whether explicit features move
general AUC. 5-fold AUC, baseline (all-features keep_nan) vs +domain features.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from harness import load_panel, _drop_leaky
from preprocess_v2 import fit_keep_nan, apply_keep_nan
from schema import partition_columns
from run_pipeline import get_model_params

CONS = ["EK_1", "EK_2", "EK_3", "EK_7", "EK_8", "EK_9"]   # signed conservation tracks


def domain_feats(Xraw):
    """Engineer from RAW columns (before keep_nan). Returns a DataFrame of new cols."""
    out = {}
    al = [c for c in Xraw.columns if c.startswith("AL_")]
    M = Xraw[al]
    af = [c for c in al if ((M[c] > 0) & (M[c] < 1)).any() and M[c].mean(skipna=True) < 0.1]
    popmax = M[af].max(axis=1) if af else pd.Series(np.nan, index=Xraw.index)
    n_present = M.notna().sum(axis=1)
    rarity = -np.log10(popmax.fillna(0) + 1e-6)               # high = rare
    out["DM_rarity"] = rarity
    cons_cols = [c for c in CONS if c in Xraw.columns]
    if cons_cols:
        C = Xraw[cons_cols]
        out["DM_cons_mean"] = C.mean(axis=1)
        out["DM_cons_min"] = C.min(axis=1)                    # weakest-link conservation
        out["DM_cons_std"] = C.std(axis=1)
        out["DM_cons_nabove2"] = (C > 2).sum(axis=1)          # consensus count
        ek7 = Xraw["EK_7"] if "EK_7" in Xraw else C.mean(axis=1)
        # conservation x rarity (the PP3+PM2 signal)
        out["DM_cons_x_rare"] = ek7.fillna(0) * rarity
        out["DM_consmean_x_rare"] = out["DM_cons_mean"].fillna(0) * rarity
        out["DM_consmin_x_present"] = out["DM_cons_min"].fillna(0) * n_present
        # depth pattern: deep (mammalian/vertebrate) minus shallow (primate)
        if "EK_9" in Xraw and "EK_8" in Xraw:
            out["DM_depth_deep_minus_primate"] = Xraw["EK_9"].fillna(0) - Xraw["EK_8"].fillna(0)
        if "EK_7" in Xraw and "EK_8" in Xraw:
            out["DM_vert_minus_primate"] = Xraw["EK_7"].fillna(0) - Xraw["EK_8"].fillna(0)
    return pd.DataFrame(out, index=Xraw.index)


def run(panel="general", n_splits=5):
    df = load_panel(panel); _, n, c, t = partition_columns(df)
    X = df[n + c]; y = df[t].astype(int)
    Xn, _ = _drop_leaky(X, y, X.copy())
    cat_cols = [col for col in c if col in Xn.columns]
    base_auc, dom_auc = [], []
    t0 = time.time()
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=1).split(Xn, y):
        Xtr, Xte = Xn.iloc[tr].reset_index(drop=True), Xn.iloc[te].reset_index(drop=True)
        ytr, yte = y.iloc[tr].values, y.iloc[te].values
        Xtr_pp, ppf = fit_keep_nan(Xtr); Xte_pp = apply_keep_nan(Xte, ppf)
        for col in cat_cols:
            if col in Xtr_pp.columns:
                Xtr_pp[col] = Xtr_pp[col].cat.codes; Xte_pp[col] = pd.Categorical(Xte_pp[col]).codes
        spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
        hp = {k: v for k, v in get_model_params(len(tr), spw).items()
              if k not in ("verbose",)}
        m = LGBMClassifier(**hp, random_state=42, verbose=-1).fit(Xtr_pp, ytr)
        base_auc.append(roc_auc_score(yte, m.predict_proba(Xte_pp)[:, 1]))
        # + domain features
        Dtr, Dte = domain_feats(Xtr), domain_feats(Xte)
        Xtr2 = pd.concat([Xtr_pp.reset_index(drop=True), Dtr.reset_index(drop=True)], axis=1)
        Xte2 = pd.concat([Xte_pp.reset_index(drop=True), Dte.reset_index(drop=True)], axis=1)
        m2 = LGBMClassifier(**hp, random_state=42, verbose=-1).fit(Xtr2, ytr)
        dom_auc.append(roc_auc_score(yte, m2.predict_proba(Xte2)[:, 1]))
    b, d = np.mean(base_auc), np.mean(dom_auc)
    print(f"{panel} {n_splits}-fold AUC: baseline={b:.4f}  +domain={d:.4f}  Δ={d-b:+.4f}  "
          f"[{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    for pn in (sys.argv[1:] or ["general"]):
        run(pn)
