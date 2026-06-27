"""STEP 2 — Feature work. Screen candidate features for lifting the PR curve (AUC-PR = primary metric).
Cross-fit target encoding (fit INSIDE folds, no leakage) of population/substitution/genotype + AF×cons
interactions. LGBM (deployed params) 5-fold OOF screen, 3 seeds, paired vs the all-features baseline.
Metrics: AUC, AUC-PR(AP), PR-curve max-F1 (the ceiling we want to lift), F1@honest-recal-thr.
Run: .venv/bin/python -u analysis/exp_step2_features.py
"""
import sys, os, warnings
sys.path.insert(0, "analysis"); sys.path.insert(0, "pipelines")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

from config import SEED
from schema import normalize_sentinels, partition_columns, drop_all_nan_rows
from utils import detect_leaky_columns, KNOWN_LEAKY, TEST_PRIORS, PANELS
from predict import _engineer, _fit_own_preprocess, _apply_own_preprocess, _lgbm_redun_keep, LGBM_PARAM_OVERRIDE
from recipe import PANEL_RECIPE
from models import get_model_params
from general_blend import _cats
from harness import f1_at_prior, tpr_fpr, pick_threshold_at_prior

PRIOR = TEST_PRIORS["general"]


def te(tr_key, tr_y, te_key, prior_rate, smooth=20.0):
    """Smoothed cross-fit target encoding: mean y per category (fit on train), applied to test."""
    s = pd.DataFrame({"k": tr_key, "y": tr_y}).groupby("k")["y"].agg(["mean", "count"])
    enc = (s["mean"] * s["count"] + prior_rate * smooth) / (s["count"] + smooth)
    return te_key.map(enc).fillna(prior_rate).values


def f1max(y, p):
    return max(f1_at_prior(*tpr_fpr(y, (p >= q).astype(int)), PRIOR) for q in np.quantile(p, np.arange(.02, .98, .005)))


def add_features(Xpp_tr, Xpp_te, raw_tr, raw_te, ytr, which):
    """Return (Xtr, Xte) with the requested candidate features appended (cross-fit on the train fold)."""
    Xtr, Xte = Xpp_tr.copy(), Xpp_te.copy()
    pr = ytr.mean()
    if "te_cat1" in which and "CAT_1" in raw_tr:
        Xtr["te_cat1"] = te(raw_tr["CAT_1"].astype(str), ytr.values, raw_tr["CAT_1"].astype(str), pr)
        Xte["te_cat1"] = te(raw_tr["CAT_1"].astype(str), ytr.values, raw_te["CAT_1"].astype(str), pr)
    if "te_aasub" in which and "AA_1" in raw_tr:
        ka = (raw_tr["AA_1"].astype(str) + "_" + raw_tr["AA_2"].astype(str))
        ka_te = (raw_te["AA_1"].astype(str) + "_" + raw_te["AA_2"].astype(str))
        Xtr["te_aasub"] = te(ka, ytr.values, ka, pr); Xte["te_aasub"] = te(ka, ytr.values, ka_te, pr)
    if "te_geno" in which and "CAT_3" in raw_tr:
        Xtr["te_geno"] = te(raw_tr["CAT_3"].astype(str), ytr.values, raw_tr["CAT_3"].astype(str), pr)
        Xte["te_geno"] = te(raw_tr["CAT_3"].astype(str), ytr.values, raw_te["CAT_3"].astype(str), pr)
    if "af_x_cons" in which:
        al = [c for c in raw_tr.columns if c.startswith("AL_")]
        afc = [c for c in al if ((raw_tr[c] > 0) & (raw_tr[c] < 1)).any() and raw_tr[c].mean(skipna=True) < 0.1]
        maf_tr = raw_tr[afc].max(axis=1).fillna(0.0); maf_te = raw_te[afc].max(axis=1).fillna(0.0)
        ek_tr = raw_tr.get("EK_7", pd.Series(0, index=raw_tr.index)).fillna(0)
        ek_te = raw_te.get("EK_7", pd.Series(0, index=raw_te.index)).fillna(0)
        Xtr["af_x_cons"] = (maf_tr.values * ek_tr.values); Xte["af_x_cons"] = (maf_te.values * ek_te.values)
        Xtr["cons_rare"] = (ek_tr.values * (maf_tr.values < 1e-4)); Xte["cons_rare"] = (ek_te.values * (maf_te.values < 1e-4))
    return Xtr, Xte


def screen(X, y, raw, cfg, which, seeds=(SEED, SEED + 1000, SEED + 2000)):
    aucs, aps, f1s, f1rs = [], [], [], []
    for seed in seeds:
        oof = np.zeros(len(y)); skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        for tr, te_ in skf.split(X, y):
            Xtr0, ytr = X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True)
            Xte0 = X.iloc[te_].reset_index(drop=True)
            Xpp, prep = _fit_own_preprocess(Xtr0, ytr, cfg)
            Xte = _apply_own_preprocess(Xte0, prep).reindex(columns=Xpp.columns)
            Xtr, Xte = add_features(Xpp, Xte, raw.iloc[tr].reset_index(drop=True), raw.iloc[te_].reset_index(drop=True), ytr, which)
            cf = _cats(Xtr)
            params = {**get_model_params(len(Xtr), 1.0), **LGBM_PARAM_OVERRIDE["general"], "verbose": -1}
            m = lgb.LGBMClassifier(**params); m.fit(Xtr, ytr, categorical_feature=cf or "auto")
            oof[te_] = m.predict_proba(Xte)[:, 1]
        yv = y.values
        aucs.append(roc_auc_score(yv, oof)); aps.append(average_precision_score(yv, oof))
        f1s.append(f1max(yv, oof))
        # honest recal threshold via inner split-free proxy: use the OOF's own prior threshold (same for all -> fair paired)
        f1rs.append(f1_at_prior(*tpr_fpr(yv, (oof >= pick_threshold_at_prior(yv, oof, PRIOR)).astype(int)), PRIOR))
    return np.array(aucs), np.array(aps), np.array(f1s), np.array(f1rs)


def main():
    df = pd.read_csv(PANELS["general"]); df = normalize_sentinels(df); df = drop_all_nan_rows(df)
    cfg = PANEL_RECIPE["general"]
    de = _engineer(df.copy(), cfg); _, num, cat, tgt = partition_columns(de)
    y = de[tgt].astype(int).reset_index(drop=True); X = de[num + cat].reset_index(drop=True)
    leaky = detect_leaky_columns(X[num], y, 0.99, verbose=False)
    X = X.drop(columns=sorted(set(leaky) | (KNOWN_LEAKY & set(X.columns))))
    raw = df.reset_index(drop=True)

    print("=== STEP 2 — feature screen (LGBM 5-fold OOF, 3 seeds, paired vs all-features baseline) ===")
    bA, bP, bF, bR = screen(X, y, raw, cfg, set())
    print(f"  baseline (all feats)  AUC={bA.mean():.4f}  AP={bP.mean():.4f}  maxF1={bF.mean():.4f}  recalF1={bR.mean():.4f}")
    cands = {"+TE(pop)": {"te_cat1"}, "+TE(AAsub)": {"te_aasub"}, "+TE(geno)": {"te_geno"},
             "+AFxCons": {"af_x_cons"}, "+ALL": {"te_cat1", "te_aasub", "te_geno", "af_x_cons"}}
    for name, which in cands.items():
        A, P, F, R = screen(X, y, raw, cfg, which)
        dP = P - bP; dF = F - bF; dR = R - bR
        sP = dP.std(ddof=1)/np.sqrt(len(dP)); sR = dR.std(ddof=1)/np.sqrt(len(dR))
        psr = [round(R[i]-bR[i], 4) for i in range(len(R))]
        win = "  <<<" if dR.mean()/sR >= 2 and all(x > 0 for x in psr) else ""
        print(f"  {name:12s} AUC={A.mean():.4f} AP={P.mean():.4f}(Δ{dP.mean():+.4f} {dP.mean()/sP if sP else 0:+.1f}σ)  "
              f"maxF1={F.mean():.4f}(Δ{dF.mean():+.4f})  recalF1={R.mean():.4f}(Δ{dR.mean():+.4f} {dR.mean()/sR if sR else 0:+.1f}σ) seeds={psr}{win}")
    print("DONE")


if __name__ == "__main__":
    main()
