"""STEP 1 — Error analysis at the 0.81 operating point. Build a 5-fold OOF of the deployed 4-tree blend
over ALL general variants, align to raw features, and dissect FN/FP at threshold 0.81:
structure (by population CAT_1, region CAT_6, conservation EK, allele-freq), and label-noise vs
honest-miss signatures near the boundary. Saves the OOF table for STEP 2.
Run: .venv/bin/python -u analysis/exp_step1_errors.py
"""
import sys, os, warnings
sys.path.insert(0, "analysis"); sys.path.insert(0, "pipelines")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold

from config import SEED
from schema import normalize_sentinels, partition_columns, drop_all_nan_rows
from utils import detect_leaky_columns, KNOWN_LEAKY, PANELS
from predict import _engineer, _fit_own_preprocess, _apply_own_preprocess, _lgbm_redun_keep, LGBM_PARAM_OVERRIDE
from recipe import PANEL_RECIPE
from models import get_model_params, train_multi_seed_ensemble
from general_blend import fit_general_blend, general_blend_probs
from calibration import fit_platt, apply_platt
from tabpfn_member import tabpfn_member_probs  # not used (4-tree) but keep import path stable

THR = 0.81


def oof_4tree(X, y, cfg):
    oof = np.zeros(len(y))
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for tr, te in skf.split(X, y):
        Xtr0, ytr = X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True)
        Xte0 = X.iloc[te].reset_index(drop=True)
        Xpp, prep = _fit_own_preprocess(Xtr0, ytr, cfg)
        Xte = _apply_own_preprocess(Xte0, prep).reindex(columns=Xpp.columns)
        spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
        params = {**get_model_params(len(Xpp), spw), **LGBM_PARAM_OVERRIDE["general"]}
        keep = _lgbm_redun_keep(Xpp)
        _, lo, lt, _ = train_multi_seed_ensemble(Xpp[keep], ytr, Xte[keep], params, n_seeds=5, n_folds=5)
        pl = fit_platt(lo, ytr.values); lgb_oof, lgb_test = apply_platt(lo, pl), apply_platt(lt, pl)
        art, _ = fit_general_blend(Xpp, ytr, lgb_oof, spw, n_splits=5)
        art["tabpfn"] = None
        oof[te] = general_blend_probs(Xte, lgb_test, art)
        print("  fold done", flush=True)
    return oof


def main():
    df = pd.read_csv(PANELS["general"]); df = normalize_sentinels(df); df = drop_all_nan_rows(df)
    cfg = PANEL_RECIPE["general"]
    de = _engineer(df.copy(), cfg); _, num, cat, tgt = partition_columns(de)
    y = de[tgt].astype(int).reset_index(drop=True); X = de[num + cat].reset_index(drop=True)
    leaky = detect_leaky_columns(X[num], y, 0.99, verbose=False)
    X = X.drop(columns=sorted(set(leaky) | (KNOWN_LEAKY & set(X.columns))))
    raw = df.reset_index(drop=True)   # raw features for interpretation

    p = oof_4tree(X, y, cfg)
    d = pd.DataFrame({"y": y.values, "p": p});
    for c in ["AA_1", "AA_2", "CAT_1", "CAT_6", "EK_7", "EK_2", "EK_9"]:
        if c in raw.columns: d[c] = raw[c].values
    # max observed AF (benign signal) from AL_ cols
    al = [c for c in raw.columns if c.startswith("AL_")]
    afcols = [c for c in al if ((raw[c] > 0) & (raw[c] < 1)).any() and raw[c].mean(skipna=True) < 0.1]
    d["max_af"] = raw[afcols].max(axis=1).fillna(0.0)
    d.to_csv("analysis/output/step1_oof.csv", index=False)

    pred = (d.p >= THR).astype(int)
    TP = ((d.y == 1) & (pred == 1)).sum(); FN = ((d.y == 1) & (pred == 0)).sum()
    FP = ((d.y == 0) & (pred == 1)).sum(); TN = ((d.y == 0) & (pred == 0)).sum()
    from sklearn.metrics import roc_auc_score
    print(f"\n=== STEP 1 — errors @ {THR} (general OOF, {len(d)} variants, OOF AUC={roc_auc_score(d.y,d.p):.4f}) ===")
    print(f"  confusion: TP={TP} FN={FN} | FP={FP} TN={TN}  (train-prior; {d.y.mean():.3f} P)")
    print(f"  FN are PATHOGENIC scored <{THR} (model not confident); FP are BENIGN scored ≥{THR}")

    # structure: enrichment of FN among P, FP among B, by segment
    fn = d[(d.y == 1) & (pred == 0)]; pp = d[d.y == 1]
    fp = d[(d.y == 0) & (pred == 1)]; bb = d[d.y == 0]
    print(f"\n  FN rate among P = {len(fn)/len(pp):.3f} | FP rate among B = {len(fp)/max(len(bb),1):.3f}")
    print("  -- conservation (EK_7 phyloP) of P variants: FN vs TP --")
    print(f"     FN mean EK_7={fn['EK_7'].mean():.2f}  TP mean EK_7={pp[pred[pp.index]==1]['EK_7'].mean():.2f}")
    print("  -- max observed allele-freq (benign signal) --")
    print(f"     FN(P) mean max_af={fn['max_af'].mean():.4f} median={fn['max_af'].median():.4f}  "
          f"| FP(B) mean max_af={fp['max_af'].mean():.4f}")
    # LABEL-NOISE signatures
    fn_benignlike = fn[(fn.max_af > 0.01) | (fn.EK_7 < 0)]    # P labeled but benign-looking
    fp_pathlike = fp[(fp.max_af < 1e-4) & (fp.EK_7 > 2)]      # B labeled but pathogenic-looking
    print(f"\n  LABEL-NOISE signatures:")
    print(f"     FN that look BENIGN (max_af>1% OR phyloP<0) = {len(fn_benignlike)}/{len(fn)} "
          f"({len(fn_benignlike)/max(len(fn),1):.2f}) -> suspected mislabel/ambiguous")
    print(f"     FP that look PATHOGENIC (max_af<1e-4 AND phyloP>2) = {len(fp_pathlike)}/{max(len(fp),1)} "
          f"({len(fp_pathlike)/max(len(fp),1):.2f}) -> honest model best-guess, likely correct call vs wrong label")
    # population structure
    if "CAT_1" in d:
        topfn = fn["CAT_1"].value_counts(normalize=True).head(3)
        topall = pp["CAT_1"].value_counts(normalize=True).head(3)
        print(f"\n  FN by popmax (CAT_1) top3: {dict(topfn.round(2))}  vs all-P: {dict(topall.round(2))}")
    print("  saved -> analysis/output/step1_oof.csv")
    print("DONE")


if __name__ == "__main__":
    main()
