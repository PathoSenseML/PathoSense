"""AL_-block structure exploitation (task #3).

The 334-col AL_ allele-frequency block dominates the feature space but mRMR keeps
only ~20-60 columns. We test compact, train-fit representations of the block,
each FORCE-ADDED to the deployed mRMR feature set (so we measure marginal value
on top of the deployed model), scored by analytic F1* at the test prior, paired
vs deployed.

Candidates:
  fam_flags    : 17-ish cohort-family present flags (group AL cols by train
                 missingness, OR over each family) + n_fams_present.
  al_aggs      : n_al_present (rarity), al_maxaf (popmax), al_meanaf, al_present_frac.
  al_pca8/16   : PCA components of the 0-filled AL block (missing=absent=0).
  fam_replace  : replace the deployed per-column missing-indicators with the 17
                 family flags (compact rarity rep instead of ~300 noisy indicators).

Everything is fit on TRAIN only and applied to TEST; leakage-safe + stateless
given the fitted family map / PCA basis.
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))

from harness import (cv_evaluate, load_panel, compare, deployed_pipeline_fn,
                     lgb_oof_and_test, _drop_leaky, fast_mrmr)
from feature_engineering import fit_score_agreement, apply_score_agreement
from calibration import fit_platt, apply_platt
from utils import TEST_PRIORS
from run_pipeline import preprocess_data, get_model_params

PANELS = ["general", "cancer", "pah", "cftr"]
REPEATS = {"general": 4, "cancer": 6, "pah": 6, "cftr": 8}
SCREEN_SEEDS = 1
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def _sa_for(panel):
    return panel in ("cancer", "cftr")


# ── AL feature builders (fit on train, apply to test) ────────────────────────

def fit_al(X_tr, kinds=("fam_flags", "al_aggs"), n_pca=8):
    al = [c for c in X_tr.columns if c.startswith("AL_")]
    fitted = {"al_cols": al, "kinds": list(kinds)}
    if "fam_flags" in kinds or "fam_replace" in kinds:
        mc = X_tr[al].isna().sum()
        fams = {}
        for c in al:
            fams.setdefault(int(mc[c]), []).append(c)
        fitted["families"] = fams
    if "al_aggs" in kinds:
        M = X_tr[al]
        af_cols = [c for c in al if ((M[c] > 0) & (M[c] < 1)).any()
                   and M[c].mean(skipna=True) < 0.1]
        fitted["af_cols"] = af_cols
    if "al_pca" in kinds:
        from sklearn.decomposition import TruncatedSVD
        Z = X_tr[al].fillna(0.0).to_numpy(dtype=float)
        # standardize columns (present-value scale varies); keep 0 for absent
        mu = Z.mean(0); sd = Z.std(0); sd[sd < 1e-9] = 1.0
        Zs = (Z - mu) / sd
        svd = TruncatedSVD(n_components=min(n_pca, Zs.shape[1] - 1), random_state=42)
        svd.fit(Zs)
        fitted["pca"] = {"mu": mu, "sd": sd, "svd": svd}
    return fitted


def apply_al(X, fitted):
    al = fitted["al_cols"]
    out = {}
    kinds = fitted["kinds"]
    if "fam_flags" in kinds:
        fl = {}
        for fid, cols in fitted["families"].items():
            cols = [c for c in cols if c in X.columns]
            fl[f"ALfam_{fid}"] = X[cols].notna().any(axis=1).astype(int) if cols else 0
        fdf = pd.DataFrame(fl, index=X.index)
        out.update(fl)
        out["ALn_fams"] = fdf.sum(axis=1)
    if "al_aggs" in kinds:
        M = X[al]
        out["ALn_present"] = M.notna().sum(axis=1)
        out["ALpresent_frac"] = M.notna().mean(axis=1)
        af = fitted.get("af_cols", [])
        if af:
            out["ALmaxaf"] = M[af].max(axis=1).fillna(0.0)
            out["ALmeanaf"] = M[af].mean(axis=1).fillna(0.0)
    df = pd.DataFrame(out, index=X.index)
    if "al_pca" in kinds:
        p = fitted["pca"]
        Z = (X[al].fillna(0.0).to_numpy(dtype=float) - p["mu"]) / p["sd"]
        comp = p["svd"].transform(Z)
        for k in range(comp.shape[1]):
            df[f"ALpca_{k}"] = comp[:, k]
    return df


def make_fn(panel, kinds, n_pca=8, fam_replace=False):
    sa = _sa_for(panel)

    def _fn(X_tr, y_tr, X_te):
        Xtr, Xte = _drop_leaky(X_tr, y_tr, X_te)
        # AL features from RAW (pre-preprocess) input
        alf = fit_al(Xtr, kinds=kinds, n_pca=n_pca)
        ALtr = apply_al(Xtr, alf); ALte = apply_al(Xte, alf)

        Xtr_pp, _, Xte_pp, info = preprocess_data(Xtr.copy(), Xtr.copy(), Xte.copy(), 0.95)
        if fam_replace:
            # drop the per-column AL missing-indicators that preprocess added
            drop = [c for c in Xtr_pp.columns if c.startswith("AL_") and c.endswith("_missing")]
            Xtr_pp = Xtr_pp.drop(columns=drop, errors="ignore")
            Xte_pp = Xte_pp.drop(columns=drop, errors="ignore")
        if sa:
            saf = fit_score_agreement(Xtr_pp, y_tr, top_k=7)
            Xtr_pp = apply_score_agreement(Xtr_pp, saf)
            Xte_pp = apply_score_agreement(Xte_pp, saf)
        Xtr_pp, _, Xte_pp, _ = fast_mrmr(Xtr_pp, y_tr, Xte_pp, Xte_pp)
        # force-add the AL features (alongside mrmr selection)
        Xtr_pp = pd.concat([Xtr_pp.reset_index(drop=True), ALtr.reset_index(drop=True)], axis=1)
        Xte_pp = pd.concat([Xte_pp.reset_index(drop=True), ALte.reset_index(drop=True)], axis=1)
        # dedupe any name collisions
        Xtr_pp = Xtr_pp.loc[:, ~Xtr_pp.columns.duplicated()]
        Xte_pp = Xte_pp.loc[:, ~Xte_pp.columns.duplicated()]

        spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        params = get_model_params(len(X_tr), spw)
        nf = 3 if len(X_tr) < 200 else 5
        test_probs, oof = lgb_oof_and_test(Xtr_pp, y_tr, Xte_pp, params,
                                           n_seeds=SCREEN_SEEDS, n_folds=nf)
        pf = fit_platt(oof, y_tr.values)
        return apply_platt(test_probs, pf), apply_platt(oof, pf)
    return _fn


def run():
    rows = []
    for pn in PANELS:
        df = load_panel(pn); rep = REPEATS[pn]
        base = cv_evaluate(pn, deployed_pipeline_fn(pn, n_seeds=SCREEN_SEEDS),
                           label="deployed", n_repeats=rep, df=df)
        bs = base.summary()
        print(f"\n=== {pn} (prior={TEST_PRIORS[pn]:.3f}) deployed F1*="
              f"{bs['f1_mean']:.4f}±{bs['f1_se']:.4f} AUC={bs['auc_mean']:.4f} ===")
        configs = {
            "fam_flags": make_fn(pn, ("fam_flags",)),
            "al_aggs": make_fn(pn, ("al_aggs",)),
            "fam+aggs": make_fn(pn, ("fam_flags", "al_aggs")),
            "al_pca8": make_fn(pn, ("al_pca",), n_pca=8),
            "al_pca16": make_fn(pn, ("al_pca",), n_pca=16),
            "fam+aggs+pca8": make_fn(pn, ("fam_flags", "al_aggs", "al_pca"), n_pca=8),
            "fam_replace_indicators": make_fn(pn, ("fam_flags",), fam_replace=True),
        }
        for name, fn in configs.items():
            try:
                r = cv_evaluate(pn, fn, label=name, n_repeats=rep, df=df)
            except Exception as e:
                print(f"  {name:24s} FAILED: {type(e).__name__}: {str(e)[:80]}")
                continue
            c = compare(base, r)
            print(f"  {name:24s} F1*={c['b_f1']:.4f}  Δ={c['delta_f1']:+.4f} "
                  f"({c['sigma']:+.2f}σ) ΔMCC={c['delta_mcc']:+.4f}  {c['verdict']}")
            rows.append({"panel": pn, "config": name, **c})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "exp_alfeatures.csv"), index=False)
    print(f"\nsaved -> {os.path.join(OUT, 'exp_alfeatures.csv')}")


if __name__ == "__main__":
    run()
