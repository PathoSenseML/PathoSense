"""PAH population (CAT_1/CAT_2) x allele-frequency (AL_) hypothesis test.

USER HYPOTHESIS (2026-06-28): a few CAT_1/CAT_2 popmax-population subcategories are more
correlated with PAH pathogenicity specifically, and those subcategories correlate more
strongly with some AL_ features. If a PAH-SPECIFIC population x frequency model captures
this, we could deploy a real blend (w<1.0) instead of the pure-transfer w=1.0.

Why this might add signal the general transfer scorer misses:
  PKU (PAH deficiency) is recessive with strong FOUNDER structure — pathogenic variants are
  enriched in specific ancestries (esp. European) at low frequency, while AFR-popmax variants
  skew toward common benign polymorphism. The general scorer DOES ingest CAT_1/CAT_2 natively,
  but it learns a GLOBAL ancestry->pathogenicity map; PAH's founder map may differ.

Why it might NOT (the skeptic's prior):
  - transfer scorer already has CAT_1/CAT_2 as native categoricals (global map learned).
  - generic PAH own-model blends already washed out (own-model-bakeoff: pah -2.4sigma).
  - ancestry x freq-band crosstab is near-degenerate (see exp scouting): almost all mass in
    one band; founder cells have n=1-2 = noise.

DECISIVE TESTS (all honest, per-fold, paired vs the DEPLOYED w=1.0; FRESH seed SEED+4242):
  (1) ORTHOGONALITY: held-out AUC of transfer alone vs a logistic stack [transfer + pop_score],
      pop_score = OOF model on the population/AF features only. + pred-corr(pop, transfer).
      If pop_score carries no signal beyond transfer, AUC does not move -> dead.
  (2) F1* w-sweep: blend w*transfer + (1-w)*own at w in {1.0(dep),.85,.75,.6,.5} + nested-w.
      own families: 'pop' (focused pop/AF features), 'full' (deployed-recipe own, control =
      already-rejected generic), 'full+pop' (does pop add to the generic own model?).
  Decision bar: a deployable win needs a w<1.0 config beating deployed by >=2sigma F1* AND a
  positive orthogonality AUC lift. Anything else = at the information ceiling (report cleanly).

Run: .venv/bin/python -u analysis/exp_pah_pop_af.py
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
from schema import (normalize_sentinels, drop_all_nan_rows, to_categorical,
                    partition_columns)
from feature_engineering import engineer_aa_features, engineer_acmg_features
from calibration import fit_platt, apply_platt
from harness import f1_at_prior, tpr_fpr, pick_threshold_at_prior
from models import get_model_params
from transfer import fit_transfer_scorer, transfer_score
from general_blend import _rf_repr, _cat_repr

PANEL = "pah"
PRIOR = TEST_PRIORS[PANEL]
RNG = SEED + 4242                      # fresh seed family (de-correlate from prior campaigns)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# ancestry super-groups from a gnomAD/AllofUs popmax label suffix
_SG = {"NFE": "EUR", "FIN": "EUR", "EUR": "EUR", "AFR": "AFR", "EAS": "EAS",
       "SAS": "SAS", "AMR": "AMR", "MID": "MID", "ASJ": "ASJ", "AMI": "AMI",
       "REMAINING": "OTH", "OTH": "OTH"}


def _supergroup(s):
    if pd.isna(s):
        return "ABSENT"
    s = str(s)
    if "&" in s:
        return "MULTI"
    return _SG.get(s.split("_")[-1], "OTH")


# ── population x frequency feature builder (leakage-safe: af_cols frozen on TRAIN) ──

def fit_pop_features(df_train_raw):
    """Freeze the minor-allele-frequency column set on the TRAIN fold. Returns fitted state."""
    al = [c for c in df_train_raw.columns if c.startswith("AL_")]
    M = df_train_raw[al].apply(pd.to_numeric, errors="coerce")
    # true MAF-like columns: a non-trivial share of values strictly in (0,1) and low mean
    af_cols = [c for c in al
               if (((M[c] > 0) & (M[c] < 1)).sum() >= 5) and (M[c].mean(skipna=True) < 0.2)]
    return {"al": al, "af_cols": af_cols}


def make_pop_features(df_raw, fitted):
    """Build PAH population x AF features. Pure replay given frozen `fitted` (batch-invariant)."""
    al, af_cols = fitted["al"], fitted["af_cols"]
    M = df_raw[al].apply(pd.to_numeric, errors="coerce")
    Maf = M[af_cols] if af_cols else M.iloc[:, :0]
    out = pd.DataFrame(index=df_raw.index)
    # popmax ancestry identity (gnomAD = CAT_1, AllofUs = CAT_2)
    a1 = df_raw["CAT_1"].map(_supergroup) if "CAT_1" in df_raw else pd.Series("ABSENT", index=df_raw.index)
    a2 = df_raw["CAT_2"].map(_supergroup) if "CAT_2" in df_raw else pd.Series("ABSENT", index=df_raw.index)
    out["pop_anc1"] = a1.astype("category")
    out["pop_anc2"] = a2.astype("category")
    out["pop_present1"] = (a1 != "ABSENT").astype(int)
    out["pop_present2"] = (a2 != "ABSENT").astype(int)
    out["pop_anc_agree"] = ((a1 == a2) & (a1 != "ABSENT")).astype(int)
    # frequency magnitude / structure across populations
    maxaf = Maf.max(axis=1)
    meanaf = Maf.mean(axis=1)
    npop = Maf.notna().sum(axis=1)
    out["pop_maxaf"] = maxaf.fillna(0.0)
    out["pop_log_maxaf"] = np.log10(maxaf.fillna(0.0) + 1e-9)
    out["pop_n_present"] = npop.astype(int)
    # founder-like concentration: popmax dominates the mean (one population carries it)
    out["pop_af_concentration"] = (maxaf / (meanaf + 1e-12)).where(npop > 0, 0.0).fillna(0.0)
    # explicit ACMG-style frequency bands (let trees split ancestry x band)
    out["pop_rare"] = ((maxaf > 0) & (maxaf < 1e-3)).astype(int)
    out["pop_common"] = (maxaf > 1e-2).astype(int)
    # direct founder signature: rare AND European popmax (PKU pathogenic-leaning)
    out["pop_rare_eur"] = (out["pop_rare"].astype(bool) & (a1 == "EUR")).astype(int)
    out["pop_common_nonabsent"] = (out["pop_common"].astype(bool) & (a1 != "ABSENT")).astype(int)
    return out


# ── models ───────────────────────────────────────────────────────────────────

def _lgbm(n, spw, seed, n_est=300):
    p = get_model_params(n, spw)
    p = {**p, "n_estimators": n_est, "random_state": seed, "n_jobs": 1, "verbose": -1}
    return lgb.LGBMClassifier(**p)


def _own_oof_test(Xtr, ytr, Xte, cat_feats, spw, seed, n_folds=4):
    """Calibrated (train-OOF, test) probs for an LGBM on (Xtr->Xte). Honest OOF for threshold."""
    oof = np.zeros(len(ytr))
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    for a, b in skf.split(Xtr, ytr):
        m = _lgbm(len(a), spw, seed)
        m.fit(Xtr.iloc[a], ytr.iloc[a], categorical_feature=cat_feats or "auto")
        oof[b] = m.predict_proba(Xtr.iloc[b])[:, 1]
    platt = fit_platt(oof, ytr.values)
    mf = _lgbm(len(ytr), spw, seed)
    mf.fit(Xtr, ytr, categorical_feature=cat_feats or "auto")
    te = mf.predict_proba(Xte)[:, 1]
    return apply_platt(oof, platt), apply_platt(te, platt)


def _f1(y, p, thr):
    return f1_at_prior(*tpr_fpr(y, (p >= thr).astype(int)), PRIOR)


WGRID = [0.5, 0.6, 0.75, 0.85, 1.0]


def main():
    t0 = time.time()
    # load PAH raw + engineered (deployed recipe: ACMG on for pah)
    df = pd.read_csv(PANELS[PANEL]); df = normalize_sentinels(df); df = drop_all_nan_rows(df, verbose=False)
    df = df.reset_index(drop=True)
    de = engineer_aa_features(to_categorical(df.copy()))
    de = engineer_acmg_features(de)
    _, num_cols, cat_cols, tgt = partition_columns(de)
    y = de[tgt].reset_index(drop=True).astype(int)
    Xfull = de[num_cols + cat_cols].reset_index(drop=True)
    leaky = detect_leaky_columns(Xfull[num_cols], y, auc_threshold=0.99, verbose=False)
    Xfull = Xfull.drop(columns=sorted(set(leaky) | (KNOWN_LEAKY & set(Xfull.columns))))
    cat_full = [c for c in Xfull.columns if isinstance(Xfull[c].dtype, pd.CategoricalDtype)]

    # OOS transfer score (leakage-correct: target Variant_IDs excluded inside fit_transfer_scorer)
    art = fit_transfer_scorer(PANEL)
    g_all = np.asarray(transfer_score(df, art))
    print(f"PAH n={len(df)} ({int(y.sum())}P/{int((y==0).sum())}B)  prior(test)={PRIOR:.3f}  "
          f"transfer OOS AUC={roc_auc_score(y, g_all):.4f}  seed={RNG}", flush=True)

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=RNG)
    # accumulators
    cfgs = ["deployed"]
    for fam in ["pop", "full", "fullpop"]:
        for w in [0.85, 0.75, 0.6, 0.5]:
            cfgs.append(f"{fam}_w{int(w*100)}")
        cfgs.append(f"{fam}_wnested")
    F1 = {k: [] for k in cfgs}
    AUC_g, AUC_stack = [], []     # orthogonality: transfer vs [transfer+pop_score] logistic
    CORR = []                     # pred corr(pop_score, transfer)
    AUC_pop = []                  # standalone pop-model AUC

    for tr, te in rskf.split(Xfull, y):
        tr, te = np.asarray(tr), np.asarray(te)
        ytr, yte = y.iloc[tr].reset_index(drop=True), y.iloc[te].reset_index(drop=True)
        spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
        seed = RNG + len(F1["deployed"])
        # transfer (Platt on train fold) — the deployed scorer
        plg = fit_platt(g_all[tr], ytr.values)
        g_tr, g_te = apply_platt(g_all[tr], plg), apply_platt(g_all[te], plg)

        # ---- population/AF own model (the hypothesis) ----
        pf = fit_pop_features(df.iloc[tr])
        Pp = make_pop_features(df, pf)
        cat_pop = [c for c in Pp.columns if isinstance(Pp[c].dtype, pd.CategoricalDtype)]
        o_tr, o_te = _own_oof_test(Pp.iloc[tr].reset_index(drop=True), ytr,
                                   Pp.iloc[te].reset_index(drop=True), cat_pop, spw, seed)

        # ---- full deployed-recipe own model (generic control) ----
        f_tr, f_te = _own_oof_test(Xfull.iloc[tr].reset_index(drop=True), ytr,
                                   Xfull.iloc[te].reset_index(drop=True), cat_full, spw, seed)

        # ---- full + pop features (does pop add to the generic own model?) ----
        Xfp = pd.concat([Xfull.reset_index(drop=True), Pp.add_prefix("Z_")], axis=1)
        cat_fp = cat_full + [f"Z_{c}" for c in cat_pop]
        fp_tr, fp_te = _own_oof_test(Xfp.iloc[tr].reset_index(drop=True), ytr,
                                     Xfp.iloc[te].reset_index(drop=True), cat_fp, spw, seed)

        # ===== orthogonality diagnostic (on the held-out fold) =====
        AUC_g.append(roc_auc_score(yte, g_te))
        AUC_pop.append(roc_auc_score(yte, o_te))
        CORR.append(np.corrcoef(o_te, g_te)[0, 1])
        # logistic stack fit on TRAIN fold OOF, scored on test
        Ztr = np.column_stack([g_tr, o_tr]); Zte = np.column_stack([g_te, o_te])
        lr = LogisticRegression(max_iter=1000).fit(Ztr, ytr.values)
        AUC_stack.append(roc_auc_score(yte, lr.predict_proba(Zte)[:, 1]))

        # ===== F1* w-sweep (threshold at prior on train-OOF blend) =====
        t_dep = pick_threshold_at_prior(ytr.values, g_tr, PRIOR)
        F1["deployed"].append(_f1(yte.values, g_te, t_dep))
        for fam, (own_tr, own_te) in [("pop", (o_tr, o_te)), ("full", (f_tr, f_te)),
                                      ("fullpop", (fp_tr, fp_te))]:
            best_w, best_f1tr = 1.0, -1
            for w in [0.85, 0.75, 0.6, 0.5]:
                b_tr = w * g_tr + (1 - w) * own_tr
                b_te = w * g_te + (1 - w) * own_te
                t = pick_threshold_at_prior(ytr.values, b_tr, PRIOR)
                F1[f"{fam}_w{int(w*100)}"].append(_f1(yte.values, b_te, t))
                # nested-w: pick w maximizing train-OOF F1*
                f1tr = _f1(ytr.values, b_tr, t)
                if f1tr > best_f1tr:
                    best_f1tr, best_w = f1tr, w
            bn_tr = best_w * g_tr + (1 - best_w) * own_tr
            bn_te = best_w * g_te + (1 - best_w) * own_te
            tn = pick_threshold_at_prior(ytr.values, bn_tr, PRIOR)
            F1[f"{fam}_wnested"].append(_f1(yte.values, bn_te, tn))

    # ── report ──
    dep = np.array(F1["deployed"]); nf = len(dep)
    print(f"\nfolds={nf}  ({time.time()-t0:.0f}s)")
    print(f"\n== ORTHOGONALITY (held-out AUC) ==")
    ag, ast, apop, corr = map(np.array, (AUC_g, AUC_stack, AUC_pop, CORR))
    dlt = ast - ag; se = dlt.std(ddof=1) / np.sqrt(nf)
    print(f"  transfer alone           AUC={ag.mean():.4f}")
    print(f"  pop-model alone          AUC={apop.mean():.4f}   pred-corr(pop,transfer)={corr.mean():.3f}")
    print(f"  logistic[transfer+pop]   AUC={ast.mean():.4f}   delta={dlt.mean():+.4f} ({dlt.mean()/se if se>0 else 0:+.2f}sigma)")
    print(f"\n== F1* at test prior (paired vs deployed; deployed={dep.mean():.4f}) ==")
    rows = []
    for k in cfgs:
        if k == "deployed":
            continue
        v = np.array(F1[k]); d = v - dep
        s = d.std(ddof=1) / np.sqrt(nf); sig = d.mean() / s if s > 0 else 0.0
        verdict = "DEPLOY" if (d.mean() > 0 and sig >= 2) else ("+" if d.mean() > 0 else "-")
        rows.append((k, v.mean(), d.mean(), sig, verdict))
    rows.sort(key=lambda r: -r[2])
    print(f"  {'config':16s} {'F1*':>7s} {'dF1':>8s} {'sigma':>7s}  verdict")
    for k, m, dm, sig, ver in rows:
        print(f"  {k:16s} {m:7.4f} {dm:+8.4f} {sig:7.2f}  {ver}")
    # save
    pd.DataFrame(rows, columns=["config", "f1", "delta", "sigma", "verdict"]).to_csv(
        os.path.join(OUT, "pah_pop_af.csv"), index=False)
    print(f"\n[saved] {os.path.join(OUT, 'pah_pop_af.csv')}")


if __name__ == "__main__":
    main()
