"""Cross-panel transfer blend (general -> small panels), LEAKAGE-CORRECTED.

Memory flagged general->cftr transfer as the top untested lever (claimed novel-cftr
AUC 0.955). But 63-69% of small-panel Variant_IDs are SHARED with general, and the
prior experiment excluded general rows by FEATURE-VECTOR match — which fails because
the AL_ block drifts across panels for the same variant, so it excluded ~0 rows and
left the general model trained on most "held-out" target variants. That inflates the
transfer benefit.

CORRECT control: finals target variants are NOVEL to general. So we train ONE general
model excluding ALL target-panel Variant_IDs and score every target fold with it —
every transfer prob is genuinely out-of-sample (the honest finals simulation).

Per target fold: target-own deployed model (e) + frozen general scorer (g, Platt-
calibrated to target on train labels). Blend w*g + (1-w)*e, w swept. Threshold on the
blend train-OOF at the test prior. Analytic F1*/MCC + finals-size bootstrap on the
held-out fold; PAIRED delta vs w=0 (deployed target alone).
"""
import os, sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")

from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from lightgbm import LGBMClassifier

from config import SEED
from schema import (normalize_sentinels, drop_all_nan_rows, to_categorical,
                    partition_columns)
from feature_engineering import (engineer_aa_features, engineer_acmg_features,
                                 engineer_interaction_features,
                                 fit_score_agreement, apply_score_agreement)
from calibration import fit_platt, apply_platt
from utils import detect_leaky_columns, KNOWN_LEAKY, PANELS, TEST_PRIORS, ACMG_PANELS
from run_pipeline import preprocess_data, get_model_params
from harness import (fast_mrmr, lgb_oof_and_test, _drop_leaky, tpr_fpr, f1_at_prior,
                     mcc_at_prior_analytic, pick_threshold_at_prior, bootstrap_f1_mcc,
                     FINALS_N)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
W_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def _load(panel):
    df = pd.read_csv(PANELS[panel]); df = normalize_sentinels(df)
    df = drop_all_nan_rows(df, verbose=False)
    return df.reset_index(drop=True)


def build_general_scorer(target_panel, sources=("general",), recipe="improved"):
    """Fit a transfer scorer on the POOLED `sources` panels MINUS all target
    Variant_IDs, and project the target rows through the SAME preprocessing +
    feature columns. Returns calibrated scores on the target rows + labels.

    recipe="deployed": median-impute + mRMR (deployed general chain).
    recipe="improved": native-NaN preprocessing + ALL features + fam_flags (the
        validated best general recipe → a stronger transfer scorer).
    sources=("general",) reproduces the general-only transfer scorer.
    """
    from pipeline_v2 import preprocess_keep_nan, fit_al, apply_al
    tids = set(_load(target_panel)["Variant_ID"])
    frames = []
    for s in sources:
        ds = _load(s)
        frames.append(ds[~ds["Variant_ID"].isin(tids)])
    dg = pd.concat(frames, ignore_index=True).reset_index(drop=True)
    dg = to_categorical(dg); dg = engineer_aa_features(dg)
    _, gn, gc, gt = partition_columns(dg)
    yg = dg[gt].reset_index(drop=True); Xg = dg[gn + gc].reset_index(drop=True)
    leaky = detect_leaky_columns(Xg, yg, auc_threshold=0.99, verbose=False)
    drop = sorted(set(leaky) | (KNOWN_LEAKY & set(Xg.columns)))
    Xg = Xg.drop(columns=drop, errors="ignore")

    dt = _load(target_panel); dt = to_categorical(dt); dt = engineer_aa_features(dt)
    yt = dt["Label"].reset_index(drop=True).astype(int)
    Xt = dt.reindex(columns=Xg.columns)  # project onto general's post-leakage cols

    if recipe == "improved":
        # match production transfer.py: native-NaN + ALL features, no fam_flags
        Xg_sel, Xt_sel = preprocess_keep_nan(Xg, Xt)
    else:
        Xg_pp, _, Xt_pp, _ = preprocess_data(Xg.copy(), Xg.copy(), Xt.copy(), 0.95)
        Xg_sel, _, Xt_sel, _ = fast_mrmr(Xg_pp, yg, Xt_pp, Xt_pp)
    feats = list(Xg_sel.columns)
    cat_feats = [c for c in feats if isinstance(Xg_sel[c].dtype, pd.CategoricalDtype)]
    for c in cat_feats:
        Xt_sel[c] = pd.Categorical(Xt_sel[c].astype(object),
                                   categories=Xg_sel[c].cat.categories)

    spw = (yg == 0).sum() / max((yg == 1).sum(), 1)
    params = get_model_params(len(Xg_sel), spw)
    i2, ies = train_test_split(np.arange(len(yg)), test_size=0.15,
                               random_state=SEED + 7, stratify=yg)
    m = LGBMClassifier(**params, random_state=SEED)
    m.fit(Xg_sel.iloc[i2], yg.iloc[i2], eval_set=[(Xg_sel.iloc[ies], yg.iloc[ies])],
          callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
          categorical_feature=cat_feats or "auto")
    g_all = m.predict_proba(Xt_sel)[:, 1]
    return g_all, yt, len(feats), len(dg)


def load_target_engineered(panel):
    df = _load(panel); df = to_categorical(df); df = engineer_aa_features(df)
    if panel in ACMG_PANELS:
        df = engineer_acmg_features(df)
    if panel in ("cancer", "cftr"):
        df = engineer_interaction_features(df)
    _, n, c, t = partition_columns(df)
    return df[n + c].reset_index(drop=True), df[t].reset_index(drop=True).astype(int)


def target_deployed_fn(panel):
    sa = panel in ("cancer", "cftr")

    def _fn(X_tr, y_tr, X_te):
        Xtr, Xte = _drop_leaky(X_tr, y_tr, X_te)
        Xtr_pp, _, Xte_pp, _ = preprocess_data(Xtr.copy(), Xtr.copy(), Xte.copy(), 0.95)
        if sa:
            saf = fit_score_agreement(Xtr_pp, y_tr, top_k=7)
            Xtr_pp = apply_score_agreement(Xtr_pp, saf); Xte_pp = apply_score_agreement(Xte_pp, saf)
        Xtr_pp, _, Xte_pp, _ = fast_mrmr(Xtr_pp, y_tr, Xte_pp, Xte_pp)
        spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        params = get_model_params(len(X_tr), spw)
        nf = 3 if len(X_tr) < 200 else 5
        tp, oof = lgb_oof_and_test(Xtr_pp, y_tr, Xte_pp, params, n_seeds=5, n_folds=nf)
        pf = fit_platt(oof, y_tr.values)
        return apply_platt(tp, pf), apply_platt(oof, pf)
    return _fn


def _sweep(panel, g_all, Xt_own, yt, fn, prior, n_target, n_repeats, tag):
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED)
    per_w = {w: {"f1": [], "mcc": [], "f1b": []} for w in W_GRID}
    for tr, te in rskf.split(Xt_own, yt):
        tr, te = np.asarray(tr), np.asarray(te)
        ytr, yte = yt.iloc[tr].reset_index(drop=True), yt.iloc[te].reset_index(drop=True)
        e_te, e_tr = fn(Xt_own.iloc[tr].reset_index(drop=True), ytr,
                        Xt_own.iloc[te].reset_index(drop=True))
        pl = fit_platt(g_all[tr], ytr.values)
        g_tr = apply_platt(g_all[tr], pl); g_te = apply_platt(g_all[te], pl)
        for w in W_GRID:
            b_tr = w * g_tr + (1 - w) * e_tr
            b_te = w * g_te + (1 - w) * e_te
            thr = pick_threshold_at_prior(ytr.values, b_tr, prior)
            preds = (b_te >= thr).astype(int)
            tpr, fpr = tpr_fpr(yte.values, preds)
            per_w[w]["f1"].append(f1_at_prior(tpr, fpr, prior))
            per_w[w]["mcc"].append(mcc_at_prior_analytic(tpr, fpr, prior))
            f1b, _ = bootstrap_f1_mcc(yte.values, preds, prior, n_target)
            per_w[w]["f1b"].append(f1b)
    base = np.array(per_w[0.0]["f1"]); nf = len(base); rows = []
    print(f"  [{tag}] paired Δ vs deployed w=0 ({nf} folds):", flush=True)
    for w in W_GRID:
        arr = np.array(per_w[w]["f1"]); d = arr - base
        se = d.std(ddof=1) / np.sqrt(nf) if nf > 1 else 0
        sigma = d.mean() / se if se > 0 else 0
        t = "(deployed)" if w == 0 else f"{sigma:+.2f}σ"
        print(f"    w={w:<4} F1*={arr.mean():.4f}  finals={np.mean(per_w[w]['f1b']):.4f}  "
              f"Δ={d.mean():+.4f}  {t}", flush=True)
        rows.append({"panel": panel, "source": tag, "w": w, "f1": arr.mean(),
                     "finals": np.mean(per_w[w]["f1b"]), "delta": d.mean(),
                     "se": se, "sigma": sigma, "mcc": np.mean(per_w[w]["mcc"])})
    return rows


def run(panel, n_repeats=4, pooled=False):
    prior = TEST_PRIORS[panel]; n_target = FINALS_N[panel]
    print(f"\n{'='*64}\n  GENERAL -> {panel.upper()} (Variant_ID-excluded, honest)\n{'='*64}", flush=True)
    Xt_own, yt = load_target_engineered(panel)
    fn = target_deployed_fn(panel)
    all_rows = []
    source_sets = [("general", ("general",))]
    if pooled:
        others = tuple(p for p in ("general", "cancer", "pah", "cftr") if p != panel)
        source_sets.append(("pooled", others))
    for tag, sources in source_sets:
        t0 = time.time()
        g_all, yt_g, nfeat, ngen = build_general_scorer(panel, sources=sources)
        assert (yt.values == yt_g.values).all(), "row order mismatch"
        print(f"  {tag} scorer ({'+'.join(sources)}): {nfeat} feats, {ngen} rows, "
              f"AUC on {panel}={roc_auc_score(yt_g, g_all):.4f}  ({time.time()-t0:.1f}s)", flush=True)
        all_rows += _sweep(panel, g_all, Xt_own, yt, fn, prior, n_target, n_repeats, tag)
    pd.DataFrame(all_rows).to_csv(os.path.join(OUT, f"exp_transfer_{panel}.csv"), index=False)
    return all_rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("panels", nargs="*", default=["cftr", "cancer", "pah"])
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--pooled", action="store_true")
    a = ap.parse_args()
    for pn in (a.panels or ["cftr", "cancer", "pah"]):
        run(pn, n_repeats=a.repeats, pooled=a.pooled)
