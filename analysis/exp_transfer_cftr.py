"""General -> CFTR transfer blend — does a frozen general single-LGBM scorer,
blended into the deployed CFTR model, beat the deployed CFTR model alone?

Mechanism: CFTR is the tiniest panel (n=111). A model trained on the large
general panel (n~2815) has seen far more variants and might generalise better on
some CFTR cases. We blend its probability into the deployed CFTR ensemble:

    blend = w * general_transfer_prob + (1 - w) * cftr_base_prob

and sweep w in {0, 0.3, 0.5, 0.7, 1.0}. w=0 is the DEPLOYED CFTR model (the
incumbent baseline); w=1 is the pure general-transfer scorer.

CROSS-PANEL LEAKAGE (handled, and verified):
  The general model scores held-out CFTR rows. In CV, the general model that
  scores a held-out CFTR test fold must NOT have trained on the same variants.
  "Same variant" is matched by FEATURE-VECTOR identity (a precision-tolerant hash
  of the shared raw anonymized columns), NOT by Variant_ID — organizers confirmed
  identical Variant_IDs across panels can be DIFFERENT variants. We verify this
  empirically: it turns out 0/111 CFTR rows match any general row by feature
  vector (the AL_ allele-frequency block differs even for shared Variant_IDs), so
  the per-fold exclusion removes ~0 general rows here. The mechanism is
  implemented and reported regardless, so the protocol is correct by construction
  and would still hold if a future panel did share variants.

  At deployment the general model trains on ALL general data and a NOVEL CFTR test
  variant gets a clean (out-of-sample) generalization prob. The CV simulates this:
  general transfer probs for held-out CFTR rows come from a general model that
  excluded any feature-vector match to that fold; the general model never trains
  on CFTR rows at all (0 matches), so every CFTR prob is out-of-sample.

HONEST EVAL: paired RepeatedStratifiedKFold(5 x 3 = 15 folds) at the CFTR TEST
prior (0.167). The CFTR base is the deployed 5-seed LGBM ensemble on the deployed
+tracks feature set (prepare_panel_tracks). The blend threshold is tuned on the
outer-train blended OOF at the test prior, frozen, applied to the held-out fold.
Paired ΔF1* vs w=0 (deployed) with SE over folds; win bar = ΔF1* > 2*SE.

Run:  .venv/bin/python -u analysis/exp_transfer_cftr.py
"""
import os
import sys
import time
import json
import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipelines"))

from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
import lightgbm as lgb
from lightgbm import LGBMClassifier

from config import SEED
from schema import (
    normalize_sentinels, drop_all_nan_rows, to_categorical, partition_columns,
)
from feature_engineering import engineer_aa_features
from calibration import fit_platt, apply_platt
from utils import (
    detect_leaky_columns, optimize_threshold_f1_balanced,
    KNOWN_LEAKY, PANELS, TEST_PRIORS,
)
from run_pipeline import preprocess_data, select_features_mrmr, get_model_params

# reuse the deployed CFTR base + scoring helpers from the existing harnesses
from fair_tuneoff import metrics_at_prior
from blend_test import prepare_panel_tracks, ensemble_fold_probs

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)

W_GRID = [0.0, 0.3, 0.5, 0.7, 1.0]
PRIOR = TEST_PRIORS["cftr"]


# ---------------------------------------------------------------------------
# Feature-vector identity (precision-tolerant string key over shared raw columns)
# ---------------------------------------------------------------------------

def _row_keys(df, cols):
    """One stable string key per row over `cols`. Floats -> %.6g, NaN -> '∅',
    so the same exported variant matches across panels despite CSV round-trip,
    while genuinely different variants do not collide."""
    parts = []
    for c in cols:
        s = df[c]
        if s.dtype.kind in "fc":
            parts.append(s.map(lambda x: "∅" if pd.isna(x) else format(float(x), ".6g")))
        else:
            parts.append(s.map(lambda x: "∅" if pd.isna(x) else str(x)))
    key = parts[0].astype(str)
    for p in parts[1:]:
        key = key.str.cat(p.astype(str), sep="|")
    return key.values


# ---------------------------------------------------------------------------
# Build general's deployed feature space and project CFTR rows into it
# ---------------------------------------------------------------------------

def _load_raw(panel):
    """read -> sentinels -> drop_all_nan -> reset.  Same canonical row order as
    prepare_panel_tracks (which applies the identical first three steps), so rows
    align by position with the CFTR base matrix."""
    df = pd.read_csv(PANELS[panel])
    df = normalize_sentinels(df)
    df = drop_all_nan_rows(df, verbose=False)
    return df.reset_index(drop=True)


def build_general_space():
    """Fit the deployed GENERAL chain (AA features, leakage drop, per-category
    preprocess, mRMR) on the general panel, and project the CFTR rows into the
    SAME feature space so the general model can score them.

    general's deployed config (BEST_CONFIG_PER_PANEL['general']) uses neither ACMG
    nor interaction nor score-agree tracks, so the transfer scorer is exactly the
    deployed general feature recipe.

    Returns:
      Xg   : general matrix in F_gen (category dtypes preserved)   [n_gen, F]
      yg   : general labels                                        [n_gen]
      Xc   : CFTR rows projected into F_gen (cats aligned to gen)  [n_cftr, F]
      cat_feats : categorical feature names in F_gen
      gen_keys, cftr_keys : feature-vector identity keys (row-aligned)
    """
    raw_cols = None

    dg = _load_raw("general")
    raw_cols = [c for c in dg.columns if c not in ("Variant_ID", "Label")]
    gen_keys = _row_keys(dg, raw_cols)

    dg = to_categorical(dg)
    dg = engineer_aa_features(dg)
    _, gnum, gcat, gtgt = partition_columns(dg)
    yg = dg[gtgt].reset_index(drop=True)
    Xg = dg[gnum + gcat].reset_index(drop=True)
    leaky = detect_leaky_columns(Xg, yg, auc_threshold=0.99, verbose=False)
    gdrop = sorted(set(leaky) | (KNOWN_LEAKY & set(Xg.columns)))
    if gdrop:
        Xg = Xg.drop(columns=gdrop)

    dc = _load_raw("cftr")
    cftr_keys = _row_keys(dc, raw_cols)
    dc = to_categorical(dc)
    dc = engineer_aa_features(dc)
    # project CFTR onto general's post-leakage column set (missing cols -> NaN)
    Xc = dc.reindex(columns=Xg.columns)

    # Fit general's preprocessing once; transform CFTR through it (3rd arg).
    Xg_pp, _, Xc_pp, _ = preprocess_data(Xg.copy(), Xg.copy(), Xc.copy(), corr_threshold=0.95)
    Xg_sel, _, _, _ = select_features_mrmr(Xg_pp, yg, Xg_pp, Xg_pp)
    feats = list(Xg_sel.columns)
    Xg_final = Xg_pp[feats].reset_index(drop=True)
    Xc_final = Xc_pp[feats].reset_index(drop=True)

    cat_feats = [c for c in feats if isinstance(Xg_final[c].dtype, pd.CategoricalDtype)]
    # Align CFTR categorical codes to general's category sets so LightGBM maps
    # them to the same integer codes it learned on (unseen -> NaN).
    for c in cat_feats:
        Xc_final[c] = pd.Categorical(Xc_final[c].astype(object),
                                     categories=Xg_final[c].cat.categories)
    return Xg_final, yg, Xc_final, cat_feats, gen_keys, cftr_keys


# ---------------------------------------------------------------------------
# Frozen general single-LGBM transfer scorer (deployed params)
# ---------------------------------------------------------------------------

def fit_general_model(Xg, yg, cat_feats, exclude_mask, spw):
    """Deployed-params single LightGBM on general data, excluding any general row
    flagged by exclude_mask (feature-vector match to the held-out CFTR fold).
    Early stopping on a 15% slice carved from the kept general rows."""
    keep = np.where(~exclude_mask)[0]
    Xt = Xg.iloc[keep].reset_index(drop=True)
    yt = yg.iloc[keep].reset_index(drop=True)
    params = get_model_params(len(keep), spw)
    i2, ies = train_test_split(np.arange(len(keep)), test_size=0.15,
                               random_state=SEED + 7, stratify=yt)
    m = LGBMClassifier(**params, random_state=SEED)
    m.fit(Xt.iloc[i2], yt.iloc[i2],
          eval_set=[(Xt.iloc[ies], yt.iloc[ies])],
          callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
          categorical_feature=cat_feats or "auto")
    return m


def main():
    ap = argparse.ArgumentParser(description="General->CFTR transfer blend")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="1 repeat (smoke test)")
    args = ap.parse_args()
    n_repeats = 1 if args.quick else args.repeats

    print(f"\n{'#'*72}\n  GENERAL -> CFTR TRANSFER BLEND\n{'#'*72}", flush=True)

    # --- CFTR base (deployed +tracks) ---
    Xc_base, yc, cat_base, prior = prepare_panel_tracks("cftr")
    assert abs(prior - PRIOR) < 1e-9
    spw_cftr = (yc == 0).sum() / max((yc == 1).sum(), 1)

    # --- general transfer space + CFTR projected into it ---
    Xg, yg, Xc_gen, cat_gen, gen_keys, cftr_keys = build_general_space()
    spw_gen = (yg == 0).sum() / max((yg == 1).sum(), 1)

    # sanity: CFTR base rows and CFTR-in-general-space rows are row-aligned
    yc2 = _load_raw("cftr")["Label"].reset_index(drop=True)
    assert len(Xc_gen) == len(Xc_base) == len(yc), (len(Xc_gen), len(Xc_base), len(yc))
    assert (yc.values == yc2.values).all(), "CFTR row order mismatch between base and general-space"

    # --- empirical feature-vector matching diagnostic ---
    gen_key_set = set(gen_keys)
    gen_label_by_key = {}
    for k, lab in zip(gen_keys, yg.values):
        gen_label_by_key.setdefault(k, set()).add(int(lab))
    matches, agree = 0, 0
    for k, lab in zip(cftr_keys, yc.values):
        if k in gen_key_set:
            matches += 1
            if gen_label_by_key[k] == {int(lab)}:
                agree += 1
    print(f"  general n={len(Xg)} (spw={spw_gen:.2f})  cftr n={len(Xc_base)} "
          f"(spw={spw_cftr:.2f})  test_prior={prior:.3f}", flush=True)
    print(f"  F_gen features={len(Xg.columns)}  cftr-base features={len(Xc_base.columns)}", flush=True)
    print(f"  feature-vector matches (cftr rows found in general): {matches}/{len(Xc_base)}  "
          f"(label-agree {agree}/{matches})", flush=True)
    print(f"  -> per-fold general-training exclusion removes ~{matches} rows "
          f"(0 means general never trains on a CFTR variant; all transfer probs OOS)\n", flush=True)

    # --- per-fold general model cache (keyed by excluded index tuple) ---
    gmodel_cache = {}

    def general_model(exclude_mask):
        key = tuple(np.where(exclude_mask)[0])
        if key not in gmodel_cache:
            gmodel_cache[key] = fit_general_model(Xg, yg, cat_gen, exclude_mask, spw_gen)
        return gmodel_cache[key]

    # --- repeated outer CV on CFTR ---
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED)
    per_fold = {w: [] for w in W_GRID}     # (f1, mcc) per fold
    rows = []
    t0 = time.time()
    for fold, (tr, te) in enumerate(outer.split(Xc_base, yc)):
        tr, te = np.asarray(tr), np.asarray(te)
        y_tr, y_te = yc.iloc[tr].values, yc.iloc[te].values
        ft = time.time()

        # CFTR base: deployed 5-seed ensemble, calibrated (train-OOF, test)
        e_oof, e_te = ensemble_fold_probs(Xc_base, yc, spw_cftr, tr, te)

        # general transfer: exclude general rows matching the held-out CFTR fold
        te_keys = set(cftr_keys[te])
        excl = np.isin(gen_keys, list(te_keys)) if te_keys else np.zeros(len(gen_keys), bool)
        gm = general_model(excl)
        g_tr_raw = gm.predict_proba(Xc_gen.iloc[tr])[:, 1]
        g_te_raw = gm.predict_proba(Xc_gen.iloc[te])[:, 1]
        # calibrate the general scorer to the CFTR distribution on TRAIN labels only
        pl = fit_platt(g_tr_raw, y_tr)
        g_tr = apply_platt(g_tr_raw, pl)
        g_te = apply_platt(g_te_raw, pl)

        line = []
        for w in W_GRID:
            b_oof = w * g_tr + (1 - w) * e_oof
            b_te = w * g_te + (1 - w) * e_te
            thr, _, _ = optimize_threshold_f1_balanced(
                y_tr, b_oof, target_prior=prior, n_bootstrap=150)
            f1, mcc = metrics_at_prior(y_te, b_te, thr, prior)
            per_fold[w].append((f1, mcc))
            rows.append({"fold": fold, "w": w, "f1": f1, "mcc": mcc, "thr": thr,
                         "n_excluded": int(excl.sum())})
            line.append(f"w{w}={f1:.3f}")
        print(f"  fold {fold+1:2d}/{5*n_repeats}: " + "  ".join(line) +
              f"  ({round(time.time()-ft,1)}s)", flush=True)

    # --- aggregate, paired vs w=0 (deployed) ---
    base = np.array([f for f, _ in per_fold[0.0]])
    base_mcc = np.array([m for _, m in per_fold[0.0]])
    print(f"\n  CFTR transfer blend — F1* mean±std over {len(base)} folds "
          f"(paired Δ vs w=0 deployed):", flush=True)
    best_w, best_mean = 0.0, base.mean()
    summary = {}
    for w in W_GRID:
        arr = np.array([f for f, _ in per_fold[w]])
        mcc = np.array([m for _, m in per_fold[w]])
        d = arr - base
        se = d.std() / max(np.sqrt(len(d)), 1)
        clears = bool(d.mean() > 2 * se)
        verdict = ("(deployed)" if w == 0.0 else
                   "HELPS (clears 2SE)" if clears else
                   "HURTS" if d.mean() < -2 * se else "no effect (noise)")
        summary[w] = {
            "f1_mean": float(arr.mean()), "f1_std": float(arr.std()),
            "mcc_mean": float(mcc.mean()), "delta": float(d.mean()),
            "se": float(se), "clears_2se": clears,
            "mcc_delta": float(mcc.mean() - base_mcc.mean()),
            "wins": int((d > 0).sum()),
        }
        dstr = "   —   " if w == 0.0 else f"{d.mean():+.4f}±{se:.4f}"
        wstr = "" if w == 0.0 else f"  wins {int((d>0).sum())}/{len(d)}"
        print(f"    w={w:<3}  F1*={arr.mean():.4f}±{arr.std():.3f}  MCC*={mcc.mean():.4f}  "
              f"Δ={dstr}{wstr}  -> {verdict}", flush=True)
        if w != 0.0 and arr.mean() > best_mean:
            best_w, best_mean = w, arr.mean()

    # pick best w by mean F1* (ties/regressions -> deployed w=0)
    chosen = best_w
    s = summary[chosen]
    print(f"\n  best blend weight by mean F1*: w={chosen}  "
          f"(F1* {s['f1_mean']:.4f} vs deployed {base.mean():.4f}, "
          f"Δ={s['delta']:+.4f}±{s['se']:.4f}, clears_2SE={s['clears_2se']})", flush=True)

    out = os.path.join(OUT_DIR, "exp_transfer_cftr.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n  Saved {out}   (total {round(time.time()-t0,1)}s)", flush=True)

    # machine-readable line for the orchestrator
    print("\nRESULT_JSON " + json.dumps({
        "panel": "cftr", "n_folds": len(base),
        "baseline_f1": float(base.mean()), "baseline_mcc": float(base_mcc.mean()),
        "best_w": chosen, "candidate_f1": float(summary[chosen]["f1_mean"]),
        "delta_f1": float(summary[chosen]["delta"]), "se": float(summary[chosen]["se"]),
        "clears_2se": bool(summary[chosen]["clears_2se"]),
        "mcc_delta": float(summary[chosen]["mcc_delta"]),
        "matches": matches,
        "per_w": {str(w): summary[w] for w in W_GRID},
    }), flush=True)


if __name__ == "__main__":
    main()
