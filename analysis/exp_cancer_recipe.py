"""Exp 3 — cancer own-model recipe upgrade.

Cancer's own model is 50% of its deployed score but still uses the legacy median+mRMR
(+score-agree+interactions) recipe. Test giving it general's reworked recipe: native-NaN
+ ALL features + RF/CatBoost blend (no mRMR/score-agree/interactions). Does the blended
cancer F1* improve, and does the optimal w shift?

Honest paired CV (identical folds): fixed baseline transfer scorer g; per fold compute the
own model both ways (e_legacy, e_rework) + sweep the blend weight w. F1* at the test prior,
threshold on the train blend. Report each own-recipe's F1* across w, paired Δ (rework−legacy).

Run:  .venv/bin/python -u analysis/exp_cancer_recipe.py
"""
import os, sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")

from sklearn.model_selection import RepeatedStratifiedKFold
from config import SEED
from schema import to_categorical, partition_columns
from feature_engineering import engineer_aa_features
from utils import TEST_PRIORS
from calibration import fit_platt, apply_platt
from harness import (pick_threshold_at_prior, tpr_fpr, f1_at_prior, FINALS_N)
from exp_transfer import _load, load_target_engineered, target_deployed_fn, W_GRID
from exp_general_blend import blend_fn          # native-NaN all-features LGBM+RF+CatBoost
from transfer import fit_transfer_scorer, transfer_score

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
PANEL = "cancer"


def aa_only_features(panel):
    """Cancer features WITHOUT interactions/score-agree (general-style recipe input)."""
    df = _load(panel); df = to_categorical(df); df = engineer_aa_features(df)
    _, n, c, t = partition_columns(df)
    return df[n + c].reset_index(drop=True), df[t].reset_index(drop=True).astype(int)


def main():
    prior = TEST_PRIORS[PANEL]; n_target = FINALS_N[PANEL]
    # fixed baseline scorer (current deployed single-seed)
    g_all = transfer_score(_load(PANEL), fit_transfer_scorer(PANEL))
    Xleg, yt = load_target_engineered(PANEL)        # AA + interactions (legacy own input)
    Xrew, yt2 = aa_only_features(PANEL)              # AA only (rework own input)
    assert (yt.values == yt2.values).all() and len(g_all) == len(yt)
    legacy_fn = target_deployed_fn(PANEL)           # median+mRMR(+score-agree)

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=8, random_state=SEED)
    per = {("legacy", w): [] for w in W_GRID}
    per.update({("rework", w): [] for w in W_GRID})
    t0 = time.time()
    for tr, te in rskf.split(Xleg, yt):
        tr, te = np.asarray(tr), np.asarray(te)
        ytr, yte = yt.iloc[tr].reset_index(drop=True), yt.iloc[te].reset_index(drop=True)
        pl = fit_platt(g_all[tr], ytr.values)
        g_tr, g_te = apply_platt(g_all[tr], pl), apply_platt(g_all[te], pl)
        el_te, el_tr = legacy_fn(Xleg.iloc[tr].reset_index(drop=True), ytr,
                                 Xleg.iloc[te].reset_index(drop=True))
        er_te, er_tr = blend_fn(Xrew.iloc[tr].reset_index(drop=True), ytr,
                                Xrew.iloc[te].reset_index(drop=True))
        # blend_fn returns (test, oof) un-Platt'd -> calibrate to train labels
        elpf = fit_platt(el_tr, ytr.values); erpf = fit_platt(er_tr, ytr.values)
        el_tr, el_te = apply_platt(el_tr, elpf), apply_platt(el_te, elpf)
        er_tr, er_te = apply_platt(er_tr, erpf), apply_platt(er_te, erpf)
        for name, e_tr, e_te in (("legacy", el_tr, el_te), ("rework", er_tr, er_te)):
            for w in W_GRID:
                b_tr = w * g_tr + (1 - w) * e_tr
                b_te = w * g_te + (1 - w) * e_te
                thr = pick_threshold_at_prior(ytr.values, b_tr, prior)
                tpr, fpr = tpr_fpr(yte.values, (b_te >= thr).astype(int))
                per[(name, w)].append(f1_at_prior(tpr, fpr, prior))

    nf = len(per[("legacy", 0.0)])
    print(f"=== cancer own-recipe upgrade — {nf} folds, F1* at prior {prior} "
          f"({time.time()-t0:.0f}s) ===", flush=True)
    print(f"{'w':>5s} {'legacy F1*':>11s} {'rework F1*':>11s} {'Δ(rew-leg)':>11s} {'σ':>7s}", flush=True)
    rows = []
    for w in W_GRID:
        leg = np.array(per[("legacy", w)]); rew = np.array(per[("rework", w)])
        d = rew - leg; se = d.std(ddof=1) / np.sqrt(nf) if nf > 1 else 0.0
        sig = d.mean() / se if se > 0 else 0.0
        print(f"{w:>5} {leg.mean():>11.4f} {rew.mean():>11.4f} {d.mean():>+11.4f} {sig:>+7.2f}", flush=True)
        rows.append({"w": w, "legacy_f1": leg.mean(), "rework_f1": rew.mean(),
                     "delta": d.mean(), "sigma": sig})
    deployed = next(r for r in rows if r["w"] == 0.5)
    best_leg = max(rows, key=lambda r: r["legacy_f1"])
    best_rew = max(rows, key=lambda r: r["rework_f1"])
    print(f"\n  deployed (legacy, w=0.5): {deployed['legacy_f1']:.4f}", flush=True)
    print(f"  best legacy: w={best_leg['w']} F1*={best_leg['legacy_f1']:.4f}", flush=True)
    print(f"  best rework: w={best_rew['w']} F1*={best_rew['rework_f1']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "exp_cancer_recipe.csv"), index=False)


if __name__ == "__main__":
    main()
