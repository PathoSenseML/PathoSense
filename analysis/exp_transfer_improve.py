"""A/B test scorer improvements (Exp 1 multi-seed + Exp 2 monotonic constraints).

Uses the DEPLOYED transfer.fit_transfer_scorer + transfer_score, so the full deployed
scorer is tested (incl. the RF+CatBoost blend for cftr/cancer — not just the LGBM member).
Each config's general scores g are built ONCE (Variant_ID-excluded). One shared paired CV
loop per panel: the target-own model `e` is computed once per fold and reused across all
configs (so deltas isolate the scorer change). Scored at the DEPLOYED w, F1* at the test
prior, threshold on the train blend. Paired Δ(σ) vs baseline (current single-seed, no monotone).

For w=1.0 panels (cftr, pah) the own model is unused, so it's skipped (pure transfer).

Run:  .venv/bin/python -u analysis/exp_transfer_improve.py [cftr pah cancer]
"""
import os, sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")

from sklearn.model_selection import RepeatedStratifiedKFold
from config import SEED
from utils import TEST_PRIORS
from calibration import fit_platt, apply_platt
from harness import (pick_threshold_at_prior, tpr_fpr, f1_at_prior,
                     mcc_at_prior_analytic, bootstrap_f1_mcc, FINALS_N)
from exp_transfer import _load, load_target_engineered, target_deployed_fn
from transfer import fit_transfer_scorer, transfer_score, TRANSFER_PANELS

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

CONFIGS = [   # (tag, n_seeds, monotone)
    ("baseline",   1,  "none"),
    ("ms5",        5,  "none"),
    ("ms10",      10,  "none"),
    ("mono_ek",    1,  "ek"),
    ("mono_ek_aa", 1,  "ek_aa"),
    ("ms5+ek",     5,  "ek"),
]


def build_scores(panel):
    raw = _load(panel)
    out = {}
    for tag, ns, mono in CONFIGS:
        t0 = time.time()
        art = fit_transfer_scorer(panel, n_seeds=ns, monotone=mono)
        out[tag] = transfer_score(raw, art)
        print(f"  [{panel}] built {tag:11s} (n_seeds={ns:2d}, mono={mono:6s}) "
              f"{time.time()-t0:.0f}s", flush=True)
    return out


def run(panel, n_repeats):
    prior = TEST_PRIORS[panel]; n_target = FINALS_N[panel]; w = TRANSFER_PANELS[panel]
    print(f"\n{'='*64}\n  {panel.upper()}  (deployed w={w}, prior={prior})\n{'='*64}", flush=True)
    Xt, yt = load_target_engineered(panel)
    scores = build_scores(panel)
    for tag in scores:
        assert len(scores[tag]) == len(yt), f"row mismatch {tag}"
    fn = target_deployed_fn(panel)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED)
    f1 = {tag: [] for tag, _, _ in CONFIGS}
    f1b = {tag: [] for tag, _, _ in CONFIGS}
    t0 = time.time()
    for tr, te in rskf.split(Xt, yt):
        tr, te = np.asarray(tr), np.asarray(te)
        ytr, yte = yt.iloc[tr].reset_index(drop=True), yt.iloc[te].reset_index(drop=True)
        if w >= 1.0:
            e_tr = np.zeros(len(tr)); e_te = np.zeros(len(te))     # own model unused
        else:
            e_te, e_tr = fn(Xt.iloc[tr].reset_index(drop=True), ytr,
                            Xt.iloc[te].reset_index(drop=True))
        for tag, _, _ in CONFIGS:
            g = scores[tag]
            pl = fit_platt(g[tr], ytr.values)
            gt, ge = apply_platt(g[tr], pl), apply_platt(g[te], pl)
            b_tr = w * gt + (1 - w) * e_tr
            b_te = w * ge + (1 - w) * e_te
            thr = pick_threshold_at_prior(ytr.values, b_tr, prior)
            preds = (b_te >= thr).astype(int)
            tpr, fpr = tpr_fpr(yte.values, preds)
            f1[tag].append(f1_at_prior(tpr, fpr, prior))
            fb, _ = bootstrap_f1_mcc(yte.values, preds, prior, n_target)
            f1b[tag].append(fb)
    base = np.array(f1["baseline"]); nf = len(base)
    rows = []
    print(f"\n  paired vs baseline ({nf} folds, {time.time()-t0:.0f}s):", flush=True)
    print(f"  {'config':11s} {'F1*':>7s} {'finals':>7s} {'Δ':>9s} {'σ':>7s}  verdict", flush=True)
    for tag, ns, mono in CONFIGS:
        arr = np.array(f1[tag]); d = arr - base
        se = d.std(ddof=1) / np.sqrt(nf) if nf > 1 else 0.0
        sig = d.mean() / se if se > 0 else 0.0
        is_mono = mono != "none" and ns == 1
        if tag == "baseline":
            v = "ref"
        elif d.mean() > 0 and sig >= 2:
            v = "DEPLOY (>=2sigma)"
        elif is_mono and sig >= -0.5:
            v = "ship-robustness (>=-0.5sigma)"   # monotone exception per plan
        elif d.mean() > 0 and sig >= 1:
            v = "marginal"
        else:
            v = "no gain"
        tagstr = "" if tag == "baseline" else f"{d.mean():+9.4f} {sig:+7.2f}"
        print(f"  {tag:11s} {arr.mean():7.4f} {np.mean(f1b[tag]):7.4f} {tagstr}  {v}", flush=True)
        rows.append({"panel": panel, "config": tag, "n_seeds": ns, "monotone": mono,
                     "f1": arr.mean(), "finals": np.mean(f1b[tag]),
                     "delta": d.mean(), "sigma": sig, "verdict": v})
    return rows


if __name__ == "__main__":
    panels = sys.argv[1:] or ["pah", "cftr", "cancer"]
    all_rows = []
    for pn in panels:
        nr = 8 if pn == "cancer" else 12     # cancer own-model per fold is the cost
        all_rows.extend(run(pn, n_repeats=nr))
    pd.DataFrame(all_rows).to_csv(os.path.join(OUT, "exp_transfer_improve.csv"), index=False)
    print(f"\nsaved -> {os.path.join(OUT, 'exp_transfer_improve.csv')}", flush=True)
