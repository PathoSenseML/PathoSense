"""STRICT re-verification of the pah importance-weighting gain (#1), removing the test-fold leak.

In exp_importance_weight.py the domain weights were derived from the FULL target distribution (incl. the
CV test folds). At finals the scorer is trained BEFORE seeing test variants, so weights must come from the
target-TRAIN portion only. Here, per outer fold:
  * domain classifier: general (0) vs TARGET-TRAIN-FOLD (1), features only -> weights on general.
  * REFIT the weighted scorer on general(-target-IDs) with those weights -> predict target-train (for
    Platt/threshold) and target-test (for eval). Unweighted scorer is frozen (fold-independent).
  * blend with own at deployed w, threshold on train-OOF at the prior, F1* on test fold. Paired.
If pah still gains, it's real and deployable; if it collapses, it was a leak artifact.
Run: .venv/bin/python -u analysis/exp_importance_weight_strict.py [panel ...]
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from config import SEED
from utils import TEST_PRIORS
from calibration import fit_platt, apply_platt
from harness import pick_threshold_at_prior, tpr_fpr, f1_at_prior
from run_pipeline import get_model_params
from exp_transfer import load_target_engineered, target_deployed_fn
from exp_transfer_blend import build_matrices

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
DEPLOY_W = {"cftr": 1.0, "pah": 1.0, "cancer": 0.5}
N_REPEATS = 10


def domain_weights(Xg, Xt_train, cats):
    X = pd.concat([Xg, Xt_train], ignore_index=True)
    d = np.r_[np.zeros(len(Xg)), np.ones(len(Xt_train))]
    i2, ies = train_test_split(np.arange(len(d)), test_size=0.2, random_state=SEED, stratify=d)
    m = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=SEED, n_jobs=4, verbosity=-1)
    m.fit(X.iloc[i2], d[i2], eval_set=[(X.iloc[ies], d[ies])],
          callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)], categorical_feature=cats or "auto")
    p = m.predict_proba(Xg)[:, 1]
    w = np.clip(p / np.clip(1 - p, 1e-3, None), 0.05, 20.0)
    return w * len(w) / w.sum()


def scorer(Xg, yg, Xpred, cats, weight=None):
    params = get_model_params(len(Xg), (yg == 0).sum() / max((yg == 1).sum(), 1))
    i2, ies = train_test_split(np.arange(len(yg)), test_size=0.15, random_state=SEED + 7, stratify=yg)
    m = LGBMClassifier(**params, random_state=SEED)
    m.fit(Xg.iloc[i2], yg.iloc[i2], sample_weight=(None if weight is None else weight[i2]),
          eval_set=[(Xg.iloc[ies], yg.iloc[ies])],
          callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)], categorical_feature=cats or "auto")
    return m.predict_proba(Xpred)[:, 1]


def run(panel):
    prior = TEST_PRIORS[panel]; w = DEPLOY_W[panel]
    Xg, yg, Xt, yt, cats = build_matrices(panel)         # Xt row-aligned with load_target_engineered
    Xt_own, yt2 = load_target_engineered(panel); fn = target_deployed_fn(panel)
    assert (yt.values == yt2.values).all()
    g_un_all = scorer(Xg, yg, Xt, cats, None)            # unweighted scorer is fold-independent
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=N_REPEATS, random_state=SEED)
    per = {"unweighted": [], "weighted_strict": []}
    t0 = time.time()
    for tr, te in rskf.split(Xt_own, yt2):
        tr, te = np.asarray(tr), np.asarray(te)
        ytr, yte = yt2.iloc[tr].reset_index(drop=True), yt2.iloc[te].reset_index(drop=True)
        e_te, e_tr = fn(Xt_own.iloc[tr].reset_index(drop=True), ytr, Xt_own.iloc[te].reset_index(drop=True))
        # weights from TARGET-TRAIN fold only -> refit weighted scorer -> predict train+test
        wt = domain_weights(Xg, Xt.iloc[tr].reset_index(drop=True), cats)
        g_w_tr = scorer(Xg, yg, Xt.iloc[tr].reset_index(drop=True), cats, wt)
        g_w_te = scorer(Xg, yg, Xt.iloc[te].reset_index(drop=True), cats, wt)
        for key, gtr, gte in (("unweighted", g_un_all[tr], g_un_all[te]),
                              ("weighted_strict", g_w_tr, g_w_te)):
            pl = fit_platt(gtr, ytr.values)
            b_tr = w * apply_platt(gtr, pl) + (1 - w) * e_tr
            b_te = w * apply_platt(gte, pl) + (1 - w) * e_te
            thr = pick_threshold_at_prior(ytr.values, b_tr, prior)
            tpr, fpr = tpr_fpr(yte.values, (b_te >= thr).astype(int))
            per[key].append(f1_at_prior(tpr, fpr, prior))
    base = np.array(per["unweighted"]); v = np.array(per["weighted_strict"]); d = v - base
    nf = len(base); se = d.std(ddof=1) / np.sqrt(nf); sig = d.mean() / se if se > 0 else 0.0
    vd = "DEPLOY" if d.mean() > 0 and sig >= 2 else "marginal" if d.mean() > 0 and sig >= 1 else "no gain"
    print(f"\n[{panel}] STRICT (no test-fold leak)  unweighted F1*={base.mean():.4f}  weighted F1*={v.mean():.4f}  "
          f"Δ={d.mean():+.4f} ({sig:+.2f}σ)  -> {vd}  ({nf} folds, {time.time()-t0:.0f}s)", flush=True)
    return {"panel": panel, "unweighted": base.mean(), "weighted_strict": v.mean(), "delta": d.mean(), "sigma": sig}


if __name__ == "__main__":
    panels = sys.argv[1:] or ["pah", "cancer", "cftr"]
    print("=== #1 STRICT — importance weighting, weights from target-TRAIN only ===", flush=True)
    rows = []
    for pn in panels:
        rows.append(run(pn))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "importance_weight_strict.csv"), index=False)
    print(f"\nsaved -> importance_weight_strict.csv", flush=True)
