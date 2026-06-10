"""Honest evaluation harness for PathoSense — the single source of truth for
comparing pipeline configurations.

Design goals (why this exists):
  * The competition test set is NOT released. We can only estimate the score we
    will get at finals. The test set is deliberately Benign-dominant (per-panel
    priors in utils.TEST_PRIORS), so we must score F1 / MCC AT THE TEST PRIOR,
    not at the training prior (which inflates F1 by 6-25 points).
  * Every stateful step (leakage drop, preprocessing, feature selection, Platt,
    threshold) MUST be fit on the TRAIN folds only and applied to the held-out
    fold — anything else leaks and manufactures false wins (the recurring lesson
    in this project's memory).
  * Single-split comparisons manufacture false winners. We use REPEATED stratified
    K-fold and PAIRED deltas (identical folds for both configs) so a config-vs-config
    comparison cancels fold-to-fold variance.

Metric — analytic F1/MCC at a target prior:
    Given a fixed threshold, a held-out fold yields TPR (recall on positives) and
    FPR (fall-out on negatives). These are prior-independent. The confusion matrix
    at ANY target prior π (for a notional N samples) is then
        TP = π·TPR,  FN = π·(1−TPR),  FP = (1−π)·FPR,  TN = (1−π)·(1−FPR)
    and F1 / MCC computed from that are EXACT functions of (TPR, FPR, π) — no
    bootstrap noise. This uses the whole fold to estimate TPR/FPR, so it is
    lower-variance than the deployed bootstrap-resample estimator while targeting
    the same quantity. (We can still cross-check against the bootstrap estimator.)

Threshold selection: swept on TRAIN out-of-fold predictions to maximize analytic
F1 at the target prior, then frozen and applied to the test fold. Honest.

Usage:
    from harness import cv_evaluate, load_panel, deployed_pipeline_fn, compare
    res = cv_evaluate("general", deployed_pipeline_fn())
    print(res.summary())
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb

from config import SEED
from schema import (normalize_sentinels, to_categorical, partition_columns,
                    drop_all_nan_rows)
from feature_engineering import (engineer_aa_features, engineer_acmg_features,
                                 engineer_interaction_features,
                                 fit_score_agreement, apply_score_agreement)
from calibration import fit_platt, apply_platt
from utils import (detect_leaky_columns, KNOWN_LEAKY, PANELS, TEST_PRIORS,
                   ACMG_PANELS)

import warnings
warnings.filterwarnings("ignore")

# LightGBM thread count, injected into every booster. Set to 1 by cv_evaluate when
# folds run in parallel (avoids oversubscription); -1 (all cores) when sequential.
_LGB_THREADS = -1


# ── data loading ────────────────────────────────────────────────────────────

_PANEL_CACHE: dict = {}

def load_panel(panel: str, acmg: Optional[bool] = None,
               interactions: bool = False, score_agree: bool = False) -> pd.DataFrame:
    """Load a panel exactly as the deployed pipeline does at LOAD time:
    sentinel-normalize -> drop all-NaN rows -> cast categoricals -> engineer AA.
    ACMG/interactions toggles match the per-panel deployed config when acmg=None.
    (score_agree is fitted per-fold, never here — it needs labels.)
    """
    key = (panel, acmg, interactions)
    if key in _PANEL_CACHE:
        return _PANEL_CACHE[key].copy()
    df = pd.read_csv(PANELS[panel])
    df = normalize_sentinels(df)
    df = drop_all_nan_rows(df, verbose=False)
    df = to_categorical(df)
    df = engineer_aa_features(df)
    use_acmg = (panel in ACMG_PANELS) if acmg is None else acmg
    if use_acmg:
        df = engineer_acmg_features(df)
    if interactions:
        df = engineer_interaction_features(df)
    _PANEL_CACHE[key] = df.copy()
    return df


# ── analytic prior-adjusted metrics ─────────────────────────────────────────

def tpr_fpr(y_true: np.ndarray, preds: np.ndarray) -> tuple:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(preds).astype(int)
    pos = y == 1; neg = y == 0
    tpr = float(p[pos].mean()) if pos.any() else 0.0
    fpr = float(p[neg].mean()) if neg.any() else 0.0
    return tpr, fpr


# Actual finals test-set sizes per panel (competition spec: P+B counts). The
# finals F1 is computed on ONE set of this size at the panel prior. Bootstrapping a
# held-out fold to THIS size (not the fold size) is the most honest finals estimate;
# the established team harness resampled to fold size, which inflated cftr/pah.
FINALS_N = {"general": 3500, "cancer": 600, "pah": 350, "cftr": 120}

from sklearn.metrics import f1_score as _f1, matthews_corrcoef as _mcc

def bootstrap_f1_mcc(y_true, preds, prior, n_target, n_boot=300, seed=SEED):
    """Mean F1/MCC over n_boot resamples of (y,preds) to n_target samples at `prior`."""
    y = np.asarray(y_true).astype(int); p = np.asarray(preds).astype(int)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0, 0.0
    n_pos = max(1, int(round(n_target * prior))); n_neg = max(1, n_target - n_pos)
    rng = np.random.RandomState(seed)
    f1s, mccs = [], []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, n_pos, replace=True),
                              rng.choice(neg, n_neg, replace=True)])
        f1s.append(_f1(y[idx], p[idx], zero_division=0)); mccs.append(_mcc(y[idx], p[idx]))
    return float(np.mean(f1s)), float(np.mean(mccs))


def f1_at_prior(tpr: float, fpr: float, prior: float) -> float:
    """Exact F1 at target prior given recall (tpr) and fall-out (fpr)."""
    tp = prior * tpr
    fp = (1 - prior) * fpr
    fn = prior * (1 - tpr)
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else 0.0


def mcc_at_prior_analytic(tpr: float, fpr: float, prior: float) -> float:
    tp = prior * tpr
    fn = prior * (1 - tpr)
    fp = (1 - prior) * fpr
    tn = (1 - prior) * (1 - fpr)
    num = tp * tn - fp * fn
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float(num / den) if den > 0 else 0.0


def pick_threshold_at_prior(y_train: np.ndarray, oof_probs: np.ndarray,
                            prior: float, step: float = 0.005) -> float:
    """Threshold maximizing analytic F1 at the target prior on TRAIN OOF probs."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.02, 0.98, step):
        tpr, fpr = tpr_fpr(y_train, (oof_probs >= t).astype(int))
        f1 = f1_at_prior(tpr, fpr, prior)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


# ── result container ─────────────────────────────────────────────────────────

@dataclass
class CVResult:
    panel: str
    label: str
    f1: np.ndarray          # per-fold analytic F1 at prior (decision metric)
    mcc: np.ndarray         # per-fold analytic MCC at prior
    auc: np.ndarray
    ap: np.ndarray
    prior: float
    seconds: float
    f1_boot: Optional[np.ndarray] = None  # per-fold F1 bootstrapped to finals size

    def summary(self) -> dict:
        n = len(self.f1)
        out = {
            "panel": self.panel, "label": self.label, "n_folds": n,
            "f1_mean": round(float(self.f1.mean()), 4),
            "f1_se":   round(float(self.f1.std(ddof=1) / np.sqrt(n)), 4),
            "mcc_mean": round(float(self.mcc.mean()), 4),
            "auc_mean": round(float(self.auc.mean()), 4),
            "ap_mean":  round(float(self.ap.mean()), 4),
            "secs": round(self.seconds, 1),
        }
        if self.f1_boot is not None:
            out["f1_finals"] = round(float(self.f1_boot.mean()), 4)
        return out


def compare(a: CVResult, b: CVResult) -> dict:
    """Paired comparison b−a on identical folds. Returns delta mean, SE, sigma."""
    assert len(a.f1) == len(b.f1), "configs must use identical folds"
    d = b.f1 - a.f1
    n = len(d)
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    mean = float(d.mean())
    sigma = (mean / se) if se > 0 else 0.0
    dm = b.mcc - a.mcc
    return {
        "delta_f1": round(mean, 4),
        "se": round(se, 4),
        "sigma": round(sigma, 2),
        "delta_mcc": round(float(dm.mean()), 4),
        "a_f1": round(float(a.f1.mean()), 4),
        "b_f1": round(float(b.f1.mean()), 4),
        "verdict": _verdict(mean, sigma),
    }


def _verdict(mean: float, sigma: float) -> str:
    if mean <= 0:
        return "REJECT (negative)"
    if sigma >= 2.0:
        return "DEPLOY (>=2sigma)"
    if sigma >= 1.5:
        return "DEPLOY (>=1.5sigma)"
    if sigma >= 1.0:
        return "MARGINAL (1-1.5sigma)"
    return "NOISE (<1sigma)"


# ── the core CV loop ──────────────────────────────────────────────────────────

def _eval_one_fold(X_all, y_all, tr_idx, te_idx, pipeline_fn, prior, n_target=0):
    X_tr = X_all.iloc[tr_idx].reset_index(drop=True)
    X_te = X_all.iloc[te_idx].reset_index(drop=True)
    y_tr = y_all.iloc[tr_idx].reset_index(drop=True)
    y_te = y_all.iloc[te_idx].reset_index(drop=True)
    test_probs, train_oof = pipeline_fn(X_tr, y_tr, X_te)
    t = pick_threshold_at_prior(y_tr.values, train_oof, prior)
    preds = (test_probs >= t).astype(int)
    tpr, fpr = tpr_fpr(y_te.values, preds)
    f1 = f1_at_prior(tpr, fpr, prior)
    mcc = mcc_at_prior_analytic(tpr, fpr, prior)
    f1b, _ = bootstrap_f1_mcc(y_te.values, preds, prior, n_target) if n_target else (f1, mcc)
    try:
        auc = roc_auc_score(y_te, test_probs); ap = average_precision_score(y_te, test_probs)
    except Exception:
        auc = 0.5; ap = float(y_te.mean())
    return f1, mcc, auc, ap, f1b


def cv_evaluate(panel: str, pipeline_fn: Callable, *,
                label: str = "", n_splits: int = 5, n_repeats: int = 6,
                seed: int = SEED, prior: Optional[float] = None,
                df: Optional[pd.DataFrame] = None, n_jobs: int = 6) -> CVResult:
    """Run repeated stratified K-fold, scoring analytic F1/MCC at the test prior.

    pipeline_fn(X_tr, y_tr, X_te) -> (test_probs, train_oof_probs)
        Must fit EVERY stateful step on (X_tr, y_tr) only. train_oof_probs are the
        out-of-fold predictions on the train rows (aligned to X_tr order), used to
        choose the threshold honestly. Both are 1-D float arrays.
    n_jobs: folds evaluated in parallel (threading backend; LightGBM forced to 1
        thread to avoid oversubscription). Set 1 for sequential / full LGBM threads.
    """
    global _LGB_THREADS
    if prior is None:
        prior = TEST_PRIORS.get(panel, 0.5)
    if df is None:
        df = load_panel(panel)
    _, num_cols, cat_cols, tgt = partition_columns(df)
    feat_cols = num_cols + cat_cols
    X_all = df[feat_cols].reset_index(drop=True)
    y_all = df[tgt].reset_index(drop=True).astype(int)

    n_target = FINALS_N.get(panel, 0)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                   random_state=seed)
    splits = list(rskf.split(X_all, y_all))
    t0 = time.time()

    if n_jobs and n_jobs > 1:
        from joblib import Parallel, delayed
        prev = _LGB_THREADS
        _LGB_THREADS = 1
        try:
            out = Parallel(n_jobs=n_jobs, backend="threading")(
                delayed(_eval_one_fold)(X_all, y_all, tr, te, pipeline_fn, prior, n_target)
                for tr, te in splits)
        finally:
            _LGB_THREADS = prev
    else:
        out = [_eval_one_fold(X_all, y_all, tr, te, pipeline_fn, prior, n_target)
               for tr, te in splits]

    f1s, mccs, aucs, aps, f1bs = (np.array(x) for x in zip(*out))
    return CVResult(panel, label or "config", f1s, mccs, aucs, aps,
                    prior, time.time() - t0, f1_boot=f1bs)


# ── reusable pipeline building blocks ────────────────────────────────────────

def _drop_leaky(X_tr, y_tr, X_te):
    leaky = detect_leaky_columns(X_tr, y_tr, auc_threshold=0.99, verbose=False)
    all_leaky = sorted(set(leaky) | (KNOWN_LEAKY & set(X_tr.columns)))
    if all_leaky:
        X_tr = X_tr.drop(columns=all_leaky)
        X_te = X_te.drop(columns=[c for c in all_leaky if c in X_te.columns])
    return X_tr, X_te


def lgb_oof_and_test(X_tr, y_tr, X_te, params, n_seeds=3, n_folds=5):
    """Multi-seed LightGBM: returns (test_probs averaged, train OOF averaged)."""
    from sklearn.model_selection import train_test_split
    patience = 50 if len(y_tr) > 300 else 30
    n_folds = min(n_folds, max(2, int(y_tr.sum()), int((y_tr == 0).sum())))
    oof_all = np.zeros((n_seeds, len(y_tr)))
    test_all = np.zeros((n_seeds, len(X_te)))
    for i in range(n_seeds):
        seed_i = SEED + i
        p = {**params, "random_state": seed_i, "n_jobs": _LGB_THREADS}
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed_i)
        oof = np.zeros(len(y_tr))
        for a, b in skf.split(X_tr, y_tr):
            m = lgb.LGBMClassifier(**p)
            m.fit(X_tr.iloc[a], y_tr.iloc[a],
                  eval_set=[(X_tr.iloc[b], y_tr.iloc[b])],
                  callbacks=[lgb.early_stopping(patience, verbose=False),
                             lgb.log_evaluation(0)])
            oof[b] = m.predict_proba(X_tr.iloc[b])[:, 1]
        oof_all[i] = oof
        X_a, X_es, y_a, y_es = train_test_split(
            X_tr, y_tr, test_size=0.1, random_state=seed_i, stratify=y_tr)
        final = lgb.LGBMClassifier(**p)
        final.fit(X_a, y_a, eval_set=[(X_es, y_es)],
                  callbacks=[lgb.early_stopping(patience, verbose=False),
                             lgb.log_evaluation(0)])
        test_all[i] = final.predict_proba(X_te)[:, 1]
    return test_all.mean(0), oof_all.mean(0)


def fast_mrmr(X_train, y_train, X_val, X_test, max_features=0):
    """Drop-in, OUTPUT-IDENTICAL replacement for run_pipeline.select_features_mrmr
    that precomputes the |correlation| matrix once instead of recomputing pairwise
    correlations inside the greedy loop (the deployed version is O(k·m·n); this is
    O(m·n) for the matrix + O(k·m) lookups). 74s/fold -> <1s/fold on general.

    Same relevance (mutual_info_classif), same redundancy (mean |Pearson| to the
    already-selected set), same tie-breaking (max() picks the first max), so it
    selects the identical feature set the deployed pipeline would.
    """
    from sklearn.feature_selection import mutual_info_classif
    n = len(X_train)
    if max_features <= 0:
        max_features = 20 if n < 200 else (40 if n < 500 else 60)
        max_features = min(max_features, len(X_train.columns))
    n_before = len(X_train.columns)
    if n_before <= max_features:
        return X_train, X_val, X_test, {"method": "none_needed",
                                        "n_before": n_before, "n_after": n_before}
    # encode (categoricals -> codes), preserving column order
    cols = list(X_train.columns)
    enc = np.empty((n, n_before), dtype=float)
    discrete = []
    for j, c in enumerate(cols):
        s = X_train[c]
        if isinstance(s.dtype, pd.CategoricalDtype):
            enc[:, j] = s.cat.codes.to_numpy()
            discrete.append(True)
        else:
            enc[:, j] = s.to_numpy(dtype=float)
            discrete.append(False)
    # mutual_info_classif / corrcoef require finite input. The deployed pipeline
    # imputes before mRMR, so enc is normally complete; but when callers leave NaN
    # (native-NaN configs), impute a COPY here for the MI/corr math only. The
    # SELECTION (column names) is what is returned, preserving the original NaN.
    if not np.isfinite(enc).all():
        col_med = np.nanmedian(np.where(np.isfinite(enc), enc, np.nan), axis=0)
        col_med = np.where(np.isfinite(col_med), col_med, 0.0)
        bad = ~np.isfinite(enc)
        enc[bad] = np.take(col_med, np.where(bad)[1])
    mi = mutual_info_classif(enc, y_train, discrete_features=discrete,
                             random_state=SEED, n_neighbors=max(3, min(5, n // 30)))
    # |Pearson| matrix once (encoded, already imputed -> no NaN). corrwith uses
    # pandas pairwise; np.corrcoef matches for complete data.
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.abs(np.corrcoef(enc, rowvar=False))
    C[np.isnan(C)] = 0.0
    selected = [int(np.argmax(mi))]
    cand = [j for j in range(n_before) if j != selected[0]]
    while len(selected) < max_features and cand:
        red = C[np.ix_(cand, selected)].mean(axis=1)
        score = mi[cand] - red
        best = cand[int(np.argmax(score))]
        selected.append(best)
        cand.remove(best)
    names = [cols[j] for j in selected]
    info = {"method": "mRMR_fast", "n_before": n_before, "n_after": len(names),
            "top_5": {cols[j]: round(float(mi[j]), 4) for j in selected[:5]}}
    return X_train[names], X_val[names], X_test[names], info


def deployed_pipeline_fn(panel: str, use_platt: bool = True,
                         use_score_agree: Optional[bool] = None,
                         n_seeds: int = 3):
    """Pipeline function replicating the DEPLOYED chain (per-fold, honest):
       leakage drop -> preprocess_data -> [score-agreement] -> mRMR -> multi-seed
       LGBM OOF+test -> Platt(train OOF). Returns (test_probs, train_oof) calibrated.
    """
    from run_pipeline import preprocess_data, get_model_params
    cfg_sa = use_score_agree
    if cfg_sa is None:
        # mirror BEST_CONFIG_PER_PANEL
        cfg_sa = panel in ("cancer", "cftr")

    def _fn(X_tr, y_tr, X_te):
        Xtr, Xte = _drop_leaky(X_tr, y_tr, X_te)
        Xtr_pp, _, Xte_pp, _ = preprocess_data(Xtr.copy(), Xtr.copy(), Xte.copy(),
                                               corr_threshold=0.95)
        if cfg_sa:
            sa = fit_score_agreement(Xtr_pp, y_tr, top_k=7)
            Xtr_pp = apply_score_agreement(Xtr_pp, sa)
            Xte_pp = apply_score_agreement(Xte_pp, sa)
        Xtr_pp, _, Xte_pp, _ = fast_mrmr(Xtr_pp, y_tr, Xte_pp, Xte_pp)
        spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        params = get_model_params(len(X_tr), spw)
        nf = 3 if len(X_tr) < 200 else 5
        test_probs, oof = lgb_oof_and_test(Xtr_pp, y_tr, Xte_pp, params,
                                           n_seeds=n_seeds, n_folds=nf)
        if use_platt:
            pf = fit_platt(oof, y_tr.values)
            return apply_platt(test_probs, pf), apply_platt(oof, pf)
        return test_probs, oof
    return _fn


if __name__ == "__main__":
    panels = sys.argv[1:] or ["general", "cancer", "pah", "cftr"]
    print(f"{'panel':8s} {'F1*ana':>8s} {'SE':>7s} {'F1finals':>9s} {'MCC*':>7s} "
          f"{'AUC':>7s} {'AP':>7s} {'secs':>6s}")
    for pn in panels:
        r = cv_evaluate(pn, deployed_pipeline_fn(pn), label="deployed",
                        n_repeats=6 if pn == "general" else 10)
        s = r.summary()
        print(f"{pn:8s} {s['f1_mean']:8.4f} {s['f1_se']:7.4f} {s.get('f1_finals',0):9.4f} "
              f"{s['mcc_mean']:7.4f} {s['auc_mean']:7.4f} {s['ap_mean']:7.4f} {s['secs']:6.1f}")
