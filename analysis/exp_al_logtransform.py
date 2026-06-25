"""Does log-transforming the AL_ allele-frequency block help?

Mechanistic prior (what we EXPECT and why):
  * The deployed models are all tree ensembles (LightGBM / RandomForest / CatBoost).
    Trees split on RANK, so any monotonic transform of a feature (log1p included) is
    invariant — it cannot change a pure GBDT's splits or predictions. Prediction:
    ~0 effect on GENERAL's native-NaN all-features recipe (and on the transfer scorer,
    which uses that same recipe).
  * Two places where log COULD nudge results, so we test rather than assume:
      - the small panels' OWN model uses mRMR, whose relevance (kNN mutual-info) and
        redundancy (Pearson) terms are NOT rank-invariant -> log can shift WHICH
        features are selected.
      - fit_keep_nan's corr>0.95 drop uses raw Pearson -> log can change which AL_
        columns are dropped.
  * Note: GENERAL's recipe already rank(pct)-transforms population-AF columns, so for
    those, log-before-rank is exactly a no-op by construction.

Honest per-fold CV, analytic F1* at the per-panel test prior, paired vs the deployed
baseline on identical folds (harness.compare). Run:
  .venv/bin/python -u analysis/exp_al_logtransform.py
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")

from harness import cv_evaluate, load_panel, compare, deployed_pipeline_fn
from exp_general_blend import lgbm_only_fn


def log_al(X: pd.DataFrame) -> pd.DataFrame:
    """log1p every numeric AL_ column with min >= 0 (preserves NaN; skips categoricals,
    presence flags stay {0,1}->{0,log2} monotonic, AN constant unaffected by rank models)."""
    X = X.copy()
    for c in X.columns:
        if not str(c).startswith("AL_"):
            continue
        if isinstance(X[c].dtype, pd.CategoricalDtype):
            continue
        col = pd.to_numeric(X[c], errors="coerce")
        mn = col.min(skipna=True)
        if pd.notna(mn) and mn >= 0:
            X[c] = np.log1p(col)
    return X


def with_log(fn):
    def _fn(X_tr, y_tr, X_te):
        return fn(log_al(X_tr), y_tr, log_al(X_te))
    return _fn


def run(panel, base_fn, n_repeats):
    df = load_panel(panel)
    base = cv_evaluate(panel, base_fn, label="baseline", n_repeats=n_repeats, df=df)
    logv = cv_evaluate(panel, with_log(base_fn), label="AL_log", n_repeats=n_repeats, df=df)
    c = compare(base, logv)
    print(f"{panel:8s} base F1*={c['a_f1']:.4f}  AL_log F1*={c['b_f1']:.4f}  "
          f"Δ={c['delta_f1']:+.4f} ({c['sigma']:+.2f}σ)  {c['verdict']}", flush=True)
    return c


if __name__ == "__main__":
    print("=== AL_ log1p transform — paired honest CV (F1* at test prior) ===", flush=True)
    # general: reworked native-NaN all-features recipe (the transfer SOURCE) -> expect no-op
    run("general", lgbm_only_fn, n_repeats=4)
    # small panels: their OWN deployed median+mRMR recipe -> where log COULD shift mRMR
    run("pah",    deployed_pipeline_fn("pah"),    n_repeats=8)
    run("cancer", deployed_pipeline_fn("cancer"), n_repeats=8)
    print("\nVerdict guide: |σ|<1 = NOISE (log is a no-op, as predicted for tree models).", flush=True)
