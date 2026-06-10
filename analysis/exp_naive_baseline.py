"""Naive 'before' baseline: raw data, NO preprocessing — just enough to make a model
run (Variant_ID dropped, categoricals -> integer codes), a default LightGBM, no
leakage removal, no transforms, no feature engineering, no calibration. Scored on the
SAME honest harness (analytic F1 at the per-panel test prior) so it is directly
comparable to the reworked pipeline.

Two naive variants: with leaky columns IN (truly no preprocessing — but the AF-
missingness leak inflates it) and with leaky columns removed (isolates real
preprocessing value). Also the trivial all-positive floor.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier

import harness
from harness import cv_evaluate, _drop_leaky
from schema import normalize_sentinels, to_categorical
from utils import PANELS, TEST_PRIORS


def raw_load(panel):
    df = pd.read_csv(PANELS[panel])
    df = normalize_sentinels(df)
    df = to_categorical(df)      # NO AA engineering, NO drop-all-nan, NO leakage removal
    return df


def naive_fn(drop_leaky=False):
    def _fn(X_tr, y_tr, X_te):
        Xtr, Xte = X_tr.copy(), X_te.copy()
        if drop_leaky:
            Xtr, Xte = _drop_leaky(Xtr, y_tr, Xte)
        for col in list(Xtr.columns):
            if isinstance(Xtr[col].dtype, pd.CategoricalDtype):
                cats = Xtr[col].cat.categories
                Xtr[col] = Xtr[col].cat.codes
                Xte[col] = pd.Categorical(Xte[col], categories=cats).codes
        nj = harness._LGB_THREADS
        m = LGBMClassifier(random_state=42, verbose=-1, n_jobs=nj)
        m.fit(Xtr, y_tr)
        tp = m.predict_proba(Xte)[:, 1]
        oof = np.zeros(len(y_tr))
        nf = 3 if len(y_tr) < 200 else 5
        for a, b in StratifiedKFold(nf, shuffle=True, random_state=0).split(Xtr, y_tr):
            mm = LGBMClassifier(random_state=42, verbose=-1, n_jobs=nj)
            mm.fit(Xtr.iloc[a], y_tr.iloc[a])
            oof[b] = mm.predict_proba(Xtr.iloc[b])[:, 1]
        return tp, oof
    return _fn


if __name__ == "__main__":
    REWORKED = {"general": 0.530, "cancer": 0.696, "pah": 0.660, "cftr": 0.797}
    DEPLOYED_HONEST = {"general": 0.49, "cancer": 0.665, "pah": 0.549, "cftr": 0.529}
    print(f"{'panel':8s} {'prior':>6s} {'all-pos':>8s} {'naive-raw':>10s} {'naive-noleak':>13s} "
          f"{'deployed':>9s} {'REWORKED':>9s}")
    for pn in ["general", "cancer", "pah", "cftr"]:
        df = raw_load(pn); prior = TEST_PRIORS[pn]
        allpos = 2 * prior / (1 + prior)
        rep = 6 if pn == "general" else 10
        r_raw = cv_evaluate(pn, naive_fn(False), n_repeats=rep, df=df).summary()["f1_mean"]
        r_nol = cv_evaluate(pn, naive_fn(True), n_repeats=rep, df=df).summary()["f1_mean"]
        print(f"{pn:8s} {prior:6.3f} {allpos:8.3f} {r_raw:10.3f} {r_nol:13.3f} "
              f"{DEPLOYED_HONEST[pn]:9.3f} {REWORKED[pn]:9.3f}", flush=True)
