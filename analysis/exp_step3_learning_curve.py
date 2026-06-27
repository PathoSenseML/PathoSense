"""STEP 3 — Learning curve. AUC + max-F1 (@ test prior 0.143) vs training-set size, on a fixed held-out
set. Rising at 100% -> more/cleaner data lifts the ceiling. Flat -> it's features/labels, not data.
LGBM (deployed params), native-NaN preprocessing fit per training subset, 4 seeds for stability.
Run: .venv/bin/python -u analysis/exp_step3_learning_curve.py
"""
import sys, os, warnings
sys.path.insert(0, "analysis"); sys.path.insert(0, "pipelines")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from config import SEED
from schema import normalize_sentinels, partition_columns, drop_all_nan_rows
from utils import detect_leaky_columns, KNOWN_LEAKY, TEST_PRIORS, PANELS
from predict import _engineer, _fit_own_preprocess, _apply_own_preprocess, LGBM_PARAM_OVERRIDE
from recipe import PANEL_RECIPE
from models import get_model_params
from general_blend import _cats
from harness import f1_at_prior, tpr_fpr

PRIOR = TEST_PRIORS["general"]
FRACS = [0.25, 0.40, 0.55, 0.70, 0.85, 1.0]


def f1max(y, p):
    return max(f1_at_prior(*tpr_fpr(y, (p >= q).astype(int)), PRIOR) for q in np.quantile(p, np.arange(.02, .98, .005)))


def main():
    df = pd.read_csv(PANELS["general"]); df = normalize_sentinels(df); df = drop_all_nan_rows(df)
    cfg = PANEL_RECIPE["general"]
    de = _engineer(df.copy(), cfg); _, num, cat, tgt = partition_columns(de)
    y = de[tgt].astype(int).reset_index(drop=True); X = de[num + cat].reset_index(drop=True)
    leaky = detect_leaky_columns(X[num], y, 0.99, verbose=False)
    X = X.drop(columns=sorted(set(leaky) | (KNOWN_LEAKY & set(X.columns))))

    res = {f: {"auc": [], "f1": []} for f in FRACS}
    for seed in (SEED, SEED + 1000, SEED + 2000, SEED + 3000):
        Xtv, Xte, ytv, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
        Xtv = Xtv.reset_index(drop=True); ytv = ytv.reset_index(drop=True)
        Xte = Xte.reset_index(drop=True); yte = yte.reset_index(drop=True)
        for f in FRACS:
            if f < 1.0:
                Xs, _, ys, _ = train_test_split(Xtv, ytv, train_size=f, random_state=seed, stratify=ytv)
                Xs = Xs.reset_index(drop=True); ys = ys.reset_index(drop=True)
            else:
                Xs, ys = Xtv, ytv
            Xpp, prep = _fit_own_preprocess(Xs, ys, cfg)
            Xte_pp = _apply_own_preprocess(Xte, prep).reindex(columns=Xpp.columns)
            cf = _cats(Xpp)
            params = {**get_model_params(len(Xpp), 1.0), **LGBM_PARAM_OVERRIDE["general"], "verbose": -1}
            m = lgb.LGBMClassifier(**params); m.fit(Xpp, ys, categorical_feature=cf or "auto")
            p = m.predict_proba(Xte_pp)[:, 1]
            res[f]["auc"].append(roc_auc_score(yte.values, p)); res[f]["f1"].append(f1max(yte.values, p))
        print(f"  seed {seed} done", flush=True)

    print(f"\n=== STEP 3 — learning curve (general, LGBM, 4 seeds, held-out 20%) ===")
    print(f"  {'n_train':>8} {'AUC':>8} {'maxF1@0.143':>12}")
    prev = None
    for f in FRACS:
        a = np.mean(res[f]["auc"]); fm = np.mean(res[f]["f1"]); ntr = int(f * len(X) * 0.8)
        slope = "" if prev is None else f"  (ΔAUC {a-prev:+.4f})"
        print(f"  {ntr:>8} {a:>8.4f} {fm:>12.4f}{slope}"); prev = a
    # is it still rising at the top?
    top = np.mean(res[FRACS[-1]]["auc"]); prev2 = np.mean(res[FRACS[-2]]["auc"])
    print(f"\n  top-step ΔAUC ({int(FRACS[-2]*100)}%→100%) = {top-prev2:+.4f}  -> "
          f"{'STILL RISING (more data would help)' if top-prev2 > 0.003 else 'FLAT (at the data ceiling; lever is features/labels)'}")
    print("DONE")


if __name__ == "__main__":
    main()
