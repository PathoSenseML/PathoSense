"""AUDIT FOLLOW-UP: is detect_leaky_columns' missingness rule (drop a col if all >10 missing samples share
a label) over-aggressive — dropping informative AL_ columns by base-rate chance rather than real leakage?
v2.0's AL_ is transformed (the old AF-missingness leak is gone), so this rule may suppress signal.
(1) diagnose: how many cols each rule drops on general/cancer + are they informative.
(2) test: F1*/AUC with vs without the missingness rule, honest harness, paired.
Run: .venv/bin/python -u analysis/exp_leakdrop_audit.py
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "analysis"); sys.path.insert(0, "pipelines")
import warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
import harness, utils
from harness import cv_evaluate, load_panel, compare
from schema import partition_columns
from meeting_lib import native_nan_oof
from run_pipeline import get_model_params

ORIG = utils.detect_leaky_columns


def detect_no_miss(X, y, auc_threshold=0.99, min_value_count=10, verbose=False):
    """utils.detect_leaky_columns WITHOUT the missingness rule (keep AUC + constant + cat P-rate)."""
    leaky = []
    for col in X.columns:
        vals = X[col]; is_cat = isinstance(vals.dtype, pd.CategoricalDtype)
        if is_cat:
            vm = vals.notna()
            if vm.sum() < 20: continue
            sub = pd.DataFrame({"v": vals[vm].astype(object), "y": y[vm].values})
            for value, g in sub.groupby("v"):
                if len(g) < min_value_count: continue
                if g["y"].mean() >= 0.99 or g["y"].mean() <= 0.01: leaky.append(col); break
            continue
        vm = vals.notna()
        if vm.sum() < 20 or y[vm].nunique() < 2: continue
        try:
            a = roc_auc_score(y[vm].values, vals[vm].values); a = max(a, 1 - a)
            if a > auc_threshold: leaky.append(col); continue
        except Exception: pass
        if vals.std() < 1e-8: leaky.append(col)
    return leaky


def diagnose(panel):
    df = load_panel(panel); _, num, cat, tgt = partition_columns(df)
    X = df[num + cat]; y = df[tgt].astype(int)
    full = set(ORIG(X, y)); nomiss = set(detect_no_miss(X, y))
    miss_only = full - nomiss
    al_miss = [c for c in miss_only if str(c).startswith("AL_")]
    aucs = []
    for c in al_miss:
        v = pd.to_numeric(X[c], errors="coerce"); m = v.notna()
        if m.sum() >= 20 and y[m].nunique() == 2:
            try: aucs.append(max(roc_auc_score(y[m], v[m]), 1 - roc_auc_score(y[m], v[m])))
            except Exception: pass
    print(f"[{panel}] full leak-drop={len(full)}  no-miss={len(nomiss)}  "
          f"dropped ONLY by missingness rule={len(miss_only)} ({len(al_miss)} are AL_)", flush=True)
    if aucs:
        print(f"          those AL_ cols' single-feature |AUC-0.5|: mean {np.mean(aucs):.3f} max {np.max(aucs):.3f} "
              f"({sum(a>0.55 for a in aucs)} have AUC>0.55 = informative)", flush=True)


def base_fn(P):
    return lambda a, b, c: native_nan_oof(a, b, c, params=P)


def main():
    for p in ("general", "cancer"):
        diagnose(p)
    print(flush=True)
    for panel in ("general", "cancer"):
        df = load_panel(panel); _, num, cat, tgt = partition_columns(df); y = df[tgt].astype(int)
        spw = (y == 0).sum() / max(y.sum(), 1)
        P = ({**get_model_params(2000, spw), "scale_pos_weight": 1, "bagging_freq": 1, "bagging_fraction": 0.8}
             if panel == "general" else get_model_params(len(y), spw))
        nrep = 6 if panel == "general" else 10
        utils.detect_leaky_columns = ORIG; harness.detect_leaky_columns = ORIG
        base = cv_evaluate(panel, base_fn(P), label="base", n_repeats=nrep, df=df, n_jobs=6)
        harness.detect_leaky_columns = detect_no_miss               # patch out the missingness rule
        nm = cv_evaluate(panel, base_fn(P), label="no_miss_rule", n_repeats=nrep, df=df, n_jobs=6)
        harness.detect_leaky_columns = ORIG                          # restore
        c = compare(base, nm)
        print(f"[{panel}] base F1*={base.f1.mean():.4f} AUC={base.auc.mean():.4f} | "
              f"keep-missingness-cols F1*={nm.f1.mean():.4f} AUC={nm.auc.mean():.4f} "
              f"ΔF1={c['delta_f1']:+.4f} ({c['sigma']:+.2f}σ) ΔAUC={nm.auc.mean()-base.auc.mean():+.4f} [{c['verdict']}]", flush=True)


if __name__ == "__main__":
    main()
