"""Pressure-test conservation x AA-severity INTERACTION features on the general panel. Hypothesis: a
variant that is BOTH a severe substitution AND highly conserved is more pathogenic — an interaction trees
*should* learn via depth, but an explicit product MAY help if depth/data is limiting. Add products of top
conservation (EK_7/9/1/2 = phyloP/GERP) x top AA severity (Grantham, volume/polarity/MW deltas, BLOSUM),
honest harness, paired vs the deployed general LGBM. Interactions are deterministic feature products (no
label, no leakage). Run: .venv/bin/python -u analysis/exp_cons_aa_interactions.py
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "analysis"); sys.path.insert(0, "pipelines")
import warnings; warnings.filterwarnings("ignore")
from harness import cv_evaluate, load_panel, compare
from schema import partition_columns
from meeting_lib import native_nan_oof
from run_pipeline import get_model_params

df = load_panel("general"); _, NUM, CAT, TGT = partition_columns(df)
y = df[TGT].astype(int)
P = {**get_model_params(2000, (y == 0).sum() / max(y.sum(), 1)),
     "scale_pos_weight": 1, "bagging_freq": 1, "bagging_fraction": 0.8}
CONS = ["EK_7", "EK_9", "EK_1", "EK_2"]
SEV = ["AA_grantham", "AA_volume_delta", "AA_polarity_delta", "AA_mw_delta", "AA_blosum62"]


def base_fn(a, b, c):
    return native_nan_oof(a, b, c, params=P)


def make_ix(cons, sev, ratio=False):
    def post(Xtr, ytr, Xte, fit_idx):
        def add(X):
            X = X.copy()
            for ec in [c for c in cons if c in X.columns]:
                ev = pd.to_numeric(X[ec], errors="coerce")
                for ac in [c for c in sev if c in X.columns]:
                    av = pd.to_numeric(X[ac], errors="coerce")
                    X[f"IX_{ec}x{ac}"] = ev * av
                    if ratio:
                        X[f"IX_{ec}d{ac}"] = av / (ev.abs() + 1.0)
            return X
        return add(Xtr), add(Xte)
    def fn(a, b, c):
        return native_nan_oof(a, b, c, params=P, post=post)
    return fn


LEVERS = {
    "ix_grantham_x_cons":  make_ix(CONS, ["AA_grantham"]),
    "ix_all_sev_x_EK7":    make_ix(["EK_7"], SEV),
    "ix_all_sev_x_allcons": make_ix(CONS, SEV),
    "ix_all_sev_x_allcons+ratio": make_ix(CONS, SEV, ratio=True),
}


def main():
    nrep = 6
    base = cv_evaluate("general", base_fn, label="base", n_repeats=nrep, df=df, n_jobs=6)
    print(f"  {'base':28s} F1*={base.f1.mean():.4f} AUC={base.auc.mean():.4f}", flush=True)
    rows = [{"lever": "base", "f1": round(base.f1.mean(), 4), "auc": round(base.auc.mean(), 4), "sigma": 0.0}]
    for name, fn in LEVERS.items():
        r = cv_evaluate("general", fn, label=name, n_repeats=nrep, df=df, n_jobs=6)
        c = compare(base, r)
        print(f"  {name:28s} F1*={r.f1.mean():.4f} AUC={r.auc.mean():.4f} "
              f"ΔF1={c['delta_f1']:+.4f} ({c['sigma']:+.2f}σ) ΔAUC={r.auc.mean()-base.auc.mean():+.4f} [{c['verdict']}]", flush=True)
        rows.append({"lever": name, "f1": round(r.f1.mean(), 4), "auc": round(r.auc.mean(), 4),
                     "d_f1": c["delta_f1"], "sigma": c["sigma"], "verdict": c["verdict"]})
    pd.DataFrame(rows).to_csv("analysis/output/cons_aa_interactions.csv", index=False)
    print("\nsaved -> analysis/output/cons_aa_interactions.csv", flush=True)


if __name__ == "__main__":
    main()
