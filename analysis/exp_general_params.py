"""General is the lowest panel AND the transfer source (double value). It's binormal-F1-ceiling-limited
by AUC, so target AUC. Test fresh LGBM-param levers on the general LGBM (the dominant blend component +
the transfer-scorer base), honest CV, AUC + F1* at prior. Levers:
  spw1        : scale_pos_weight=1 (rework found +0.017/1.6σ; NOT deployed)
  bagging     : bagging_freq=1 + bagging_fraction=0.8 (the deployed subsample=0.8 is INERT without it)
  spw1_bag    : both
  leaves63    : num_leaves=63, max_depth=-1 (more capacity for the 518-col matrix)
  slow        : lr=0.01, n_estimators=4000 (slower/deeper fit)
  lessreg     : reg_alpha=0, reg_lambda=0.5
Paired vs the deployed default (params=None). 2 seeds (n_repeats=8 each). Report AUC Δ (binding) + F1* Δ.
Run: .venv/bin/python -u analysis/exp_general_params.py
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
from config import SEED
from harness import cv_evaluate, load_panel, compare
from run_pipeline import get_model_params
from meeting_lib import native_nan_oof

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

df = load_panel("general")
y = df[[c for c in df.columns if c.lower() in ("label", "target")][0]] if False else None
# spw ratio for general (benign/pathogenic) on full panel
from schema import partition_columns
_, num, cat, tgt = partition_columns(df)
yv = df[tgt].astype(int)
SPW = (yv == 0).sum() / max((yv == 1).sum(), 1)
print(f"general spw(ratio)={SPW:.3f}  n={len(yv)}  P={int(yv.sum())} B={int((yv==0).sum())}", flush=True)
BASE = get_model_params(2000, SPW)          # large-tier deployed params

VARIANTS = {
    "spw1":     {**BASE, "scale_pos_weight": 1.0},
    "bagging":  {**BASE, "bagging_freq": 1, "bagging_fraction": 0.8},
    "spw1_bag": {**BASE, "scale_pos_weight": 1.0, "bagging_freq": 1, "bagging_fraction": 0.8},
    "leaves63": {**BASE, "num_leaves": 63, "max_depth": -1},
    "slow":     {**BASE, "learning_rate": 0.01, "n_estimators": 4000},
    "lessreg":  {**BASE, "reg_alpha": 0.0, "reg_lambda": 0.5},
}


def mk(params):
    def fn(Xtr, ytr, Xte):
        return native_nan_oof(Xtr, ytr, Xte, params=params)
    return fn


if __name__ == "__main__":
    NR = 8
    base = cv_evaluate("general", mk(BASE), label="base", n_repeats=NR, df=df)
    print(f"\nBASE (deployed params): F1*={base.f1.mean():.4f}  AUC={base.auc.mean():.4f}  ({len(base.f1)} folds)", flush=True)
    rows = [{"variant": "base", "f1": base.f1.mean(), "auc": base.auc.mean(), "df1": 0, "sig_f1": 0, "dauc": 0}]
    for name, p in VARIANTS.items():
        v = cv_evaluate("general", mk(p), label=name, n_repeats=NR, df=df)
        c = compare(base, v)
        dauc = v.auc.mean() - base.auc.mean()
        # auc sigma (paired)
        da = v.auc - base.auc
        sa = da.mean() / (da.std(ddof=1)/np.sqrt(len(da))) if da.std() > 0 else 0
        print(f"  {name:9s} F1*={v.f1.mean():.4f} ΔF1*={c['delta_f1']:+.4f} ({c['sigma']:+.2f}σ)  | "
              f"AUC={v.auc.mean():.4f} ΔAUC={dauc:+.4f} ({sa:+.2f}σ)", flush=True)
        rows.append({"variant": name, "f1": v.f1.mean(), "auc": v.auc.mean(),
                     "df1": c["delta_f1"], "sig_f1": c["sigma"], "dauc": dauc, "sig_auc": sa})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "general_params.csv"), index=False)
    print(f"\nsaved -> general_params.csv", flush=True)
