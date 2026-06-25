"""Idea #5 — per-variant confidence flags. Does the deployed model's reliability actually differ on
flagged variants? Collect OOF predictions (own native-NaN model), then compare AUC / Brier / accuracy
on subsets: CAT_6 low-quality regions, EK-missing, AL-missing, MNV/complex AA. A reliability GAP
justifies emitting lower-confidence flags on those variants at finals (reporting/calibration, not F1).
Run: .venv/bin/python -u analysis/exp_confidence.py [panel ...]
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from config import SEED
from schema import partition_columns
from harness import load_panel
from meeting_lib import native_nan_oof

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
STD = set("ACDEFGHIKLMNPQRSTVWY") | {"*"}


def oof_preds(df):
    _, num, cat, tgt = partition_columns(df)
    X = df[num + cat].reset_index(drop=True); y = df[tgt].astype(int).reset_index(drop=True)
    oof = np.zeros(len(y)); skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for tr, te in skf.split(X, y):
        tp, _ = native_nan_oof(X.iloc[tr], y.iloc[tr], X.iloc[te], n_seeds=2)
        oof[te] = tp
    return X, y, oof


def metrics(y, p):
    if len(np.unique(y)) < 2 or len(y) < 15:
        return (np.nan, np.nan, len(y))
    return (roc_auc_score(y, p), brier_score_loss(y, p), len(y))


if __name__ == "__main__":
    panels = sys.argv[1:] or ["general", "cancer", "pah", "cftr"]
    print("=== Idea #5: reliability on flagged vs clean variants (deployed-style OOF) ===", flush=True)
    rows = []
    for pn in panels:
        df = load_panel(pn); X, y, p = oof_preds(df)
        ml = (~X["AA_1"].astype(str).isin(STD)) | (~X["AA_2"].astype(str).isin(STD))
        masks = {"ALL": np.ones(len(y), bool),
                 "CAT_6 flagged": X["CAT_6"].notna().values if "CAT_6" in X else np.zeros(len(y), bool),
                 "EK_7 missing": X["EK_7"].isna().values if "EK_7" in X else np.zeros(len(y), bool),
                 "MNV/complex AA": ml.values}
        print(f"\n[{pn}] n={len(y)}", flush=True)
        for name, m in masks.items():
            auc, brier, n = metrics(y[m].values, p[m])
            tag = "" if name == "ALL" else "  <- flag candidate"
            print(f"  {name:16s} n={n:5d}  AUC={auc if not np.isnan(auc) else float('nan'):.3f}  Brier={brier:.3f}{tag}"
                  if not np.isnan(auc) else f"  {name:16s} n={n:5d}  (too few/one-class)", flush=True)
            rows.append({"panel": pn, "subset": name, "n": n, "auc": auc, "brier": brier})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "confidence.csv"), index=False)
    print(f"\nsaved -> confidence.csv", flush=True)
