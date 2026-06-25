"""Idea #2 — seed-bagging for variance reduction. Deployed uses n_seeds=2-3. More seeds won't move
the CV mean much, but reduces the variance of the single finals prediction (cftr is scored on ~20
positives -> high variance). Run the own model at n_seeds in {2,5,12}; report mean F1* + per-fold SE.
Expectation: mean ~flat, SE shrinks (stability gain).
Run: .venv/bin/python -u analysis/exp_seed_bag.py [panel ...]
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
from harness import cv_evaluate, load_panel
from meeting_lib import native_nan_oof

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def fn(ns):
    return lambda X_tr, y_tr, X_te: native_nan_oof(X_tr, y_tr, X_te, n_seeds=ns)


if __name__ == "__main__":
    panels = sys.argv[1:] or ["cftr", "cancer", "general"]
    print("=== Idea #2: seed-bagging — mean F1* and per-fold SE vs n_seeds ===", flush=True)
    rows = []
    for pn in panels:
        df = load_panel(pn); nr = 4 if pn == "general" else 10
        for ns in (2, 5, 12):
            s = cv_evaluate(pn, fn(ns), label=f"ns{ns}", n_repeats=nr, df=df).summary()
            print(f"  [{pn}] n_seeds={ns:2d}  F1*={s['f1_mean']:.4f}  SE={s['f1_se']:.4f}", flush=True)
            rows.append({"panel": pn, "n_seeds": ns, "f1": s["f1_mean"], "se": s["f1_se"]})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "seed_bag.csv"), index=False)
    print(f"\nsaved -> seed_bag.csv", flush=True)
