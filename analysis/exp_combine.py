"""Validate COMBINED recipes per panel — do the individually-winning preprocessing
levers stack, or do they overlap (all recovering the same AL_ signal)?

Tests recipe combinations via the unified pipeline_v2.make_fn, paired vs deployed,
analytic F1* at the test prior. Picks the best recipe per panel.
"""
import os, sys, itertools
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")

from harness import cv_evaluate, load_panel, compare, deployed_pipeline_fn
from pipeline_v2 import make_fn
from utils import TEST_PRIORS

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
PANELS = ["general", "cancer", "pah", "cftr"]
REPEATS = {"general": 6, "cancer": 8, "pah": 8, "cftr": 10}
SEEDS = 2  # screening; final winner re-validated at deploy seeds separately

# Candidate recipes (deployed-equivalent baseline = make_fn with all defaults).
RECIPES = {
    "deployed_v2base":   dict(),                                            # sanity: == deployed
    "noFS":              dict(feature_select="none"),
    "nativeNaN":         dict(impute="none"),
    "famflags":          dict(al_features=("fam_flags",)),
    "noFS+nativeNaN":    dict(feature_select="none", impute="none"),
    "noFS+famflags":     dict(feature_select="none", al_features=("fam_flags",)),
    "nativeNaN+famflags": dict(impute="none", al_features=("fam_flags",)),
    "noFS+nativeNaN+famflags": dict(feature_select="none", impute="none",
                                    al_features=("fam_flags",)),
}


def run(panel):
    df = load_panel(panel); rep = REPEATS[panel]
    base = cv_evaluate(panel, deployed_pipeline_fn(panel, n_seeds=SEEDS),
                       label="deployed", n_repeats=rep, df=df)
    bs = base.summary()
    print(f"\n=== {panel} (prior={TEST_PRIORS[panel]:.3f}) deployed F1*="
          f"{bs['f1_mean']:.4f}±{bs['f1_se']:.4f} finals={bs.get('f1_finals'):.4f} ===", flush=True)
    rows = []
    for name, rec in RECIPES.items():
        try:
            r = cv_evaluate(panel, make_fn(panel, rec, n_seeds=SEEDS), label=name,
                            n_repeats=rep, df=df)
        except Exception as e:
            print(f"  {name:28s} FAILED: {type(e).__name__}: {str(e)[:70]}", flush=True)
            continue
        c = compare(base, r)
        print(f"  {name:28s} F1*={c['b_f1']:.4f} finals={r.summary().get('f1_finals'):.4f} "
              f"Δ={c['delta_f1']:+.4f} ({c['sigma']:+.2f}σ)  {c['verdict']}", flush=True)
        rows.append({"panel": panel, "recipe": name, **c,
                     "finals": r.summary().get("f1_finals")})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"exp_combine_{panel}.csv"), index=False)
    return rows


if __name__ == "__main__":
    for pn in (sys.argv[1:] or PANELS):
        run(pn)
