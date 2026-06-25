"""Group-level (leave-one-group-out) ablation — "ablation study -> grup grup" /
CFTR: check every feature group.

For each panel, drop a whole FEATURE GROUP and measure the honest ΔF1* at the test
prior (paired CV vs the full feature set). This is the counterpart to the SHAP
group-attribution in explainability.py: SHAP says how much a group CONTRIBUTES to the
model's output; ablation says how much TEST-PRIOR F1 we actually lose without it.

Groups (by column prefix, matching explainability.group_of):
  EK   conservation (phyloP / phastCons / GERP++)
  AL   allele frequency block (+ engineered ALfam/ALn aggregates)
  AA   amino-acid engineered features (BLOSUM/Grantham/hydrophobicity/...)
  CAT  categorical (population / region flags)

Each panel uses its OWN-model recipe (general = native-NaN all-features; small panels =
deployed median+mRMR). Honest per-fold CV, F1* at the per-panel test prior, paired vs
the full set on identical folds.

Run:  .venv/bin/python -u analysis/exp_group_ablation.py
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")

from harness import cv_evaluate, load_panel, compare, deployed_pipeline_fn
from exp_general_blend import lgbm_only_fn

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

GROUPS = {
    "EK":  ("EK_",),
    "AL":  ("AL_", "ALfam", "ALn", "ALmax", "ALmean", "ALpresent"),
    "AA":  ("AA_",),
    "CAT": ("CAT_",),
}

RECIPES = {
    "general": lambda: lgbm_only_fn,
    "cancer":  lambda: deployed_pipeline_fn("cancer"),
    "pah":     lambda: deployed_pipeline_fn("pah"),
    "cftr":    lambda: deployed_pipeline_fn("cftr"),
}


def drop_group_fn(recipe_fn, prefixes):
    def _fn(X_tr, y_tr, X_te):
        keep = [c for c in X_tr.columns
                if not any(str(c).startswith(p) for p in prefixes)]
        return recipe_fn(X_tr[keep], y_tr, X_te[keep])
    return _fn


def run(panel, n_repeats):
    df = load_panel(panel)
    recipe = RECIPES[panel]()
    full = cv_evaluate(panel, recipe, label="full", n_repeats=n_repeats, df=df)
    fs = full.summary()
    print(f"\n[{panel}] FULL  F1*={fs['f1_mean']:.4f}  AUC={fs['auc_mean']:.4f}", flush=True)
    rows = [{"panel": panel, "group": "(full)", "f1": fs["f1_mean"],
             "auc": fs["auc_mean"], "delta_f1": 0.0, "sigma": 0.0, "verdict": "reference"}]
    for g, prefixes in GROUPS.items():
        # skip groups absent from this panel
        if not any(str(c).startswith(p) for c in df.columns for p in prefixes):
            continue
        r = cv_evaluate(panel, drop_group_fn(recipe, prefixes), label=f"-{g}",
                        n_repeats=n_repeats, df=df)
        c = compare(full, r)              # b - a = (drop g) - full ; negative => group helps
        verdict = ("CRITICAL" if c["delta_f1"] < 0 and c["sigma"] <= -2 else
                   "helps" if c["delta_f1"] < 0 and c["sigma"] <= -1 else
                   "noise/redundant")
        print(f"[{panel}] drop {g:3s} F1*={c['b_f1']:.4f}  ΔF1*={c['delta_f1']:+.4f} "
              f"({c['sigma']:+.2f}σ)  -> {verdict}", flush=True)
        rows.append({"panel": panel, "group": g, "f1": c["b_f1"],
                     "auc": r.summary()["auc_mean"], "delta_f1": c["delta_f1"],
                     "sigma": c["sigma"], "verdict": verdict})
    return rows


if __name__ == "__main__":
    print("=== Group leave-one-out ablation — honest CV (F1* at test prior) ===", flush=True)
    print("ΔF1* = (drop group) - full.  Negative ΔF1* with σ≤-2 = the group is CRITICAL.", flush=True)
    all_rows = []
    for pn, nr in [("general", 4), ("cancer", 8), ("pah", 8), ("cftr", 8)]:
        all_rows.extend(run(pn, nr))
    out = os.path.join(OUT, "group_ablation.csv")
    pd.DataFrame(all_rows).to_csv(out, index=False)
    print(f"\nsaved -> {out}", flush=True)
