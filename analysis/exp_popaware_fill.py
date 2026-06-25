"""Part B — population-aware AL imputation (Arc's idea). Using the KW AL->population mapping, a missing
AL column for a population the variant is NOT popmax-in = "observed elsewhere but absent here" = AF 0
(not unknown). Fill those with 0; leave unmapped AL native-NaN.

Two variants vs native-NaN LGBM (general + cancer = deployed-relevant own-models), paired ΔF1*:
  popaware : for each variant, zero-fill the population-AL columns owned by populations != its popmax
  popzero  : simpler — zero-fill ALL missing values in the mapped population-specific AL columns
Mapping derived label-free (CAT+AL only) from the full data; if positive we'd re-verify per-fold.
Run: .venv/bin/python -u analysis/exp_popaware_fill.py [panel ...]
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
from harness import cv_evaluate, load_panel, compare
from meeting_lib import native_nan_oof

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# owner mapping from the KW analysis (significant p<1e-4), CAT_1 + CAT_2 combined
OWNER = {}
for cc in ("CAT_1", "CAT_2"):
    d = pd.read_csv(os.path.join(OUT, f"al_cat_anova_{cc}.csv"))
    for _, r in d[d["p"] < 1e-4].iterrows():
        OWNER.setdefault(r["AL"], set()).add(str(r["owner"]))   # an AL col may map to a pop in each CAT
POP_AL = sorted(OWNER.keys())
print(f"mapped population-specific AL columns: {len(POP_AL)}", flush=True)


def _suffix(s):
    return s.astype(str).str.split("_").str[-1]


def base_fn(X_tr, y_tr, X_te):
    return native_nan_oof(X_tr, y_tr, X_te)


def make_fn(mode):
    def _fn(X_tr, y_tr, X_te):
        def fill(X):
            X = X.copy()
            if mode == "popzero":
                for c in [c for c in POP_AL if c in X.columns]:
                    X[c] = X[c].fillna(0.0)
            else:  # popaware: zero only the AL owned by populations != the variant's popmax
                pm = _suffix(X["CAT_1"]) if "CAT_1" in X.columns else pd.Series("", index=X.index)
                pm2 = _suffix(X["CAT_2"]) if "CAT_2" in X.columns else pd.Series("", index=X.index)
                for c in [c for c in POP_AL if c in X.columns]:
                    owners = OWNER[c]
                    not_owner = ~(pm.isin(owners) | pm2.isin(owners))   # variant's popmax != this col's pop
                    m = X[c].isna() & not_owner
                    X.loc[m, c] = 0.0
            return X
        return native_nan_oof(fill(X_tr), y_tr, fill(X_te))
    return _fn


def run(panel, nr):
    df = load_panel(panel)
    base = cv_evaluate(panel, base_fn, label="base", n_repeats=nr, df=df)
    print(f"\n[{panel}] BASE (native-NaN) F1*={base.summary()['f1_mean']:.4f}", flush=True)
    rows = [{"panel": panel, "variant": "base", "f1": base.summary()["f1_mean"], "delta": 0.0, "sigma": 0.0}]
    for mode in ("popzero", "popaware"):
        r = cv_evaluate(panel, make_fn(mode), label=mode, n_repeats=nr, df=df)
        c = compare(base, r)
        print(f"[{panel}] {mode:9s} F1*={c['b_f1']:.4f}  Δ={c['delta_f1']:+.4f} ({c['sigma']:+.2f}σ)  {c['verdict']}", flush=True)
        rows.append({"panel": panel, "variant": mode, "f1": c["b_f1"], "delta": c["delta_f1"], "sigma": c["sigma"]})
    return rows


if __name__ == "__main__":
    panels = sys.argv[1:] or ["cancer", "general", "pah", "cftr"]
    print("=== Part B: population-aware AL fill — paired ΔF1* ===", flush=True)
    allrows = []
    for pn in panels:
        allrows += run(pn, 4 if pn == "general" else 10)
    pd.DataFrame(allrows).to_csv(os.path.join(OUT, "popaware_fill.csv"), index=False)
    print(f"\nsaved -> popaware_fill.csv", flush=True)
