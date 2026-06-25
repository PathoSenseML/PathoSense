"""AL <-> CAT (population) association via Kruskal-Wallis, to test the idea that specific AL columns
are specific-population allele data (e.g. "AL_X is European AF"), enabling population-aware imputation.

CAT_1 = gnomAD popmax population, CAT_2 = AllofUs popmax population (= argmax over per-population AL).
So an AL column that IS population P's frequency should be systematically HIGHEST for variants whose
popmax is P. Kruskal-Wallis tests whether an AL column's value distribution differs across popmax groups;
the group with the highest median = the column's candidate "owner" population.

Outputs, per AL column: KW H-stat (population-discriminativeness), and the owner population. Then a
population -> top-AL mapping. Strong, clean ownership => the mapping (and population-aware fill) is viable.
Run: .venv/bin/python -u analysis/exp_al_cat_anova.py
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
from scipy.stats import kruskal
from schema import normalize_sentinels, drop_all_nan_rows
from utils import PANELS

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
ALL = pd.concat([drop_all_nan_rows(normalize_sentinels(pd.read_csv(p)), verbose=False)
                 for p in PANELS.values()], ignore_index=True)
AL = [c for c in ALL.columns if c.startswith("AL_")]


def pop_suffix(s):
    return s.dropna().astype(str).str.split("_").str[-1]


def analyse(cat_col, min_group=15):
    print(f"\n###### AL x {cat_col} (Kruskal-Wallis across popmax groups) ######")
    grp = pop_suffix(ALL[cat_col]).reindex(ALL.index)
    pops = [p for p, n in grp.value_counts().items() if n >= min_group]
    print(f"  populations (n>={min_group}): {pops}")
    rows = []
    for c in AL:
        v = ALL[c]
        groups = [v[(grp == p) & v.notna()].astype(float).values for p in pops]
        groups = [g for g in groups if len(g) >= 8]
        if len(groups) < 3:
            continue
        try:
            H, pval = kruskal(*groups)
        except Exception:
            continue
        # owner population = highest median among present values
        med = {p: np.nanmedian(v[(grp == p) & v.notna()].astype(float)) for p in pops
               if (((grp == p) & v.notna()).sum() >= 8)}
        owner = max(med, key=med.get) if med else None
        rows.append({"AL": c, "H": H, "p": pval, "owner": owner,
                     "owner_med": med.get(owner, np.nan), "n_present": int(v.notna().sum())})
    d = pd.DataFrame(rows).sort_values("H", ascending=False)
    # significant + discriminative
    sig = d[(d["p"] < 1e-4)]
    print(f"  AL cols tested: {len(d)};  KW-significant (p<1e-4): {len(sig)}  "
          f"({100*len(sig)/max(len(d),1):.0f}%)")
    print(f"  top-15 most population-discriminative AL columns:")
    print(d.head(15)[["AL", "H", "p", "owner", "n_present"]].to_string(index=False))
    # population -> its top AL columns (by H, owned by that pop)
    print(f"\n  population -> top owned AL columns (the soft mapping):")
    for p in pops:
        owned = sig[sig["owner"] == p].sort_values("H", ascending=False)["AL"].head(6).tolist()
        if owned:
            print(f"    {p:10s}: {owned}")
    d.to_csv(os.path.join(OUT, f"al_cat_anova_{cat_col}.csv"), index=False)
    return d


if __name__ == "__main__":
    print(f"pooled n={len(ALL)}  AL cols={len(AL)}", flush=True)
    analyse("CAT_1")
    analyse("CAT_2")
    print("\n[done] saved al_cat_anova_CAT_1.csv / _CAT_2.csv", flush=True)
