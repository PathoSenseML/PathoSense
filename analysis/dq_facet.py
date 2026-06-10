"""Data-quality / cleaning facet analysis (TRAIN-time only).

Computes, per panel:
  1. ref==alt rows (AA_1==AA_2): count + label distribution (candidate noise to drop)
  2. exact-duplicate feature ROWS with conflicting labels (after all-NaN drop)
  3. near-duplicate feature ROWS with conflicting labels (rounded numeric hash)
  4. exact-duplicate / near-constant / near-duplicate COLUMNS (CAT_3/4/5, AL_185, etc.)
  5. obvious outliers / impossible values (AF>1, conservation out of known ceilings,
     duplicate Variant_ID WITHIN a panel)
"""
import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipelines"))
from schema import normalize_sentinels, drop_all_nan_rows, ID_COLS, TARGET_COL, CATEGORICAL_COLS

PANELS = {
    "general": "data/panels/YARISMA_TRAIN_MASTER.csv",
    "cancer":  "data/panels/YARISMA_TRAIN_KANSER.csv",
    "pah":     "data/panels/YARISMA_TRAIN_PAH.csv",
    "cftr":    "data/panels/YARISMA_TRAIN_CFTR.csv",
}
ROOT = os.path.join(os.path.dirname(__file__), "..")

def feat_cols(df):
    return [c for c in df.columns if c not in ID_COLS and c != TARGET_COL]

def load(panel):
    df = pd.read_csv(os.path.join(ROOT, PANELS[panel]))
    df = normalize_sentinels(df)
    return df

print("="*80)
print("CHECK 0: basic shapes + label balance")
print("="*80)
for p in PANELS:
    df = load(p)
    n = len(df)
    y = df[TARGET_COL]
    print(f"{p:8s} n={n:5d}  P={int((y==1).sum()):5d} B={int((y==0).sum()):5d}  "
          f"train_prior_P={(y==1).mean():.3f}  dup_VariantID_within={df['Variant_ID'].duplicated().sum()}")

print()
print("="*80)
print("CHECK 1: ref==alt (AA_1==AA_2) rows — 'no-change' technical errors")
print("="*80)
for p in PANELS:
    df = load(p)
    a1 = df["AA_1"].astype(object); a2 = df["AA_2"].astype(object)
    both = a1.notna() & a2.notna()
    same = both & (a1 == a2)
    y = df[TARGET_COL]
    ys = y[same]
    print(f"{p:8s} ref==alt n={int(same.sum()):4d}  labels={ys.value_counts().to_dict()}  "
          f"(P_frac={ys.mean() if len(ys) else float('nan'):.3f})  "
          f"| missing AA (either null)={int((~both).sum())}")
    # show a few examples
    if same.sum() > 0:
        ex = df.loc[same, ["Variant_ID","AA_1","AA_2",TARGET_COL]].head(8)
        print("   examples:", list(zip(ex["AA_1"],ex["AA_2"],ex[TARGET_COL])))

print()
print("="*80)
print("CHECK 1b: multi-letter / non-standard AA codes (also dubious)")
print("="*80)
STD = set("ARNDCQEGHILKMFPSTWYV") | {"*"}
for p in PANELS:
    df = load(p)
    a1 = df["AA_1"].astype(object); a2 = df["AA_2"].astype(object)
    bad = (a1.notna() & ~a1.isin(STD)) | (a2.notna() & ~a2.isin(STD))
    y = df[TARGET_COL]
    vc = y[bad].value_counts().to_dict()
    uniq = sorted(set(a1[bad].dropna().tolist()+a2[bad].dropna().tolist()))[:20]
    print(f"{p:8s} multiletter/nonstd n={int(bad.sum()):4d} labels={vc} codes={uniq}")

print()
print("="*80)
print("CHECK 2: exact-duplicate feature rows w/ conflicting labels (AFTER all-NaN drop)")
print("="*80)
for p in PANELS:
    df = load(p)
    n0 = len(df)
    df = drop_all_nan_rows(df, verbose=False)
    n1 = len(df)
    fcols = feat_cols(df)
    # hash each row's feature tuple (NaN==NaN). Fill NaN with sentinel string.
    key = df[fcols].astype(object).where(df[fcols].notna(), "<<NA>>")
    key = key.apply(lambda r: hash(tuple(r.values)), axis=1)
    grp = df.groupby(key)[TARGET_COL]
    sizes = grp.size()
    dup_keys = sizes[sizes>=2].index
    conflicted = 0; conflict_rows=0; same=0; same_rows=0
    for k in dup_keys:
        labs = df.loc[key==k, TARGET_COL]
        if labs.nunique()>1: conflicted+=1; conflict_rows+=len(labs)
        else: same+=1; same_rows+=len(labs)-1  # extra redundant rows
    print(f"{p:8s} all-NaN dropped={n0-n1:3d}  exact-dup groups: conflicted={conflicted} "
          f"(rows={conflict_rows})  same-label={same} (redundant_extra_rows={same_rows})")

print()
print("="*80)
print("CHECK 3: near-duplicate rows w/ conflicting labels (rounded numeric, top conservation+AA)")
print("="*80)
# Use a stable, leakage-free signature: rounded conservation cols + AA pair.
SIG_COLS = ["EK_7","EK_9","EK_1","EK_2","EK_6","AA_1","AA_2"]
for p in PANELS:
    df = load(p)
    df = drop_all_nan_rows(df, verbose=False)
    cols = [c for c in SIG_COLS if c in df.columns]
    sig = df.copy()
    for c in cols:
        if c in ("AA_1","AA_2"):
            sig[c] = sig[c].astype(object).where(sig[c].notna(), "<NA>")
        else:
            sig[c] = sig[c].round(2).astype(object).where(sig[c].notna(), "<NA>")
    key = sig[cols].apply(lambda r: hash(tuple(r.values)), axis=1)
    grp_sizes = df.groupby(key)[TARGET_COL].size()
    dup_keys = grp_sizes[grp_sizes>=2].index
    conflicted=0; conflict_rows=0
    examples=[]
    for k in dup_keys:
        labs = df.loc[key==k, TARGET_COL]
        if labs.nunique()>1:
            conflicted+=1; conflict_rows+=len(labs)
            if len(examples)<3:
                examples.append(labs.value_counts().to_dict())
    print(f"{p:8s} near-dup(conservation+AA) conflicted groups={conflicted} rows={conflict_rows} ex={examples}")

print()
print("="*80)
print("CHECK 4: duplicate / near-constant / near-duplicate COLUMNS")
print("="*80)
for p in PANELS:
    df = load(p)
    df = drop_all_nan_rows(df, verbose=False)
    fcols = feat_cols(df)
    # 4a. constant / near-constant columns (1 unique non-null value, or all-NaN)
    const_cols=[]; allnan=[]
    for c in fcols:
        nun = df[c].nunique(dropna=True)
        if df[c].notna().sum()==0: allnan.append(c)
        elif nun<=1: const_cols.append(c)
    # 4b. exact-duplicate columns (identical values incl NaN pattern)
    # compare by hashing column content
    colhash={}
    dup_pairs=[]
    for c in fcols:
        h = hash(tuple(df[c].astype(object).where(df[c].notna(),"<NA>").values))
        if h in colhash:
            dup_pairs.append((colhash[h], c))
        else:
            colhash[h]=c
    # 4c. categorical near-duplicates CAT_3/4/5 explicit
    cat_overlap={}
    catc=[c for c in ("CAT_3","CAT_4","CAT_5") if c in df.columns]
    for i in range(len(catc)):
        for j in range(i+1,len(catc)):
            a=df[catc[i]].astype(object); b=df[catc[j]].astype(object)
            both=a.notna()&b.notna()
            disagree=int((both&(a!=b)).sum())
            nan_diff=int((a.isna()!=b.isna()).sum())
            cat_overlap[f"{catc[i]}~{catc[j]}"]=(disagree,nan_diff)
    print(f"{p:8s} allNaN_cols={len(allnan)} const_cols={len(const_cols)} exact_dup_col_pairs={len(dup_pairs)}")
    print(f"   const examples: {const_cols[:8]}")
    print(f"   exact-dup col pairs (first 10): {dup_pairs[:10]}")
    print(f"   CAT_3/4/5 disagreement (val_disagree,nan_pattern_diff): {cat_overlap}")

print()
print("="*80)
print("CHECK 5: outliers / impossible values")
print("="*80)
# AF columns should be in [0,1]; conservation ceilings known. Flag violations.
for p in PANELS:
    df = load(p)
    df = drop_all_nan_rows(df, verbose=False)
    al_cols=[c for c in df.columns if c.startswith("AL_")]
    af_gt1=0
    for c in al_cols:
        v=df[c]
        af_gt1 += int((v>1.0001).sum())
    # conservation ceiling checks
    notes=[]
    if "EK_7" in df: notes.append(f"EK_7(phyloP100,max~10.0) max={df['EK_7'].max():.3f} min={df['EK_7'].min():.3f}")
    if "EK_1" in df: notes.append(f"EK_1(GERP_NR>=0,ceil6.17) min={df['EK_1'].min():.3f} max={df['EK_1'].max():.3f}")
    if "EK_2" in df: notes.append(f"EK_2(GERP_RS,ceil6.17) min={df['EK_2'].min():.3f} max={df['EK_2'].max():.3f}")
    print(f"{p:8s} AL_*>1 count(over all AF cols)={af_gt1}")
    for nt in notes: print("   ", nt)
