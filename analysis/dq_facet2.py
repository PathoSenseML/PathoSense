import sys, os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipelines"))
from schema import (normalize_sentinels, drop_all_nan_rows, to_categorical,
                    partition_columns, ID_COLS, TARGET_COL, CATEGORICAL_COLS)
from run_pipeline import preprocess_data, select_features_mrmr
from feature_engineering import engineer_aa_features, engineer_acmg_features
from utils import detect_leaky_columns, KNOWN_LEAKY, ACMG_PANELS

PANELS = {
    "general": "data/panels/YARISMA_TRAIN_MASTER.csv",
    "cancer":  "data/panels/YARISMA_TRAIN_KANSER.csv",
    "pah":     "data/panels/YARISMA_TRAIN_PAH.csv",
    "cftr":    "data/panels/YARISMA_TRAIN_CFTR.csv",
}
ROOT = os.path.join(os.path.dirname(__file__), "..")
def load(p):
    df = pd.read_csv(os.path.join(ROOT, PANELS[p]))
    return normalize_sentinels(df)

print("="*80); print("AL_*>1 characterization (which cols exceed 1, are they AN/counts vs AF?)"); print("="*80)
for p in PANELS:
    df = drop_all_nan_rows(load(p), verbose=False)
    al=[c for c in df.columns if c.startswith("AL_")]
    over=[]
    for c in al:
        n=int((df[c]>1.0001).sum())
        if n>0: over.append((c,n,round(float(df[c].max()),2)))
    over.sort(key=lambda t:-t[1])
    print(f"{p:8s} #AL cols with any value>1: {len(over)} | top: {over[:12]}")

print(); print("="*80); print("Missing-AA rows that are NOT all-NaN (general): are they noise?"); print("="*80)
for p in ["general","cancer","pah"]:
    df = drop_all_nan_rows(load(p), verbose=False)
    a1=df["AA_1"].astype(object); a2=df["AA_2"].astype(object)
    miss = a1.isna()|a2.isna()
    y=df[TARGET_COL]
    # how many feature cols are populated for these rows on avg?
    fcols=[c for c in df.columns if c not in ID_COLS and c!=TARGET_COL]
    pop = df.loc[miss, fcols].notna().sum(axis=1)
    print(f"{p:8s} missing-AA(not all-NaN) n={int(miss.sum())} labels={y[miss].value_counts().to_dict()} "
          f"median_populated_featcols={pop.median() if miss.sum() else 0:.0f}/{len(fcols)}")

print(); print("="*80); print("Proper near-dup conflict: rows with POPULATED identical (AA pair + rounded EK_7/EK_9/EK_2)"); print("="*80)
SIG=["EK_7","EK_9","EK_2","AA_1","AA_2"]
for p in PANELS:
    df = drop_all_nan_rows(load(p), verbose=False).reset_index(drop=True)
    cols=[c for c in SIG if c in df.columns]
    # require ALL signature cols populated
    mask = df[cols].notna().all(axis=1)
    sub=df[mask].copy()
    for c in cols:
        if c in ("AA_1","AA_2"): sub[c]=sub[c].astype(object)
        else: sub[c]=sub[c].round(1)
    key=sub[cols].apply(lambda r: tuple(r.values), axis=1)
    g=sub.groupby(key)[TARGET_COL]
    sizes=g.size(); dup=sizes[sizes>=2].index
    conf=0; confrows=0; ex=[]
    for k in dup:
        labs=sub.loc[key==k,TARGET_COL]
        if labs.nunique()>1:
            conf+=1; confrows+=len(labs)
            if len(ex)<4: ex.append(dict(sig=k, labels=labs.value_counts().to_dict()))
    print(f"{p:8s} populated rows={int(mask.sum())} | conflicting near-dup groups={conf} rows={confrows}")
    for e in ex: print("    ",e)

print(); print("="*80); print("Do redundant categoricals (CAT_3/4/5) survive into the DEPLOYED selected feature set?"); print("="*80)
for p in PANELS:
    df = drop_all_nan_rows(load(p), verbose=False)
    df = to_categorical(df)
    df = engineer_aa_features(df)
    if p in ACMG_PANELS:
        df = engineer_acmg_features(df)
    _, num_cols, cat_cols, tgt = partition_columns(df)
    y=df[tgt].reset_index(drop=True)
    X=df[num_cols+cat_cols].reset_index(drop=True)
    leaky=detect_leaky_columns(X,y,auc_threshold=0.99,verbose=False)
    drop=sorted(set(leaky)|(KNOWN_LEAKY&set(X.columns)))
    if drop: X=X.drop(columns=drop)
    Xpp,_,_,_=preprocess_data(X.copy(),X.copy(),X.copy(),corr_threshold=0.95)
    Xsel,_,_,_=select_features_mrmr(Xpp,y,Xpp,Xpp)
    feats=list(Xsel.columns)
    cats_in=[c for c in ("CAT_1","CAT_2","CAT_3","CAT_4","CAT_5","CAT_6") if c in feats]
    print(f"{p:8s} n_selected={len(feats)} | CAT_* selected: {cats_in}")
