import sys, os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "/mnt/SSD/Dev/PathoSense/pipelines")
from schema import normalize_sentinels, drop_all_nan_rows, ID_COLS, TARGET_COL
P={"general":"YARISMA_TRAIN_MASTER.csv","cancer":"YARISMA_TRAIN_KANSER.csv","pah":"YARISMA_TRAIN_PAH.csv","cftr":"YARISMA_TRAIN_CFTR.csv"}
for p,f in P.items():
    df=normalize_sentinels(pd.read_csv(f"/mnt/SSD/Dev/PathoSense/data/panels/{f}"))
    df=drop_all_nan_rows(df,verbose=False)
    fcols=[c for c in df.columns if c not in ID_COLS and c!=TARGET_COL]
    pop=df[fcols].notna().sum(axis=1)
    y=df[TARGET_COL].values
    auc=roc_auc_score(y,pop) ; auc=max(auc,1-auc)
    q=pop.quantile([0,.05,.1,.25,.5]).astype(int).to_dict()
    # label balance among sparsest decile
    thr=pop.quantile(.10)
    sp=pop<=thr
    bal=df.loc[sp,TARGET_COL].value_counts().to_dict()
    print(f"{p:8s} n={len(df):4d} pop[min/5%/10%/25%/50%]={[q[k] for k in (0,.05,.1,.25,.5)]} "
          f"AUC(pop vs label)={auc:.3f} | sparsest~10% (n={int(sp.sum())}) labels={bal} P_overall={y.mean():.2f}")
    # how many rows have <=3 populated
    for k in (1,2,3,5):
        m=pop<=k
        if m.sum()>0:
            print(f"          rows with <= {k} populated: {int(m.sum())} labels={df.loc[m,TARGET_COL].value_counts().to_dict()}")
