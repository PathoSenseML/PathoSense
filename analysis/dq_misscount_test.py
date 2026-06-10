"""Test: does an AGGREGATE per-row missingness feature improve F1* at the test prior,
on top of the deployed features (which already carry per-column missing indicators)?
Honest paired repeated-CV via feature_ablation.eval_fold."""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, "/mnt/SSD/Dev/PathoSense/pipelines")
sys.path.insert(0, "/mnt/SSD/Dev/PathoSense/analysis")
from sklearn.model_selection import RepeatedStratifiedKFold
from schema import normalize_sentinels, drop_all_nan_rows, to_categorical, partition_columns
from feature_engineering import engineer_aa_features, engineer_acmg_features
from utils import detect_leaky_columns, KNOWN_LEAKY, PANELS, TEST_PRIORS, ACMG_PANELS
from run_pipeline import preprocess_data, select_features_mrmr
from config import SEED
from feature_ablation import eval_fold

def add_dq(df, feat_cols):
    al=[c for c in feat_cols if c.startswith("AL_")]
    df["DQ_n_missing"]   = df[feat_cols].isna().sum(axis=1).astype(float)
    df["DQ_frac_af_miss"]= df[al].isna().mean(axis=1).astype(float) if al else 0.0
    return df

def prepare(panel, use_dq):
    df = pd.read_csv(PANELS[panel]); df = normalize_sentinels(df)
    df = drop_all_nan_rows(df, verbose=False)
    feat_cols=[c for c in df.columns if c not in ("Variant_ID","Label")]
    df = to_categorical(df); df = engineer_aa_features(df)
    if panel in ACMG_PANELS: df = engineer_acmg_features(df)
    if use_dq: df = add_dq(df, feat_cols)
    _, num, cat, tgt = partition_columns(df)
    y=df[tgt].reset_index(drop=True); X=df[num+cat].reset_index(drop=True)
    leaky=detect_leaky_columns(X,y,auc_threshold=0.99,verbose=False)
    drop=sorted(set(leaky)|(KNOWN_LEAKY&set(X.columns)))
    if drop: X=X.drop(columns=drop)
    Xpp,_,_,_=preprocess_data(X.copy(),X.copy(),X.copy(),corr_threshold=0.95)
    Xsel,_,_,_=select_features_mrmr(Xpp,y,Xpp,Xpp)
    feats=list(Xsel.columns)
    Xf=Xpp[feats].reset_index(drop=True)
    catf=[c for c in feats if isinstance(Xf[c].dtype,pd.CategoricalDtype)]
    dq_sel=[c for c in ("DQ_n_missing","DQ_frac_af_miss") if c in feats]
    return Xf,y,catf,feats,dq_sel

def run(panel,n_splits=5,n_repeats=1):
    prior=TEST_PRIORS[panel]
    res={}
    for tag,use in [("deployed",False),("+misscount",True)]:
        X,y,catf,feats,dq_sel=prepare(panel,use)
        spw=(y==0).sum()/max((y==1).sum(),1)
        outer=RepeatedStratifiedKFold(n_splits=n_splits,n_repeats=n_repeats,random_state=SEED)
        f1s=[]
        for fold,(tr,te) in enumerate(outer.split(X,y)):
            f1,mcc,auc=eval_fold(X,y,spw,catf,np.asarray(tr),np.asarray(te),prior,seed=SEED+fold)
            f1s.append(f1)
        res[tag]=(np.array(f1s),feats,dq_sel)
        print(f"  {panel:8s} {tag:11s} F1*={np.mean(f1s):.4f}±{np.std(f1s):.3f} nfeat={len(feats)} dq_selected={dq_sel}",flush=True)
    d=res["+misscount"][0]-res["deployed"][0]
    se=d.std()/max(np.sqrt(len(d)),1)
    verdict=("HELPS" if d.mean()>2*se else "HURTS" if d.mean()<-2*se else "noise")
    print(f"  {panel:8s} Δ(misscount)= {d.mean():+.4f} ± {se:.4f} SE  wins {int((d>0).sum())}/{len(d)} -> {verdict}\n",flush=True)

import sys as _s
for p in (_s.argv[1:] or ["general","cancer"]):
    run(p)
