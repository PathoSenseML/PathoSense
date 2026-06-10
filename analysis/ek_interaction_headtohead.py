"""Head-to-head: deployed FX_cons_x_disrupt (raw EK_7 x disruption) vs a
rank-normalised multi-track conservation consensus x disruption. Univariate sAUC,
plus the consensus term alone vs raw EK_7 alone.
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipelines"))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from schema import normalize_sentinels, drop_all_nan_rows
from feature_engineering import engineer_aa_features, engineer_interaction_features
from utils import PANELS

EK = [f"EK_{i}" for i in range(1, 10)]


def sauc(y, v):
    v = np.asarray(v, float); y = np.asarray(y); m = ~np.isnan(v)
    if m.sum() < 20 or pd.Series(y[m]).nunique() < 2 or np.nanstd(v[m]) < 1e-12:
        return np.nan
    a = roc_auc_score(y[m], v[m]); return max(a, 1 - a)


def rank_apply(df, cols):
    out = pd.DataFrame(index=df.index)
    for c in cols:
        sv = np.sort(df[c].dropna().astype(float).values)
        v = df[c].astype(float).values
        pct = np.searchsorted(sv, v, side="right") / max(len(sv), 1)
        pct = pct.astype(float); pct[np.isnan(v)] = np.nan
        out[c] = pct
    return out


for panel in ["general", "cancer", "pah", "cftr"]:
    df = pd.read_csv(PANELS[panel])
    df = normalize_sentinels(df); df = drop_all_nan_rows(df, verbose=False)
    df = engineer_aa_features(df)
    df = engineer_interaction_features(df)         # deployed: makes FX_cons_x_disrupt from raw EK_7
    y = df["Label"].astype(int).values
    present = [c for c in EK if c in df.columns]
    ranks = rank_apply(df, present)
    core = [c for c in ("EK_7", "EK_9", "EK_2") if c in ranks.columns]
    cons = ranks[core].mean(axis=1).fillna(ranks.mean(axis=1))      # consensus, [0,1]
    disrupt = ((4.0 - df["AA_blosum62"].astype(float)) / 15.0).clip(0, 1)
    prop = cons * disrupt

    print(f"\n{panel.upper()}  n={len(df)}")
    print(f"  cons term:   raw EK_7 sAUC={sauc(y, df['EK_7'].values):.3f}   "
          f"rank-consensus(EK7,EK9,EK2) sAUC={sauc(y, cons.values):.3f}")
    print(f"  interaction: deployed FX_cons_x_disrupt sAUC={sauc(y, df['FX_cons_x_disrupt'].values):.3f}   "
          f"PROPOSED consensus*disrupt sAUC={sauc(y, prop.values):.3f}")
