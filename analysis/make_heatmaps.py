"""Per-panel feature-correlation heatmaps (top label-relevant features, hierarchically clustered).
Shows the redundancy/block structure (e.g. the correlated AL_ block, the EK_ conservation tracks).

Run:  .venv/bin/python -u analysis/make_heatmaps.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from sklearn.metrics import roc_auc_score

from schema import normalize_sentinels, drop_all_nan_rows, to_categorical, partition_columns
from feature_engineering import engineer_aa_features
from utils import PANELS

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "docs", "panel_research", "explainability")
os.makedirs(OUT, exist_ok=True)
DICT = {"EK_1": "GERP-NR", "EK_2": "GERP-RS", "EK_3": "phyloP", "EK_4": "phastCons",
        "EK_5": "phastCons", "EK_6": "phastC470", "EK_7": "phyloP100", "EK_8": "phyloP17",
        "EK_9": "phyloP470", "CAT_1": "gnomAD-pop", "CAT_2": "AoU-pop", "CAT_6": "region",
        "AA_1": "refAA", "AA_2": "altAA"}
GCOL = {"EK": "#1f77b4", "AL": "#d62728", "AA": "#2ca02c", "CAT": "#9467bd", "FX": "#ff7f0e"}


def grp(c):
    c = str(c)
    return ("EK" if c.startswith("EK_") else "AL" if c.startswith(("AL_", "ALfam", "ALn"))
            else "AA" if c.startswith("AA_") else "CAT" if c.startswith("CAT_") else "FX")


def panel_heatmap(panel, k=32):
    df = drop_all_nan_rows(normalize_sentinels(pd.read_csv(PANELS[panel])), verbose=False)
    df = to_categorical(df); df = engineer_aa_features(df)
    _, num, cat, tgt = partition_columns(df)
    y = df[tgt].astype(int).values
    # build a numeric frame: numerics as-is; categoricals ORDINAL-encoded (by frequency -> no label leak)
    enc = pd.DataFrame(index=df.index)
    for c in num:
        enc[c] = pd.to_numeric(df[c], errors="coerce")
    cat_cols = []
    for c in cat:
        order = {v: i for i, v in enumerate(df[c].value_counts().index)}
        col = df[c].map(order).astype(float)
        if col.notna().sum() >= 30 and col.nunique(dropna=True) >= 2:   # skip near-constant cats
            enc[c] = col; cat_cols.append(c)
    # rank by |label AUC - 0.5|
    scores = {}
    for c in enc.columns:
        v = enc[c]; m = v.notna().values
        if m.sum() < 30 or len(np.unique(y[m])) < 2 or pd.Series(v[m]).nunique() < 2:
            continue
        try:
            a = roc_auc_score(y[m], v[m].values); scores[c] = abs(max(a, 1 - a) - 0.5)
        except Exception:
            pass
    # force ALL (usable) CAT columns in + fill the rest with top numerics
    top_num = [c for c, _ in sorted(scores.items(), key=lambda x: -x[1]) if c not in cat_cols]
    top = top_num[:max(1, k - len(cat_cols))] + cat_cols
    corr = enc[top].corr(min_periods=30).fillna(0.0)
    # hierarchical cluster ordering on 1-|corr|
    D = 1 - corr.abs().values; np.fill_diagonal(D, 0.0)
    order = leaves_list(linkage(squareform(D, checks=False), method="average"))
    corr = corr.iloc[order, order]
    labels = [f"{c} ({DICT[c]})" if c in DICT else c for c in corr.columns]
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7); ax.set_yticklabels(labels, fontsize=7)
    for t in ax.get_xticklabels():
        t.set_color(GCOL.get(grp(t.get_text().split(" ")[0]), "#333"))
    for t in ax.get_yticklabels():
        t.set_color(GCOL.get(grp(t.get_text().split(" ")[0]), "#333"))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    ax.set_title(f"{panel.upper()} — feature correlation ({len(top)} feats incl. all CAT_; "
                 f"categoricals ordinal-encoded)\n"
                 f"clustered; colors: EK=blue  AL=red  AA=green  CAT=purple", fontsize=12, weight="bold")
    fig.tight_layout()
    path = os.path.join(OUT, f"heatmap_{panel}.png"); fig.savefig(path, dpi=140); plt.close(fig)
    print(f"saved -> {path}  ({len(top)} feats)", flush=True)
    return path


if __name__ == "__main__":
    for p in ["general", "cancer", "pah", "cftr"]:
        panel_heatmap(p)
