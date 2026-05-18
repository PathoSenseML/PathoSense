"""D3: Correlation clustering on the in-silico bulk.

The 179 "in_silico_or_phastcons" + 27 "population_frequency" columns in MASTER
probably collapse to a much smaller number of independent signals (dbNSFP often
publishes both raw scores and rank-scored versions, and many in-silico tools
correlate strongly because they share training data).

This script:
  - Restricts to the columns categorized as in-silico or population MAF (skip
    constants/near-constants).
  - Computes pairwise Pearson |r| on the master panel (NaN-aware via pairwise).
  - Hierarchical clustering with average linkage on (1 - |r|).
  - Cuts at a configurable threshold (default 0.3, ~= |r|>=0.7 within cluster).
  - For each cluster, pick a representative = the column with highest label-AUC.
  - Save: cluster assignments, representatives, top-AUC per cluster, the
    dendrogram as PNG, and a heatmap of the |r| matrix.
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipelines"))
from schema import normalize_sentinels, to_categorical
from utils import PANELS

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Columns we want to cluster: anything tagged in-silico or population in D2.
CLUSTERABLE_CATEGORIES = {
    "in_silico_or_phastcons",
    "in_silico_probability",
    "in_silico_sift_inverted",
    "in_silico_cadd_phred",
    "population_frequency",
    "conservation_or_in_silico_neg",
    "score_centered",
    "bounded_positive_other",
}

# Distance threshold for cluster cut: smaller = stricter (more clusters)
CLUSTER_DISTANCE_THRESHOLD = 0.3   # 1 - |r| < 0.3 means |r| > 0.7


def safe_auc(y, vals):
    mask = ~pd.isna(vals)
    if mask.sum() < 20 or pd.Series(y[mask]).nunique() < 2:
        return float("nan")
    try:
        a = roc_auc_score(y[mask], vals[mask])
        return float(max(a, 1 - a))
    except Exception:
        return float("nan")


def main():
    print("loading MASTER...")
    df = pd.read_csv(PANELS["general"])
    df = normalize_sentinels(df)
    df = to_categorical(df)

    cats = pd.read_csv(os.path.join(OUT_DIR, "column_categories.csv"))
    target_cols = cats[cats["claimed_category"].isin(CLUSTERABLE_CATEGORIES)]["column"].tolist()
    target_cols = [c for c in target_cols if c in df.columns]
    print(f"  clustering {len(target_cols)} numeric columns "
          f"(in-silico + population + conservation candidates)")

    X = df[target_cols].astype(float)
    y = df["Label"].values

    # Pairwise |Pearson r| (skip pairs with < 30 shared non-NaN samples)
    print("  computing pairwise |r|...")
    n_cols = len(target_cols)
    R = np.full((n_cols, n_cols), np.nan)
    valid = ~np.isnan(X.values)
    arr = X.values
    for i in range(n_cols):
        R[i, i] = 1.0
        for j in range(i + 1, n_cols):
            m = valid[:, i] & valid[:, j]
            if m.sum() < 30:
                continue
            vi = arr[m, i]; vj = arr[m, j]
            if vi.std() < 1e-9 or vj.std() < 1e-9:
                continue
            r = np.corrcoef(vi, vj)[0, 1]
            R[i, j] = R[j, i] = abs(r)

    # Fill NaN distances with 1.0 (treat as fully different = no link)
    distmat = 1.0 - np.nan_to_num(R, nan=0.0)
    # Hierarchical clustering (need a condensed distance vector)
    from scipy.spatial.distance import squareform
    np.fill_diagonal(distmat, 0)
    distmat = (distmat + distmat.T) / 2.0  # symmetrise
    condensed = squareform(distmat, checks=False)
    Z = linkage(condensed, method="average")

    # Cut tree
    labels = fcluster(Z, t=CLUSTER_DISTANCE_THRESHOLD, criterion="distance")
    print(f"  cut at distance {CLUSTER_DISTANCE_THRESHOLD} -> {len(set(labels))} clusters")

    # Build assignment table
    aucs = {c: safe_auc(y, X[c].values) for c in target_cols}
    miss_rates = {c: float(X[c].isna().mean()) for c in target_cols}
    assignments = pd.DataFrame({
        "column": target_cols,
        "cluster": labels,
        "label_auc": [aucs[c] for c in target_cols],
        "missing_rate": [miss_rates[c] for c in target_cols],
    })

    # Cluster summaries
    summaries = []
    for cid, group in assignments.groupby("cluster"):
        members = group.sort_values("label_auc", ascending=False)
        rep = members.iloc[0]["column"]
        summaries.append({
            "cluster_id": int(cid),
            "size": len(group),
            "representative": rep,
            "rep_label_auc": round(float(members.iloc[0]["label_auc"]), 4),
            "mean_label_auc": round(float(members["label_auc"].mean()), 4),
            "max_label_auc": round(float(members["label_auc"].max()), 4),
            "members_first5": ",".join(members.head(5)["column"].tolist()),
        })
    summary_df = pd.DataFrame(summaries).sort_values("size", ascending=False).reset_index(drop=True)

    # Save
    assignments.to_csv(os.path.join(OUT_DIR, "cluster_assignments.csv"), index=False)
    summary_df.to_csv(os.path.join(OUT_DIR, "clusters.csv"), index=False)

    print(f"\n  wrote cluster_assignments.csv ({len(assignments)} rows)")
    print(f"  wrote clusters.csv ({len(summary_df)} clusters)\n")

    # Print top clusters
    print("-- top 20 clusters by size --")
    print(summary_df.head(20).to_string(index=False))
    print()
    big = summary_df[summary_df["size"] > 1]
    print(f"-- clusters with size > 1: {len(big)} (covering {big['size'].sum()} cols) --")
    singletons = summary_df[summary_df["size"] == 1]
    print(f"-- singleton clusters: {len(singletons)} --")

    # Dendrogram (truncated to 30 leaves so it's readable)
    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(Z, ax=ax, truncate_mode="lastp", p=30, leaf_font_size=8, show_contracted=True)
    ax.set_title(f"Hierarchical clustering of in-silico/population/conservation cols "
                 f"(cut at d={CLUSTER_DISTANCE_THRESHOLD})")
    ax.set_ylabel("1 - |Pearson r|")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "feature_dendrogram.png"), dpi=140)
    plt.close(fig)

    # Heatmap of |r| (re-order by cluster)
    order = np.argsort(labels)
    R_sorted = R[order][:, order]
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(R_sorted, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_title("|Pearson r| matrix, columns reordered by cluster")
    plt.colorbar(im, ax=ax, shrink=0.7, label="|r|")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "feature_corr_heatmap.png"), dpi=140)
    plt.close(fig)

    print(f"\n  plots -> {PLOTS_DIR}/feature_dendrogram.png, feature_corr_heatmap.png")


if __name__ == "__main__":
    main()
