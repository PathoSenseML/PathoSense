"""FULL-column correlation heatmaps, done properly.

Why the existing scripts are not enough:
  * make_heatmaps.py uses only the top-32 label-relevant features and does
    `corr(min_periods=30).fillna(0)` -> sparse AL_ pairs (57% NaN, 0 fully-present
    cols) silently become 0, HIDING the redundant families.
  * cluster_features.py covers only the in-silico subset and likewise zeroes low-
    overlap pairs.

"Properly" here means, for ALL feature columns (AL_/EK_/CAT_/AA_):
  1. ALL columns, not a top-k subset.
  2. SPEARMAN value correlation — AL_ is an organizer-transformed (rank-preserving)
     allele frequency and EK_ conservation is skewed, so rank corr is the honest
     measure; categoricals are ordinal-encoded by frequency (no label leak).
  3. HONEST sparse handling — pairs with < MIN_OVERLAP co-present rows are NOT
     zeroed; they are rendered GREY (masked) so missing structure is visible.
  4. A second CO-OCCURRENCE (missingness) heatmap — for a 57%-NaN block the
     present/absent pattern is itself the database x population family structure;
     value-corr alone misses it. This matrix is complete (no NaN).
  5. Hierarchical clustering (avg linkage on 1-|r|) to surface the blocks, plus a
     family colour ribbon so the 351x351 grid is readable without per-cell labels.
  6. A redundancy report: connected components at |r| > 0.99 (overlap >= MIN_OVERLAP)
     -> the genuinely-duplicated clusters, written to CSV.

Run: .venv/bin/python -u analysis/make_full_heatmap.py [panel ...]   (default: general)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from schema import normalize_sentinels, drop_all_nan_rows, to_categorical
from utils import PANELS

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "heatmaps")
os.makedirs(OUT, exist_ok=True)
REDUNDANT_R = 0.99         # |r| threshold for "duplicate" clustering
FAM_COLORS = {"AL": "#d62728", "EK": "#1f77b4", "CAT": "#9467bd", "AA": "#2ca02c"}


def family(c):
    c = str(c)
    return ("AL" if c.startswith("AL_") else "EK" if c.startswith("EK_")
            else "CAT" if c.startswith("CAT_") else "AA" if c.startswith("AA_") else "FX")


def encode(df):
    """All feature columns -> numeric, NaN PRESERVED. Categoricals ordinal by frequency."""
    feat = [c for c in df.columns if family(c) in ("AL", "EK", "CAT", "AA")]
    E = pd.DataFrame(index=df.index)
    for c in feat:
        s = df[c]
        if s.dtype == object or str(s.dtype) == "category":
            order = {v: i for i, v in enumerate(s.value_counts().index)}   # freq rank, no label use
            E[c] = s.map(order).astype(float)
        else:
            E[c] = pd.to_numeric(s, errors="coerce")
    # drop zero-variance columns (constant or all-NaN) — undefined correlation
    keep = [c for c in feat if E[c].notna().sum() >= 2 and E[c].nunique(dropna=True) >= 2]
    return E[keep], keep


def cluster_order(absR):
    """Leaf order from average-linkage hierarchical clustering on 1-|r| (NaN->0 link)."""
    D = 1.0 - np.nan_to_num(absR, nan=0.0)
    np.fill_diagonal(D, 0.0); D = (D + D.T) / 2.0
    Z = linkage(squareform(D, checks=False), method="average")
    return leaves_list(Z)


def redundant_clusters(R, overlap, cols, min_overlap):
    """Connected components where |r| > REDUNDANT_R and overlap >= min_overlap."""
    n = len(cols); adj = (np.abs(R) > REDUNDANT_R) & (overlap >= min_overlap)
    np.fill_diagonal(adj, False)
    seen, comps = set(), []
    for i in range(n):
        if i in seen or not adj[i].any():
            continue
        stack, comp = [i], []
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u); comp.append(u)
            stack.extend(np.where(adj[u])[0].tolist())
        if len(comp) > 1:
            comps.append(sorted(comp))
    comps.sort(key=len, reverse=True)
    return comps


def _ribbon(ax, fams, horizontal):
    """Draw a thin family colour strip aligned to the (already-ordered) columns."""
    keys = list(FAM_COLORS); idx = np.array([keys.index(f) if f in keys else len(keys) for f in fams])
    cmap = ListedColormap([FAM_COLORS[k] for k in keys] + ["#bbbbbb"])
    norm = BoundaryNorm(range(len(keys) + 2), cmap.N)
    img = idx.reshape(1, -1) if horizontal else idx.reshape(-1, 1)
    ax.imshow(img, cmap=cmap, norm=norm, aspect="auto"); ax.set_xticks([]); ax.set_yticks([])


def heatmap(M, cols, fams, title, fname, cbar_label, cmap, vmin, vmax, min_overlap):
    """351x351 imshow with family ribbons top+left, colorbar, legend. No per-cell labels."""
    Mm = np.ma.masked_invalid(M)
    cm = plt.get_cmap(cmap).copy(); cm.set_bad("#dddddd")          # NaN/low-overlap -> light grey
    fig = plt.figure(figsize=(16, 15))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.4, 20], height_ratios=[0.4, 20],
                          wspace=0.01, hspace=0.01)
    ax_top = fig.add_subplot(gs[0, 1]); _ribbon(ax_top, fams, True)
    ax_left = fig.add_subplot(gs[1, 0]); _ribbon(ax_left, fams, False)
    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(Mm, cmap=cm, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(f"{len(cols)} feature columns")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.set_label(cbar_label)
    leg = [Patch(color=v, label=k) for k, v in FAM_COLORS.items()] + [Patch(color="#dddddd", label="n<%d (grey)" % min_overlap)]
    ax.legend(handles=leg, loc="upper right", fontsize=9, framealpha=0.9, title="family")
    ax_top.set_title(title, fontsize=14, weight="bold", loc="left")
    path = os.path.join(OUT, fname); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
    return path


def run(panel):
    df = drop_all_nan_rows(normalize_sentinels(pd.read_csv(PANELS[panel])), verbose=False)
    df = to_categorical(df)
    E, cols = encode(df)
    n_cols = len(cols); fams0 = [family(c) for c in cols]
    mino = int(max(20, min(50, len(df) // 15)))                       # adapt overlap floor to panel size
    print(f"\n[{panel}] {len(df)} rows x {n_cols} feature cols "
          f"({', '.join(f'{k}:{fams0.count(k)}' for k in FAM_COLORS)}); min_overlap={mino}", flush=True)

    # ---- value Spearman correlation (pairwise, honest low-overlap mask) ----
    Rv = E.corr(method="spearman", min_periods=mino).values           # NaN where overlap<mino
    pres = E.notna().values.astype(np.uint8)
    overlap = pres.T @ pres                                            # co-present counts per pair
    order = cluster_order(np.where(np.isnan(Rv), 0.0, np.abs(Rv)))
    cols_o = [cols[i] for i in order]; fams_o = [fams0[i] for i in order]
    Rv_o = Rv[np.ix_(order, order)]
    p1 = heatmap(Rv_o, cols_o, fams_o,
                 f"{panel.upper()} — Spearman value correlation, ALL {n_cols} cols "
                 f"(clustered; grey = <{mino} co-present rows)",
                 f"heatmap_{panel}_value_spearman.png", "Spearman ρ", "RdBu_r", -1, 1, mino)

    # ---- co-occurrence (missingness) phi correlation — complete, reveals families ----
    pres_df = E.notna().astype(float)
    pres_df = pres_df.loc[:, pres_df.nunique() > 1]                    # drop all-present/all-absent
    Rm = pres_df.corr().values; mcols = list(pres_df.columns); mfam = [family(c) for c in mcols]
    mo = cluster_order(np.abs(Rm))
    Rm_o = Rm[np.ix_(mo, mo)]; mfam_o = [mfam[i] for i in mo]
    p2 = heatmap(Rm_o, [mcols[i] for i in mo], mfam_o,
                 f"{panel.upper()} — co-occurrence (present/absent) phi corr, {len(mcols)} cols "
                 f"(clustered) — shows DB x population families",
                 f"heatmap_{panel}_cooccurrence.png", "phi (φ)", "RdBu_r", -1, 1, mino)

    # ---- redundancy report ----
    comps = redundant_clusters(Rv, overlap, cols, mino)
    n_dup = sum(len(c) for c in comps)
    rows = []
    for k, comp in enumerate(comps):
        members = [cols[i] for i in comp]
        rows.append({"cluster": k, "size": len(comp), "families": "/".join(sorted(set(family(m) for m in members))),
                     "members": ",".join(members[:12]) + ("..." if len(members) > 12 else "")})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"redundant_clusters_{panel}.csv"), index=False)
    nzero = int((~np.isnan(Rv)).sum() - n_cols)
    npairs = n_cols * (n_cols - 1)
    print(f"  value-corr: {100*nzero/npairs:.1f}% of pairs have >= {mino} overlap "
          f"(rest grey, NOT zeroed)", flush=True)
    print(f"  redundancy: {len(comps)} clusters of |ρ|>{REDUNDANT_R} covering {n_dup} cols "
          f"(largest {len(comps[0]) if comps else 0}); -> redundant_clusters_{panel}.csv", flush=True)
    print(f"  saved -> {os.path.basename(p1)}, {os.path.basename(p2)}", flush=True)
    return p1, p2


if __name__ == "__main__":
    panels = sys.argv[1:] or ["general"]
    for p in panels:
        run(p)
    print(f"\nall heatmaps in {OUT}", flush=True)
