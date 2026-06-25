"""Full-attention statistical reanalysis of all 4 panels — hunting for structural
opportunities (the kind Arc flagged: "wide columns that want a log transform").

Sections:
  A. Per-family profile (dtype, missingness, range, skew) — see the data.
  B. The log/transform question, settled with numbers: our models are ALL tree-based
     (LGBM/RF/CatBoost/ExtraTrees) and AUC is a RANK statistic, so ANY monotonic transform
     (log, rank, sqrt) leaves single-feature AUC AND tree splits unchanged. Demonstrated.
     => transforming single columns is inert for us; the lever is COMBINING columns.
  C. Candidate DERIVED features trees can't synthesize themselves, single-feature AUC vs the
     best single RAW feature, per panel:
       - EK conservation CONSENSUS (mean/median/max/min/std over EK_1..9) — denoise correlated tracks
       - AL RARITY aggregates (n databases present / n zero / max & mean AF) — absence = pathogenic
       - conservation x rarity interaction
     Plus orthogonality (|corr| to the best raw feature) — a high-AUC candidate only helps if it's
     not already captured.

Run: .venv/bin/python -u analysis/exp_data_reanalysis.py
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score
from scipy.stats import skew
from schema import normalize_sentinels, drop_all_nan_rows
from utils import PANELS

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
PANEL_LIST = ["general", "cancer", "pah", "cftr"]


def load(panel):
    df = drop_all_nan_rows(normalize_sentinels(pd.read_csv(PANELS[panel])), verbose=False)
    return df.reset_index(drop=True)


def fam(c):
    for p in ("EK_", "AL_", "AA_", "CAT_"):
        if str(c).startswith(p):
            return p[:-1]
    return "OTHER"


def auc1(y, x):
    """Single-feature AUC on non-null rows, direction-agnostic."""
    m = ~pd.isna(x)
    if m.sum() < 30 or y[m].nunique() < 2:
        return np.nan
    try:
        a = roc_auc_score(y[m], pd.Series(x)[m].astype(float))
        return max(a, 1 - a)
    except Exception:
        return np.nan


# ── A. family profile ────────────────────────────────────────────────────────
def profile():
    print("\n###### A. PER-FAMILY PROFILE ######")
    for pn in PANEL_LIST:
        df = load(pn); y = df["Label"].astype(int)
        feats = [c for c in df.columns if c not in ("Variant_ID", "Label")]
        rows = []
        for f in set(fam(c) for c in feats):
            cols = [c for c in feats if fam(c) == f]
            num = df[cols].select_dtypes(include=[np.number])
            sk = num.apply(lambda s: skew(s.dropna()) if s.notna().sum() > 10 else np.nan)
            rows.append({"fam": f, "n_cols": len(cols), "n_numeric": num.shape[1],
                         "miss%": round(100 * df[cols].isna().mean().mean(), 1),
                         "med_skew": round(float(sk.abs().median()), 1) if num.shape[1] else np.nan,
                         "max|skew|": round(float(sk.abs().max()), 1) if num.shape[1] else np.nan})
        print(f"\n[{pn}] n={len(df)}")
        print(pd.DataFrame(rows).set_index("fam").reindex(["EK", "AL", "AA", "CAT", "OTHER"]).to_string())


# ── B. monotonic-transform invariance ────────────────────────────────────────
def transform_invariance():
    print("\n###### B. DOES LOG/RANK HELP? (single-feature AUC under monotonic transforms) ######")
    df = load("general"); y = df["Label"].astype(int)
    num = df.select_dtypes(include=[np.number]).drop(columns=["Label"], errors="ignore")
    # pick the 8 most skewed numeric columns (the ones that "look like they want a log")
    sk = num.apply(lambda s: abs(skew(s.dropna())) if s.notna().sum() > 30 else 0).sort_values(ascending=False)
    cols = list(sk.head(8).index)
    print(f"{'column':10s} {'|skew|':>7s} {'AUC raw':>8s} {'AUC log':>8s} {'AUC rank':>9s} {'AUC sqrt':>9s}")
    for c in cols:
        x = df[c].astype(float)
        shift = x.min(skipna=True)
        xl = np.log1p(x - shift) if pd.notna(shift) else x
        xr = x.rank()
        xs = np.sqrt(x - shift) if pd.notna(shift) else x
        print(f"{c:10s} {sk[c]:>7.1f} {auc1(y, x):>8.4f} {auc1(y, xl):>8.4f} "
              f"{auc1(y, xr):>9.4f} {auc1(y, xs):>9.4f}")
    print("  -> AUC is a rank statistic; tree splits are rank-based. Monotonic transforms are INERT")
    print("     for our tree ensemble. (This is why every log-transform experiment was rejected.)")


# ── C. candidate derived features ────────────────────────────────────────────
def candidate_features():
    print("\n###### C. CANDIDATE DERIVED FEATURES (single-feature AUC vs best RAW) ######")
    allrows = []
    for pn in PANEL_LIST:
        df = load(pn); y = df["Label"].astype(int)
        ek = [c for c in df.columns if c.startswith("EK_")]
        al = [c for c in df.columns if c.startswith("AL_")]
        af = [c for c in al if ((df[c] > 0) & (df[c] <= 1)).any() and df[c].mean(skipna=True) < 0.1]

        cand = {}
        # EK conservation consensus (trees can't average 9 correlated tracks)
        E = df[ek].astype(float)
        cand["EK_mean"] = E.mean(axis=1); cand["EK_median"] = E.median(axis=1)
        cand["EK_max"] = E.max(axis=1);   cand["EK_min"] = E.min(axis=1)
        cand["EK_std"] = E.std(axis=1);   cand["EK_range"] = E.max(axis=1) - E.min(axis=1)
        # AL rarity aggregates (absence from population DBs => pathogenic)
        cand["AL_n_present"] = df[al].notna().sum(axis=1)
        cand["AL_n_zero"] = (df[al] == 0).sum(axis=1)
        if af:
            cand["AL_maxAF"] = df[af].max(axis=1); cand["AL_meanAF"] = df[af].mean(axis=1)
        # conservation x rarity (conserved AND rare = most pathogenic)
        maxaf = df[af].max(axis=1).fillna(0.0) if af else 0.0
        cand["EKmean_x_rare"] = cand["EK_mean"] * (1 - maxaf)

        best_ek = max((auc1(y, df[c]) for c in ek), default=np.nan)
        best_al = max((auc1(y, df[c]) for c in al), default=np.nan)
        bestraw = np.nanmax([best_ek, best_al])
        rows = []
        for name, x in cand.items():
            a = auc1(y, x)
            # orthogonality: |corr| of candidate to the single best raw feature
            ref = max(ek + al, key=lambda c: (auc1(y, df[c]) or 0))
            corr = abs(pd.Series(x).corr(df[ref].astype(float)))
            rows.append((name, a, a - bestraw, corr))
        rows.sort(key=lambda r: -(r[1] if not np.isnan(r[1]) else 0))
        print(f"\n[{pn}]  best raw single-feat AUC: EK={best_ek:.4f} AL={best_al:.4f}")
        print(f"  {'candidate':14s} {'AUC':>7s} {'Δ vs best-raw':>13s} {'|corr| to best-raw':>18s}")
        for name, a, d, corr in rows:
            flag = "  <== beats best raw" if (not np.isnan(d) and d > 0.005) else ""
            print(f"  {name:14s} {a:>7.4f} {d:>+13.4f} {corr:>18.3f}{flag}")
            allrows.append({"panel": pn, "candidate": name, "auc": a, "delta_vs_bestraw": d,
                            "corr_to_bestraw": corr})
    pd.DataFrame(allrows).to_csv(os.path.join(OUT, "reanalysis_candidates.csv"), index=False)


if __name__ == "__main__":
    print("###### FULL DATA REANALYSIS ######", flush=True)
    profile()
    transform_invariance()
    candidate_features()
    print(f"\nsaved -> {os.path.join(OUT, 'reanalysis_candidates.csv')}", flush=True)
