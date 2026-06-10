"""EK_* conservation-block signal probe (read-only analysis, no pipeline mutation).

Measures, per panel, leakage-safe conservation signal vs Label:
  1. per-EK single-feature AUC (sign-aware) + missingness
  2. missingness-of-EK as a signal
  3. aggregate tracks: mean / max / min over rank-normalised tracks; n_present
  4. cross-method AGREEMENT: fraction of tracks above their train-quantile threshold
  5. conservation x AA-severity interaction (EK x BLOSUM/Grantham)
  6. does any derived feature beat the single best raw EK column?

AUC is computed on non-missing rows for raw cols; derived cols fill missing with a
neutral value (rank 0.5 / count 0). Rank/quantile thresholds are fit on the SAME
rows scored here (in-sample) only to gauge ceiling signal — the deployed version
would freeze them on train. We report symmetric magnitude max(auc,1-auc) and sign.
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipelines"))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from schema import normalize_sentinels, drop_all_nan_rows
from feature_engineering import engineer_aa_features
from utils import PANELS

EK_COLS = [f"EK_{i}" for i in range(1, 10)]
EK_LABEL = {
    "EK_1": "GERP++ NR", "EK_2": "GERP++ RS", "EK_3": "phyloP 470-fam(sparse)",
    "EK_4": "phastCons A", "EK_5": "phastCons B", "EK_6": "phastCons 470way",
    "EK_7": "phyloP100way vert", "EK_8": "phyloP17way primate", "EK_9": "phyloP470way mamm",
}


def auc_signed(y, v):
    """Return (auc_with_higher=pathogenic, n_used). NaN-dropped."""
    v = np.asarray(v, float); y = np.asarray(y)
    m = ~np.isnan(v)
    if m.sum() < 20 or pd.Series(y[m]).nunique() < 2 or np.nanstd(v[m]) < 1e-12:
        return np.nan, int(m.sum())
    return float(roc_auc_score(y[m], v[m])), int(m.sum())


def oof_auc(y, V, fill, n_splits=5, seed=42):
    """Honest-ish AUC for a feature that requires fitting (rank/quantile thresholds):
    fit transform on train folds, evaluate on held-out fold. V is a callable
    fit(train_df)->params, transform(df,params)->series. Here we pass precomputed
    in-sample feature (ceiling) AND an OOF version. Simpler: caller gives a fit/apply."""
    raise NotImplementedError


def rank_normalize_fit(df, cols):
    """Frozen ECDF per col: store sorted non-null values. Stateless at apply."""
    params = {}
    for c in cols:
        v = df[c].dropna().astype(float).values
        params[c] = np.sort(v) if len(v) else np.array([0.0])
    return params


def rank_normalize_apply(df, cols, params):
    """Map each value to its percentile within frozen train distribution. NaN->NaN."""
    out = pd.DataFrame(index=df.index)
    for c in cols:
        sv = params[c]
        v = df[c].astype(float).values
        pct = np.searchsorted(sv, v, side="right") / max(len(sv), 1)
        pct = pct.astype(float)
        pct[np.isnan(v)] = np.nan
        out[c] = pct
    return out


def main():
    for panel in ["general", "cancer", "pah", "cftr"]:
        df = pd.read_csv(PANELS[panel])
        df = normalize_sentinels(df)
        df = drop_all_nan_rows(df, verbose=False)
        df = engineer_aa_features(df)
        y = df["Label"].astype(int).values
        present = [c for c in EK_COLS if c in df.columns]
        n = len(df); pos = int(y.sum())
        print(f"\n{'='*78}\n{panel.upper()}  n={n}  pathogenic={pos} ({pos/n:.1%})\n{'='*78}")

        # ---- 1. per-EK raw single-feature AUC + missingness ----
        print(f"{'col':6s} {'name':22s} {'miss%':>6s} {'range':>20s} {'sAUC':>6s} {'dir':>4s} {'nUsed':>6s}")
        raw_aucs = {}
        for c in present:
            v = df[c].astype(float)
            miss = v.isna().mean()
            a, nu = auc_signed(y, v.values)
            s_auc = max(a, 1 - a) if not np.isnan(a) else np.nan
            direction = "+" if (not np.isnan(a) and a >= 0.5) else "-"
            raw_aucs[c] = s_auc
            rng = f"[{np.nanmin(v):.2f},{np.nanmax(v):.2f}]" if v.notna().any() else "NA"
            print(f"{c:6s} {EK_LABEL.get(c,''):22s} {miss*100:6.1f} {rng:>20s} "
                  f"{s_auc:6.3f} {direction:>4s} {nu:6d}")
        best_raw = max((v for v in raw_aucs.values() if not np.isnan(v)), default=np.nan)
        best_raw_col = max(raw_aucs, key=lambda k: (raw_aucs[k] if not np.isnan(raw_aucs[k]) else -1))

        # ---- 2. missingness-of-EK as signal ----
        # fraction of EK tracks missing per row, and a "all phyloP/GERP missing" flag
        miss_frac = df[present].isna().mean(axis=1).values
        a_miss, _ = auc_signed(y, miss_frac)
        print(f"\n[2] EK missing-fraction per row: sAUC={max(a_miss,1-a_miss):.3f} "
              f"(dir {'+' if a_miss>=0.5 else '-'})  "
              f"mean_missfrac path={miss_frac[y==1].mean():.3f} benign={miss_frac[y==0].mean():.3f}")

        # ---- 3. aggregate over RANK-normalised tracks (in-sample ceiling) ----
        params = rank_normalize_fit(df, present)
        ranks = rank_normalize_apply(df, present, params)
        agg = {}
        agg["mean_rank"] = ranks.mean(axis=1)              # NaN-skipping mean
        agg["max_rank"]  = ranks.max(axis=1)
        agg["min_rank"]  = ranks.min(axis=1)
        # core phyloP+GERP subset (exclude phastCons which is 0/1 heavy)
        core = [c for c in ("EK_7", "EK_9", "EK_2", "EK_1") if c in ranks.columns]
        agg["mean_rank_core"] = ranks[core].mean(axis=1) if core else pd.Series(np.nan, df.index)
        for name, s in agg.items():
            a, nu = auc_signed(y, s.values)
            print(f"[3] {name:16s} sAUC={max(a,1-a):6.3f} (dir {'+' if a>=0.5 else '-'})  nUsed={nu}")

        # ---- 4. cross-method AGREEMENT: count of tracks above train-median ----
        # frozen per-track median (here in-sample) as the 'conserved' threshold
        med = {c: np.nanmedian(df[c].astype(float)) for c in present}
        q75 = {c: np.nanquantile(df[c].astype(float), 0.75) for c in present}
        above_med = pd.DataFrame({c: (df[c].astype(float) > med[c]).astype(float) for c in present})
        above_75  = pd.DataFrame({c: (df[c].astype(float) > q75[c]).astype(float) for c in present})
        # fraction of NON-MISSING tracks above threshold
        notna = df[present].notna()
        frac_above_med = (above_med * notna).sum(axis=1) / notna.sum(axis=1).clip(lower=1)
        frac_above_75  = (above_75 * notna).sum(axis=1) / notna.sum(axis=1).clip(lower=1)
        for name, s in [("frac_above_median", frac_above_med), ("frac_above_q75", frac_above_75)]:
            a, nu = auc_signed(y, s.values)
            print(f"[4] {name:18s} sAUC={max(a,1-a):6.3f} (dir {'+' if a>=0.5 else '-'})")
        # unanimity flags
        all_above_med = ((above_med * notna).sum(axis=1) == notna.sum(axis=1)).astype(int)
        a, _ = auc_signed(y, all_above_med.values)
        print(f"[4] all_tracks_above_median  sAUC={max(a,1-a):6.3f}  "
              f"rate path={all_above_med[y==1].mean():.2f} benign={all_above_med[y==0].mean():.2f}")

        # ---- 5. conservation x AA-severity interaction ----
        cons = agg["mean_rank_core"].fillna(agg["mean_rank"])   # 0..1 consensus
        inter = {}
        if "AA_grantham" in df.columns:
            g = df["AA_grantham"].astype(float)
            gnorm = (g / 215.0).clip(0, 1)            # grantham max ~215
            inter["cons_x_grantham"] = cons * gnorm
        if "AA_blosum62" in df.columns:
            b = df["AA_blosum62"].astype(float)
            disrupt = ((4.0 - b) / 15.0).clip(0, 1)
            inter["cons_x_disrupt"] = cons * disrupt
            inter["cons_x_conservative"] = cons * (1 - disrupt)
        for name, s in inter.items():
            a, nu = auc_signed(y, s.values)
            print(f"[5] {name:20s} sAUC={max(a,1-a):6.3f} (dir {'+' if a>=0.5 else '-'})  nUsed={nu}")

        # ---- 6. verdict: best derived vs best raw ----
        derived = {**{k: auc_signed(y, v.values)[0] for k, v in agg.items()},
                   "frac_above_median": auc_signed(y, frac_above_med.values)[0],
                   "frac_above_q75": auc_signed(y, frac_above_75.values)[0],
                   **{k: auc_signed(y, v.values)[0] for k, v in inter.items()}}
        derived_sym = {k: max(v, 1 - v) for k, v in derived.items() if not np.isnan(v)}
        best_der = max(derived_sym, key=derived_sym.get)
        print(f"\n[6] best RAW = {best_raw_col} ({EK_LABEL.get(best_raw_col,'')}) sAUC={best_raw:.3f}  |  "
              f"best DERIVED = {best_der} sAUC={derived_sym[best_der]:.3f}  |  "
              f"Δ={derived_sym[best_der]-best_raw:+.3f}")


if __name__ == "__main__":
    main()
