"""EK_* probe round 2: disagreement, EK7/EK9 combos, per-col missingness, and a
MULTIVARIATE OOF-AUC test (does a derived conservation feature add over the raw
EK block a tree already sees?).
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipelines"))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier

from schema import normalize_sentinels, drop_all_nan_rows
from feature_engineering import engineer_aa_features
from utils import PANELS

EK_COLS = [f"EK_{i}" for i in range(1, 10)]


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


def oof_auc(X, y, seed=42, n_splits=5):
    """5-fold OOF AUC of a small LightGBM on feature matrix X."""
    y = np.asarray(y); oof = np.zeros(len(y))
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    for tr, te in skf.split(X, y):
        m = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
                           min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
                           reg_lambda=1.0, scale_pos_weight=spw, random_state=seed,
                           verbose=-1)
        m.fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    return roc_auc_score(y, oof)


def main():
    for panel in ["general", "cancer", "pah", "cftr"]:
        df = pd.read_csv(PANELS[panel])
        df = normalize_sentinels(df); df = drop_all_nan_rows(df, verbose=False)
        df = engineer_aa_features(df)
        y = df["Label"].astype(int).values
        present = [c for c in EK_COLS if c in df.columns]
        print(f"\n{'='*70}\n{panel.upper()}  n={len(df)}  P={y.sum()}\n{'='*70}")

        ranks = rank_apply(df, present)
        core = [c for c in ("EK_7", "EK_9", "EK_2") if c in ranks.columns]

        # --- disagreement: std across rank-normalised tracks ---
        disagree = ranks.std(axis=1)
        disagree_core = ranks[core].std(axis=1) if len(core) > 1 else pd.Series(np.nan, df.index)
        print(f"[A] disagree_std(all)   sAUC={sauc(y, disagree.values):.3f}")
        print(f"[A] disagree_std(core)  sAUC={sauc(y, disagree_core.values):.3f}")
        # signed: phyloP100way rank minus phyloP17way rank (deep-vs-primate conflict)
        if "EK_7" in ranks and "EK_8" in ranks:
            deep_vs_prim = ranks["EK_7"] - ranks["EK_8"]
            print(f"[A] EK7-EK8 (deep>primate) sAUC={sauc(y, deep_vs_prim.values):.3f}")

        # --- EK7/EK9 combos (two strongest tracks) ---
        if "EK_7" in ranks and "EK_9" in ranks:
            mn = ranks[["EK_7", "EK_9"]].min(axis=1)
            mx = ranks[["EK_7", "EK_9"]].max(axis=1)
            pr = ranks["EK_7"] * ranks["EK_9"]
            print(f"[B] min(EK7r,EK9r)  sAUC={sauc(y, mn.values):.3f}   "
                  f"max sAUC={sauc(y, mx.values):.3f}   prod sAUC={sauc(y, pr.values):.3f}   "
                  f"(EK7 {sauc(y, df['EK_7'].values):.3f} EK9 {sauc(y, df['EK_9'].values):.3f})")

        # --- per-column missing indicator signal ---
        print("[C] per-col missing-indicator sAUC (and miss%):")
        for c in present:
            miss = df[c].isna().astype(int)
            if miss.nunique() < 2:
                continue
            a = sauc(y, miss.values)
            mr = df[c].isna().mean()
            mp = df.loc[y == 1, c].isna().mean(); mb = df.loc[y == 0, c].isna().mean()
            if not np.isnan(a) and a >= 0.54:
                print(f"     {c}: miss-ind sAUC={a:.3f}  miss%={mr*100:.0f}  "
                      f"(path miss {mp*100:.0f}% / benign miss {mb*100:.0f}%)")

        # --- MULTIVARIATE OOF: raw EK block vs +derived ---
        Xraw = df[present].astype(float).copy()
        oof_raw = oof_auc(Xraw, y)
        # add the best derived candidates
        Xplus = Xraw.copy()
        Xplus["d_mean_core"] = ranks[core].mean(axis=1)
        Xplus["d_disagree"] = disagree
        if "EK_7" in ranks and "EK_9" in ranks:
            Xplus["d_min79"] = ranks[["EK_7", "EK_9"]].min(axis=1)
        if "AA_blosum62" in df.columns:
            cons = ranks[core].mean(axis=1).fillna(ranks.mean(axis=1))
            disrupt = ((4.0 - df["AA_blosum62"].astype(float)) / 15.0).clip(0, 1)
            Xplus["d_cons_x_disrupt"] = cons * disrupt
        oof_plus = oof_auc(Xplus, y)
        # also: full AA+EK baseline vs +derived (closer to deployed conditions)
        aa_cols = [c for c in df.columns if c.startswith("AA_") and df[c].dtype != object
                   and not isinstance(df[c].dtype, pd.CategoricalDtype)]
        Xbase = df[present + aa_cols].astype(float).copy()
        Xbase2 = Xbase.copy()
        for k in ("d_mean_core", "d_disagree", "d_min79", "d_cons_x_disrupt"):
            if k in Xplus:
                Xbase2[k] = Xplus[k]
        ob, ob2 = oof_auc(Xbase, y), oof_auc(Xbase2, y)
        print(f"[D] OOF-AUC EKraw={oof_raw:.4f} -> +derived={oof_plus:.4f} (Δ{oof_plus-oof_raw:+.4f})")
        print(f"[D] OOF-AUC EK+AA={ob:.4f} -> +derived={ob2:.4f} (Δ{ob2-ob:+.4f})")


if __name__ == "__main__":
    main()
