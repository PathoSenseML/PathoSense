"""Lean multivariate OOF-AUC: does a derived conservation feature add over raw EK?"""
import sys, os, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipelines"))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from schema import normalize_sentinels, drop_all_nan_rows
from feature_engineering import engineer_aa_features
from utils import PANELS

EK = [f"EK_{i}" for i in range(1, 10)]


def rank_apply(df, cols):
    out = pd.DataFrame(index=df.index)
    for c in cols:
        sv = np.sort(df[c].dropna().astype(float).values)
        v = df[c].astype(float).values
        pct = np.searchsorted(sv, v, side="right") / max(len(sv), 1)
        pct = pct.astype(float); pct[np.isnan(v)] = np.nan
        out[c] = pct
    return out


def oof(X, y, reps=3):
    y = np.asarray(y); aucs = []
    for r in range(reps):
        o = np.zeros(len(y))
        skf = StratifiedKFold(5, shuffle=True, random_state=42 + r)
        spw = (y == 0).sum() / max((y == 1).sum(), 1)
        for tr, te in skf.split(X, y):
            m = LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=15,
                               min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
                               reg_lambda=1.0, scale_pos_weight=spw, n_jobs=2,
                               random_state=42 + r, verbose=-1)
            m.fit(X.iloc[tr], y[tr])
            o[te] = m.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y, o))
    return np.mean(aucs), np.std(aucs)


def main():
    for panel in ["general", "cancer", "pah", "cftr"]:
        t0 = time.time()
        df = pd.read_csv(PANELS[panel])
        df = normalize_sentinels(df); df = drop_all_nan_rows(df, verbose=False)
        df = engineer_aa_features(df)
        y = df["Label"].astype(int).values
        present = [c for c in EK if c in df.columns]
        ranks = rank_apply(df, present)
        core = [c for c in ("EK_7", "EK_9", "EK_2") if c in ranks.columns]

        derived = pd.DataFrame(index=df.index)
        derived["d_mean_core"] = ranks[core].mean(axis=1)
        derived["d_disagree_core"] = ranks[core].std(axis=1)
        if "EK_7" in ranks and "EK_9" in ranks:
            derived["d_min79"] = ranks[["EK_7", "EK_9"]].min(axis=1)
        cons = ranks[core].mean(axis=1).fillna(ranks.mean(axis=1))
        if "AA_blosum62" in df.columns:
            disrupt = ((4.0 - df["AA_blosum62"].astype(float)) / 15.0).clip(0, 1)
            derived["d_cons_x_disrupt"] = cons * disrupt
        if "AA_grantham" in df.columns:
            derived["d_cons_x_grantham"] = cons * (df["AA_grantham"].astype(float) / 215).clip(0, 1)

        Xek = df[present].astype(float)
        aa_num = [c for c in df.columns if c.startswith("AA_")
                  and c not in ("AA_1", "AA_2") and pd.api.types.is_numeric_dtype(df[c])]
        Xeka = df[present + aa_num].astype(float)

        m0, s0 = oof(Xek, y)
        m1, s1 = oof(pd.concat([Xek, derived], axis=1), y)
        m2, s2 = oof(Xeka, y)
        m3, s3 = oof(pd.concat([Xeka, derived], axis=1), y)
        # isolate each derived feature added to EK+AA
        per = {}
        for k in derived.columns:
            mk, _ = oof(pd.concat([Xeka, derived[[k]]], axis=1), y)
            per[k] = mk - m2
        print(f"\n{panel.upper()}  n={len(df)}  ({time.time()-t0:.0f}s)", flush=True)
        print(f"  EK only       AUC={m0:.4f}±{s0:.3f}")
        print(f"  EK +derived   AUC={m1:.4f}±{s1:.3f}  (Δ{m1-m0:+.4f})")
        print(f"  EK+AA         AUC={m2:.4f}±{s2:.3f}")
        print(f"  EK+AA+derived AUC={m3:.4f}±{s3:.3f}  (Δ{m3-m2:+.4f})")
        print(f"  per-feature Δ over EK+AA: " +
              "  ".join(f"{k}={v:+.4f}" for k, v in sorted(per.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
