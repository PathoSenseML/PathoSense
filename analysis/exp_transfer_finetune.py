"""#3 — LightGBM continued-training fine-tuning (real transfer learning vs fixed blend).
Train scorer on general (excl target IDs), then CONTINUE boosting on the target panel's own rows
(init_model). AUC probe first (threshold-free, cheap): does fine-tuning improve the scorer's
held-out AUC on the target? If yes (>=2sigma), build the full honest F1* pipeline.

Run:  .venv/bin/python -u analysis/exp_transfer_finetune.py [pah cftr cancer]
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from lightgbm import LGBMClassifier
from config import SEED
from schema import to_categorical
from feature_engineering import engineer_aa_features
from preprocess_v2 import apply_keep_nan, apply_al
from exp_transfer import _load, load_target_engineered
from transfer import fit_transfer_scorer


def project_matrix(df_raw, art):
    df = to_categorical(df_raw.copy()); df = engineer_aa_features(df)
    X = df.reindex(columns=art["raw_cols"]); Xs = apply_keep_nan(X, art["ppf"])
    if art["alf"] is not None:
        Xs = pd.concat([Xs.reset_index(drop=True), apply_al(X, art["alf"]).reset_index(drop=True)], axis=1)
        Xs = Xs.loc[:, ~Xs.columns.duplicated()]
    for col in art["feats"]:
        if col not in Xs.columns:
            Xs[col] = np.nan
    Xs = Xs[art["feats"]]
    for col, cats in art["cat_cats"].items():
        Xs[col] = pd.Categorical(Xs[col].astype(object), categories=cats)
    return Xs


def run(panel, K_list=(30, 80, 150), lr=0.02, n_repeats=10):
    art = fit_transfer_scorer(panel)
    base = art["models"][0]
    cat_feats = [c for c in art["feats"] if c in art["cat_cats"]]
    Xs = project_matrix(_load(panel), art)
    _, y = load_target_engineered(panel)
    print(f"\n[{panel}] base scorer AUC (general) baseline below; fine-tune adds K trees on target",
          flush=True)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED)
    base_a, ft_a = [], {K: [] for K in K_list}
    for tr, te in rskf.split(Xs, y):
        ytr = y.iloc[tr]
        base_a.append(roc_auc_score(y.iloc[te], base.predict_proba(Xs.iloc[te])[:, 1]))
        spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
        for K in K_list:
            ft = LGBMClassifier(n_estimators=K, learning_rate=lr, num_leaves=15,
                                min_child_samples=10, reg_lambda=2.0, scale_pos_weight=spw,
                                random_state=SEED, n_jobs=4, verbose=-1)
            ft.fit(Xs.iloc[tr], ytr, init_model=base.booster_,
                   categorical_feature=cat_feats or "auto")
            ft_a[K].append(roc_auc_score(y.iloc[te], ft.predict_proba(Xs.iloc[te])[:, 1]))
    b = np.array(base_a); nf = len(b)
    print(f"  base AUC={b.mean():.4f}  ({nf} folds)", flush=True)
    for K in K_list:
        a = np.array(ft_a[K]); d = a - b
        se = d.std(ddof=1) / np.sqrt(nf) if nf > 1 else 0.0
        sig = d.mean() / se if se > 0 else 0.0
        print(f"  +{K:3d} trees AUC={a.mean():.4f}  ΔAUC={d.mean():+.4f} ({sig:+.2f}σ)", flush=True)


if __name__ == "__main__":
    for pn in (sys.argv[1:] or ["pah", "cftr", "cancer"]):
        run(pn)
