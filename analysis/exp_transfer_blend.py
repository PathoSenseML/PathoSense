"""Does a BLENDED transfer scorer (LGBM+RF+CatBoost on general-minus-target) beat the
single-LGBM transfer scorer? pah (w=1.0) and cftr (w=0.75) lean heavily on transfer,
so a stronger/decorrelated transfer scorer directly raises their F1*. AUC is the
rank-based quality signal (threshold-independent), compared per panel.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import lightgbm as lgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from config import SEED
from schema import to_categorical, partition_columns
from feature_engineering import engineer_aa_features
from utils import detect_leaky_columns, KNOWN_LEAKY
from run_pipeline import get_model_params
from preprocess_v2 import fit_keep_nan, apply_keep_nan
from exp_transfer import _load


def build_matrices(panel):
    """general-minus-target keep_nan matrix + projected target (improved recipe)."""
    tids = set(_load(panel)["Variant_ID"])
    dg = _load("general"); dg = dg[~dg["Variant_ID"].isin(tids)].reset_index(drop=True)
    dg = to_categorical(dg); dg = engineer_aa_features(dg)
    _, gn, gc, gt = partition_columns(dg)
    yg = dg[gt].reset_index(drop=True).astype(int); Xg = dg[gn + gc].reset_index(drop=True)
    leaky = sorted(set(detect_leaky_columns(Xg, yg, 0.99, verbose=False)) | (KNOWN_LEAKY & set(Xg.columns)))
    Xg = Xg.drop(columns=leaky, errors="ignore")
    dt = _load(panel); dt = to_categorical(dt); dt = engineer_aa_features(dt)
    yt = dt["Label"].reset_index(drop=True).astype(int); Xt = dt.reindex(columns=Xg.columns)
    Xg_pp, ppf = fit_keep_nan(Xg); Xt_pp = apply_keep_nan(Xt, ppf)
    cats = [c for c in Xg_pp.columns if isinstance(Xg_pp[c].dtype, pd.CategoricalDtype)]
    for c in cats:
        Xt_pp[c] = pd.Categorical(Xt_pp[c].astype(object), categories=Xg_pp[c].cat.categories)
    return Xg_pp, yg, Xt_pp, yt, cats


def lgb_score(Xg, yg, Xt, cats):
    spw = (yg == 0).sum() / max((yg == 1).sum(), 1)
    params = get_model_params(len(Xg), spw)
    i2, ies = train_test_split(np.arange(len(yg)), test_size=0.15, random_state=SEED+7, stratify=yg)
    m = LGBMClassifier(**params, random_state=SEED)
    m.fit(Xg.iloc[i2], yg.iloc[i2], eval_set=[(Xg.iloc[ies], yg.iloc[ies])],
          callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
          categorical_feature=cats or "auto")
    return m.predict_proba(Xt)[:, 1]


def rf_score(Xg, yg, Xt, cats):
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-2)
    Xg2, Xt2 = Xg.copy(), Xt.copy()
    if cats:
        Xg2[cats] = enc.fit_transform(Xg[cats].astype(object).where(Xg[cats].notna(), "NA"))
        Xt2[cats] = enc.transform(Xt[cats].astype(object).where(Xt[cats].notna(), "NA"))
    Xg2 = Xg2.fillna(-999.0); Xt2 = Xt2.fillna(-999.0)
    m = RandomForestClassifier(n_estimators=600, max_features="sqrt", n_jobs=6,
                               class_weight="balanced", random_state=SEED).fit(Xg2, yg)
    return m.predict_proba(Xt2)[:, 1]


def cat_score(Xg, yg, Xt, cats):
    spw = (yg == 0).sum() / max((yg == 1).sum(), 1)
    Xg2, Xt2 = Xg.copy(), Xt.copy()
    for c in cats:
        Xg2[c] = Xg[c].astype(object).where(Xg[c].notna(), "NA").astype(str)
        Xt2[c] = Xt[c].astype(object).where(Xt[c].notna(), "NA").astype(str)
    idx = [Xg2.columns.get_loc(c) for c in cats]
    i2, ies = train_test_split(np.arange(len(yg)), test_size=0.15, random_state=SEED+5, stratify=yg)
    m = CatBoostClassifier(iterations=1000, depth=6, learning_rate=0.05, scale_pos_weight=spw,
                           cat_features=idx, random_seed=SEED, verbose=0, allow_writing_files=False,
                           early_stopping_rounds=40, thread_count=6)
    m.fit(Xg2.iloc[i2], yg.iloc[i2], eval_set=(Xg2.iloc[ies], yg.iloc[ies]))
    return m.predict_proba(Xt2)[:, 1]


if __name__ == "__main__":
    for panel in (sys.argv[1:] or ["pah", "cftr", "cancer"]):
        t0 = time.time()
        Xg, yg, Xt, yt, cats = build_matrices(panel)
        pl = lgb_score(Xg, yg, Xt, cats); pr = rf_score(Xg, yg, Xt, cats); pc = cat_score(Xg, yg, Xt, cats)
        a = lambda p: roc_auc_score(yt, p)
        rk = lambda p: rankdata(p)        # rank-average for the blend (scale-free)
        blend = (rk(pl) + rk(pr) + rk(pc)) / 3
        blend2 = (rk(pl) + rk(pc)) / 2    # lgbm+catboost (drop RF if it's weak)
        print(f"{panel:7s} LGBM={a(pl):.4f} RF={a(pr):.4f} Cat={a(pc):.4f} | "
              f"blend(L+R+C)={a(blend):.4f} blend(L+C)={a(blend2):.4f}  [{time.time()-t0:.0f}s]", flush=True)
