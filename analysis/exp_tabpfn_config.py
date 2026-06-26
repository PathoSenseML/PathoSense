"""Find TabPFN's best config on general before committing to the full ensemble re-run.
Default model_path='auto' already = v3 (newest). Sweep: version (v3/v2.6/v2.5), native categoricals
vs ordinal-encoded, balance_probabilities (prior-shift), n_estimators, softmax_temperature.
3 train/test splits per config (cut single-split noise). AUC on general.

Run: TABPFN_TOKEN=<key> .venv/bin/python -u analysis/exp_tabpfn_config.py
"""
import sys, os, warnings
sys.path.insert(0, "analysis"); sys.path.insert(0, "pipelines")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score
from tabpfn import TabPFNClassifier
import torch
DEV = "cuda" if torch.cuda.is_available() else "cpu"

from schema import normalize_sentinels, partition_columns, drop_all_nan_rows
from utils import detect_leaky_columns, KNOWN_LEAKY, PANELS
from predict import _engineer, _fit_own_preprocess, _apply_own_preprocess
from recipe import PANEL_RECIPE
from general_blend import _cats

V3 = "tabpfn-v3-classifier-v3_default.ckpt"
V26 = "tabpfn-v2.6-classifier-v2.6_default.ckpt"
V25 = "tabpfn-v2.5-classifier-v2.5_default.ckpt"


def build(Xtr, Xte, cf, native_cat):
    num = [c for c in Xtr.columns if c not in cf]
    A = Xtr[num].to_numpy(float); B = Xte[num].to_numpy(float)
    cat_idx = None
    if cf:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        Ac = enc.fit_transform(Xtr[cf].astype(object).where(Xtr[cf].notna(), "NA"))
        Bc = enc.transform(Xte[cf].astype(object).where(Xte[cf].notna(), "NA"))
        A = np.hstack([A, Ac]); B = np.hstack([B, Bc])
        cat_idx = list(range(len(num), len(num) + len(cf)))
    return A, B, (cat_idx if native_cat else None)


def main():
    if not os.environ.get("TABPFN_TOKEN"):
        print("set TABPFN_TOKEN"); return
    df = pd.read_csv(PANELS["general"]); df = normalize_sentinels(df); df = drop_all_nan_rows(df)
    de = _engineer(df.copy(), PANEL_RECIPE["general"]); _, num, cat, tgt = partition_columns(de)
    y = de[tgt].astype(int).reset_index(drop=True); X = de[num + cat].reset_index(drop=True)
    leaky = detect_leaky_columns(X[num], y, 0.99, verbose=False)
    X = X.drop(columns=sorted(set(leaky) | (KNOWN_LEAKY & set(X.columns))))

    configs = [
        ("v3 base (current)", dict(model_path=V3, n_estimators=8), False),
        ("v3 +native_cat",    dict(model_path=V3, n_estimators=8), True),
        ("v3 +balance",       dict(model_path=V3, n_estimators=8, balance_probabilities=True), False),
        ("v3 n_est=16",       dict(model_path=V3, n_estimators=16), False),
        ("v3 temp=0.75",      dict(model_path=V3, n_estimators=8, softmax_temperature=0.75), False),
        ("v3 temp=1.1",       dict(model_path=V3, n_estimators=8, softmax_temperature=1.1), False),
        ("v2.6 base",         dict(model_path=V26, n_estimators=8), False),
        ("v2.5 base",         dict(model_path=V25, n_estimators=8), False),
    ]
    skf = StratifiedKFold(3, shuffle=True, random_state=42)
    splits = list(skf.split(X, y))
    print(f"=== TabPFN config sweep on general (3 splits, device={DEV}) ===")
    for name, kw, native in configs:
        aucs = []
        for tr, te in splits:
            Xtr, prep = _fit_own_preprocess(X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True), PANEL_RECIPE["general"])
            Xte = _apply_own_preprocess(X.iloc[te].reset_index(drop=True), prep).reindex(columns=Xtr.columns)
            cf = _cats(Xtr)
            A, B, cidx = build(Xtr, Xte, cf, native)
            clf = TabPFNClassifier(device=DEV, random_state=42, ignore_pretraining_limits=True,
                                   categorical_features_indices=cidx, **kw)
            clf.fit(A, y.iloc[tr].values); p = clf.predict_proba(B)[:, 1]
            aucs.append(roc_auc_score(y.iloc[te].values, p))
        print(f"  {name:20s} AUC={np.mean(aucs):.4f}  (per-split {[round(a,3) for a in aucs]})", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
