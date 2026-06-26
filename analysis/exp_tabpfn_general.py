"""Evaluate TabPFN v2 on the general panel — the one untested model that could move AUC past the
information ceiling (it's SOTA for small tabular: 2815 samples, ~500 feats is its sweet spot).

REQUIRES a one-time license + token:  export TABPFN_TOKEN="<key>"  (from https://ux.priorlabs.ai)

Honest per-fold CV (native-NaN preprocessing, cats ordinal-encoded), AUC + F1* at prior 0.143, 3 seeds.
Reports TabPFN alone, and TabPFN added to the deployed 4-tree blend, vs the deployed blend.
F1>0.55 needs AUC ~0.866 (deployed ~0.864) — so we need TabPFN to lift blended AUC by ~+0.003.

Run: TABPFN_TOKEN=<key> .venv/bin/python -u analysis/exp_tabpfn_general.py
"""
import sys, os, warnings
sys.path.insert(0, "analysis"); sys.path.insert(0, "pipelines")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

from config import SEED
from schema import normalize_sentinels, partition_columns, drop_all_nan_rows
from utils import detect_leaky_columns, KNOWN_LEAKY, TEST_PRIORS, PANELS
from predict import _engineer, _fit_own_preprocess, _apply_own_preprocess, _lgbm_redun_keep
from recipe import PANEL_RECIPE
from general_blend import _cats, _rf_repr
from calibration import fit_platt, apply_platt
from harness import f1_at_prior, tpr_fpr, pick_threshold_at_prior
from exp_general_push_055 import _oof_test, _f1_swept

PANEL = "general"; PRIOR = TEST_PRIORS[PANEL]
try:
    import torch; DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"


# Best config from exp_tabpfn_config.py: v2.6 (beats v3 +0.0054 on general) + native cats + n_est=16.
TABPFN_MODEL = "tabpfn-v2.6-classifier-v2.6_default.ckpt"
TABPFN_NEST = 16


def _tabpfn_member(Xtr, ytr, Xte, cats, seed):
    """Calibrated (train-OOF, test) for TabPFN v2.6. Native-NaN numeric kept; cats passed NATIVE."""
    from tabpfn import TabPFNClassifier
    num = [c for c in Xtr.columns if c not in cats]
    A = Xtr[num].to_numpy(dtype=float); B = Xte[num].to_numpy(dtype=float)   # TabPFN handles NaN
    cidx = None
    if cats:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        Ac = enc.fit_transform(Xtr[cats].astype(object).where(Xtr[cats].notna(), "NA"))
        Bc = enc.transform(Xte[cats].astype(object).where(Xte[cats].notna(), "NA"))
        A = np.hstack([A, Ac]); B = np.hstack([B, Bc])
        cidx = list(range(len(num), len(num) + len(cats)))

    def fp(Xa, ya, Xb):
        clf = TabPFNClassifier(device=DEVICE, random_state=seed, ignore_pretraining_limits=True,
                               model_path=TABPFN_MODEL, n_estimators=TABPFN_NEST,
                               categorical_features_indices=cidx)
        clf.fit(Xa, ya); return clf.predict_proba(Xb)[:, 1]

    oof = np.zeros(len(ytr)); skf = StratifiedKFold(3, shuffle=True, random_state=seed)
    for a, b in skf.split(A, ytr):
        oof[b] = fp(A[a], ytr.iloc[a].values, A[b])
    pl = fit_platt(oof, ytr.values)
    return apply_platt(oof, pl), apply_platt(fp(A, ytr.values, B), pl)


def main():
    if not os.environ.get("TABPFN_TOKEN"):
        print("ERROR: set TABPFN_TOKEN (one-time license at https://ux.priorlabs.ai). Aborting."); return
    print(f"TabPFN eval on general — device={DEVICE}")
    df = pd.read_csv(PANELS[PANEL]); df = normalize_sentinels(df); df = drop_all_nan_rows(df)
    de = _engineer(df.copy(), PANEL_RECIPE[PANEL]); _, num, cat, tgt = partition_columns(de)
    y = de[tgt].astype(int).reset_index(drop=True); X = de[num + cat].reset_index(drop=True)
    leaky = detect_leaky_columns(X[num], y, auc_threshold=0.99, verbose=False)
    X = X.drop(columns=sorted(set(leaky) | (KNOWN_LEAKY & set(X.columns))))

    res = {k: {"auc": [], "f1o": [], "f1h": []} for k in ["4tree(dep)", "tabpfn", "4tree+tabpfn"]}
    for seed in (SEED, SEED + 1000, SEED + 2000):
        outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=seed)
        for tr, te in outer.split(X, y):
            Xtr0, ytr = X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True)
            Xte0, yte = X.iloc[te].reset_index(drop=True), y.iloc[te].reset_index(drop=True)
            Xtr, prep = _fit_own_preprocess(Xtr0, ytr, PANEL_RECIPE[PANEL])
            Xte = _apply_own_preprocess(Xte0, prep).reindex(columns=Xtr.columns)
            cf = _cats(Xtr); ci = [Xtr.columns.get_loc(c) for c in cf]
            _, enc = _rf_repr(Xtr, cf, fit=True)
            keep = _lgbm_redun_keep(Xtr)
            yv_tr, yv_te = ytr.values, yte.values
            trees_o, trees_e = [], []
            for kind, (Xa, Xb) in [("lgbm", (Xtr[keep], Xte[keep])), ("rf", (Xtr, Xte)),
                                   ("cat", (Xtr, Xte)), ("xgb", (Xtr, Xte))]:
                o, e = _oof_test(kind, Xa, ytr, Xb, cf, ci, enc, 1.0, seed); trees_o.append(o); trees_e.append(e)
            to, teo = _tabpfn_member(Xtr, ytr, Xte, cf, seed)

            def record(key, btr, bte):
                thr = pick_threshold_at_prior(yv_tr, btr, PRIOR)
                res[key]["auc"].append(roc_auc_score(yv_te, bte)); res[key]["f1o"].append(_f1_swept(yv_te, bte))
                res[key]["f1h"].append(f1_at_prior(*tpr_fpr(yv_te, (bte >= thr).astype(int)), PRIOR))
            record("4tree(dep)", sum(trees_o)/4, sum(trees_e)/4)
            record("tabpfn", to, teo)
            record("4tree+tabpfn", (sum(trees_o)+to)/5, (sum(trees_e)+teo)/5)
        print(f"  seed done", flush=True)

    print(f"\n=== general + TabPFN ({len(res['tabpfn']['auc'])} folds) — F1>0.55 needs AUC~0.866 ===")
    base = np.array(res["4tree(dep)"]["auc"])
    for k in res:
        a = np.array(res[k]["auc"]); fo = np.array(res[k]["f1o"]); fh = np.array(res[k]["f1h"])
        dA = a - base; se = dA.std(ddof=1)/np.sqrt(len(dA)) if len(dA) > 1 else 0
        print(f"  {k:16s} AUC={a.mean():.4f} (Δ{dA.mean():+.4f} {dA.mean()/se if se>0 else 0:+.1f}σ)  "
              f"oracle-F1={fo.mean():.4f}{'  >0.55!' if fo.mean()>0.55 else ''}  honest-F1={fh.mean():.4f}")
    print("DONE")


if __name__ == "__main__":
    main()
