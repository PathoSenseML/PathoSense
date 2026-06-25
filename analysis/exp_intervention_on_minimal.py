"""Phase 2 — re-test the session's FEATURE interventions on the MINIMAL baseline (raw native-NaN
LGBM, no fit_keep_nan steps), to close the loophole that the deployed corr>0.95 drop / rank-transform
could have silently dropped or diluted a new feature on the full pipeline.

Minimal baseline = drop_leaky -> LGBM on raw columns (native cats, native NaN numerics). Interventions
add their feature to the RAW matrix (so corr-drop can't remove it). Paired ΔF1* vs minimal.
Tests: +MNV combined-substitution, +EK×rarity interaction, +AL-PCA(10) on raw AL, geomean AL-impute.
Run: .venv/bin/python -u analysis/exp_intervention_on_minimal.py [panel ...]
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from harness import cv_evaluate, load_panel, compare, _drop_leaky, lgb_oof_and_test
from calibration import fit_platt, apply_platt
from run_pipeline import get_model_params
from config import SEED
from feature_engineering import grantham, blosum62, _STD_AA_SET

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def _lgb(Xtr, ytr, Xte, n):
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    tp, oof = lgb_oof_and_test(Xtr, ytr, Xte, get_model_params(n, spw), n_seeds=2, n_folds=3 if n < 200 else 5)
    pf = fit_platt(oof, ytr.values); return apply_platt(tp, pf), apply_platt(oof, pf)


def _combined(a1, a2, fnc):
    a1, a2 = str(a1), str(a2)
    if a1 in ("nan", "") or a2 in ("nan", ""):
        return np.nan
    pairs = list(zip(a1, a2)) if len(a1) == len(a2) else [(x, y) for x in a1 for y in a2]
    v = [fnc(a, b) for a, b in pairs if a in _STD_AA_SET and b in _STD_AA_SET]
    v = [x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(v)) if v else np.nan


def minimal(add=None):
    def _fn(X_tr, y_tr, X_te):
        y_tr = y_tr.reset_index(drop=True)
        Xtr, Xte = X_tr.reset_index(drop=True), X_te.reset_index(drop=True)
        if add is not None:                      # derive the feature from RAW cols (before leak-drop)
            Xtr, Xte = add(Xtr.copy(), y_tr, Xte.copy())
        Xtr, Xte = _drop_leaky(Xtr, y_tr, Xte)
        return _lgb(Xtr, y_tr, Xte, len(X_tr))
    return _fn


def add_mnv(Xtr, ytr, Xte):
    for X in (Xtr, Xte):
        a1, a2 = X["AA_1"].astype(object), X["AA_2"].astype(object)
        X["AA_grantham_mnv"] = [_combined(r, a, grantham) for r, a in zip(a1, a2)]
        X["AA_blosum_mnv"] = [_combined(r, a, blosum62) for r, a in zip(a1, a2)]
    return Xtr, Xte


def add_interaction(Xtr, ytr, Xte):
    ek = [c for c in Xtr.columns if str(c).startswith("EK_")]
    al = [c for c in Xtr.columns if str(c).startswith("AL_")]
    af = [c for c in al if ((Xtr[c] > 0) & (Xtr[c] <= 1)).any() and Xtr[c].mean(skipna=True) < 0.1]
    for X in (Xtr, Xte):
        ekm = X[ek].astype(float).mean(axis=1)
        rare = 1.0 - (X[af].max(axis=1).fillna(0.0) if af else 0.0)
        X["EK_x_rare"] = ekm * rare
    return Xtr, Xte


def add_alpca(Xtr, ytr, Xte, k=10):
    al = [c for c in Xtr.columns if str(c).startswith("AL_")]
    if len(al) <= k:
        return Xtr, Xte
    med = Xtr[al].median()
    Atr = Xtr[al].fillna(med).fillna(0.0).values; Ate = Xte[al].fillna(med).fillna(0.0).values
    sc = StandardScaler().fit(Atr); pca = PCA(k, random_state=SEED).fit(sc.transform(Atr))
    Ctr, Cte = pca.transform(sc.transform(Atr)), pca.transform(sc.transform(Ate))
    Xtr = Xtr.drop(columns=al); Xte = Xte.drop(columns=al)
    for i in range(k):
        Xtr[f"ALpca{i}"] = Ctr[:, i]; Xte[f"ALpca{i}"] = Cte[:, i]
    return Xtr, Xte


def add_geomean(Xtr, ytr, Xte):
    al = [c for c in Xtr.columns if str(c).startswith("AL_")]
    for c in al:
        pos = Xtr[c].dropna(); pos = pos[pos > 0]
        fv = float(np.exp(np.log(pos.clip(lower=1e-9)).mean())) if len(pos) else 0.0
        Xtr[c] = Xtr[c].fillna(fv); Xte[c] = Xte[c].fillna(fv)
    return Xtr, Xte


VARIANTS = {"+MNV": add_mnv, "+EKxrare": add_interaction, "+AL_PCA10": add_alpca, "geomean_impute": add_geomean}


def run(panel, nr):
    df = load_panel(panel)
    base = cv_evaluate(panel, minimal(), label="minimal", n_repeats=nr, df=df)
    print(f"\n[{panel}] MINIMAL(raw) F1*={base.summary()['f1_mean']:.4f}", flush=True)
    rows = [{"panel": panel, "variant": "minimal", "f1": base.summary()["f1_mean"], "delta": 0.0, "sigma": 0.0}]
    for name, add in VARIANTS.items():
        r = cv_evaluate(panel, minimal(add), label=name, n_repeats=nr, df=df)
        c = compare(base, r)
        print(f"[{panel}] {name:16s} F1*={c['b_f1']:.4f}  Δ={c['delta_f1']:+.4f} ({c['sigma']:+.2f}σ)  {c['verdict']}", flush=True)
        rows.append({"panel": panel, "variant": name, "f1": c["b_f1"], "delta": c["delta_f1"], "sigma": c["sigma"]})
    return rows


if __name__ == "__main__":
    panels = sys.argv[1:] or ["cftr", "pah", "cancer", "general"]
    print("=== Phase 2: feature interventions on MINIMAL baseline (no corr-drop to undercut them) ===", flush=True)
    allrows = []
    for pn in panels:
        allrows += run(pn, 4 if pn == "general" else 8)
    pd.DataFrame(allrows).to_csv(os.path.join(OUT, "intervention_on_minimal.csv"), index=False)
    print(f"\nsaved -> intervention_on_minimal.csv", flush=True)
