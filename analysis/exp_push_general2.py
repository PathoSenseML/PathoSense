"""Campaign wave 2 for general AUC. Genuinely-new levers (honest harness, paired vs base):
  - confident-learning DATA CLEANING (drop train rows whose label the model confidently contradicts;
    fit on the train fold only — no test leakage).
  - engineered CONSERVATION / LINEAGE / RARITY features from what we already have (EK cross-track
    agreement, primate-vs-mammalian phyloP ratios, AL POPMAX rank + databases-present count).
  - OPERATING-REGION focus: at pi=0.143 the F1-optimal threshold sits at high specificity, so train an
    asymmetric/focal-weighted LGBM that sharpens the high-confidence tail, and blend it in.
Run: .venv/bin/python -u analysis/exp_push_general2.py
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
from scipy.stats import rankdata
from config import SEED
from harness import cv_evaluate, load_panel, compare, _drop_leaky
from schema import partition_columns
from meeting_lib import native_nan_oof
from run_pipeline import get_model_params
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

df0 = load_panel("general"); _, NUM, CAT, TGT = partition_columns(df0)
yfull = df0[TGT].astype(int)
P = {**get_model_params(2000, (yfull == 0).sum() / max(yfull.sum(), 1)),
     "scale_pos_weight": 1, "bagging_freq": 1, "bagging_fraction": 0.8}


def base_fn(a, b, c):
    return native_nan_oof(a, b, c, params=P)


# ── confident-learning data cleaning (honest: train-fold only) ──
def clean_fn(thr_lo=0.08, thr_hi=0.92):
    def fn(X_tr, y_tr, X_te):
        X_tr = X_tr.reset_index(drop=True); y_tr = y_tr.reset_index(drop=True).astype(int)
        tp0, oof = native_nan_oof(X_tr, y_tr, X_te, params=P)       # oof aligned to full train
        y = y_tr.values
        bad = ((y == 1) & (oof < thr_lo)) | ((y == 0) & (oof > thr_hi))
        if bad.sum() == 0 or bad.sum() > 0.2 * len(y):
            return tp0, oof
        keep = ~bad
        tp, _ = native_nan_oof(X_tr.iloc[keep].reset_index(drop=True),
                               y_tr.iloc[keep].reset_index(drop=True), X_te, params=P)
        return tp, oof                                              # threshold from full-train oof
    return fn


# ── engineered conservation / lineage / rarity features ──
def add_eng(X):
    X = X.copy()
    ek = [c for c in X.columns if str(c).startswith("EK_") and X[c].dtype != object]
    if len(ek) >= 3:
        E = X[ek].apply(pd.to_numeric, errors="coerce")
        X["ENG_ek_mean"] = E.mean(axis=1); X["ENG_ek_max"] = E.max(axis=1)
        X["ENG_ek_min"] = E.min(axis=1); X["ENG_ek_std"] = E.std(axis=1)
        X["ENG_ek_range"] = X["ENG_ek_max"] - X["ENG_ek_min"]
    # lineage ratios: primate(EK_8) vs mammalian(EK_9) vs vertebrate(EK_7) phyloP
    for a, b, nm in [("EK_8", "EK_9", "prim_mamm"), ("EK_7", "EK_9", "vert_mamm"), ("EK_8", "EK_7", "prim_vert")]:
        if a in X and b in X:
            va = pd.to_numeric(X[a], errors="coerce"); vb = pd.to_numeric(X[b], errors="coerce")
            X[f"ENG_{nm}_diff"] = va - vb
    # AL rarity: how many DBs report it + max AF rank (common -> benign)
    al = [c for c in X.columns if str(c).startswith("AL_")]
    if al:
        AL = X[al].apply(pd.to_numeric, errors="coerce")
        X["ENG_al_present"] = AL.notna().sum(axis=1).astype(float)
        X["ENG_al_max"] = AL.max(axis=1)
        X["ENG_al_mean"] = AL.mean(axis=1)
    return X


def eng_fn(X_tr, y_tr, X_te):
    return native_nan_oof(add_eng(X_tr), y_tr, add_eng(X_te), params=P)


# ── operating-region focus: asymmetric LGBM (sharpen high-confidence tail) + blend ──
def _rank(x): return rankdata(x) / (len(x) + 1)

def asym_fn(spw=2.5):
    Pa = {**P, "scale_pos_weight": spw}     # more conservative -> fewer FP in the high-spec region
    def fn(a, b, c): return native_nan_oof(a, b, c, params=Pa)
    return fn

def blend_base_asym(X_tr, y_tr, X_te):
    t1, o1 = base_fn(X_tr, y_tr, X_te)
    t2, o2 = asym_fn(2.5)(X_tr, y_tr, X_te)
    return 0.6 * _rank(t1) + 0.4 * _rank(t2), 0.6 * _rank(o1) + 0.4 * _rank(o2)

def blend_eng_clean(X_tr, y_tr, X_te):
    t1, o1 = eng_fn(X_tr, y_tr, X_te)
    t2, o2 = clean_fn()(X_tr, y_tr, X_te)
    return 0.5 * _rank(t1) + 0.5 * _rank(t2), 0.5 * _rank(o1) + 0.5 * _rank(o2)


LEVERS = {
    "data_clean":     clean_fn(),
    "cons_rarity_eng": eng_fn,
    "asym_spw2.5":    asym_fn(2.5),
    "blend_base_asym": blend_base_asym,
    "eng_plus_clean":  blend_eng_clean,
}


def main():
    nrep = 4
    base = cv_evaluate("general", base_fn, label="base", n_repeats=nrep, df=df0, n_jobs=6)
    print(f"  {'base':18s} F1*={base.f1.mean():.4f}  AUC={base.auc.mean():.4f}", flush=True)
    rows = [{"lever": "base", "f1": round(base.f1.mean(), 4), "auc": round(base.auc.mean(), 4), "sigma": 0.0}]
    for name, fn in LEVERS.items():
        try:
            r = cv_evaluate("general", fn, label=name, n_repeats=nrep, df=df0, n_jobs=6)
            c = compare(base, r)
            print(f"  {name:18s} F1*={r.f1.mean():.4f}  AUC={r.auc.mean():.4f}  "
                  f"ΔF1={c['delta_f1']:+.4f} ({c['sigma']:+.2f}σ) ΔAUC={r.auc.mean()-base.auc.mean():+.4f} "
                  f"[{c['verdict']}]", flush=True)
            rows.append({"lever": name, "f1": round(r.f1.mean(), 4), "auc": round(r.auc.mean(), 4),
                         "d_f1": c["delta_f1"], "sigma": c["sigma"], "verdict": c["verdict"]})
        except Exception as e:
            print(f"  {name:18s} FAILED: {type(e).__name__}: {str(e)[:90]}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "push_general2.csv"), index=False)
    print("\nsaved -> push_general2.csv", flush=True)


if __name__ == "__main__":
    main()
