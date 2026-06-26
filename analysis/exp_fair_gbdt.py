"""Fair GBDT re-run: do the non-LGBM trees lose only because the bake-off under-tuned them + gave them
the dense matrix? Re-run RF / CatBoost / XGBoost with their DEPLOYED-TUNED params (general_blend.py
constructors) on the native-NaN matrix (their proper preprocessing), paired vs the LightGBM base.
AUC is rank-based (calibration-independent), so this is the fair single-model comparison."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "analysis"); sys.path.insert(0, "pipelines")
import warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold, train_test_split
import harness
from harness import cv_evaluate, load_panel, compare, _drop_leaky
from meeting_lib import native_nan_oof
from preprocess_v2 import fit_keep_nan, apply_keep_nan
from general_blend import _rf_general, _cat_general, _xgb_model, _rf_repr, _cat_repr, _cats
from config import SEED

df = load_panel("general")


def base_lgbm(a, b, c):
    return native_nan_oof(a, b, c)        # deployed-tuned LightGBM on native-NaN (the bake-off base)


def member_fn(kind):
    """Standalone tuned member on its deployed preprocessing; internal 3-fold OOF + full-fit test."""
    def fn(X_tr, y_tr, X_te):
        Xtr, Xte = _drop_leaky(X_tr.reset_index(drop=True), y_tr.reset_index(drop=True), X_te.reset_index(drop=True))
        _, ppf = fit_keep_nan(Xtr); Xpp_tr = apply_keep_nan(Xtr, ppf); Xpp_te = apply_keep_nan(Xte, ppf)
        y = y_tr.reset_index(drop=True).astype(int)
        spw = (y == 0).sum() / max(y.sum(), 1)
        cat_feats = _cats(Xpp_tr)
        if kind == "cat":
            Atr = _cat_repr(Xpp_tr, cat_feats); Ate = _cat_repr(Xpp_te, cat_feats)
            cat_idx = [Atr.columns.get_loc(c) for c in cat_feats]
        else:                                      # rf / xgb use ordinal repr (NaN->-999)
            Atr, enc = _rf_repr(Xpp_tr, cat_feats, fit=True)
            Ate, _ = _rf_repr(Xpp_te, cat_feats, enc=enc, fit=False)
        def mk():
            return (_rf_general() if kind == "rf" else
                    _xgb_model() if kind == "xgb" else _cat_general(cat_idx, spw))
        oof = np.zeros(len(y)); test = []
        skf = StratifiedKFold(3, shuffle=True, random_state=SEED)
        for a, b in skf.split(Atr, y):
            m = mk()
            if kind == "cat":
                i2, ies = train_test_split(a, test_size=0.15, random_state=SEED, stratify=y.iloc[a])
                m.fit(Atr.iloc[i2], y.iloc[i2], eval_set=(Atr.iloc[ies], y.iloc[ies]))
            else:
                m.fit(Atr.iloc[a], y.iloc[a])
            oof[b] = m.predict_proba(Atr.iloc[b])[:, 1]
        mf = mk()
        if kind == "cat":
            i2, ies = train_test_split(np.arange(len(y)), test_size=0.15, random_state=SEED + 9, stratify=y)
            mf.fit(Atr.iloc[i2], y.iloc[i2], eval_set=(Atr.iloc[ies], y.iloc[ies]))
        else:
            mf.fit(Atr, y)
        return mf.predict_proba(Ate)[:, 1], oof
    return fn


def main():
    nrep = 4
    base = cv_evaluate("general", base_lgbm, label="lgbm", n_repeats=nrep, df=df, n_jobs=6)
    print(f"  {'LightGBM (deployed, native-NaN)':34s} AUC={base.auc.mean():.4f} F1*={base.f1.mean():.4f}", flush=True)
    for kind, nm in [("cat", "CatBoost tuned (native-NaN)"), ("rf", "RandomForest tuned (native-NaN)"),
                     ("xgb", "XGBoost tuned (native-NaN)")]:
        r = cv_evaluate("general", member_fn(kind), label=nm, n_repeats=nrep, df=df, n_jobs=4)
        c = compare(base, r)
        print(f"  {nm:34s} AUC={r.auc.mean():.4f} F1*={r.f1.mean():.4f}  "
              f"ΔAUC={r.auc.mean()-base.auc.mean():+.4f} ΔF1={c['delta_f1']:+.4f} ({c['sigma']:+.2f}σ)", flush=True)
    print("\n  (deployed model = the 4-way BLEND of all of these, AUC ~0.864 > any single member)", flush=True)


if __name__ == "__main__":
    main()
