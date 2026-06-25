"""MAE training-objective test. Our task is F1-classification at the test prior; MAE is a
regression (L1) objective. Test swapping the LGBM training objective: binary(logloss, deployed) vs
regression_l1 (MAE) vs regression_l2 (MSE). Honest per-fold CV (native-NaN, same matrix), AUC (ranking)
+ analytic F1* at the per-panel prior. Regression outputs are used as scores (rank + threshold-at-prior).
2 seeds. general + cancer (own-model panels; the same objective feeds the transfer scorer).
Run: .venv/bin/python -u analysis/exp_mae.py [panel ...]
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from config import SEED
from harness import (load_panel, _drop_leaky, cv_evaluate, compare, pick_threshold_at_prior,
                     tpr_fpr, f1_at_prior)
from schema import partition_columns
from preprocess_v2 import fit_keep_nan, apply_keep_nan
from run_pipeline import get_model_params

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def make_fn(objective):
    """pipeline_fn: per fold, native-NaN preprocess + LGBM with the given objective; OOF+test scores."""
    def fn(X_tr, y_tr, X_te):
        Xtr, Xte = _drop_leaky(X_tr, y_tr, X_te)
        Xtr_pp, ppf = fit_keep_nan(Xtr); Xte_pp = apply_keep_nan(Xte, ppf)
        for c in Xtr_pp.columns:
            if str(Xtr_pp[c].dtype) == "category":
                Xte_pp[c] = pd.Categorical(Xte_pp[c].astype(object), categories=Xtr_pp[c].cat.categories)
        catf = [c for c in Xtr_pp.columns if str(Xtr_pp[c].dtype) == "category"]
        spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        base = get_model_params(len(Xtr_pp), spw)
        common = dict(n_estimators=base["n_estimators"], learning_rate=base["learning_rate"],
                      num_leaves=base["num_leaves"], max_depth=base["max_depth"],
                      min_child_samples=base["min_child_samples"], reg_alpha=base["reg_alpha"],
                      reg_lambda=base["reg_lambda"], random_state=SEED, n_jobs=-1, verbose=-1)
        n = len(y_tr); oof = np.zeros(n)
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
        test_acc = []
        for a, b in skf.split(Xtr_pp, y_tr):
            if objective == "binary":
                m = lgb.LGBMClassifier(objective="binary", scale_pos_weight=spw, **common)
                m.fit(Xtr_pp.iloc[a], y_tr.iloc[a], categorical_feature=catf or "auto")
                oof[b] = m.predict_proba(Xtr_pp.iloc[b])[:, 1]
            else:
                m = lgb.LGBMRegressor(objective=objective, **common)
                m.fit(Xtr_pp.iloc[a], y_tr.iloc[a], categorical_feature=catf or "auto")
                oof[b] = m.predict(Xtr_pp.iloc[b])
            test_acc.append(m.predict_proba(Xte_pp)[:, 1] if objective == "binary" else m.predict(Xte_pp))
        test = np.mean(test_acc, axis=0)
        return test, oof
    return fn


if __name__ == "__main__":
    panels = sys.argv[1:] or ["general", "cancer"]
    print("=== MAE/L2 vs logloss objective (honest CV, AUC + F1* at prior) ===", flush=True)
    rows = []
    for pn in panels:
        df = load_panel(pn); nr = 6 if pn == "general" else 10
        base = cv_evaluate(pn, make_fn("binary"), label="logloss", n_repeats=nr, df=df)
        print(f"\n[{pn}] logloss(deployed): F1*={base.f1.mean():.4f}  AUC={base.auc.mean():.4f}", flush=True)
        for obj, name in (("regression_l1", "MAE"), ("regression_l2", "MSE")):
            v = cv_evaluate(pn, make_fn(obj), label=name, n_repeats=nr, df=df)
            c = compare(base, v); da = v.auc.mean() - base.auc.mean()
            print(f"  {name:4s}: F1*={v.f1.mean():.4f} Δ={c['delta_f1']:+.4f} ({c['sigma']:+.2f}σ) | "
                  f"AUC={v.auc.mean():.4f} ΔAUC={da:+.4f}", flush=True)
            rows.append({"panel": pn, "objective": name, "f1": v.f1.mean(), "df1": c["delta_f1"],
                         "sigma": c["sigma"], "auc": v.auc.mean(), "dauc": da})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "mae.csv"), index=False)
    print(f"\nsaved -> mae.csv", flush=True)
