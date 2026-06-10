"""Focal loss for LightGBM (proper test). Research flagged focal loss as strong for
imbalanced F1/PR-AUC. Our problem is prior-shift not training-imbalance, but focal's
hard-example focus could still sharpen the decision boundary → higher AUC → higher F1*.

LightGBM 4.x removed `fobj`; the custom objective is passed via the sklearn
`LGBMClassifier(objective=callable)` API, signature (y_true, y_pred_raw) -> (grad, hess)
(Max Halford / jrzaurin formulation). AUC is rank-invariant, so we compare on raw
scores (no sigmoid ambiguity). Hessian floored for split stability.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import numpy as np, pandas as pd
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

from harness import load_panel, _drop_leaky
from preprocess_v2 import fit_keep_nan, apply_keep_nan
from schema import partition_columns
from run_pipeline import get_model_params


class FocalLoss:
    def __init__(self, gamma, alpha):
        self.g = gamma; self.a = alpha
    def __call__(self, y_true, y_pred):          # sklearn custom-objective signature
        y = 2 * y_true - 1
        at = np.where(y_true == 1, self.a, 1 - self.a)
        p = np.clip(expit(y_pred), 1e-6, 1 - 1e-6)
        pt = np.where(y_true == 1, p, 1 - p)
        g = self.g
        grad = at * y * (1 - pt) ** g * (g * pt * np.log(pt) + pt - 1)
        u = at * y * (1 - pt) ** g
        du = -at * y * g * (1 - pt) ** (g - 1)
        v = g * pt * np.log(pt) + pt - 1
        dv = g * np.log(pt) + g + 1
        hess = (du * v + u * dv) * y * (pt * (1 - pt))
        hess = np.maximum(np.abs(hess), 1e-6)    # ensure positive-definite splits
        return grad, hess


def _prep_fold(Xn, cat_cols, tr, te):
    Xtr_pp, ppf = fit_keep_nan(Xn.iloc[tr].reset_index(drop=True))
    Xte_pp = apply_keep_nan(Xn.iloc[te].reset_index(drop=True), ppf)
    for col in cat_cols:
        if col in Xtr_pp.columns:
            Xtr_pp[col] = Xtr_pp[col].cat.codes
            Xte_pp[col] = pd.Categorical(Xte_pp[col]).codes
    return Xtr_pp, Xte_pp


def run(panel, n_splits=5):
    df = load_panel(panel); _, n, c, t = partition_columns(df)
    X = df[n + c]; y = df[t].astype(int)
    Xn, _ = _drop_leaky(X, y, X.copy())
    cat_cols = [col for col in c if col in Xn.columns]
    configs = {"binary(deployed)": None, "focal g1 a.5": (1.0, .5),
               "focal g2 a.5": (2.0, .5), "focal g2 a.25": (2.0, .25),
               "focal g3 a.5": (3.0, .5)}
    res = {k: [] for k in configs}
    t0 = time.time()
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=1).split(Xn, y):
        ytr, yte = y.iloc[tr].values, y.iloc[te].values
        Xtr_pp, Xte_pp = _prep_fold(Xn, cat_cols, tr, te)
        spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
        base = get_model_params(len(tr), spw)
        hp = {k: v for k, v in base.items()
              if k in ("n_estimators", "learning_rate", "num_leaves", "max_depth",
                       "min_child_samples", "subsample", "colsample_bytree",
                       "reg_alpha", "reg_lambda")}
        for name, fa in configs.items():
            if fa is None:
                m = LGBMClassifier(**hp, objective="binary", scale_pos_weight=spw,
                                   random_state=42, verbose=-1)
                m.fit(Xtr_pp, ytr)
                p = m.predict_proba(Xte_pp)[:, 1]
            else:
                m = LGBMClassifier(**hp, objective=FocalLoss(*fa), random_state=42, verbose=-1)
                m.fit(Xtr_pp, ytr)
                p = m.predict(Xte_pp, raw_score=True)   # AUC is rank-invariant
            res[name].append(roc_auc_score(yte, p))
    print(f"\n{panel} {n_splits}-fold AUC ({time.time()-t0:.0f}s):", flush=True)
    b = np.mean(res["binary(deployed)"])
    for name in configs:
        m = np.mean(res[name])
        tag = "(baseline)" if name.startswith("binary") else f"Δ={m-b:+.4f}"
        print(f"  {name:18s} AUC={m:.4f}  {tag}", flush=True)


if __name__ == "__main__":
    for pn in (sys.argv[1:] or ["general"]):
        run(pn)
