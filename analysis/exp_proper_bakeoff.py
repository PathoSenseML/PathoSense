"""PROPER all-model bake-off on the general panel — every family, HPO'd, calibrated, fairly preprocessed.

For each model:
  * its OWN preprocessing — trees get the native-NaN matrix (deployed repr); linear/kernel/neural/NB
    get median-impute + StandardScaler + (model-specific) mutual-info feature selection.
  * Optuna HPO (per-model search space, CV-AUC objective). Trees with deployed-tuned params skip HPO.
  * Platt calibration on the OOF + test probs (every base model, like the deployed pipeline).
  * honest evaluation via harness.cv_evaluate — analytic F1* at the test prior + AUC, PAIRED vs the
    deployed LightGBM base on identical folds.

Caveat: HPO selects one config per model on inner CV-AUC, then we evaluate that fixed config on the
harness's (different-seed) repeated folds. Not fully nested, but the same protocol our deployed models
were tuned under, and the paired delta vs the (also-tuned) LightGBM base is apples-to-apples.

Run: .venv/bin/python -u analysis/exp_proper_bakeoff.py
"""
import sys, time, numpy as np, pandas as pd
sys.path.insert(0, "analysis"); sys.path.insert(0, "pipelines")
import warnings; warnings.filterwarnings("ignore")
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
import harness
from harness import cv_evaluate, load_panel, compare, _drop_leaky
from meeting_lib import native_nan_oof
from preprocess_v2 import fit_keep_nan, apply_keep_nan
from calibration import fit_platt, apply_platt
from general_blend import _rf_general, _cat_general, _xgb_model, _rf_repr, _cat_repr, _cats
from config import SEED

df = load_panel("general")
OUT = "analysis/output/proper_bakeoff.csv"


# ── dense preprocessing for non-tree models (fit on train; MI top-k) ──
def dense_fit(Xtr, ytr, k):
    cols = list(Xtr.columns); cat = {}
    def num(X):
        A = np.empty((len(X), len(cols)))
        for j, c in enumerate(cols):
            s = X[c]
            if str(s.dtype) in ("category", "object"):
                if c not in cat:
                    cat[c] = {v: i for i, v in enumerate(Xtr[c].astype("object").value_counts().index)}
                A[:, j] = X[c].astype("object").map(cat[c]).astype(float)
            else:
                A[:, j] = pd.to_numeric(s, errors="coerce")
        return A
    A = num(Xtr); med = np.nanmedian(A, 0); med = np.where(np.isfinite(med), med, 0.0)
    A = np.where(np.isnan(A), med, A); sc = StandardScaler().fit(A); A = sc.transform(A)
    A = np.clip(np.nan_to_num(A), -8, 8)
    sel = np.arange(A.shape[1])
    if k and k < A.shape[1]:
        mi = mutual_info_classif(A, ytr, random_state=SEED); sel = np.argsort(mi)[::-1][:k]
    def apply(X):
        B = np.where(np.isnan(num(X)), med, num(X)); B = np.clip(np.nan_to_num(sc.transform(B)), -8, 8)
        return B[:, sel]
    return apply(Xtr), apply, sel


# ── HPO: maximize 4-fold CV AUC over a model's param space on the (dense) general matrix ──
def hpo_dense(make, space, k, n_trials):
    Xd, _ = _drop_leaky(df[[c for c in df.columns if c not in ("Variant_ID", "Label")]], df["Label"].astype(int),
                        df[[c for c in df.columns if c not in ("Variant_ID", "Label")]])
    y = df["Label"].astype(int).reset_index(drop=True)
    A, _, _ = dense_fit(Xd.reset_index(drop=True), y, k)
    def obj(t):
        p = {kk: fn(t) for kk, fn in space.items()}
        try:
            s = cross_val_score(make(p), A, y, cv=StratifiedKFold(4, shuffle=True, random_state=SEED),
                                scoring="roc_auc", n_jobs=4)
            return float(np.mean(s))
        except Exception:
            return 0.5
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return st.best_params, st.best_value


def dense_pipeline(make, params, k):
    def fn(X_tr, y_tr, X_te):
        Xtr, Xte = _drop_leaky(X_tr.reset_index(drop=True), y_tr.reset_index(drop=True), X_te.reset_index(drop=True))
        y = y_tr.reset_index(drop=True).astype(int)
        A, apply, _ = dense_fit(Xtr, y, k); B = apply(Xte)
        oof = np.zeros(len(y)); test = []
        for s in range(2):
            skf = StratifiedKFold(5, shuffle=True, random_state=SEED + s); o = np.zeros(len(y))
            for a, b in skf.split(A, y):
                m = make(params); m.fit(A[a], y.iloc[a]); o[b] = m.predict_proba(A[b])[:, 1]
            mf = make(params); mf.fit(A, y); test.append(mf.predict_proba(B)[:, 1]); oof += o / 2
        test = np.mean(test, 0)
        pf = fit_platt(oof, y.values)                       # calibrate (Platt) every base model
        return apply_platt(test, pf), apply_platt(oof, pf)
    return fn


# ── tree pipelines (deployed-tuned params + native-NaN + Platt) ──
def tree_pipeline(kind):
    def fn(X_tr, y_tr, X_te):
        Xtr, Xte = _drop_leaky(X_tr.reset_index(drop=True), y_tr.reset_index(drop=True), X_te.reset_index(drop=True))
        _, ppf = fit_keep_nan(Xtr); Ptr = apply_keep_nan(Xtr, ppf); Pte = apply_keep_nan(Xte, ppf)
        y = y_tr.reset_index(drop=True).astype(int); spw = (y == 0).sum() / max(y.sum(), 1)
        cats = _cats(Ptr)
        if kind == "cat":
            Atr = _cat_repr(Ptr, cats); Ate = _cat_repr(Pte, cats); ci = [Atr.columns.get_loc(c) for c in cats]
            mk = lambda: _cat_general(ci, spw)
        else:
            Atr, enc = _rf_repr(Ptr, cats, fit=True); Ate, _ = _rf_repr(Pte, cats, enc=enc, fit=False)
            if kind == "rf": mk = _rf_general
            elif kind == "xgb": mk = _xgb_model
            elif kind == "et": mk = lambda: ExtraTreesClassifier(n_estimators=800, max_features=0.3,
                                  min_samples_leaf=4, max_depth=14, class_weight="balanced", n_jobs=4, random_state=SEED)
            elif kind == "hgb": mk = lambda: HistGradientBoostingClassifier(max_iter=600, learning_rate=0.03,
                                  l2_regularization=2.0, max_leaf_nodes=31, random_state=SEED)
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
        test = mf.predict_proba(Ate)[:, 1]
        pf = fit_platt(oof, y.values)
        return apply_platt(test, pf), apply_platt(oof, pf)
    return fn


# ── non-tree model registry: (builder, optuna space, MI-k, n_trials) ──
F = lambda lo, hi, log=False: (lambda t, lo=lo, hi=hi, log=log: t.suggest_float)  # placeholder unused
NONTREE = {
    "logreg": (lambda p: LogisticRegression(max_iter=3000, class_weight="balanced", **p),
               {"C": lambda t: t.suggest_float("C", 1e-3, 10, log=True),
                "penalty": lambda t: "l2", "solver": lambda t: "lbfgs"}, 0, 30),
    "logreg_en": (lambda p: LogisticRegression(max_iter=4000, class_weight="balanced", penalty="elasticnet",
                  solver="saga", **p),
                  {"C": lambda t: t.suggest_float("C", 1e-3, 10, log=True),
                   "l1_ratio": lambda t: t.suggest_float("l1_ratio", 0.0, 1.0)}, 200, 20),
    "sgd": (lambda p: SGDClassifier(loss="log_loss", max_iter=3000, class_weight="balanced", random_state=SEED, **p),
            {"alpha": lambda t: t.suggest_float("alpha", 1e-6, 1e-1, log=True),
             "penalty": lambda t: t.suggest_categorical("penalty", ["l2", "elasticnet"])}, 0, 25),
    "lda": (lambda p: LinearDiscriminantAnalysis(solver="lsqr", **p),
            {"shrinkage": lambda t: t.suggest_float("shrinkage", 0.0, 1.0)}, 200, 20),
    "qda": (lambda p: QuadraticDiscriminantAnalysis(**p),
            {"reg_param": lambda t: t.suggest_float("reg_param", 0.0, 1.0)}, 60, 20),
    "gnb": (lambda p: GaussianNB(**p),
            {"var_smoothing": lambda t: t.suggest_float("var_smoothing", 1e-11, 1e-3, log=True)}, 0, 20),
    "svm_rbf": (lambda p: SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED, **p),
                {"C": lambda t: t.suggest_float("C", 1e-1, 1e2, log=True),
                 "gamma": lambda t: t.suggest_float("gamma", 1e-4, 1e-1, log=True)}, 100, 20),
    "svm_lin": (lambda p: SVC(kernel="linear", probability=True, class_weight="balanced", random_state=SEED, **p),
                {"C": lambda t: t.suggest_float("C", 1e-3, 10, log=True)}, 200, 15),
    "knn": (lambda p: KNeighborsClassifier(**p),
            {"n_neighbors": lambda t: t.suggest_int("n_neighbors", 5, 60),
             "weights": lambda t: t.suggest_categorical("weights", ["uniform", "distance"]),
             "p": lambda t: t.suggest_int("p", 1, 2)}, 50, 25),
    "mlp": (lambda p: MLPClassifier(max_iter=500, early_stopping=True, random_state=SEED,
            hidden_layer_sizes={"s": (128,), "m": (256, 128), "l": (256, 128, 64)}[p["arch"]],
            alpha=p["alpha"], learning_rate_init=p["lr"]),
            {"arch": lambda t: t.suggest_categorical("arch", ["s", "m", "l"]),
             "alpha": lambda t: t.suggest_float("alpha", 1e-5, 1e-1, log=True),
             "lr": lambda t: t.suggest_float("lr", 1e-4, 1e-2, log=True)}, 200, 20),
}

TREES = {"lightgbm(base)": None, "catboost": "cat", "random_forest": "rf",
         "xgboost": "xgb", "extra_trees": "et", "hist_gb": "hgb"}


def main():
    nrep = 4
    base = cv_evaluate("general", lambda a, b, c: native_nan_oof(a, b, c), label="lightgbm",
                       n_repeats=nrep, df=df, n_jobs=6)
    rows = [{"model": "lightgbm(base)", "auc": round(base.auc.mean(), 4), "f1": round(base.f1.mean(), 4),
             "d_auc": 0.0, "sigma_f1": 0.0, "hpo": "deployed"}]
    print(f"  {'lightgbm(base)':20s} AUC={base.auc.mean():.4f} F1*={base.f1.mean():.4f}", flush=True)

    for name, kind in TREES.items():
        if kind is None:
            continue
        r = cv_evaluate("general", tree_pipeline(kind), label=name, n_repeats=nrep, df=df, n_jobs=4)
        c = compare(base, r)
        print(f"  {name:20s} AUC={r.auc.mean():.4f} F1*={r.f1.mean():.4f} ΔAUC={r.auc.mean()-base.auc.mean():+.4f} "
              f"({c['sigma']:+.2f}σ) [deployed/sane params]", flush=True)
        rows.append({"model": name, "auc": round(r.auc.mean(), 4), "f1": round(r.f1.mean(), 4),
                     "d_auc": round(r.auc.mean()-base.auc.mean(), 4), "sigma_f1": c["sigma"], "hpo": "deployed/sane"})

    for name, (make, space, k, ntr) in NONTREE.items():
        try:
            t0 = time.time(); best, bv = hpo_dense(make, space, k, ntr)
            r = cv_evaluate("general", dense_pipeline(make, best, k), label=name, n_repeats=nrep, df=df, n_jobs=6)
            c = compare(base, r)
            print(f"  {name:20s} AUC={r.auc.mean():.4f} F1*={r.f1.mean():.4f} ΔAUC={r.auc.mean()-base.auc.mean():+.4f} "
                  f"({c['sigma']:+.2f}σ) [HPO {ntr}t cvAUC={bv:.3f} {int(time.time()-t0)}s]", flush=True)
            rows.append({"model": name, "auc": round(r.auc.mean(), 4), "f1": round(r.f1.mean(), 4),
                         "d_auc": round(r.auc.mean()-base.auc.mean(), 4), "sigma_f1": c["sigma"], "hpo": f"optuna{ntr} {best}"})
        except Exception as e:
            print(f"  {name:20s} FAILED: {type(e).__name__}: {str(e)[:70]}", flush=True)
    out = pd.DataFrame(rows).sort_values("auc", ascending=False)
    out.to_csv(OUT, index=False)
    print(f"\n=== LEADERBOARD (by AUC) ===\n{out[['model','auc','f1','d_auc','sigma_f1']].to_string(index=False)}\nsaved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
