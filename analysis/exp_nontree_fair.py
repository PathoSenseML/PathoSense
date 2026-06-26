"""FAIR non-tree bake-off — give linear/kernel/distance/probabilistic/neural models their OWN optimal
preprocessing (like CatBoost got native-NaN), then HPO + Platt-calibrate them.

Proper non-tree preprocessing (fit on train fold only):
  * numerics: median-impute -> QuantileTransformer(normal)  [fixes the skewed AL_ block]
  * + missingness INDICATORS for columns with >5% missing   [trees read NaN; linear models need flags]
  * categoricals: ONE-HOT encode (nominal — ordinal is wrong for these models)
  * optional mutual-info top-k selection (for curse-of-dim-sensitive models: SVM-RBF/kNN/QDA)
Per-model Optuna HPO (CV-AUC). Honest harness eval (F1* @ 0.143 + AUC), paired vs the deployed LightGBM.
Compare to the previous tree-flavored non-tree numbers to see how much proper prep helps.
Run: .venv/bin/python -u analysis/exp_nontree_fair.py
"""
import sys, time, numpy as np, pandas as pd
sys.path.insert(0, "analysis"); sys.path.insert(0, "pipelines")
import warnings; warnings.filterwarnings("ignore")
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
import harness
from harness import cv_evaluate, load_panel, compare, _drop_leaky
from meeting_lib import native_nan_oof
from preprocess_v2 import fit_keep_nan, apply_keep_nan
from calibration import fit_platt, apply_platt
from schema import partition_columns
from config import SEED

df = load_panel("general"); _, NUM, CAT, TGT = partition_columns(df)


def prep_fit(Xtr, ytr, k):
    """Fit proper non-tree preprocessing on the train fold; return (transformed_train, apply_fn)."""
    num = [c for c in Xtr.columns if str(Xtr[c].dtype) not in ("category", "object")]
    cat = [c for c in Xtr.columns if str(Xtr[c].dtype) in ("category", "object")]
    A = Xtr[num].apply(pd.to_numeric, errors="coerce")
    med = A.median(); miss_cols = [c for c in num if A[c].isna().mean() > 0.05]
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=min(1000, len(Xtr)), random_state=SEED)
    qt.fit(A.fillna(med))
    ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False)
    if cat: ohe.fit(Xtr[cat].astype(str))

    def build(X):
        N = X[num].apply(pd.to_numeric, errors="coerce")
        Q = qt.transform(N.fillna(med))
        M = N[miss_cols].isna().astype(float).values if miss_cols else np.empty((len(X), 0))
        C = ohe.transform(X[cat].astype(str)) if cat else np.empty((len(X), 0))
        return np.clip(np.nan_to_num(np.hstack([Q, M, C])), -8, 8)

    Z = build(Xtr); sel = np.arange(Z.shape[1])
    if k and k < Z.shape[1]:
        mi = mutual_info_classif(Z, ytr, random_state=SEED); sel = np.argsort(mi)[::-1][:k]
    return Z[:, sel], (lambda X: build(X)[:, sel])


def hpo(make, space, k, ntr):
    feat = [c for c in df.columns if c in (NUM + CAT)]
    Xd, _ = _drop_leaky(df[feat], df[TGT].astype(int), df[feat])
    y = df[TGT].astype(int).reset_index(drop=True)
    Z, _ = prep_fit(Xd.reset_index(drop=True), y, k)
    def obj(t):
        p = {kk: fn(t) for kk, fn in space.items()}
        try:
            return float(np.mean(cross_val_score(make(p), Z, y, cv=StratifiedKFold(4, shuffle=True, random_state=SEED),
                                                  scoring="roc_auc", n_jobs=4)))
        except Exception:
            return 0.5
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(obj, n_trials=ntr, show_progress_bar=False)
    return st.best_params, st.best_value


def pipeline(make, params, k):
    def fn(X_tr, y_tr, X_te):
        Xtr, Xte = _drop_leaky(X_tr.reset_index(drop=True), y_tr.reset_index(drop=True), X_te.reset_index(drop=True))
        y = y_tr.reset_index(drop=True).astype(int)
        Z, apply = prep_fit(Xtr, y, k); Zte = apply(Xte)
        oof = np.zeros(len(y)); test = []
        for s in range(2):
            skf = StratifiedKFold(5, shuffle=True, random_state=SEED + s); o = np.zeros(len(y))
            for a, b in skf.split(Z, y):
                m = _proba(make(params)); m.fit(Z[a], y.iloc[a]); o[b] = m.predict_proba(Z[b])[:, 1]
            mf = _proba(make(params)); mf.fit(Z, y); test.append(mf.predict_proba(Zte)[:, 1]); oof += o / 2
        test = np.mean(test, 0); pf = fit_platt(oof, y.values)
        return apply_platt(test, pf), apply_platt(oof, pf)
    return fn


def _proba(m):
    """Wrap models without predict_proba (RidgeClassifier, NearestCentroid, LinearSVC-like) in calibration."""
    return m if hasattr(m, "predict_proba") else CalibratedClassifierCV(m, cv=3, method="sigmoid")


MODELS = {
    "logreg_l2":  (lambda p: LogisticRegression(max_iter=3000, class_weight="balanced", C=p["C"]),
                   {"C": lambda t: t.suggest_float("C", 1e-3, 30, log=True)}, 0, 30),
    "logreg_l1":  (lambda p: LogisticRegression(penalty="l1", solver="saga", max_iter=4000,
                   class_weight="balanced", C=p["C"]), {"C": lambda t: t.suggest_float("C", 1e-3, 10, log=True)}, 0, 25),
    "logreg_en":  (lambda p: LogisticRegression(penalty="elasticnet", solver="saga", max_iter=4000,
                   class_weight="balanced", C=p["C"], l1_ratio=p["l1"]),
                   {"C": lambda t: t.suggest_float("C", 1e-3, 10, log=True), "l1": lambda t: t.suggest_float("l1", 0, 1)}, 0, 25),
    "ridge":      (lambda p: RidgeClassifier(class_weight="balanced", alpha=p["alpha"]),
                   {"alpha": lambda t: t.suggest_float("alpha", 1e-2, 1e3, log=True)}, 0, 20),
    "sgd":        (lambda p: SGDClassifier(loss="log_loss", max_iter=4000, class_weight="balanced",
                   random_state=SEED, alpha=p["alpha"], penalty=p["pen"]),
                   {"alpha": lambda t: t.suggest_float("alpha", 1e-6, 1e-1, log=True),
                    "pen": lambda t: t.suggest_categorical("pen", ["l2", "elasticnet"])}, 0, 25),
    "lda":        (lambda p: LinearDiscriminantAnalysis(solver="lsqr", shrinkage=p["s"]),
                   {"s": lambda t: t.suggest_float("s", 0, 1)}, 300, 20),
    "qda":        (lambda p: QuadraticDiscriminantAnalysis(reg_param=p["r"]),
                   {"r": lambda t: t.suggest_float("r", 0, 1)}, 60, 20),
    "gnb":        (lambda p: GaussianNB(var_smoothing=p["v"]),
                   {"v": lambda t: t.suggest_float("v", 1e-11, 1e-2, log=True)}, 0, 20),
    "bnb":        (lambda p: BernoulliNB(alpha=p["a"], binarize=p["b"]),
                   {"a": lambda t: t.suggest_float("a", 1e-3, 5, log=True), "b": lambda t: t.suggest_float("b", -1, 1)}, 0, 20),
    "svm_rbf":    (lambda p: SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED,
                   C=p["C"], gamma=p["g"]), {"C": lambda t: t.suggest_float("C", 1e-1, 1e2, log=True),
                   "g": lambda t: t.suggest_float("g", 1e-4, 1, log=True)}, 150, 22),
    "svm_lin":    (lambda p: SVC(kernel="linear", probability=True, class_weight="balanced", random_state=SEED, C=p["C"]),
                   {"C": lambda t: t.suggest_float("C", 1e-3, 10, log=True)}, 250, 15),
    "knn":        (lambda p: KNeighborsClassifier(n_neighbors=p["n"], weights=p["w"], p=p["p"]),
                   {"n": lambda t: t.suggest_int("n", 5, 80), "w": lambda t: t.suggest_categorical("w", ["uniform", "distance"]),
                    "p": lambda t: t.suggest_int("p", 1, 2)}, 60, 25),
    "ncentroid":  (lambda p: NearestCentroid(shrink_threshold=p["s"]),
                   {"s": lambda t: t.suggest_float("s", 0, 1)}, 200, 15),
    "mlp":        (lambda p: MLPClassifier(max_iter=600, early_stopping=True, random_state=SEED,
                   hidden_layer_sizes={"s": (128,), "m": (256, 128), "l": (256, 128, 64)}[p["arch"]],
                   alpha=p["alpha"], learning_rate_init=p["lr"]),
                   {"arch": lambda t: t.suggest_categorical("arch", ["s", "m", "l"]),
                    "alpha": lambda t: t.suggest_float("alpha", 1e-5, 1e-1, log=True),
                    "lr": lambda t: t.suggest_float("lr", 1e-4, 1e-2, log=True)}, 250, 20),
}


def main():
    nrep = 4
    base = cv_evaluate("general", lambda a, b, c: native_nan_oof(a, b, c), label="lightgbm", n_repeats=nrep, df=df, n_jobs=6)
    print(f"  {'lightgbm (deployed)':16s} AUC={base.auc.mean():.4f} F1*={base.f1.mean():.4f}  [tree baseline]", flush=True)
    rows = [{"model": "lightgbm", "auc": round(base.auc.mean(), 4), "f1": round(base.f1.mean(), 4), "d_auc": 0.0}]
    for name, (make, space, k, ntr) in MODELS.items():
        try:
            t0 = time.time(); best, bv = hpo(make, space, k, ntr)
            nj = 1 if name in ("svm_rbf", "svm_lin") else 6
            r = cv_evaluate("general", pipeline(make, best, k), label=name, n_repeats=nrep, df=df, n_jobs=nj)
            print(f"  {name:16s} AUC={r.auc.mean():.4f} F1*={r.f1.mean():.4f} ΔAUC={r.auc.mean()-base.auc.mean():+.4f} "
                  f"[HPO cvAUC={bv:.3f} {int(time.time()-t0)}s]", flush=True)
            rows.append({"model": name, "auc": round(r.auc.mean(), 4), "f1": round(r.f1.mean(), 4),
                         "d_auc": round(r.auc.mean()-base.auc.mean(), 4)})
        except Exception as e:
            print(f"  {name:16s} FAILED: {type(e).__name__}: {str(e)[:70]}", flush=True)
    R = pd.DataFrame(rows).sort_values("auc", ascending=False)
    R.to_csv("analysis/output/nontree_fair.csv", index=False)
    print(f"\n=== NON-TREE FAIR LEADERBOARD (proper prep + HPO + Platt) ===\n{R.to_string(index=False)}", flush=True)
    print("\nsaved -> analysis/output/nontree_fair.csv", flush=True)


if __name__ == "__main__":
    main()
