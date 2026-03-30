"""
Honest SHAP + error analysis at competition sizes using OOF predictions.
OOF = each sample predicted by a model that never saw it during training.
"""
import sys, os, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from config import SEED, RESULTS_DIR
import lightgbm as lgb

try:
    import shap
except ImportError:
    raise ImportError("shap is required for analysis_competition.py: pip install shap")

from run_pipeline import (
    preprocess_data, select_features_mrmr, get_model_params
)
from utils import detect_leaky_columns, optimize_threshold_f1, subsample_random

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANON_TRAIN = {
    "general": os.path.join(BASE, "data/panels/anon_general_dataset.csv"),
    "cancer": os.path.join(BASE, "data/panels/anon_hereditary_cancer_panel.csv"),
    "pah": os.path.join(BASE, "data/panels/anon_pah_panel.csv"),
    "cftr": os.path.join(BASE, "data/panels/anon_cftr_panel.csv"),
}
COMP_SIZES = {
    "general": (2400, 600),
    "cancer": (320, 80),
    "pah": (320, 80),
    "cftr": (112, 28),
}


def run_analysis(panel_name):
    n_p, n_b = COMP_SIZES[panel_name]
    df, _, _ = subsample_random(pd.read_csv(ANON_TRAIN[panel_name]), n_p, n_b, SEED)
    y = df["Label"]
    X = df[[c for c in df.columns if c != "Label"]]
    print(f"\n{'='*60}")
    print(f"  {panel_name.upper()}: {len(df)} samples ({int(y.sum())}P/{int((y==0).sum())}B)")

    # Leakage
    leaky = detect_leaky_columns(X, y, auc_threshold=0.99, verbose=False)
    X = X.drop(columns=leaky)

    # Split for preprocessing
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=SEED, stratify=y)
    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)

    # Preprocess
    X_train_pp, X_val_pp, _, _ = preprocess_data(
        X_train.copy(), X_val.copy(), X_val.copy(), corr_threshold=0.99)
    X_all = pd.concat([X_train_pp, X_val_pp], ignore_index=True)
    y_all = pd.concat([y_train, y_val], ignore_index=True)

    # mRMR
    X_train_pp, X_val_pp, _, fs_info = select_features_mrmr(
        X_train_pp, y_train, X_val_pp, X_val_pp)
    remaining = list(X_train_pp.columns)
    X_all = X_all[[c for c in remaining if c in X_all.columns]]

    spw = (y_all == 0).sum() / max((y_all == 1).sum(), 1)
    params = get_model_params(len(X_all), spw)
    n_folds = 3 if len(X_all) < 200 else 5

    print(f"  Features: {len(remaining)}, Folds: {n_folds}")

    # ── OOF predictions + SHAP ──
    oof_probs = np.zeros(len(y_all))
    all_shap = np.zeros((len(y_all), len(remaining)))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, y_all)):
        p = {**params, "random_state": SEED}
        m = lgb.LGBMClassifier(**p)
        m.fit(X_all.iloc[tr_idx], y_all.iloc[tr_idx],
              eval_set=[(X_all.iloc[va_idx], y_all.iloc[va_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        oof_probs[va_idx] = m.predict_proba(X_all.iloc[va_idx])[:, 1]

        # SHAP on OOF samples (model never saw these)
        explainer = shap.TreeExplainer(m)
        sv = explainer.shap_values(X_all.iloc[va_idx])
        if isinstance(sv, list):
            sv = sv[1]
        all_shap[va_idx] = sv

    # ── OOF metrics ──
    opt_t, opt_f1 = optimize_threshold_f1(y_all, oof_probs)
    oof_preds = (oof_probs >= opt_t).astype(int)
    oof_f1 = f1_score(y_all, oof_preds, zero_division=0)
    oof_auc = roc_auc_score(y_all, oof_probs)
    cm = confusion_matrix(y_all, oof_preds)
    tn, fp, fn, tp = cm.ravel()

    print(f"  OOF F1={oof_f1:.4f} AUC={oof_auc:.4f} (t={opt_t})")
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")

    # ── SHAP analysis ──
    mean_abs_shap = np.abs(all_shap).mean(axis=0)
    shap_ranking = sorted(zip(remaining, mean_abs_shap), key=lambda x: -x[1])
    print(f"  Top 5 SHAP: {[(f, round(s, 4)) for f, s in shap_ranking[:5]]}")

    # ── Error analysis (normalized) ──
    error_info = {}
    if fn > 3:
        fn_mask = (oof_preds == 0) & (y_all.values == 1)
        tp_mask = (oof_preds == 1) & (y_all.values == 1)
        # Z-score normalize features first
        feature_means = X_all.mean()
        feature_stds = X_all.std().replace(0, 1)
        X_norm = (X_all - feature_means) / feature_stds

        fn_mean = X_norm[fn_mask].mean()
        tp_mean = X_norm[tp_mask].mean()
        diff = (fn_mean - tp_mean).abs().sort_values(ascending=False)
        top_diff = {k: round(float(v), 3) for k, v in diff.head(5).items()}
        print(f"  FN vs TP normalized diffs: {top_diff}")
        error_info["fn_vs_tp_normalized"] = top_diff

    if fp > 3:
        fp_mask = (oof_preds == 1) & (y_all.values == 0)
        tn_mask = (oof_preds == 0) & (y_all.values == 0)
        X_norm = (X_all - X_all.mean()) / X_all.std().replace(0, 1)
        fp_mean = X_norm[fp_mask].mean()
        tn_mean = X_norm[tn_mask].mean()
        diff_fp = (fp_mean - tn_mean).abs().sort_values(ascending=False)
        top_diff_fp = {k: round(float(v), 3) for k, v in diff_fp.head(5).items()}
        print(f"  FP vs TN normalized diffs: {top_diff_fp}")
        error_info["fp_vs_tn_normalized"] = top_diff_fp

    # ── Save plots ──
    plots_dir = os.path.join(RESULTS_DIR, panel_name, "plots_analysis")
    os.makedirs(plots_dir, exist_ok=True)

    # SHAP bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(all_shap, X_all, plot_type="bar", show=False, max_display=15)
    plt.title(f"{panel_name.upper()} - SHAP Feature Importance (OOF, Competition Size)")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "shap_bar_oof.png"), dpi=150)
    plt.close()

    # SHAP beeswarm
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(all_shap, X_all, show=False, max_display=15)
    plt.title(f"{panel_name.upper()} - SHAP Summary (OOF)")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "shap_beeswarm_oof.png"), dpi=150)
    plt.close()

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=16)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Benign', 'Pathogenic'])
    ax.set_yticklabels(['Benign', 'Pathogenic'])
    ax.set_title(f"{panel_name.upper()} - OOF Confusion Matrix (t={opt_t})")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "confusion_oof.png"), dpi=150)
    plt.close()

    return {
        "panel": panel_name, "n": len(y_all),
        "n_p": int(y_all.sum()), "n_b": int((y_all == 0).sum()),
        "oof_f1": round(oof_f1, 4), "oof_auc": round(oof_auc, 4),
        "opt_t": opt_t,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "shap_top10": {f: round(float(s), 4) for f, s in shap_ranking[:10]},
        "error_analysis": error_info,
        "features": remaining,
    }


if __name__ == "__main__":
    print("SHAP + Error Analysis @ Competition Sizes (OOF, Anonymous)")
    results = {}
    for p in ["general", "cancer", "pah", "cftr"]:
        results[p] = run_analysis(p)

    # Summary
    print(f"\n{'='*80}")
    print(f"  SUMMARY: OOF Analysis @ Competition Sizes")
    print(f"{'='*80}")
    print(f"  {'Panel':<10} {'n':<8} {'F1':<8} {'AUC':<8} {'TP':<6} {'FP':<6} {'TN':<6} {'FN':<6} {'Top SHAP feature':<30}")
    print(f"  {'-'*82}")
    for p in ["general", "cancer", "pah", "cftr"]:
        r = results[p]
        top_feat = list(r["shap_top10"].keys())[0]
        print(f"  {p:<10} {r['n']:<8} {r['oof_f1']:<8.4f} {r['oof_auc']:<8.4f} "
              f"{r['tp']:<6} {r['fp']:<6} {r['tn']:<6} {r['fn']:<6} {top_feat:<30}")

    out = os.path.join(RESULTS_DIR, "analysis_competition_oof.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out}")
