"""
Evaluation at competition training sizes (80/20 P/B), tested on external PHACTboost test set.
Anonymous column names. This is the most realistic competition simulation.
"""
import sys, os, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from config import SEED, RESULTS_DIR
from run_pipeline import (
    preprocess_data, select_features_mrmr, get_model_params,
    train_multi_seed_ensemble
)
from utils import detect_leaky_columns, optimize_threshold_f1, compute_metrics, subsample_random

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ANON_TRAIN = {
    "general": os.path.join(BASE, "data/panels/anon_general_dataset.csv"),
    "cancer": os.path.join(BASE, "data/panels/anon_hereditary_cancer_panel.csv"),
    "pah": os.path.join(BASE, "data/panels/anon_pah_panel.csv"),
    "cftr": os.path.join(BASE, "data/panels/anon_cftr_panel.csv"),
}
ANON_TEST = {
    "general": os.path.join(BASE, "data/panels/anon_test_general.csv"),
    "cancer": os.path.join(BASE, "data/panels/anon_test_cancer.csv"),
    "pah": os.path.join(BASE, "data/panels/anon_test_pah.csv"),
    "cftr": os.path.join(BASE, "data/panels/anon_test_cftr.csv"),
}

# Sartname sizes with 80/20 P/B
COMP_SIZES = {
    "general": (2400, 600),
    "cancer":  (320, 80),
    "pah":     (320, 80),
    "cftr":    (112, 28),
}


def run_competition_eval(panel_name):
    n_p, n_b = COMP_SIZES[panel_name]

    df_full = pd.read_csv(ANON_TRAIN[panel_name])
    df_ext = pd.read_csv(ANON_TEST[panel_name])

    df_train, actual_p, actual_b = subsample_random(df_full, n_p, n_b, SEED)
    y_train_all = df_train["Label"]
    X_train_all = df_train[[c for c in df_train.columns if c != "Label"]]
    y_ext = df_ext["Label"]
    X_ext = df_ext[[c for c in df_ext.columns if c != "Label"]]

    total = len(df_train)
    print(f"\n  {panel_name.upper()}: {total} train ({actual_p}P/{actual_b}B) | "
          f"{len(df_ext)} ext test ({int(y_ext.sum())}P/{int((y_ext==0).sum())}B)")

    # Leakage
    leaky = detect_leaky_columns(X_train_all, y_train_all, auc_threshold=0.99, verbose=False)
    X_train_all = X_train_all.drop(columns=leaky)
    X_ext = X_ext.drop(columns=[c for c in leaky if c in X_ext.columns])

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all, y_train_all, test_size=0.15, random_state=SEED, stratify=y_train_all)
    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    X_ext = X_ext.reset_index(drop=True)
    y_ext = y_ext.reset_index(drop=True)

    # Preprocess (relaxed corr=0.99)
    X_train_pp, X_val_pp, X_ext_pp, prep_info = preprocess_data(
        X_train.copy(), X_val.copy(), X_ext.copy(), corr_threshold=0.99)

    X_tv = pd.concat([X_train_pp, X_val_pp], ignore_index=True)
    y_tv = pd.concat([y_train, y_val], ignore_index=True)

    # mRMR feature selection
    X_train_pp, X_val_pp, X_ext_pp, fs_info = select_features_mrmr(
        X_train_pp, y_train, X_val_pp, X_ext_pp)
    remaining = list(X_train_pp.columns)
    X_tv = X_tv[[c for c in remaining if c in X_tv.columns]]

    print(f"  Features: {prep_info['final_features']} -> {fs_info['n_after']} (mRMR)")

    # Multi-seed ensemble
    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    params = get_model_params(len(X_tv), spw)
    n_folds = 3 if len(X_tv) < 200 else 5

    models, oof_probs, ext_probs, oof_std = train_multi_seed_ensemble(
        X_tv, y_tv, X_ext_pp, params, n_seeds=5, n_folds=n_folds)

    # OOF threshold
    opt_t, opt_oof_f1 = optimize_threshold_f1(y_tv, oof_probs)

    ext_def = compute_metrics(y_ext, ext_probs, 0.5)
    ext_opt = compute_metrics(y_ext, ext_probs, opt_t)

    oof_auc = roc_auc_score(y_tv, oof_probs)
    oof_f1 = f1_score(y_tv, (oof_probs >= 0.5).astype(int), zero_division=0)

    print(f"  OOF: F1={oof_f1:.4f} AUC={oof_auc:.4f}")
    print(f"  Ext @0.5: F1={ext_def['f1']:.4f} AUC={ext_def['roc_auc']:.4f} "
          f"P={ext_def['precision']:.4f} R={ext_def['recall']:.4f}")
    print(f"  Ext @{opt_t}: F1={ext_opt['f1']:.4f} AUC={ext_opt['roc_auc']:.4f} "
          f"P={ext_opt['precision']:.4f} R={ext_opt['recall']:.4f}")

    return {
        "panel": panel_name, "total": total,
        "n_p": actual_p, "n_b": actual_b,
        "features": len(remaining), "n_folds": n_folds,
        "oof_f1": round(oof_f1, 4), "oof_auc": round(oof_auc, 4),
        "ext_f1_def": ext_def["f1"], "ext_auc": ext_def["roc_auc"],
        "ext_prec_def": ext_def["precision"], "ext_rec_def": ext_def["recall"],
        "ext_f1_opt": ext_opt["f1"], "opt_t": opt_t,
        "ext_prec_opt": ext_opt["precision"], "ext_rec_opt": ext_opt["recall"],
        "ext_n": len(y_ext), "ext_b": int((y_ext == 0).sum()),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("  PathoSense @ Competition Sizes — Anonymous Columns — External Test")
    print("=" * 70)

    results = {}
    for p in ["general", "cancer", "pah", "cftr"]:
        results[p] = run_competition_eval(p)

    # Summary
    print("\n" + "=" * 90)
    print("  FINAL: mRMR Multi-Seed (PathoSense) @ Competition Training Sizes")
    print("  Anonymous columns | External PHACTboost test")
    print("=" * 90)
    print(f"  {'Panel':<10} {'Train':<14} {'Feat':<6} "
          f"{'F1 @0.5':<10} {'AUC':<10} {'P':<8} {'R':<8} "
          f"{'F1 @opt':<10} {'t_opt':<8} {'Ext Test':<10}")
    print(f"  {'-'*94}")
    for p in ["general", "cancer", "pah", "cftr"]:
        r = results[p]
        pb = f"{r['n_p']}P/{r['n_b']}B"
        print(f"  {p:<10} {pb:<14} {r['features']:<6} "
              f"{r['ext_f1_def']:<10.4f} {r['ext_auc']:<10.4f} "
              f"{r['ext_prec_def']:<8.4f} {r['ext_rec_def']:<8.4f} "
              f"{r['ext_f1_opt']:<10.4f} {r['opt_t']:<8} "
              f"{r['ext_n']:<10}")

    out = os.path.join(RESULTS_DIR, "competition_size_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out}")
