"""
Pseudo-labeling experiment.
Train on labeled data, predict test set, add high-confidence predictions back, retrain.
Evaluated at competition sizes with anonymous columns.
"""
import sys, os, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import train_test_split
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

COMP_SIZES = {
    "general": (2400, 600),
    "cancer": (320, 80),
    "pah": (320, 80),
    "cftr": (112, 28),
}


def run_pseudo_label(panel_name, conf_threshold=0.85, max_rounds=2):
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
          f"{len(df_ext)} ext test")

    # Leakage
    leaky = detect_leaky_columns(X_train_all, y_train_all, auc_threshold=0.99, verbose=False)
    X_train_all = X_train_all.drop(columns=leaky)
    X_ext = X_ext.drop(columns=[c for c in leaky if c in X_ext.columns])

    # Val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all, y_train_all, test_size=0.15, random_state=SEED, stratify=y_train_all)
    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    X_ext = X_ext.reset_index(drop=True)
    y_ext = y_ext.reset_index(drop=True)

    # Preprocess
    X_train_pp, X_val_pp, X_ext_pp, _ = preprocess_data(
        X_train.copy(), X_val.copy(), X_ext.copy(), corr_threshold=0.99)
    X_tv = pd.concat([X_train_pp, X_val_pp], ignore_index=True)
    y_tv = pd.concat([y_train, y_val], ignore_index=True)

    # mRMR
    X_train_pp, X_val_pp, X_ext_pp, fs_info = select_features_mrmr(
        X_train_pp, y_train, X_val_pp, X_ext_pp)
    remaining = list(X_train_pp.columns)
    X_tv = X_tv[[c for c in remaining if c in X_tv.columns]]
    X_ext_sel = X_ext_pp[remaining].copy()

    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    params = get_model_params(len(X_tv), spw)
    n_folds = 3 if len(X_tv) < 200 else 5

    # ── Round 0: Baseline (no pseudo-labels) ──
    models, oof, test_probs, oof_std = train_multi_seed_ensemble(
        X_tv, y_tv, X_ext_sel, params, n_seeds=5, n_folds=n_folds)

    opt_t, _ = optimize_threshold_f1(y_tv, oof)
    baseline = compute_metrics(y_ext, test_probs, 0.5)
    baseline_opt = compute_metrics(y_ext, test_probs, opt_t)

    print(f"  Round 0 (baseline): F1@0.5={baseline['f1']:.4f} "
          f"AUC={baseline['roc_auc']:.4f} F1@opt={baseline_opt['f1']:.4f}")

    results = [{
        "round": 0, "train_size": len(X_tv), "pseudo_added": 0,
        "f1_def": baseline["f1"], "auc": baseline["roc_auc"],
        "f1_opt": baseline_opt["f1"], "opt_t": opt_t,
    }]

    # ── Pseudo-labeling rounds ──
    X_tv_current = X_tv.copy()
    y_tv_current = y_tv.copy()

    for rnd in range(1, max_rounds + 1):
        # Select high-confidence predictions
        high_patho = test_probs >= conf_threshold
        high_benign = test_probs <= (1 - conf_threshold)
        high_conf = high_patho | high_benign

        n_pseudo = int(high_conf.sum())
        if n_pseudo == 0:
            print(f"  Round {rnd}: No high-confidence samples (threshold={conf_threshold})")
            break

        pseudo_labels = (test_probs[high_conf] >= 0.5).astype(int)
        n_pseudo_p = int(pseudo_labels.sum())
        n_pseudo_b = n_pseudo - n_pseudo_p
        X_pseudo = X_ext_sel[high_conf].reset_index(drop=True)

        # Augment training data
        X_aug = pd.concat([X_tv_current, X_pseudo], ignore_index=True)
        y_aug = pd.concat([y_tv_current, pd.Series(pseudo_labels)], ignore_index=True)

        # Retrain
        spw_aug = (y_aug == 0).sum() / max((y_aug == 1).sum(), 1)
        params_aug = get_model_params(len(X_aug), spw_aug)
        n_folds_aug = 3 if len(X_aug) < 200 else 5

        models_aug, oof_aug, test_probs_aug, _ = train_multi_seed_ensemble(
            X_aug, y_aug, X_ext_sel, params_aug, n_seeds=5, n_folds=n_folds_aug)

        opt_t_aug, _ = optimize_threshold_f1(y_aug, oof_aug)
        metrics_aug = compute_metrics(y_ext, test_probs_aug, 0.5)
        metrics_aug_opt = compute_metrics(y_ext, test_probs_aug, opt_t_aug)

        print(f"  Round {rnd}: +{n_pseudo} pseudo ({n_pseudo_p}P/{n_pseudo_b}B) "
              f"→ train={len(X_aug)} | F1@0.5={metrics_aug['f1']:.4f} "
              f"AUC={metrics_aug['roc_auc']:.4f} F1@opt={metrics_aug_opt['f1']:.4f}")

        results.append({
            "round": rnd, "train_size": len(X_aug),
            "pseudo_added": n_pseudo, "pseudo_p": n_pseudo_p, "pseudo_b": n_pseudo_b,
            "f1_def": metrics_aug["f1"], "auc": metrics_aug["roc_auc"],
            "f1_opt": metrics_aug_opt["f1"], "opt_t": opt_t_aug,
        })

        # Update for next round
        test_probs = test_probs_aug
        X_tv_current = X_aug
        y_tv_current = y_aug

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("  Pseudo-Labeling Experiment — PathoSense @ Competition Sizes")
    print("  Anonymous columns | External PHACTboost test")
    print("=" * 70)

    all_results = {}
    for panel in ["general", "cancer", "pah", "cftr"]:
        all_results[panel] = run_pseudo_label(panel, conf_threshold=0.85, max_rounds=2)

    # Summary
    print("\n" + "=" * 90)
    print("  PSEUDO-LABELING RESULTS: Baseline vs Round 1 vs Round 2")
    print("=" * 90)
    print(f"  {'Panel':<10} {'Round':<8} {'Train':<8} {'Pseudo':<8} "
          f"{'F1@0.5':<10} {'AUC':<10} {'F1@opt':<10} {'delta F1':<10}")
    print(f"  {'-'*74}")

    for panel in ["general", "cancer", "pah", "cftr"]:
        baseline_f1 = all_results[panel][0]["f1_def"]
        for r in all_results[panel]:
            delta = r["f1_def"] - baseline_f1
            pseudo_str = str(r.get("pseudo_added", 0))
            print(f"  {panel:<10} {r['round']:<8} {r['train_size']:<8} {pseudo_str:<8} "
                  f"{r['f1_def']:<10.4f} {r['auc']:<10.4f} {r['f1_opt']:<10.4f} "
                  f"{delta:+10.4f}")

    out = os.path.join(RESULTS_DIR, "pseudo_labeling_experiment.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out}")
