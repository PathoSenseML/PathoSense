"""Recalibrate the deployed FIXED thresholds for the POST-AUDIT-FIX (retrained) models.

The audit fixes (#6 nested-ES OOF, #3 fast_mrmr) changed general's & cancer's calibrated
probability distribution, but THRESHOLD_OVERRIDE (general 0.82, cancer 0.81) was carried over
from the OLD models. The retrain selector now picks 0.83 (general) / 0.72 (cancer) — so the
fixed threshold may no longer be optimal, which can show up as an F1 "drop" while ranking (AUC)
held or improved. Re-derive the optimal fixed threshold honestly.

Faithful: replicates the ACTUAL deployed pipeline per fold (general = LGBM-redun-prune + RF +
CatBoost + XGB blend; cancer = 0.5*transfer + 0.5*own-LGBM), computes train blend-OOF (for the
selector) + held-out blend-test, sweeps a fixed-threshold grid on the held-out folds, 2 seeds.
pah/cftr unchanged (own model not built) -> their thresholds stand; not re-run here.

Run: .venv/bin/python -u analysis/exp_threshold_recalibrate.py
"""
import sys, warnings
sys.path.insert(0, "analysis"); sys.path.insert(0, "pipelines")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score

from config import SEED
from utils import TEST_PRIORS, optimize_threshold_f1_balanced
from predict import (_fit_own_preprocess, _apply_own_preprocess, _lgbm_redun_keep,
                     REDUN_PRUNE_LGBM, LGBM_PARAM_OVERRIDE)
from transfer import TRANSFER_PANELS
from run_pipeline import get_model_params, train_multi_seed_ensemble
from general_blend import fit_general_blend, general_blend_probs
from calibration import fit_platt, apply_platt
from harness import tpr_fpr, f1_at_prior, load_panel
from schema import partition_columns
from exp_own_model_bakeoff import _load            # df, X, y, cfg, g (OOS transfer)

N_SEEDS = 3
SEEDS = [SEED, SEED + 1000, SEED + 2000]
GRID = {"general": [0.78, 0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86],
        "cancer":  [0.76, 0.78, 0.80, 0.81, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92]}
DEPLOYED = {"general": 0.82, "cancer": 0.81}


def load_panel_xy(panel):
    """general has no transfer scorer -> engineer + leaky-drop directly (g=None), mirroring _load."""
    if panel != "general":
        df, X, y, cfg, g = _load(panel)
        return df, X, y, cfg, np.asarray(g)
    from schema import normalize_sentinels, drop_all_nan_rows
    from utils import detect_leaky_columns, KNOWN_LEAKY, PANELS
    from recipe import PANEL_RECIPE
    from predict import _engineer
    cfg = PANEL_RECIPE["general"]
    df = pd.read_csv(PANELS["general"]); df = normalize_sentinels(df); df = drop_all_nan_rows(df)
    de = _engineer(df.copy(), cfg); _, num, cat, tgt = partition_columns(de)
    y = de[tgt].astype(int).reset_index(drop=True); X = de[num + cat].reset_index(drop=True)
    leaky = detect_leaky_columns(X[num], y, auc_threshold=0.99, verbose=False)
    X = X.drop(columns=sorted(set(leaky) | (KNOWN_LEAKY & set(X.columns))))
    return df, X, y, cfg, None


def fold_blend(panel, Xtr0, ytr, Xte0, cfg, g_tr, g_te):
    """Return (blend_oof_on_train, blend_test_on_heldout) replicating the deployed pipeline."""
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    Xtr, prep = _fit_own_preprocess(Xtr0, ytr, cfg)
    Xte = _apply_own_preprocess(Xte0, prep).reindex(columns=Xtr.columns)
    params = get_model_params(len(Xtr), spw)
    if panel in LGBM_PARAM_OVERRIDE:
        params = {**params, **LGBM_PARAM_OVERRIDE[panel]}
    n_folds = 3 if len(Xtr) < 200 else 5
    Xtr_l, Xte_l = Xtr, Xte
    if panel in REDUN_PRUNE_LGBM:
        keep = _lgbm_redun_keep(Xtr); Xtr_l, Xte_l = Xtr[keep], Xte[keep]
    _, oof, test, _ = train_multi_seed_ensemble(Xtr_l, ytr, Xte_l, params, n_seeds=N_SEEDS, n_folds=n_folds)
    pl = fit_platt(oof, ytr.values)
    own_oof, own_test = apply_platt(oof, pl), apply_platt(test, pl)
    if cfg.get("blend"):
        art, own_oof = fit_general_blend(Xtr, ytr, own_oof, spw, n_splits=n_folds)
        own_test = general_blend_probs(Xte, own_test, art)
    w = TRANSFER_PANELS.get(panel)
    if w is None:                                   # general: own only
        return own_oof, own_test
    plg = fit_platt(g_tr, ytr.values)
    bo = w * apply_platt(g_tr, plg) + (1 - w) * own_oof
    bt = w * apply_platt(g_te, plg) + (1 - w) * own_test
    return bo, bt


def run(panel):
    df, X, y, cfg, g = load_panel_xy(panel)
    prior = TEST_PRIORS[panel]; grid = GRID[panel]; dep = DEPLOYED[panel]
    nrep = 2 if panel == "cancer" else 1            # general fold is heavy
    print(f"\n==== {panel}  (prior {prior:.3f}, deployed override t={dep}) ====", flush=True)
    best_per_seed = []; pooled = {t: [] for t in grid}
    for seed in SEEDS:
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=nrep, random_state=seed)
        sel_f1, sel_t = [], []; fixed = {t: [] for t in grid}; aucs = []
        for tr, te in rskf.split(X, y):
            ytr = y.iloc[tr].reset_index(drop=True); yte = y.iloc[te].values
            gtr = g[tr] if g is not None else None; gte = g[te] if g is not None else None
            bo, bt = fold_blend(panel, X.iloc[tr].reset_index(drop=True), ytr,
                                X.iloc[te].reset_index(drop=True), cfg, gtr, gte)
            aucs.append(roc_auc_score(yte, bt))
            td, _, _ = optimize_threshold_f1_balanced(ytr.values, bo, target_prior=prior, n_bootstrap=200)
            sel_t.append(td); sel_f1.append(f1_at_prior(*tpr_fpr(yte, (bt >= td).astype(int)), prior))
            for t in grid:
                v = f1_at_prior(*tpr_fpr(yte, (bt >= t).astype(int)), prior)
                fixed[t].append(v); pooled[t].append(v)
        sel_f1 = np.array(sel_f1); n = len(sel_f1)
        bestt = max(grid, key=lambda t: np.mean(fixed[t])); best_per_seed.append(bestt)
        print(f"  seed{seed}: AUC={np.mean(aucs):.4f}  selector~{np.mean(sel_t):.3f} "
              f"(F1*={sel_f1.mean():.4f}) | dep t={dep} F1*={np.mean(fixed[dep]):.4f} "
              f"| best fixed t={bestt} F1*={np.mean(fixed[bestt]):.4f}", flush=True)
    # pooled across all seeds, paired vs deployed
    depp = np.array(pooled[dep]); print(f"  --- pooled ({len(depp)} folds), Δ vs deployed t={dep} ---")
    for t in grid:
        a = np.array(pooled[t]); d = a - depp
        se = d.std(ddof=1)/np.sqrt(len(d)) if len(d) > 1 else 0
        sig = d.mean()/se if se > 0 else 0
        allseed = "stable+" if all(bt >= dep for bt in best_per_seed) else ""
        mark = "  <-pooled best" if t == max(grid, key=lambda u: np.mean(pooled[u])) else ""
        flag = "  <<< ROBUST" if sig >= 2 and d.mean() > 0 else ""
        print(f"      t={t:.2f}  F1*={a.mean():.4f}  Δ={d.mean():+.4f} ({sig:+.2f}σ){mark}{flag}", flush=True)
    print(f"  >> best per seed: {best_per_seed}  (deployed {dep})", flush=True)


if __name__ == "__main__":
    for p in ("cancer", "general"):
        run(p)
    print("\nDONE")
