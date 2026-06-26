"""AUDIT FOLLOW-UP: the transfer-scorer blend RF (transfer.py:273) uses the UN-regularized config
(n_est=600, max_features='sqrt', unlimited depth) that HPO proved over-fits the general matrix and which
we already fixed on the general OWN model (_rf_general: depth=9, max_features=0.3, min_leaf=10). Does
regularizing the TRANSFER RF improve the deployed cftr/cancer scorers? Honest eval: target rows are held
out from the general scorer via Variant_ID exclusion, so we score the FULL target panel. 5 seeds, paired
(current vs regularized RF, identical LGBM/CatBoost per seed → isolates the RF change). AUC + max-F1*.
Run: .venv/bin/python -u analysis/exp_transfer_rf.py
"""
import sys, numpy as np
sys.path.insert(0, "analysis"); sys.path.insert(0, "pipelines")
import warnings; warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier as RealRF
from sklearn.metrics import roc_auc_score
import transfer, config
from transfer import fit_transfer_scorer, transfer_score, _load_raw, TRANSFER_BLEND
from schema import partition_columns
from utils import TEST_PRIORS
from harness import f1_at_prior, tpr_fpr

print("TRANSFER_BLEND panels (use RF):", TRANSFER_BLEND, flush=True)


def reg_rf(*a, **kw):
    """Regularized RF (mirror general_blend._rf_general), preserving the call's random_state/n_jobs/cw."""
    return RealRF(n_estimators=1100, max_features=0.3, max_depth=9, min_samples_leaf=10,
                  n_jobs=kw.get("n_jobs", 4), class_weight=kw.get("class_weight", "balanced"),
                  random_state=kw.get("random_state"))


def eval_target(panel, rf_cls, seed_off):
    transfer.RandomForestClassifier = rf_cls
    transfer.SEED = config.SEED + seed_off
    try:
        art = fit_transfer_scorer(panel)
        raw = _load_raw(panel)
        probs = transfer_score(raw, art)
    finally:
        transfer.RandomForestClassifier = RealRF; transfer.SEED = config.SEED
    _, n, c, t = partition_columns(raw)
    y = raw[t].astype(int).values
    prior = TEST_PRIORS[panel]
    auc = float(roc_auc_score(y, probs))
    f1 = max(f1_at_prior(*tpr_fpr(y, (probs >= q).astype(int)), prior)
             for q in np.quantile(probs, np.arange(0.02, 0.98, 0.01)))
    return auc, f1


def main():
    SEEDS = range(5)
    for panel in ("cftr", "cancer"):
        cur = [eval_target(panel, RealRF, s) for s in SEEDS]
        reg = [eval_target(panel, reg_rf, s) for s in SEEDS]
        ca = np.array([x[0] for x in cur]); ra = np.array([x[0] for x in reg])
        cf = np.array([x[1] for x in cur]); rf = np.array([x[1] for x in reg])
        dA = ra - ca; dF = rf - cf
        seA = dA.std(ddof=1) / np.sqrt(len(dA)); seF = dF.std(ddof=1) / np.sqrt(len(dF))
        print(f"\n[{panel}] (transfer scorer on full held-out target panel, {len(list(SEEDS))} seeds)")
        print(f"  AUC : current {ca.mean():.4f}  regularized {ra.mean():.4f}  "
              f"Δ={dA.mean():+.4f} ({dA.mean()/seA if seA>0 else 0:+.2f}σ)")
        print(f"  F1* : current {cf.mean():.4f}  regularized {rf.mean():.4f}  "
              f"Δ={dF.mean():+.4f} ({dF.mean()/seF if seF>0 else 0:+.2f}σ)", flush=True)


if __name__ == "__main__":
    main()
