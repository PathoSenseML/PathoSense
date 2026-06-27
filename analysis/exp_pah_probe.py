"""STEP B — pah ranking rule-out (strict, time-boxed). pah is pure general-transfer (w=1.0), AUC 0.802,
the lowest -> the only panel with nominal un-chased ranking headroom. Prior is LOW: ~10 documented pah
ranking levers already rejected (pooled source -2.1sigma, hetero-blend -0.012, HPO -7sigma, domain-adapt
artifact, own-model blends -2.4sigma, fam_flags-replace/drop_al -8sigma, spw_bag -7sigma). The transfer
docstring even records additive fam_flags as 'slightly hurts'. Remaining knobs only logged as 'marginal':
multi-seed scorer (n_seeds) + additive fam_flags. Test BOTH, paired, primary metric AUC (prevalence-
invariant) + recal-F1 @0.286. STOP on first dry signal.

Run: .venv/bin/python -u analysis/exp_pah_probe.py
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from config import SEED
from utils import TEST_PRIORS
from harness import pick_threshold_at_prior, tpr_fpr, f1_at_prior
from exp_transfer import _load
from transfer import fit_transfer_scorer, transfer_score

PRIOR = TEST_PRIORS["pah"]; PANEL = "pah"
t0 = time.time()
df = _load(PANEL)
y = df["Pathogenicity"].astype(int).values if "Pathogenicity" in df else None
if y is None:                                   # resolve target column robustly
    from schema import partition_columns
    _, _, _, tgt = partition_columns(df); y = df[tgt].astype(int).values
print(f"pah rows={len(y)}  prior(train)={y.mean():.3f}  test-prior={PRIOR}", flush=True)


def score_variant(**kw):
    art = fit_transfer_scorer(PANEL, **kw)
    return transfer_score(_load(PANEL), art)


def auc_boot(y, ga, gb, nboot=2000, seed=0):
    """paired bootstrap of AUC(gb)-AUC(ga)."""
    rng = np.random.RandomState(seed); n = len(y); d = []
    base = roc_auc_score(y, gb) - roc_auc_score(y, ga)
    for _ in range(nboot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        d.append(roc_auc_score(y[idx], gb[idx]) - roc_auc_score(y[idx], ga[idx]))
    d = np.array(d); se = d.std(ddof=1)
    return base, se, (base / se if se > 0 else 0.0)


def recalF1(y, g, seeds=(SEED, SEED + 1000, SEED + 2000), nr=12):
    """recal-F1 @ pah prior: threshold on train-rows score, eval on test-rows. pure-transfer => g is
    held-out for all rows. paired-ready per-fold array returned."""
    out = []
    for s in seeds:
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=nr, random_state=s)
        for tr, te in rskf.split(g.reshape(-1, 1), y):
            t = pick_threshold_at_prior(y[tr], g[tr], PRIOR)
            out.append(f1_at_prior(*tpr_fpr(y[te], (g[te] >= t).astype(int)), PRIOR))
    return np.array(out)


print("\n[baseline] deployed pah scorer (n_seeds=1, no fam_flags)...", flush=True)
g_base = score_variant()
auc_base = roc_auc_score(y, g_base); f1_base = recalF1(y, g_base)
print(f"  AUC={auc_base:.4f}  recalF1@{PRIOR}={f1_base.mean():.4f}  ({time.time()-t0:.0f}s)", flush=True)

variants = [("multiseed_n5", dict(n_seeds=5)),
            ("famflags_additive", dict(al_features=("fam_flags",)))]
rows = [{"variant": "baseline", "AUC": auc_base, "dAUC": 0.0, "sigma_AUC": 0.0,
         "recalF1": f1_base.mean(), "dF1": 0.0, "sigma_F1": 0.0}]
for name, kw in variants:
    print(f"\n[{name}] {kw} ...", flush=True)
    g = score_variant(**kw)
    auc = roc_auc_score(y, g)
    dauc, se_auc, sig_auc = auc_boot(y, g_base, g)
    f1 = recalF1(y, g)
    df1 = (f1 - f1_base); se_f1 = df1.std(ddof=1) / np.sqrt(len(df1)); sig_f1 = df1.mean() / se_f1 if se_f1 > 0 else 0.0
    print(f"  AUC={auc:.4f} (Δ{dauc:+.4f}, {sig_auc:+.2f}σ)   recalF1={f1.mean():.4f} (Δ{df1.mean():+.4f}, {sig_f1:+.2f}σ)  ({time.time()-t0:.0f}s)", flush=True)
    rows.append({"variant": name, "AUC": auc, "dAUC": dauc, "sigma_AUC": sig_auc,
                 "recalF1": f1.mean(), "dF1": df1.mean(), "sigma_F1": sig_f1})
    # STOP-on-dry rule: a lever is a confirmed dry well if AUC doesn't robustly rise OR the F1 doesn't follow
    if not (sig_auc >= 2.0 and sig_f1 >= 2.0 and df1.mean() > 0):
        print(f"  -> DRY (need ≥2σ on BOTH AUC and recal-F1 with +F1; this lever is closed for pah).", flush=True)

out = pd.DataFrame(rows)
out.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "pah_probe.csv"), index=False)
print("\n" + out.to_string(index=False), flush=True)
print(f"\nVERDICT: {'pah RANKING CLOSED (no lever robust on both AUC and F1)' if all((r['sigma_AUC']<2 or r['sigma_F1']<2 or r['dF1']<=0) for r in rows[1:]) else 'CHECK — a lever moved both metrics'}", flush=True)
print(f"total {time.time()-t0:.0f}s", flush=True)
