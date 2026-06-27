"""Instant analysis off the deployment-faithful member cache (/tmp/general_faithful_cache.npz):
is the 5-member F1-weight win REAL vs the actually-DEPLOYED 4-tree@0.81 (not the weak @prior baseline)?
Seed-stability + the optimized weight vector + a FIXED (mean-weight) deployable form.
"""
import sys, os, warnings
sys.path.insert(0, "analysis"); sys.path.insert(0, "pipelines")
warnings.filterwarnings("ignore")
import numpy as np
from scipy.optimize import minimize
from utils import TEST_PRIORS
from harness import f1_at_prior, tpr_fpr, pick_threshold_at_prior

PRIOR = TEST_PRIORS["general"]; CACHE = "/tmp/general_faithful_cache.npz"
NAMES = ["lgbm", "rf", "cat", "xgb", "tabpfn"]; IDX4 = [0, 1, 2, 3]; IDX5 = [0, 1, 2, 3, 4]


def f1_at(y, p, t): return f1_at_prior(*tpr_fpr(y, (p >= t).astype(int)), PRIOR)


def opt_w_f1(O, y):
    k = O.shape[0]
    def neg(w):
        w = np.clip(w, 0, None); s = w.sum()
        if s <= 0: return 0
        p = (w / s) @ O; return -f1_at(y, p, pick_threshold_at_prior(y, p, PRIOR))
    r = minimize(neg, np.ones(k) / k, method="Nelder-Mead", options={"maxiter": 400, "xatol": 1e-3})
    w = np.clip(r.x, 0, None); return w / w.sum()


z = np.load(CACHE); n = int(z["n"])
folds = [(z[f"O{i}"], z[f"E{i}"], z[f"ytr{i}"], z[f"yte{i}"]) for i in range(n)]
# folds are 3 seeds x 5; group by seed
seeds = [folds[0:5], folds[5:10], folds[10:15]]

# collect per-fold F1 for each config + the per-fold optimized weights
opt_ws = []
def cfg_f1(idx, mode):  # mode: '0.81' fixed / 'prior' / 'f1wts@prior'
    out = []
    for O, E, ytr, yte in folds:
        if mode == "f1wts@prior":
            w = opt_w_f1(O[idx], ytr); opt_ws.append(w); otr = w @ O[idx]; ote = w @ E[idx]
            t = pick_threshold_at_prior(ytr, otr, PRIOR)
        else:
            otr = O[idx].mean(0); ote = E[idx].mean(0)
            t = 0.81 if mode == "0.81" else pick_threshold_at_prior(ytr, otr, PRIOR)
        out.append(f1_at(yte, ote, t))
    return np.array(out)

dep = cfg_f1(IDX4, "0.81")   # DEPLOYED 4-tree @ fixed 0.81
configs = {
    "4tree @0.81 (DEPLOYED)": dep,
    "5-eq @0.81": cfg_f1(IDX5, "0.81"),
    "5-eq @prior": cfg_f1(IDX5, "prior"),
    "5-F1wts @prior": cfg_f1(IDX5, "f1wts@prior"),
}
print(f"=== vs DEPLOYED 4-tree@0.81 (n={len(dep)} folds, deployment-faithful cache) ===")
for name, f in configs.items():
    d = f - dep; se = d.std(ddof=1)/np.sqrt(len(d)) if len(d) > 1 else 1
    # per-seed mean deltas (seed-stability)
    ps = [np.mean(f[s*5:(s+1)*5] - dep[s*5:(s+1)*5]) for s in range(3)]
    allpos = all(x > 0 for x in ps)
    tag = "" if name.startswith("4tree") else f"  Δ={d.mean():+.4f} ({d.mean()/se:+.2f}σ)  per-seed={[round(x,4) for x in ps]} all+={allpos}"
    print(f"  {name:24s} F1={f.mean():.4f}{tag}")

mw = np.mean(opt_ws, axis=0)
print(f"\n  mean F1-opt weight vector [lgbm,rf,cat,xgb,tabpfn] = {np.round(mw,3).tolist()}")
# FIXED deployable: the mean optimized weights applied as a fixed vector
fixed = []
for O, E, ytr, yte in folds:
    ote = mw @ E[IDX5]; otr = mw @ O[IDX5]
    fixed.append(f1_at(yte, ote, pick_threshold_at_prior(ytr, otr, PRIOR)))
fixed = np.array(fixed); d = fixed - dep; se = d.std(ddof=1)/np.sqrt(len(d))
ps = [np.mean(fixed[s*5:(s+1)*5] - dep[s*5:(s+1)*5]) for s in range(3)]
print(f"  5-member @ FIXED mean-weights @prior  F1={fixed.mean():.4f}  Δ={d.mean():+.4f} ({d.mean()/se:+.2f}σ)  "
      f"per-seed={[round(x,4) for x in ps]} all+={all(x>0 for x in ps)}")
print("DONE")
