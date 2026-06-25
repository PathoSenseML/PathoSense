"""Re-verify multi-seed transfer-scorer averaging in the DEPLOYED config (variance reduction → should
be non-negative). The scorer's lgbm_p is averaged over n_seeds models (varying model seed + ES-split
seed). rf/cat components are fit once (deterministic), unchanged by n_seeds. Test n_seeds 1 vs 10.

Deployed: cftr = pure transfer (w=1.0, equal-weight blend); cancer = 0.5·transfer(RF-up 0.25/0.5/0.25)
+ 0.5·own. Paired Δ(ms10 − ms1) in deployed F1* at the test prior, TWO seeds. Deploy per-panel if Δ is
non-negative in BOTH seeds (the new bar: doesn't drop the score).
Run: .venv/bin/python -u analysis/exp_multiseed_deployed.py [panel ...]
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import RepeatedStratifiedKFold
from config import SEED
from utils import TEST_PRIORS
from calibration import fit_platt, apply_platt
from harness import pick_threshold_at_prior, tpr_fpr, f1_at_prior
from transfer import fit_transfer_scorer, TRANSFER_BLEND_WEIGHTS, TRANSFER_PANELS
from exp_transfer import _load, load_target_engineered, target_deployed_fn
from exp_transfer_blendw import transfer_components

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def blended_g(art, df, panel):
    L, R, C = transfer_components(df, art)
    w = TRANSFER_BLEND_WEIGHTS.get(panel, (1/3, 1/3, 1/3))
    return w[0]*L + w[1]*R + w[2]*C


def evaluate(panel, nseeds_list=(1, 10)):
    prior = TEST_PRIORS[panel]; w_t = TRANSFER_PANELS[panel]
    df = _load(panel)
    gmap = {}
    for ns in nseeds_list:
        art = fit_transfer_scorer(panel, n_seeds=ns)
        gmap[ns] = blended_g(art, df, panel)
    yt_full = df["Label"].astype(int).values
    from sklearn.metrics import roc_auc_score
    print(f"[{panel}] scorer AUC " + " ".join(f"ms{ns}={roc_auc_score(yt_full,gmap[ns]):.4f}" for ns in nseeds_list), flush=True)

    need_own = w_t < 1.0
    if need_own:
        Xt_own, yt = load_target_engineered(panel); fn = target_deployed_fn(panel)
        assert (yt.values == yt_full).all()
    rows = []
    for bs in (SEED, SEED + 1000):
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=12, random_state=bs)
        acc = {ns: [] for ns in nseeds_list}
        for tr, te in rskf.split(np.zeros(len(yt_full)), yt_full):
            tr, te = np.asarray(tr), np.asarray(te)
            ytr, yte = yt_full[tr], yt_full[te]
            if need_own:
                e_te, e_tr = fn(Xt_own.iloc[tr].reset_index(drop=True),
                                pd.Series(ytr), Xt_own.iloc[te].reset_index(drop=True))
            for ns in nseeds_list:
                g = gmap[ns]
                pl = fit_platt(g[tr], ytr); gtr, gte = apply_platt(g[tr], pl), apply_platt(g[te], pl)
                if need_own:
                    b_tr = w_t*gtr + (1-w_t)*e_tr; b_te = w_t*gte + (1-w_t)*e_te
                else:
                    b_tr, b_te = gtr, gte
                t = pick_threshold_at_prior(ytr, b_tr, prior)
                acc[ns].append(f1_at_prior(*tpr_fpr(yte, (b_te >= t).astype(int)), prior))
        base = np.array(acc[nseeds_list[0]]); nf = len(base)
        print(f"  seed{bs}: ms{nseeds_list[0]} F1*={base.mean():.4f}", flush=True)
        for ns in nseeds_list[1:]:
            a = np.array(acc[ns]); d = a - base
            se = d.std(ddof=1)/np.sqrt(nf); sig = d.mean()/se if se > 0 else 0.0
            print(f"    ms{ns} F1*={a.mean():.4f}  Δ={d.mean():+.4f} ({sig:+.2f}σ)", flush=True)
            rows.append({"panel": panel, "seed": bs, "nseeds": ns, "f1": a.mean(),
                         "delta": d.mean(), "sigma": sig})
    return rows


if __name__ == "__main__":
    panels = sys.argv[1:] or ["cftr", "cancer"]
    print("=== multi-seed transfer scorer in deployed config (non-negative test) ===", flush=True)
    allrows = []
    for pn in panels:
        t0 = time.time(); allrows += evaluate(pn)
        print(f"  ({pn} {time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(allrows).to_csv(os.path.join(OUT, "multiseed_deployed.csv"), index=False)
    print(f"\nsaved -> multiseed_deployed.csv", flush=True)
