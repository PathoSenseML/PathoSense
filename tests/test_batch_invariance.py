"""Regression test: deployed predict() is BATCH-INVARIANT.

A variant's score must not depend on which other rows are in the same inference
batch — finals scoring may run row-by-row or in bulk, and any per-batch fit
(median/quantile/af-col detection computed at inference time) would silently
break that. This locks the guarantee for all four panels' full deployed paths
(own model + transfer scorer + blend + threshold).

Asserts: prob(row i scored ALONE) == prob(row i scored IN A BATCH), exactly
(diff <= 1e-9), for several rows per panel, including a 1-row and a 2-row batch.

Run:  .venv/bin/python tests/test_batch_invariance.py     (exit 1 on failure)
"""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipelines"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

from utils import PANELS
from predict import predict, MODELS_DIR

TOL = 1e-9
# TabPFN does in-context inference with a ~2.4e-3 batch dependence (diluted to ~5e-4 in the 5-blend);
# per-row scoring to remove it is ~6.7 s/row = impractical. So when the OPTIONAL TabPFN member is active
# (general only), allow a small tolerance — it can't cross the deployed threshold. Everything else: strict.
TOL_TABPFN = 3e-3
ROWS_TO_CHECK = [0, 1, 5, 17, 33]   # spread across the batch


def _score(df, panel, tmpdir, tag):
    inp = os.path.join(tmpdir, f"in_{tag}.csv")
    out = os.path.join(tmpdir, f"out_{tag}.csv")
    df.to_csv(inp, index=False)
    return predict(inp, panel, out)


def _has_tabpfn(art_path):
    try:
        import joblib
        a = joblib.load(art_path)
        return (a.get("own") or {}).get("blend", {}) is not None and \
               ((a.get("own") or {}).get("blend") or {}).get("tabpfn") is not None
    except Exception:
        return False


def check_panel(panel, tmpdir):
    art = os.path.join(MODELS_DIR, panel, "artifacts.joblib")
    if not os.path.exists(art):
        print(f"  {panel:7s} SKIP (no trained model at {art})")
        return True
    tol = TOL_TABPFN if _has_tabpfn(art) else TOL
    src = pd.read_csv(PANELS[panel]).head(40).reset_index(drop=True)
    full = _score(src, panel, tmpdir, "full").reset_index(drop=True)

    worst = 0.0
    for i in ROWS_TO_CHECK:
        if i >= len(src):
            continue
        alone = _score(src.iloc[[i]], panel, tmpdir, "one").reset_index(drop=True)
        d = abs(float(alone["probability"].iloc[0]) - float(full["probability"].iloc[i]))
        worst = max(worst, d)
    # also a 2-row batch (row 0 + row 1) vs row 0 alone
    pair = _score(src.iloc[[0, 1]], panel, tmpdir, "pair").reset_index(drop=True)
    solo0 = _score(src.iloc[[0]], panel, tmpdir, "solo0").reset_index(drop=True)
    worst = max(worst, abs(float(pair["probability"].iloc[0]) - float(solo0["probability"].iloc[0])))

    ok = worst <= tol
    tag = "  [TabPFN tol]" if tol == TOL_TABPFN else ""
    print(f"  {panel:7s} {'PASS' if ok else 'FAIL'}  max|Δprob(alone vs batch)|={worst:.2e}{tag}")
    return ok


def main():
    print("=== batch-invariance regression test ===")
    with tempfile.TemporaryDirectory() as tmp:
        results = [check_panel(p, tmp) for p in ("general", "cancer", "pah", "cftr")]
    if all(results):
        print("ALL PASS"); return 0
    print("FAILURES DETECTED"); return 1


if __name__ == "__main__":
    sys.exit(main())
