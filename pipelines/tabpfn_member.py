"""OPTIONAL TabPFN-v2.6 member for the GENERAL blend.

The only model lift past the GBDT information ceiling (analysis/exp_tabpfn_general.py): added as an
equal-weight 5th member it gives +0.0025 AUC (+6.4σ) / ~+0.01 honest F1 (general ~0.539→~0.549). It is
OPTIONAL and gracefully falls back to the 4-tree blend when unavailable — it needs a TabPFN license +
`TABPFN_TOKEN` + a CUDA GPU (so the finals jury's offline re-run, lacking those, just runs the clean
4-tree blend = unchanged behavior).

TabPFN does IN-CONTEXT learning: the "model" is the training matrix itself, stored in the artifact and
replayed at inference. Inference is BATCH-scored (per-row would be ~6.7 s/row = impractical at finals
scale). Batch scoring carries a ~2.4e-3 in-context batch-dependence in the member → ~5e-4 in the 5-blend
(negligible — can't cross the threshold; the batch-invariance test allows general a small tolerance only
when this member is present).
"""
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder

from config import SEED
from calibration import fit_platt, apply_platt

TABPFN_MODEL = "tabpfn-v2.6-classifier-v2.6_default.ckpt"   # v2.6 beats v3 +0.0054 AUC here
TABPFN_NEST = 16


def tabpfn_usable() -> bool:
    """True only if TabPFN can actually run here (license token + import + CUDA)."""
    if not os.environ.get("TABPFN_TOKEN"):
        return False
    try:
        import tabpfn  # noqa: F401
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _encode(X_pp, cat_feats, num_cols, enc=None):
    """Native-NaN numeric (NaN kept — TabPFN handles it) + ordinal-encoded categoricals -> float32."""
    A = X_pp[num_cols].to_numpy(dtype=float)
    cidx = None
    if cat_feats:
        cats = X_pp[cat_feats].astype(object).where(X_pp[cat_feats].notna(), "NA")
        if enc is None:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            Ac = enc.fit_transform(cats)
        else:
            Ac = enc.transform(cats)
        A = np.hstack([A, Ac])
        cidx = list(range(len(num_cols), len(num_cols) + len(cat_feats)))
    return A.astype(np.float32), enc, cidx


def _clf(cidx, seed=SEED):
    from tabpfn import TabPFNClassifier
    return TabPFNClassifier(device=_device(), model_path=TABPFN_MODEL, n_estimators=TABPFN_NEST,
                            ignore_pretraining_limits=True, categorical_features_indices=cidx,
                            random_state=seed)


def fit_tabpfn_member(X_pp, y, cat_feats, n_splits=5):
    """Return (art, oof_cal) or (None, None) if TabPFN is unavailable/fails (graceful fallback).
    art stores the encoded TRAIN matrix + labels (TabPFN is in-context) + Platt + encoder."""
    if not tabpfn_usable():
        return None, None
    try:
        num_cols = [c for c in X_pp.columns if c not in cat_feats]
        A, enc, cidx = _encode(X_pp, cat_feats, num_cols)
        yv = np.asarray(y).astype(int)
        oof = np.zeros(len(yv))
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=SEED)
        for a, b in skf.split(A, yv):
            m = _clf(cidx); m.fit(A[a], yv[a]); oof[b] = m.predict_proba(A[b])[:, 1]
        platt = fit_platt(oof, yv)
        art = {"train_X": A, "train_y": yv, "encoder": enc, "cat_idx": cidx,
               "cat_feats": list(cat_feats), "num_cols": num_cols, "platt": platt,
               "model": TABPFN_MODEL, "n_est": TABPFN_NEST}
        return art, apply_platt(oof, platt)
    except Exception as e:
        print(f"  [tabpfn] member fit failed ({type(e).__name__}: {e}); 4-tree blend fallback")
        return None, None


def tabpfn_member_probs(X_pp, art):
    """Inference: reconstruct TabPFN from the stored train matrix, batch-score X_pp. Platt-calibrated.
    Deterministic given fixed train+seed+device (batch-scored; ~5e-4 batch-dependence — see module doc)."""
    A, _, _ = _encode(X_pp, art["cat_feats"], art["num_cols"], enc=art["encoder"])
    m = _clf(art["cat_idx"]); m.fit(art["train_X"], art["train_y"])
    return apply_platt(m.predict_proba(A)[:, 1], art["platt"])
