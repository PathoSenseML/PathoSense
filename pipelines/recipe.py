"""Per-panel pipeline recipe — 2026-06-10 rework (validated in analysis/, honest
per-fold CV at the per-panel test prior).

Own-model preprocessing/feature recipe per panel:
  general — native-NaN + ALL features (no mRMR). The deployed mRMR-60 over-pruned the
            334-col AL_ block; using all features recovers it (+0.029 F1*, 5.5σ).
            native-NaN is neutral-but-cleaner on top (no imputation; honest CV Δ≈0).
            AL_ fam_flags help only WITH mRMR (they recover what mRMR pruned); with all
            features they are redundant and slightly hurt — so excluded. No score-agree.
            general is the transfer SOURCE (can't transfer to itself), so this is its lever.
  cancer/pah/cftr — deployed median+mRMR own model kept; cross-panel TRANSFER (below)
            dominates these small panels, so their own model is a minor (pah) or
            unused (cftr/cancer at w=1.0) blend component.

Cross-panel transfer weights live in transfer.TRANSFER_PANELS.
"""

PANEL_RECIPE = {
    "general": dict(prep="keep_nan", feature_select="none", al_features=(),
                    score_agree=False, interactions=False, acmg=False, blend=True),
    "cancer":  dict(prep="median",  feature_select="mrmr", al_features=(),
                    score_agree=True,  interactions=True,  acmg=False),
    "pah":     dict(prep="median",  feature_select="mrmr", al_features=(),
                    score_agree=False, interactions=False, acmg=True),
    "cftr":    dict(prep="median",  feature_select="mrmr", al_features=(),
                    score_agree=True,  interactions=True,  acmg=False),
}
