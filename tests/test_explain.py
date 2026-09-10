"""Regression checks for explain.py against all four saved deployment artifacts.

Run: .venv/bin/python tests/test_explain.py (exit 1 on failure).
Requires locally trained models and panel CSVs; missing inputs fail, never silently skip.
Only the first 25 rows per panel are read. No restricted data is embedded here.

TreeSHAP is additive in raw log-odds. Each seed round-trips to its probability,
while averaged contributions round-trip to the MEAN LOGIT, not the mean
probability: sigmoid(mean(logits)) != mean(sigmoid(logits)). Consequently these
LightGBM explanations do not reconstruct the final calibrated blend probability.
"""
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipelines"))

import numpy as np
import pandas as pd
from scipy.special import expit

from explain import (FLAG_KEYS, af_columns, confidence_flags, explain_frame,
                     f1_at_prior, family_importance, family_of, global_importance,
                     lgbm_contribs, readable)
from predict import load_artifacts, score_frame
from utils import PANELS

# Rates from analysis/output/panel_cm_rates.csv are a separate confusion-rate run,
# not canonical CV means. General/cancer approximate canonical F1 within 0.01;
# PAH/CFTR require their own analytic expectations rather than canonical assertions.
RATES = {
    "general": (0.7353, 0.1714, 0.143, 0.537),
    "cancer": (0.8216, 0.0953, 0.167, 0.725),
    "pah": (0.8325, 0.2801, 0.286, 0.669),
    "cftr": (0.8403, 0.0407, 0.167, 0.851),
}


def check_panel(panel):
    art = load_artifacts(panel)
    df = pd.read_csv(PANELS[panel], nrows=25)
    assert len(df) == 25, "Expected 25 local input rows"
    scored = score_frame(df, art)
    explanation = explain_frame(art, scored)
    contribs, base = explanation["contribs"], explanation["base"]
    assert contribs.shape[0] == len(df) and base.shape == (len(df),)
    assert np.isfinite(contribs.to_numpy()).all() and np.isfinite(base).all()

    w = art["transfer_w"]
    expected = {}
    if art["own"] and (w is None or w < 1):
        expected["own"] = 1.0 if w is None else 1 - w
    if art["transfer"] and w > 0:
        expected["transfer"] = w
    members = explanation["members"]
    assert len(members) == len(expected)
    assert {name: weight for name, weight, _, _ in members} == expected

    worst = 0.0
    weighted_contribs = pd.DataFrame(0.0, index=contribs.index, columns=contribs.columns)
    weighted_base = np.zeros(len(df))
    for name, weight, member_contribs, member_base in members:
        assert weight > 0
        models = art["own"]["models"] if name == "own" else art["transfer"]["art"]["models"]
        X = scored["X_lgbm" if name == "own" else "Xs"]
        logits = []
        seed_contribs, seed_bases = [], []
        for model in models:
            c, b = lgbm_contribs([model], X)
            raw_logit = model.booster_.predict(X, raw_score=True)
            reconstructed = expit(b + c.sum(axis=1).to_numpy())
            actual = model.predict_proba(X)[:, 1]
            np.testing.assert_allclose(reconstructed, actual, atol=1e-6, rtol=0)
            worst = max(worst, float(np.max(np.abs(reconstructed - actual))))
            logits.append(raw_logit)
            seed_contribs.append(c.to_numpy())
            seed_bases.append(b)
        np.testing.assert_allclose(member_contribs.to_numpy(), np.mean(seed_contribs, axis=0), atol=1e-9, rtol=0)
        np.testing.assert_allclose(member_base, np.mean(seed_bases, axis=0), atol=1e-9, rtol=0)
        np.testing.assert_allclose(member_base + member_contribs.sum(axis=1).to_numpy(),
                                   np.mean(logits, axis=0), atol=1e-6, rtol=0)
        weighted_contribs += weight * member_contribs.reindex(columns=contribs.columns, fill_value=0)
        weighted_base += weight * member_base
    np.testing.assert_allclose(contribs, weighted_contribs, atol=1e-9, rtol=0)
    np.testing.assert_allclose(base, weighted_base, atol=1e-9, rtol=0)

    importance = global_importance(art)
    assert np.isfinite(importance).all() and (importance >= 0).all()
    assert abs(importance.sum() - 1) <= 1e-9
    grouped = family_importance(importance)
    assert abs(grouped.sum() - importance.sum()) <= 1e-9
    assert (grouped >= 0).all()

    # Deliberately non-contiguous index exercises flag alignment, not just row count.
    indexed = df.set_axis(pd.Index(np.arange(len(df)) * 3 + 7, name="source_row"))
    flags = confidence_flags(indexed, scored["probs"], art["threshold"])
    assert tuple(flags.columns) == FLAG_KEYS and flags.index.equals(indexed.index)
    assert all(dtype == bool for dtype in flags.dtypes)
    np.testing.assert_array_equal(flags["borderline"],
                                  np.abs(scored["probs"] - art["threshold"]) < 0.05)

    tpr, fpr, prior, canonical = RATES[panel]
    projected = f1_at_prior(tpr, fpr, prior)[2]
    if panel in ("pah", "cftr"):
        analytic = {"pah": 0.6576425133796188, "cftr": 0.8224879642426353}[panel]
        assert abs(projected - analytic) < 1e-6, (projected, analytic)
    else:
        assert abs(projected - canonical) < 0.01, (projected, canonical)
    af = af_columns(df)
    for feature in contribs.columns:
        for lang in ("en", "tr"):
            assert readable(feature, lang, af).strip(), (feature, lang)
    print(f"  {panel:7s} PASS  rows={len(df)} members={','.join(expected)} "
          f"max|seed round-trip|={worst:.2e} F1*={projected:.3f}")


def main():
    print("=== explainability regression test ===")
    try:
        sample = pd.DataFrame({"AL_1": [0.0, 1.0, np.nan],
                               "AL_2": [1e-5, 1.0, np.nan],
                               "AL_3": [1e-40, 1.0, np.nan]})
        assert af_columns(sample) == {"AL_2", "AL_3"}
        print("  AF classification PASS (binary, fractional, sub-1e-12 values)")
        # FX_score_agree_* aggregates AUC-selected numeric features; only the FX_cons_* pair is chemistry.
        for feature in ("FX_score_agree_count", "FX_score_agree_mean", "FX_score_agree_std"):
            assert family_of(feature) == "other", feature
            assert "scaled features" in readable(feature), feature
            assert "ölçeklenmiş özellik" in readable(feature, "tr"), feature
        assert family_of("FX_score_agree_mean_missing") == "missingness"
        assert family_of("FX_cons_x_disrupt") == "conservation" and family_of("AA_grantham") == "amino_acid"
        print("  score-agreement taxonomy PASS (other family, honest name in both languages)")
        # The AL_ family flags are named as missingness patterns, so they are drawn in that family:
        # one screen cannot say allele frequency in its legend and missingness in its label.
        for feature in ("ALfam_1040", "ALn_fams"):
            assert family_of(feature) == "missingness", feature
            assert "missingness" in readable(feature).lower(), feature
            assert "eksiklik" in readable(feature, "tr").lower(), feature
        assert family_of("AL_327") == "allele_frequency" and family_of("AL_327_missing") == "missingness"
        print("  AL_ family-flag taxonomy PASS (missingness, matching their own names)")
    except Exception:
        traceback.print_exc()
        return 1
    results = []
    for panel in RATES:
        try:
            check_panel(panel)
            results.append(True)
        except Exception:
            print(f"  {panel:7s} FAIL", flush=True)
            traceback.print_exc()
            results.append(False)
    print("ALL PASS" if all(results) else "FAILURES DETECTED")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
