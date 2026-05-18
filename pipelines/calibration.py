"""Probability calibration utilities.

Implements Platt scaling (Platt 1999) — a 2-parameter logistic regression
fit on log-odds of predicted probabilities. Standard formulation:

    calibrated_p = 1 / (1 + exp(a * logit(raw_p) + b))

Where:
    logit(p) = log(p / (1 - p))
    a, b are fitted by maximum likelihood on (logit(oof_probs), y_true).

When the underlying model already produces well-calibrated probabilities,
Platt fits ~(a=-1, b=0) -- effectively the identity in sigmoid space.
When the model systematically under- or over-predicts, Platt corrects it.

For PathoSense the model is trained with `scale_pos_weight = n_neg/n_pos`
which empirically yields probabilities tracking the *training prior*, not 0.5
(see analysis/output/threshold_diagnostics.json). Platt corrects this and
gives probabilities centered around the true conditional likelihood.

Why fit on log-odds, not raw probability:
    - Raw-probability inputs saturate near 0 and 1 (sigmoid floor/ceiling).
    - Log-odds input is unbounded, lets the calibrator use the full sigmoid
      curve symmetrically.
    - Standard Platt formulation in the literature uses log-odds.
"""
from __future__ import annotations
from typing import Dict
import numpy as np
from sklearn.linear_model import LogisticRegression


_EPS = 1e-7


def _logit(p):
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def fit_platt(probs: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Fit Platt sigmoid on (oof probabilities, labels). Returns (a, b)."""
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=int)
    X = _logit(probs).reshape(-1, 1)
    # LogisticRegression with L2=infinity (C=1e6) → MLE fit, no regularisation.
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(X, y)
    a = float(lr.coef_[0, 0])
    b = float(lr.intercept_[0])
    return {"a": a, "b": b}


def apply_platt(probs: np.ndarray, fitted: Dict[str, float]) -> np.ndarray:
    """Apply saved Platt parameters to new probabilities."""
    probs = np.asarray(probs, dtype=float)
    z = fitted["a"] * _logit(probs) + fitted["b"]
    return 1.0 / (1.0 + np.exp(-z))


def calibration_summary(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    """Mean predicted probability, mean actual rate, and Expected Calibration Error."""
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=int)
    bins = np.linspace(0, 1, n_bins + 1)
    ece, mean_pred, mean_act = 0.0, float(probs.mean()), float(y.mean())
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if i == n_bins - 1:
            mask |= (probs == bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_pred = float(probs[mask].mean())
        bin_act = float(y[mask].mean())
        ece += (mask.sum() / len(probs)) * abs(bin_pred - bin_act)
    return {
        "mean_predicted": round(mean_pred, 4),
        "mean_actual":    round(mean_act, 4),
        "ece":            round(ece, 4),
    }


if __name__ == "__main__":
    # Smoke test: model that over-predicts a 50/50 dataset; Platt should pull it back
    rng = np.random.RandomState(42)
    y = rng.randint(0, 2, size=2000)
    raw = 0.3 + 0.4 * y + 0.05 * rng.randn(2000)
    raw = np.clip(raw, 0.01, 0.99)
    fitted = fit_platt(raw, y)
    calibrated = apply_platt(raw, fitted)
    print(f"Fitted Platt: a={fitted['a']:.3f}, b={fitted['b']:.3f}")
    print(f"Raw       : {calibration_summary(raw, y)}")
    print(f"Calibrated: {calibration_summary(calibrated, y)}")
