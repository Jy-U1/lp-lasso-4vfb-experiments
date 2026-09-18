"""Safeguarded scalar and vector lp proximity operators."""

from __future__ import annotations

import numpy as np


def lp_prox_threshold(tau: float, p: float) -> tuple[float, float]:
    if tau <= 0.0 or not 0.0 < p < 1.0:
        raise ValueError("need tau>0 and 0<p<1")
    gap = (2.0 * tau * (1.0 - p)) ** (1.0 / (2.0 - p))
    threshold = gap + tau * p * gap ** (p - 1.0)
    return float(gap), float(threshold)


def prox_lp_scalar(
    r: float, tau: float, p: float, tol: float = 1e-14, max_iter: int = 150
) -> float:
    a = abs(float(r))
    gap, threshold = lp_prox_threshold(tau, p)
    if a <= threshold:  # zero-at-threshold convention
        return 0.0
    lo, hi, x = gap, a, a
    for _ in range(max_iter):
        value = x + tau * p * x ** (p - 1.0) - a
        if abs(value) <= tol * max(1.0, a):
            break
        if value > 0.0:
            hi = x
        else:
            lo = x
        deriv = 1.0 - tau * p * (1.0 - p) * x ** (p - 2.0)
        trial = x - value / deriv if deriv > 0.0 and np.isfinite(deriv) else np.nan
        if not np.isfinite(trial) or trial <= lo or trial >= hi:
            trial = 0.5 * (lo + hi)
        x = trial
        if hi - lo <= tol * max(1.0, a):
            x = 0.5 * (lo + hi)
            break
    return float(np.copysign(x, r))


def prox_lp(
    r: np.ndarray, tau: float, p: float, tol: float = 1e-14, max_iter: int = 150
) -> np.ndarray:
    """Vectorized global prox with the zero-at-threshold selection."""
    r = np.asarray(r, dtype=float)
    a = np.abs(r)
    gap, threshold = lp_prox_threshold(tau, p)
    mask = a > threshold
    out = np.zeros_like(r)
    if not np.any(mask):
        return out
    target = a[mask]
    lo = np.full_like(target, gap)
    hi = target.copy()
    x = target.copy()
    for _ in range(max_iter):
        value = x + tau * p * x ** (p - 1.0) - target
        done = np.abs(value) <= tol * np.maximum(1.0, target)
        if np.all(done):
            break
        hi = np.where((value > 0.0) & ~done, x, hi)
        lo = np.where((value <= 0.0) & ~done, x, lo)
        deriv = 1.0 - tau * p * (1.0 - p) * x ** (p - 2.0)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            trial = x - value / deriv
        bad = (~np.isfinite(trial)) | (deriv <= 0.0) | (trial <= lo) | (trial >= hi)
        trial = np.where(bad, 0.5 * (lo + hi), trial)
        x = np.where(done, x, trial)
        if np.all((hi - lo) <= tol * np.maximum(1.0, target)):
            x = 0.5 * (lo + hi)
            break
    out[mask] = np.copysign(x, r[mask])
    return out

