"""Active-set geometry and numerical local-minimum diagnostics."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh

from .model import LpLassoProblem


def active_support(x: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    return np.flatnonzero(np.abs(x) > threshold)


def active_hessian(
    problem: LpLassoProblem, x: np.ndarray, threshold: float = 0.0
) -> tuple[np.ndarray, np.ndarray | None]:
    support = active_support(x, threshold)
    if support.size == 0:
        return support, None
    u = x[support]
    matrix = problem.A[:, support]
    hessian = matrix.T @ matrix - problem.lam * problem.p * (1.0 - problem.p) * np.diag(
        np.abs(u) ** (problem.p - 2.0)
    )
    return support, hessian


def smallest_eigenpair(hessian: np.ndarray) -> tuple[float, np.ndarray]:
    if hessian.shape == (1, 1):
        return float(hessian[0, 0]), np.ones(1)
    values, vectors = eigh(hessian, subset_by_index=[0, 0], check_finite=False)
    return float(values[0]), vectors[:, 0].copy()


def active_hessian_stats(
    problem: LpLassoProblem, x: np.ndarray, threshold: float = 0.0
) -> dict[str, float | int]:
    support, hessian = active_hessian(problem, x, threshold)
    if hessian is None:
        return {
            "support_size": 0,
            "lambda_min": np.nan,
            "hessian_norm": np.nan,
            "relative_lambda_min": np.nan,
        }
    eigenvalues = np.linalg.eigvalsh(hessian)
    norm = float(np.max(np.abs(eigenvalues)))
    return {
        "support_size": int(support.size),
        "lambda_min": float(eigenvalues[0]),
        "hessian_norm": norm,
        "relative_lambda_min": float(eigenvalues[0] / norm) if norm > 0.0 else 0.0,
    }

