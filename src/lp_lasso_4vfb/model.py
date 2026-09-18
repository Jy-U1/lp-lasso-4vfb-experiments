"""The nonconvex lp-Lasso model and objective-scale transformations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class LpLassoProblem:
    A: np.ndarray
    y: np.ndarray
    p: float
    lam: float
    name: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.A = np.asarray(self.A, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        if self.A.ndim != 2 or self.y.shape != (self.A.shape[0],):
            raise ValueError("incompatible A and y")
        if not 0.0 < self.p < 1.0 or self.lam <= 0.0:
            raise ValueError("need 0<p<1 and lambda>0")
        self._L = float(np.linalg.norm(self.A, 2) ** 2)

    @property
    def m(self) -> int:
        return int(self.A.shape[0])

    @property
    def n(self) -> int:
        return int(self.A.shape[1])

    @property
    def L(self) -> float:
        return self._L


def objective(problem: LpLassoProblem, x: np.ndarray) -> float:
    residual = problem.A @ x - problem.y
    return 0.5 * float(residual @ residual) + problem.lam * float(
        np.sum(np.abs(x) ** problem.p)
    )


def smooth_gradient(problem: LpLassoProblem, x: np.ndarray) -> np.ndarray:
    return problem.A.T @ (problem.A @ x - problem.y)


def choose_gamma(problem: LpLassoProblem, step_fraction: float, override: float | None) -> float:
    gamma = float(override) if override is not None else step_fraction / problem.L
    if problem.L > 0.0 and not gamma * problem.L < 1.0:
        raise ValueError("need gamma*||A||_2^2 < 1")
    return gamma


def rescale_objective(problem: LpLassoProblem, scale: float) -> LpLassoProblem:
    """Return data representing F_scale(x)=scale*F(x)."""
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    return LpLassoProblem(
        np.sqrt(scale) * problem.A,
        np.sqrt(scale) * problem.y,
        problem.p,
        scale * problem.lam,
        name=f"{problem.name}_scale{scale:g}",
        meta={**problem.meta, "objective_scale": float(scale)},
    )

