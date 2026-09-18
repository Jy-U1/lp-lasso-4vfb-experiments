"""Common result container and finite-tolerance endpoint classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class RunResult:
    method: str
    problem_name: str
    x: np.ndarray
    status: str
    iterations: int
    gamma: float
    final_objective: float
    final_residual: float
    final_raw_residual: float
    final_normalized_residual: float
    final_stats: dict[str, float | int]
    runtime: float
    algorithm_runtime: float
    diagnostic_time: float = 0.0
    history: pd.DataFrame = field(default_factory=pd.DataFrame)
    iterates: list[np.ndarray] | None = None
    verification_log: pd.DataFrame = field(default_factory=pd.DataFrame)
    verification_calls: int = 0
    scheduled_verifications: int = 0
    forced_final_verifications: int = 0
    accepted_corrections: int = 0
    eligible_iterations: int = 0
    eig_time: float = 0.0
    probe_time: float = 0.0
    verification_time: float = 0.0
    fb_core_time: float = 0.0
    prox_gap: float = np.nan
    eta0: float = np.nan
    kappa0: float = np.nan
    sigma: float = np.nan
    classification_support_threshold: float = np.nan
    thresholded_tail_norm: float = np.nan
    raw_nonzero_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def passes_nlm(self, residual_tol: float = 1e-11, eig_tol: float = 1e-8) -> bool:
        if self.final_residual > residual_tol:
            return False
        support = int(self.final_stats["support_size"])
        if support == 0:
            return bool(np.array_equal(self.x, np.zeros_like(self.x)))
        return bool(float(self.final_stats["relative_lambda_min"]) > eig_tol)

