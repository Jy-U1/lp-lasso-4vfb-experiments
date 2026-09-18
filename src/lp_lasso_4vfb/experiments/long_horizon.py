"""Long-horizon floating-point FB control."""

from __future__ import annotations

from decimal import Decimal, localcontext
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from ..algorithms.common import fb_candidate, gamma_and_gap, residual_values
from ..config import legacy_solver
from ..geometry import active_hessian_stats, active_support
from ..model import LpLassoProblem, objective
from ..problems import make_exact_saddle_problem, normalized_gaussian_direction
from ..reporting import OutputStore
from .common import ExperimentProfile


def _high_precision_difference(
    problem: LpLassoProblem, x: np.ndarray, reference: np.ndarray, digits: int = 80
) -> float:
    with localcontext() as context:
        context.prec = digits
        p = Decimal(str(problem.p))
        lam = Decimal(str(problem.lam))

        def value(vector: np.ndarray) -> Decimal:
            vector_mp = [Decimal.from_float(float(entry)) for entry in vector]
            residual = [
                sum(
                    (
                        Decimal.from_float(float(problem.A[i, j])) * vector_mp[j]
                        for j in range(problem.n)
                    ),
                    Decimal(0),
                )
                - Decimal.from_float(float(problem.y[i]))
                for i in range(problem.m)
            ]
            smooth = Decimal("0.5") * sum((entry * entry for entry in residual), Decimal(0))
            penalty = lam * sum((abs(entry) ** p for entry in vector_mp), Decimal(0))
            return smooth + penalty

        return float(value(x) - value(reference))


def run_long_horizon(profile: ExperimentProfile, output: Path) -> dict[str, Any]:
    store = OutputStore(output)
    problem = make_exact_saddle_problem(20, p=0.5, lam=0.1, kind="quartic")
    reference = np.asarray(problem.meta["x_saddle"])
    x = reference + 1e-5 * normalized_gaussian_direction(20, 4101)
    cfg = legacy_solver(max_iter=profile.long_horizon)
    gamma, _ = gamma_and_gap(problem, cfg)
    checkpoints = np.unique(
        np.r_[
            np.geomspace(1, profile.long_horizon, 180).astype(int),
            [value for value in (10_000, 100_000, 1_000_000) if value <= profile.long_horizon],
            profile.long_horizon,
        ]
    )
    checkpoint_set = set(map(int, checkpoints))
    rows = []
    first_fixed = None
    start = time.perf_counter()
    reference_objective = objective(problem, reference)
    for iteration in range(1, profile.long_horizon + 1):
        candidate = fb_candidate(problem, x, gamma, cfg)
        step = float(np.linalg.norm(candidate - x))
        x = candidate
        if step == 0.0 and first_fixed is None:
            first_fixed = iteration
        if iteration in checkpoint_set:
            selected, raw, normalized = residual_values(problem, x, gamma, cfg)
            stats = active_hessian_stats(problem, x)
            value = objective(problem, x)
            rows.append(
                {
                    "iteration": iteration,
                    "objective": value,
                    "objective_change_double": value - reference_objective,
                    "objective_change_high_precision": _high_precision_difference(
                        problem, x, reference
                    ),
                    "distance_to_saddle": float(np.linalg.norm(x - reference)),
                    "step_norm": step,
                    "fb_residual": raw,
                    "normalized_residual": normalized,
                    "support_size": int(active_support(x).size),
                    "relative_lambda_min": stats["relative_lambda_min"],
                    "elapsed_seconds": time.perf_counter() - start,
                }
            )
    frame = pd.DataFrame(rows)
    frame["first_machine_fixed_iteration"] = first_fixed
    frame["total_runtime_seconds"] = time.perf_counter() - start
    store.csv("raw", "long_horizon_fb", frame)
    last = frame.iloc[-1]
    return {
        "requested_steps": profile.long_horizon,
        "first_machine_fixed_iteration": int(first_fixed or -1),
        "final_distance": float(last.distance_to_saddle),
        "final_raw_residual": float(last.fb_residual),
        "final_relative_lambda_min": float(last.relative_lambda_min),
        "high_precision_objective_change": float(last.objective_change_high_precision),
    }

