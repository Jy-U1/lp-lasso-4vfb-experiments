"""Shared numerical primitives for FB-family algorithms."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from ..config import SolverConfig
from ..geometry import active_hessian_stats
from ..model import LpLassoProblem, choose_gamma, objective, smooth_gradient
from ..prox import lp_prox_threshold, prox_lp
from ..results import RunResult


def fb_candidate(
    problem: LpLassoProblem, x: np.ndarray, gamma: float, cfg: SolverConfig
) -> np.ndarray:
    return prox_lp(
        x - gamma * smooth_gradient(problem, x),
        gamma * problem.lam,
        problem.p,
        cfg.prox_tol,
        cfg.prox_max_iter,
    )


def residual_values(
    problem: LpLassoProblem,
    x: np.ndarray,
    gamma: float,
    cfg: SolverConfig,
) -> tuple[float, float, float]:
    candidate = fb_candidate(problem, x, gamma, cfg)
    step = float(np.linalg.norm(candidate - x))
    gap, _ = lp_prox_threshold(gamma * problem.lam, problem.p)
    raw = step / gamma
    normalized = step / gap
    selected = raw if cfg.residual_mode == "raw" else normalized
    return selected, raw, normalized


def step_metric(step: float, gamma: float, gap: float, mode: str) -> float:
    return step / gamma if mode == "raw" else step / gap


def finalize_result(
    *,
    method: str,
    problem: LpLassoProblem,
    x: np.ndarray,
    status: str,
    iterations: int,
    gamma: float,
    cfg: SolverConfig,
    runtime: float,
    algorithm_runtime: float,
    diagnostic_time: float = 0.0,
    history: list[dict[str, Any]] | None = None,
    iterates: list[np.ndarray] | None = None,
    **kwargs: Any,
) -> RunResult:
    start = time.perf_counter()
    final_objective = objective(problem, x)
    selected, raw, normalized = residual_values(problem, x, gamma, cfg)
    gap, _ = lp_prox_threshold(gamma * problem.lam, problem.p)
    threshold = 0.5 * gap
    classified = x.copy()
    tail = np.abs(classified) < threshold
    tail_norm = float(np.linalg.norm(classified[tail]))
    classified[tail] = 0.0
    stats = active_hessian_stats(problem, classified)
    elapsed = time.perf_counter() - start
    return RunResult(
        method=method,
        problem_name=problem.name,
        x=x.copy(),
        status=status,
        iterations=iterations,
        gamma=gamma,
        final_objective=final_objective,
        final_residual=selected,
        final_raw_residual=raw,
        final_normalized_residual=normalized,
        final_stats=stats,
        runtime=runtime + elapsed,
        algorithm_runtime=algorithm_runtime,
        diagnostic_time=diagnostic_time + elapsed,
        history=pd.DataFrame(history or []),
        iterates=iterates,
        classification_support_threshold=threshold,
        thresholded_tail_norm=tail_norm,
        raw_nonzero_count=int(np.count_nonzero(x)),
        **kwargs,
    )


def gamma_and_gap(problem: LpLassoProblem, cfg: SolverConfig) -> tuple[float, float]:
    gamma = choose_gamma(problem, cfg.step_fraction, cfg.gamma_override)
    gap, _ = lp_prox_threshold(gamma * problem.lam, problem.p)
    return gamma, gap

