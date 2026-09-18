"""Damped IRL1 and IRL2 baselines used in the manuscript."""

from __future__ import annotations

import time

import numpy as np

from ..config import DIRLConfig, SolverConfig
from ..geometry import active_support
from ..model import LpLassoProblem, objective, smooth_gradient
from ..results import RunResult
from .common import finalize_result, gamma_and_gap, residual_values


def _soft_threshold(x: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def run_dirl(
    problem: LpLassoProblem,
    x0: np.ndarray,
    solver_cfg: SolverConfig,
    cfg: DIRLConfig,
    kind: str,
) -> RunResult:
    cfg.validate()
    solver_cfg.validate()
    if kind not in {"DIRL1", "DIRL2"}:
        raise ValueError("kind must be DIRL1 or DIRL2")
    gamma, _ = gamma_and_gap(problem, solver_cfg)
    beta = cfg.beta_factor * problem.L
    x = np.asarray(x0, dtype=float).copy()
    epsilon = np.full(problem.n, cfg.epsilon0, dtype=float)
    history: list[dict[str, float | int]] = []
    diagnostic_time = 0.0
    status = "max_iter"
    iterations = 0
    start = time.perf_counter()
    for k in range(cfg.max_iter):
        gradient = smooth_gradient(problem, x)
        if kind == "DIRL1":
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                weights = problem.p * (np.abs(x) + epsilon) ** (problem.p - 1.0)
            center = x - gradient / beta
            y = _soft_threshold(center, problem.lam * weights / beta)
            x_next = (1.0 - cfg.alpha) * x + cfg.alpha * y
            epsilon_next = (1.0 - cfg.alpha * (1.0 - cfg.mu)) * epsilon
        else:
            radius_sq = x * x + epsilon * epsilon
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                weights = 0.5 * problem.p * radius_sq ** (0.5 * problem.p - 1.0)
            center = x - gradient / beta
            denominator = 1.0 + 2.0 * problem.lam * weights / beta
            y = np.divide(
                center,
                denominator,
                out=np.zeros_like(center),
                where=np.isfinite(denominator),
            )
            x_next = (1.0 - cfg.alpha) * x + cfg.alpha * y
            # This follows displayed Algorithm 2 in arXiv:2401.09274v5.
            epsilon_next = cfg.mu * epsilon
        step = float(np.linalg.norm(x_next - x))
        x, epsilon = x_next, epsilon_next
        iterations = k + 1
        check = (
            iterations % max(1, cfg.diagnostic_stride) == 0
            or step <= cfg.tol_step
            or iterations == cfg.max_iter
        )
        selected = np.nan
        if check:
            diagnostic_start = time.perf_counter()
            selected, _, _ = residual_values(problem, x, gamma, solver_cfg)
            diagnostic_time += time.perf_counter() - diagnostic_start
        if cfg.store_history and (
            iterations % max(1, cfg.history_stride) == 0 or check
        ):
            history.append(
                {
                    "iter": iterations,
                    "objective": objective(problem, x),
                    "algorithm_step": step,
                    "fb_residual": selected,
                    "epsilon_max": float(np.max(epsilon)),
                    "support": int(active_support(x).size),
                }
            )
        if (
            np.isfinite(selected)
            and selected <= cfg.tol_residual
            and float(np.max(epsilon)) <= cfg.epsilon_stop
        ):
            status = "converged"
            break
    runtime = time.perf_counter() - start
    return finalize_result(
        method=kind,
        problem=problem,
        x=x,
        status=status,
        iterations=iterations,
        gamma=gamma,
        cfg=solver_cfg,
        runtime=runtime,
        algorithm_runtime=max(0.0, runtime - diagnostic_time),
        diagnostic_time=diagnostic_time,
        history=history,
    )

