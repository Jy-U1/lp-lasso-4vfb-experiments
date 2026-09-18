"""Ordinary forward--backward splitting."""

from __future__ import annotations

import time

import numpy as np

from ..config import SolverConfig
from ..geometry import active_support
from ..model import LpLassoProblem, objective
from ..results import RunResult
from .common import fb_candidate, finalize_result, gamma_and_gap, step_metric


def run_fb(problem: LpLassoProblem, x0: np.ndarray, cfg: SolverConfig) -> RunResult:
    cfg.validate()
    gamma, gap = gamma_and_gap(problem, cfg)
    x = np.asarray(x0, dtype=float).copy()
    history: list[dict[str, float | int]] = []
    iterates = [x.copy()] if cfg.store_iterates else None
    status = "max_iter"
    iterations = 0
    core_time = 0.0
    start = time.perf_counter()
    for k in range(cfg.max_iter):
        core_start = time.perf_counter()
        candidate = fb_candidate(problem, x, gamma, cfg)
        step = float(np.linalg.norm(candidate - x))
        residual = step_metric(step, gamma, gap, cfg.residual_mode)
        core_time += time.perf_counter() - core_start
        x = candidate
        iterations = k + 1
        if iterates is not None:
            iterates.append(x.copy())
        terminal = residual <= cfg.stop_tol
        if cfg.store_history and (
            iterations % max(1, cfg.history_stride) == 0 or terminal
        ):
            history.append(
                {
                    "iter": iterations,
                    "objective": objective(problem, x),
                    "residual": residual,
                    "actual_step": step,
                    "support": int(active_support(x).size),
                }
            )
        if terminal:
            status = "converged"
            break
    runtime = time.perf_counter() - start
    return finalize_result(
        method="FB",
        problem=problem,
        x=x,
        status=status,
        iterations=iterations,
        gamma=gamma,
        cfg=cfg,
        runtime=runtime,
        algorithm_runtime=runtime,
        history=history,
        iterates=iterates,
        fb_core_time=core_time,
    )

