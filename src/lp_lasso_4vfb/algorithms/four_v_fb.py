"""Fourth-order verified forward--backward splitting (4V-FB)."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import pandas as pd

from ..config import FourVFBConfig
from ..geometry import active_hessian, active_support, smallest_eigenpair
from ..model import LpLassoProblem, objective
from ..results import RunResult
from .common import fb_candidate, finalize_result, gamma_and_gap, step_metric


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _verified_constants(
    problem: LpLassoProblem,
    x0: np.ndarray,
    gap: float,
    cfg: FourVFBConfig,
) -> dict[str, float]:
    eta0 = cfg.eta0_factor * gap
    initial_objective = objective(problem, x0)
    m0 = (initial_objective / problem.lam) ** (1.0 / problem.p)
    coefficient = problem.p * (1.0 - problem.p) * (2.0 - problem.p) * (3.0 - problem.p)
    kappa0 = problem.lam * coefficient / (
        problem.n * (m0 + eta0) ** (4.0 - problem.p)
    )
    sigma = cfg.sigma_fraction * kappa0 / 24.0
    if not np.isfinite(kappa0) or kappa0 <= 0.0:
        raise FloatingPointError("kappa0 underflowed; rescale the problem")
    return {"eta0": eta0, "M0": m0, "kappa0": kappa0, "sigma": sigma}


def _acceptance_threshold(
    *,
    required: float,
    fz: float,
    fplus: float,
    fminus: float,
    cfg: FourVFBConfig,
) -> tuple[float, float]:
    absolute = max(cfg.guard_abs, cfg.guard_rel * max(1.0, abs(fz)))
    if cfg.guard_mode == "legacy_additive":
        return required + absolute, absolute
    if cfg.guard_mode == "absolute_max":
        return max(required, absolute), absolute
    scale = fz + max(fplus, fminus)
    relative = cfg.guard_rel * scale
    return max(required, relative), relative


def run_four_v_fb(
    problem: LpLassoProblem, x0: np.ndarray, cfg: FourVFBConfig
) -> RunResult:
    cfg.validate()
    solver = cfg.solver
    gamma, gap = gamma_and_gap(problem, solver)
    constants = _verified_constants(problem, np.asarray(x0, dtype=float), gap, cfg)
    x = np.asarray(x0, dtype=float).copy()
    history: list[dict[str, Any]] = []
    verification_rows: list[dict[str, Any]] = []
    iterates = [x.copy()] if solver.store_iterates else None
    verification_calls = scheduled_calls = forced_calls = corrections = 0
    eligible_total = eligible_since_correction = 0
    eig_time = probe_time = verification_time = fb_core_time = 0.0
    status = "max_iter"
    iterations = 0
    start = time.perf_counter()

    for k in range(solver.max_iter):
        core_start = time.perf_counter()
        z = fb_candidate(problem, x, gamma, solver)
        fb_step = float(np.linalg.norm(z - x))
        residual = step_metric(fb_step, gamma, gap, solver.residual_mode)
        fb_core_time += time.perf_counter() - core_start
        support = active_support(z)
        iterations = k + 1

        if support.size == 0 and cfg.stop_at_origin:
            x = z
            if iterates is not None:
                iterates.append(x.copy())
            status = "origin_local_min"
            break

        eligible = residual <= cfg.eligibility_tol
        if eligible:
            eligible_total += 1
            eligible_since_correction += 1
        scheduled = eligible and (
            cfg.schedule == "every_eligible" or _is_power_of_two(eligible_since_correction)
        )
        forced_final = residual <= solver.stop_tol and not scheduled
        verify_now = bool(scheduled or forced_final)
        corrected = False
        eta = eigenvalue = actual_drop = required = guard = threshold = math.nan
        chosen_sign = 0
        f_next = math.nan
        x_next = z

        if verify_now:
            verification_calls += 1
            scheduled_calls += int(scheduled)
            forced_calls += int(forced_final)
            verify_start = time.perf_counter()
            fz = objective(problem, z)
            f_next = fz
            eig_start = time.perf_counter()
            u = z[support]
            _, hessian = active_hessian(problem, z)
            if hessian is None:
                raise RuntimeError("verification requires a nonempty support")
            eigenvalue, eigenvector = smallest_eigenpair(hessian)
            eig_elapsed = time.perf_counter() - eig_start
            eig_time += eig_elapsed
            nonzero = np.abs(eigenvector) > cfg.eigvec_zero_tol
            if not np.any(nonzero):
                raise RuntimeError("numerical eigenvector vanished")
            sign_safe = cfg.rho * float(
                np.min(np.abs(u[nonzero]) / np.abs(eigenvector[nonzero]))
            )
            eta = min(constants["eta0"], sign_safe)
            plus, minus = z.copy(), z.copy()
            plus[support] = u + eta * eigenvector
            minus[support] = u - eta * eigenvector
            probe_start = time.perf_counter()
            fplus, fminus = objective(problem, plus), objective(problem, minus)
            probe_elapsed = time.perf_counter() - probe_start
            probe_time += probe_elapsed
            if fplus <= fminus:
                best, fbest, chosen_sign = plus, fplus, 1
            else:
                best, fbest, chosen_sign = minus, fminus, -1
            actual_drop = fz - fbest
            required = constants["sigma"] * eta**4
            threshold, guard = _acceptance_threshold(
                required=required,
                fz=fz,
                fplus=fplus,
                fminus=fminus,
                cfg=cfg,
            )
            corrected = bool(actual_drop >= threshold)
            if corrected:
                x_next, f_next = best, fbest
                corrections += 1
            verification_elapsed = time.perf_counter() - verify_start
            verification_time += verification_elapsed
            verification_rows.append(
                {
                    "iter": iterations,
                    "eligible_q": eligible_since_correction,
                    "scheduled": scheduled,
                    "forced_final": forced_final,
                    "support_size": int(support.size),
                    "lambda_min": eigenvalue,
                    "eta": eta,
                    "chosen_sign": chosen_sign,
                    "actual_drop": actual_drop,
                    "required_drop": required,
                    "guard": guard,
                    "acceptance_threshold": threshold,
                    "guard_mode": cfg.guard_mode,
                    "accepted": corrected,
                    "eig_time": eig_elapsed,
                    "probe_time": probe_elapsed,
                    "verification_time": verification_elapsed,
                }
            )
            if corrected:
                eligible_since_correction = 0

        actual_step = float(np.linalg.norm(x_next - x))
        x = x_next
        if iterates is not None:
            iterates.append(x.copy())
        terminal = residual <= solver.stop_tol and not corrected
        if solver.store_history and (
            iterations % max(1, solver.history_stride) == 0 or verify_now or terminal
        ):
            if not np.isfinite(f_next):
                f_next = objective(problem, x)
            history.append(
                {
                    "iter": iterations,
                    "objective": f_next,
                    "residual": residual,
                    "actual_step": actual_step,
                    "support": int(active_support(x).size),
                    "verified": verify_now,
                    "scheduled": scheduled,
                    "forced_final": forced_final,
                    "corrected": corrected,
                    "eligible_q": eligible_since_correction,
                    "eta": eta,
                    "lambda_min": eigenvalue,
                    "actual_drop": actual_drop,
                    "required_drop": required,
                    "guard": guard,
                }
            )
        if terminal:
            status = "converged"
            break

    runtime = time.perf_counter() - start
    return finalize_result(
        method="4V-FB",
        problem=problem,
        x=x,
        status=status,
        iterations=iterations,
        gamma=gamma,
        cfg=solver,
        runtime=runtime,
        algorithm_runtime=runtime,
        history=history,
        iterates=iterates,
        verification_log=pd.DataFrame(verification_rows),
        verification_calls=verification_calls,
        scheduled_verifications=scheduled_calls,
        forced_final_verifications=forced_calls,
        accepted_corrections=corrections,
        eligible_iterations=eligible_total,
        eig_time=eig_time,
        probe_time=probe_time,
        verification_time=verification_time,
        fb_core_time=fb_core_time,
        prox_gap=gap,
        eta0=constants["eta0"],
        kappa0=constants["kappa0"],
        sigma=constants["sigma"],
        metadata={"guard_mode": cfg.guard_mode, "residual_mode": solver.residual_mode},
    )

