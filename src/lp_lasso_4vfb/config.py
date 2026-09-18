"""Typed solver configurations shared by every experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


ResidualMode = Literal["raw", "normalized"]
GuardMode = Literal["legacy_additive", "absolute_max", "homogeneous_max"]
ScheduleMode = Literal["powers_of_two", "every_eligible"]


@dataclass(frozen=True)
class SolverConfig:
    max_iter: int = 10_000
    stop_tol: float = 1e-11
    step_fraction: float = 0.95
    gamma_override: Optional[float] = None
    prox_tol: float = 1e-14
    prox_max_iter: int = 150
    residual_mode: ResidualMode = "raw"
    store_history: bool = False
    history_stride: int = 1
    store_iterates: bool = False

    def validate(self) -> None:
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive")
        if self.gamma_override is None and not 0.0 < self.step_fraction < 1.0:
            raise ValueError("step_fraction must lie in (0,1)")
        if self.stop_tol <= 0.0:
            raise ValueError("stop_tol must be positive")
        if self.residual_mode not in {"raw", "normalized"}:
            raise ValueError("unknown residual mode")


@dataclass(frozen=True)
class FourVFBConfig:
    solver: SolverConfig = SolverConfig()
    eligibility_tol: float = 1e-7
    schedule: ScheduleMode = "powers_of_two"
    rho: float = 0.5
    eta0_factor: float = 1.0
    sigma_fraction: float = 0.5
    eigvec_zero_tol: float = 1e-15
    guard_mode: GuardMode = "legacy_additive"
    guard_abs: float = 1e-15
    guard_rel: float = 1e-13
    stop_at_origin: bool = True

    def validate(self) -> None:
        self.solver.validate()
        if self.eligibility_tol < self.solver.stop_tol:
            raise ValueError("eligibility_tol must be at least stop_tol")
        if self.schedule not in {"powers_of_two", "every_eligible"}:
            raise ValueError("unknown verification schedule")
        if not 0.0 < self.rho < 1.0:
            raise ValueError("rho must lie in (0,1)")
        if not 0.0 < self.sigma_fraction < 1.0:
            raise ValueError("sigma_fraction must lie in (0,1)")


@dataclass(frozen=True)
class DIRLConfig:
    max_iter: int = 10_000
    alpha: float = 0.2
    beta_factor: float = 1.0
    mu: float = 0.9
    epsilon0: float = 1e-2
    tol_step: float = 1e-13
    tol_residual: float = 1e-11
    epsilon_stop: float = 1e-14
    diagnostic_stride: int = 100
    store_history: bool = False
    history_stride: int = 1

    def validate(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie in (0,1)")
        if not 0.0 < self.mu < 1.0:
            raise ValueError("mu must lie in (0,1)")
        if self.beta_factor <= self.alpha / 2.0:
            raise ValueError("beta_factor must exceed alpha/2")
        if self.epsilon0 <= 0.0:
            raise ValueError("epsilon0 must be positive")


def legacy_solver(max_iter: int = 10_000, **kwargs: object) -> SolverConfig:
    return SolverConfig(max_iter=max_iter, residual_mode="raw", **kwargs)


def legacy_four_v(max_iter: int = 10_000, **kwargs: object) -> FourVFBConfig:
    solver = legacy_solver(max_iter=max_iter)
    return FourVFBConfig(
        solver=solver,
        eligibility_tol=1e-7,
        guard_mode="legacy_additive",
        **kwargs,
    )


def scale_solver(max_iter: int = 10_000, **kwargs: object) -> SolverConfig:
    return SolverConfig(max_iter=max_iter, residual_mode="normalized", **kwargs)


def scale_four_v(max_iter: int = 10_000, **kwargs: object) -> FourVFBConfig:
    solver = scale_solver(max_iter=max_iter)
    return FourVFBConfig(
        solver=solver,
        eligibility_tol=1e-4,
        guard_mode="homogeneous_max",
        **kwargs,
    )

