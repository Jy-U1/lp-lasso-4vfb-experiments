"""Experiment profiles and shared dispatch helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ..algorithms import run_dirl, run_fb, run_four_v_fb
from ..config import DIRLConfig, FourVFBConfig, SolverConfig
from ..model import LpLassoProblem
from ..reporting import OutputStore
from ..results import RunResult


@dataclass(frozen=True)
class ExperimentProfile:
    name: str
    ordinary_dims: tuple[int, ...]
    ordinary_p: tuple[float, ...]
    ordinary_seeds: tuple[int, ...]
    exact_dims: tuple[int, ...]
    exact_p: tuple[float, ...]
    quartic_eps: tuple[float, ...]
    quartic_seeds: tuple[int, ...]
    strict_eps: tuple[float, ...]
    strict_seeds: tuple[int, ...]
    timing_repeats: int
    max_iter: int
    ordinary_max_iter_p08: int
    long_horizon: int
    rescaling_factors: tuple[float, ...]
    correlations: tuple[float, ...]
    data_nu: tuple[float, ...]
    data_seeds: tuple[int, ...]


PAPER_PROFILE = ExperimentProfile(
    name="paper",
    ordinary_dims=(50, 100, 200, 400),
    ordinary_p=(0.3, 0.5, 0.8),
    ordinary_seeds=(1101, 1102, 1103, 1104, 1105),
    exact_dims=(20, 50),
    exact_p=(0.3, 0.5, 0.8),
    quartic_eps=(1e-2, 1e-3, 1e-4, 1e-5),
    quartic_seeds=(2101, 2102, 2103, 2104, 2105),
    strict_eps=(1e-3, 1e-5),
    strict_seeds=(3101, 3102, 3103, 3104, 3105),
    timing_repeats=3,
    max_iter=10_000,
    ordinary_max_iter_p08=50_000,
    long_horizon=1_000_000,
    rescaling_factors=(1e-12, 1e-8, 1e-4, 1.0, 1e4, 1e8, 1e12),
    correlations=(0.8, 0.9, 0.95),
    data_nu=(1e-8, 1e-6, 1e-4),
    data_seeds=tuple(260910 + j for j in range(20)),
)

SMOKE_PROFILE = replace(
    PAPER_PROFILE,
    name="smoke",
    ordinary_dims=(20,),
    ordinary_p=(0.5,),
    ordinary_seeds=(1101,),
    exact_dims=(10,),
    exact_p=(0.5,),
    quartic_eps=(1e-3,),
    quartic_seeds=(2101,),
    strict_eps=(1e-3,),
    strict_seeds=(3101,),
    timing_repeats=1,
    max_iter=3_000,
    ordinary_max_iter_p08=8_000,
    long_horizon=20_000,
    rescaling_factors=(1e-8, 1.0, 1e8),
    correlations=(0.9,),
    data_nu=(1e-6,),
    data_seeds=(260910, 260911),
)


def get_profile(name: str) -> ExperimentProfile:
    if name == "paper":
        return PAPER_PROFILE
    if name == "smoke":
        return SMOKE_PROFILE
    raise ValueError("profile must be 'smoke' or 'paper'")


def make_store(output: Path) -> OutputStore:
    return OutputStore(output)


def run_method(
    method: str,
    problem: LpLassoProblem,
    x0: np.ndarray,
    solver: SolverConfig,
    four_v: FourVFBConfig,
    dirl: DIRLConfig,
) -> RunResult:
    if method == "FB":
        return run_fb(problem, x0, solver)
    if method == "4V-FB":
        return run_four_v_fb(problem, x0, four_v)
    if method in {"DIRL1", "DIRL2"}:
        return run_dirl(problem, x0, solver, dirl, method)
    raise ValueError(f"unknown method: {method}")

