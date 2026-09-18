"""Deterministic problem generators used by the paper experiments."""

from __future__ import annotations

import numpy as np

from .config import SolverConfig
from .geometry import active_hessian, smallest_eigenpair
from .model import LpLassoProblem, choose_gamma, objective, smooth_gradient
from .prox import prox_lp


def make_random_gaussian_problem(
    n: int,
    m: int | None = None,
    *,
    p: float = 0.5,
    lam: float = 0.05,
    seed: int = 0,
) -> LpLassoProblem:
    rng = np.random.default_rng(seed)
    m = n if m is None else int(m)
    A = rng.normal(size=(m, n)) / np.sqrt(m)
    y = rng.normal(size=m)
    return LpLassoProblem(
        A,
        y,
        p,
        lam,
        name=f"random_gaussian_n{n}_m{m}_p{p}_seed{seed}",
        meta={"regime": "ordinary", "seed": seed},
    )


def make_exact_saddle_problem(
    n: int,
    *,
    p: float = 0.5,
    lam: float = 0.1,
    kind: str = "quartic",
    amplitude: float = 1.0,
) -> LpLassoProblem:
    if n < 2 or kind not in {"strict", "quartic"}:
        raise ValueError("need n>=2 and kind strict/quartic")
    a = float(amplitude)
    d = lam * p * (1.0 - p) * a ** (p - 2.0)
    beta = 4.0 / p
    Q = np.eye(n)
    v = np.zeros(n)
    v[0], v[1] = 1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0)
    w = np.zeros(n)
    w[0], w[1] = 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)
    Q[:, 0], Q[:, 1] = v, w
    target_hessian = np.full(n, beta * d)
    target_hessian[0] = 0.0 if kind == "quartic" else -0.5 * d
    gram_eigenvalues = d + target_hessian
    if np.min(gram_eigenvalues) <= 0.0:
        raise RuntimeError("nonpositive Gram eigenvalue")
    A = np.diag(np.sqrt(gram_eigenvalues)) @ Q.T
    x_saddle = a * np.ones(n)
    penalty_gradient = lam * p * np.sign(x_saddle) * np.abs(x_saddle) ** (p - 1.0)
    y = A @ x_saddle + np.linalg.solve(A.T, penalty_gradient)
    return LpLassoProblem(
        A,
        y,
        p,
        lam,
        name=f"exact_{kind}_n{n}_p{p}",
        meta={
            "regime": f"exact_{kind}",
            "geometry": kind,
            "x_saddle": x_saddle,
            "escape_direction": v,
            "target_hessian_eigenvalues": target_hessian,
        },
    )


def make_correlated_problem(
    *,
    c: float,
    p: float,
    lam: float = 0.1,
    b_ratio: float = 1.0,
) -> LpLassoProblem:
    if not 0.0 < c < 1.0:
        raise ValueError("c must lie in (0,1)")
    # Use the symmetric Gram factor from the validation script.  Although any
    # left-orthogonal rotation gives the same exact model, the factor itself is
    # part of the seeded finite-precision data-perturbation experiment.
    common = np.sqrt((1.0 + c) / 2.0)
    contrast = np.sqrt((1.0 - c) / 2.0)
    a1 = np.array([common, contrast])
    a2 = np.array([common, -contrast])
    A = np.column_stack([a1, a2])
    t_c = (lam * p * (1.0 - p) / (1.0 - c)) ** (1.0 / (2.0 - p))
    b_c = (1.0 + c) * t_c + lam * p * t_c ** (p - 1.0)
    b = b_ratio * b_c
    y = b * (a1 + a2) / (1.0 + c)
    return LpLassoProblem(
        A,
        y,
        p,
        lam,
        name=f"correlated_c{c}_p{p}_bratio{b_ratio}",
        meta={
            "regime": "correlated",
            "c": c,
            "t_c": t_c,
            "b_c": b_c,
            "b": b,
            "b_ratio": b_ratio,
            "x_transition": t_c * np.ones(2),
            "escape_direction": np.array([1.0, -1.0]) / np.sqrt(2.0),
        },
    )


def perturb_data(
    base: LpLassoProblem,
    *,
    nu: float,
    seed: int,
) -> LpLassoProblem:
    rng = np.random.default_rng(seed)
    E = rng.normal(size=base.A.shape)
    e = rng.normal(size=base.y.shape)
    A = base.A + nu * np.linalg.norm(base.A, "fro") * E / np.linalg.norm(E, "fro")
    y = base.y + nu * np.linalg.norm(base.y) * e / np.linalg.norm(e)
    return LpLassoProblem(
        A,
        y,
        base.p,
        base.lam,
        name=f"{base.name}_nu{nu:g}_seed{seed}",
        meta={**base.meta, "nu": nu, "perturbation_seed": seed},
    )


def normalized_gaussian_direction(n: int, seed: int) -> np.ndarray:
    direction = np.random.default_rng(seed).normal(size=n)
    return direction / np.linalg.norm(direction)


def diagnose_constructed_saddle(
    problem: LpLassoProblem, cfg: SolverConfig
) -> dict[str, float | int | str]:
    x_saddle = np.asarray(problem.meta["x_saddle"])
    direction = np.asarray(problem.meta["escape_direction"])
    gamma = choose_gamma(problem, cfg.step_fraction, cfg.gamma_override)
    candidate = prox_lp(
        x_saddle - gamma * smooth_gradient(problem, x_saddle),
        gamma * problem.lam,
        problem.p,
        cfg.prox_tol,
        cfg.prox_max_iter,
    )
    stationarity = smooth_gradient(problem, x_saddle) + problem.lam * problem.p * (
        np.sign(x_saddle) * np.abs(x_saddle) ** (problem.p - 1.0)
    )
    _, hessian = active_hessian(problem, x_saddle)
    if hessian is None:
        raise RuntimeError("constructed point must have full support")
    eigenvalues = np.linalg.eigvalsh(hessian)
    d3_coefficient = problem.lam * problem.p * (problem.p - 1.0) * (problem.p - 2.0)
    d4_coefficient = d3_coefficient * (problem.p - 3.0)
    d3 = d3_coefficient * np.sum(
        np.sign(x_saddle) * np.abs(x_saddle) ** (problem.p - 3.0) * direction**3
    )
    d4 = d4_coefficient * np.sum(
        np.abs(x_saddle) ** (problem.p - 4.0) * direction**4
    )
    return {
        "problem": problem.name,
        "geometry": str(problem.meta["geometry"]),
        "n": problem.n,
        "p": problem.p,
        "stationarity_norm": float(np.linalg.norm(stationarity)),
        "fb_fixed_residual": float(np.linalg.norm(candidate - x_saddle)),
        "lambda_min": float(eigenvalues[0]),
        "lambda_second": float(eigenvalues[1]),
        "D3_along_escape": float(d3),
        "D4_along_escape": float(d4),
        "F_saddle": objective(problem, x_saddle),
        "gamma": gamma,
    }
