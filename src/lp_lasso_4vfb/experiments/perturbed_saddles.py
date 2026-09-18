"""Perturbed strict/quartic saddles with FB, DIRL1/2, and 4V-FB."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..config import DIRLConfig, legacy_four_v, legacy_solver
from ..model import objective
from ..problems import make_exact_saddle_problem, normalized_gaussian_direction
from ..reporting import OutputStore, attach_verification_context, summarize_run
from .common import ExperimentProfile, run_method


def _grid(
    *,
    geometry: str,
    dimensions: Iterable[int],
    p_values: Iterable[float],
    epsilons: Iterable[float],
    seeds: Iterable[int],
    profile: ExperimentProfile,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    solver = legacy_solver(max_iter=profile.max_iter)
    four_v = legacy_four_v(max_iter=profile.max_iter)
    dirl = DIRLConfig(max_iter=profile.max_iter)
    rows: list[dict[str, Any]] = []
    logs: list[pd.DataFrame] = []
    for n in dimensions:
        for p in p_values:
            problem = make_exact_saddle_problem(n, p=p, lam=0.1, kind=geometry)
            x_saddle = np.asarray(problem.meta["x_saddle"])
            f_saddle = objective(problem, x_saddle)
            for epsilon in epsilons:
                for seed in seeds:
                    direction = normalized_gaussian_direction(n, seed)
                    x0 = x_saddle + epsilon * direction
                    run_key = f"perturbed_{geometry}_n{n}_p{p}_eps{epsilon:.0e}_seed{seed}"
                    for method in ("FB", "DIRL1", "DIRL2", "4V-FB"):
                        result = run_method(method, problem, x0, solver, four_v, dirl)
                        distance = float(np.linalg.norm(result.x - x_saddle))
                        rows.append(
                            summarize_run(
                                problem,
                                result,
                                run_key,
                                initial_objective=objective(problem, x0),
                                experiment="perturbed_saddle",
                                geometry=geometry,
                                perturbation=epsilon,
                                seed=seed,
                                distance_to_saddle=distance,
                                escaped_radius_0p1=bool(distance > 0.1),
                                F_saddle=f_saddle,
                                objective_change_from_saddle=result.final_objective - f_saddle,
                            )
                        )
                        log = attach_verification_context(
                            result,
                            run_key,
                            experiment="perturbed_saddle",
                            geometry=geometry,
                            n=n,
                            p=p,
                            perturbation=epsilon,
                            seed=seed,
                        )
                        if not log.empty:
                            logs.append(log)
    return pd.DataFrame(rows), pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()


def run_perturbed_saddles(profile: ExperimentProfile, output: Path) -> dict[str, Any]:
    store = OutputStore(output)
    strict, strict_logs = _grid(
        geometry="strict",
        dimensions=(20,) if profile.name == "paper" else profile.exact_dims,
        p_values=(0.5,),
        epsilons=profile.strict_eps,
        seeds=profile.strict_seeds,
        profile=profile,
    )
    quartic, quartic_logs = _grid(
        geometry="quartic",
        dimensions=profile.exact_dims,
        p_values=profile.exact_p,
        epsilons=profile.quartic_eps,
        seeds=profile.quartic_seeds,
        profile=profile,
    )
    store.csv("raw", "perturbed_strict_runs", strict)
    store.csv("raw", "perturbed_quartic_runs", quartic)
    store.csv(
        "raw",
        "perturbed_verification_attempts",
        pd.concat([strict_logs, quartic_logs], ignore_index=True),
    )
    summary: dict[str, Any] = {"strict": {}, "quartic": {}}
    for label, frame in (("strict", strict), ("quartic", quartic)):
        for method, group in frame.groupby("method"):
            summary[label][method] = {
                "runs": len(group),
                "escaped": int(group.escaped_radius_0p1.sum()),
                "nlm": int(group.numerical_locmin.sum()),
                "corrected": int((group.accepted_corrections > 0).sum()),
                "capped": int((group.status == "max_iter").sum()),
            }
    return summary

