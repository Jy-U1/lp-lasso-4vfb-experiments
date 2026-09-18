"""Ordinary Gaussian instances and sparse-schedule cost comparison."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..algorithms import run_fb, run_four_v_fb
from ..config import legacy_four_v, legacy_solver
from ..model import objective
from ..problems import make_random_gaussian_problem
from ..reporting import OutputStore, attach_verification_context, summarize_run
from .common import ExperimentProfile


def _timed_repeats(problem, x0, solver, four_v, repeats: int, seed: int):
    by_method = {"FB": [], "4V-FB": []}
    for repetition in range(repeats):
        order = ("FB", "4V-FB") if (seed + repetition) % 2 else ("4V-FB", "FB")
        for method in order:
            result = (
                run_fb(problem, x0, solver)
                if method == "FB"
                else run_four_v_fb(problem, x0, four_v)
            )
            by_method[method].append(result)
    return by_method


def run_ordinary(profile: ExperimentProfile, output: Path) -> dict[str, Any]:
    store = OutputStore(output)
    rows: list[dict[str, Any]] = []
    logs: list[pd.DataFrame] = []
    # Unreported warm-up avoids assigning one-time SciPy/BLAS setup to a method.
    warm = make_random_gaussian_problem(8, 6, p=0.5, seed=999)
    warm_solver = legacy_solver(max_iter=2)
    warm_four_v = legacy_four_v(max_iter=2)
    run_fb(warm, np.zeros(8), warm_solver)
    run_four_v_fb(warm, np.zeros(8), warm_four_v)
    for n in profile.ordinary_dims:
        m = int(round(0.8 * n))
        for p in profile.ordinary_p:
            cap = profile.ordinary_max_iter_p08 if np.isclose(p, 0.8) else profile.max_iter
            solver = legacy_solver(max_iter=cap)
            four_v = legacy_four_v(max_iter=cap)
            for seed in profile.ordinary_seeds:
                problem = make_random_gaussian_problem(n, m, p=p, lam=0.05, seed=seed)
                x0 = 0.2 * np.random.default_rng(100_000 + seed).normal(size=n)
                run_key = f"ordinary_n{n}_p{p}_seed{seed}"
                repeated = _timed_repeats(
                    problem, x0, solver, four_v, profile.timing_repeats, seed
                )
                for method in ("FB", "4V-FB"):
                    results = repeated[method]
                    representative = results[0]
                    if any(
                        not np.allclose(representative.x, item.x, rtol=0.0, atol=1e-12)
                        for item in results[1:]
                    ):
                        raise AssertionError(f"nondeterministic terminal point: {run_key}")
                    row = summarize_run(
                        problem,
                        representative,
                        run_key,
                        initial_objective=objective(problem, x0),
                        experiment="ordinary",
                        seed=seed,
                        iteration_cap=cap,
                    )
                    for field, attribute in (
                        ("runtime_total", "runtime"),
                        ("runtime_algorithm", "algorithm_runtime"),
                        ("diagnostic_time", "diagnostic_time"),
                        ("eig_time", "eig_time"),
                        ("probe_time", "probe_time"),
                        ("verification_time", "verification_time"),
                    ):
                        row[field] = float(np.median([getattr(item, attribute) for item in results]))
                    row["verification_time_fraction"] = float(
                        np.median(
                            [
                                item.verification_time / item.runtime
                                if item.runtime > 0.0
                                else np.nan
                                for item in results
                            ]
                        )
                    )
                    row["timing_repeats"] = len(results)
                    runtimes = [item.algorithm_runtime for item in results]
                    row["runtime_algorithm_min"] = float(np.min(runtimes))
                    row["runtime_algorithm_max"] = float(np.max(runtimes))
                    row["runtime_algorithm_q25"] = float(np.quantile(runtimes, 0.25))
                    row["runtime_algorithm_q75"] = float(np.quantile(runtimes, 0.75))
                    rows.append(row)
                    log = attach_verification_context(
                        representative,
                        run_key,
                        experiment="ordinary",
                        n=n,
                        p=p,
                        seed=seed,
                    )
                    if not log.empty:
                        logs.append(log)
    frame = pd.DataFrame(rows)
    verification = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()
    store.csv("raw", "ordinary_runs", frame)
    store.csv("raw", "ordinary_verification_attempts", verification)

    n = 100 if profile.name == "paper" else profile.ordinary_dims[0]
    m = int(round(0.8 * n))
    seed = profile.ordinary_seeds[0]
    problem = make_random_gaussian_problem(n, m, p=0.5, lam=0.05, seed=seed)
    x0 = 0.2 * np.random.default_rng(100_000 + seed).normal(size=n)
    ablation_rows = []
    ablation_logs: list[pd.DataFrame] = []
    for schedule in ("powers_of_two", "every_eligible"):
        config = legacy_four_v(max_iter=profile.max_iter, schedule=schedule)
        results = [
            run_four_v_fb(problem, x0, config) for _ in range(profile.timing_repeats)
        ]
        row = summarize_run(
            problem,
            results[0],
            f"schedule_ablation_{schedule}",
            initial_objective=objective(problem, x0),
            experiment="schedule_ablation",
            schedule=schedule,
        )
        row["runtime_algorithm"] = float(np.median([item.algorithm_runtime for item in results]))
        ablation_rows.append(row)
        log = attach_verification_context(
            results[0],
            row["run_key"],
            experiment="schedule_ablation",
            schedule=schedule,
            n=n,
            p=0.5,
            seed=seed,
        )
        if not log.empty:
            ablation_logs.append(log)
    ablation = pd.DataFrame(ablation_rows)
    store.csv("raw", "schedule_ablation_runs", ablation)
    store.csv(
        "raw",
        "schedule_ablation_verification_attempts",
        pd.concat(ablation_logs, ignore_index=True) if ablation_logs else pd.DataFrame(),
    )
    return {
        "ordinary_rows": len(frame),
        "ordinary_pairs": int((frame.method == "4V-FB").sum()),
        "ordinary_nlm": int(frame[frame.method == "4V-FB"].numerical_locmin.sum()),
        "ordinary_corrections": int(frame[frame.method == "4V-FB"].accepted_corrections.sum()),
        "ablation_calls": {
            row.schedule: int(row.verification_calls) for row in ablation.itertuples()
        },
    }
