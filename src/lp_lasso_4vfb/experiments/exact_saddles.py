"""Exact strict and quartic-degenerate FB fixed-point tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..algorithms import run_fb, run_four_v_fb
from ..config import legacy_four_v, legacy_solver
from ..model import objective
from ..problems import diagnose_constructed_saddle, make_exact_saddle_problem
from ..reporting import OutputStore, attach_verification_context, summarize_run
from .common import ExperimentProfile


def run_exact_saddles(profile: ExperimentProfile, output: Path) -> dict[str, Any]:
    store = OutputStore(output)
    solver = legacy_solver(max_iter=profile.max_iter)
    four_v = legacy_four_v(max_iter=profile.max_iter)
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    logs: list[pd.DataFrame] = []
    for geometry in ("strict", "quartic"):
        for n in profile.exact_dims:
            for p in profile.exact_p:
                problem = make_exact_saddle_problem(n, p=p, lam=0.1, kind=geometry)
                x_saddle = np.asarray(problem.meta["x_saddle"])
                diagnostic = diagnose_constructed_saddle(problem, solver)
                diagnostics.append(diagnostic)
                run_key = f"exact_{geometry}_n{n}_p{p}"
                for method in ("FB", "4V-FB"):
                    result = (
                        run_fb(problem, x_saddle, solver)
                        if method == "FB"
                        else run_four_v_fb(problem, x_saddle, four_v)
                    )
                    rows.append(
                        summarize_run(
                            problem,
                            result,
                            run_key,
                            initial_objective=objective(problem, x_saddle),
                            experiment="exact_saddle",
                            geometry=geometry,
                            distance_to_saddle=float(np.linalg.norm(result.x - x_saddle)),
                            F_saddle=objective(problem, x_saddle),
                            objective_change_from_saddle=(
                                result.final_objective - objective(problem, x_saddle)
                            ),
                        )
                    )
                    log = attach_verification_context(
                        result, run_key, experiment="exact_saddle", geometry=geometry, n=n, p=p
                    )
                    if not log.empty:
                        logs.append(log)
    frame = pd.DataFrame(rows)
    diagnostic_frame = pd.DataFrame(diagnostics)
    verification = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()
    store.csv("raw", "exact_saddle_runs", frame)
    store.csv("raw", "exact_verification_attempts", verification)
    store.csv("tables", "construction_diagnostics", diagnostic_frame)
    v4 = frame[frame.method == "4V-FB"]
    fb = frame[frame.method == "FB"]
    return {
        "instances": len(v4),
        "fb_nlm": int(fb.numerical_locmin.sum()),
        "four_v_nlm": int(v4.numerical_locmin.sum()),
        "four_v_corrected": int((v4.accepted_corrections == 1).sum()),
        "max_final_residual": float(v4.final_residual.max()),
    }

