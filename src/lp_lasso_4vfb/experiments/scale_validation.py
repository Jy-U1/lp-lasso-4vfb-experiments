"""Objective-rescaling, correlated-response, and data-perturbation tests."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..algorithms import run_fb, run_four_v_fb
from ..config import FourVFBConfig, scale_four_v, scale_solver
from ..model import objective, rescale_objective
from ..problems import make_correlated_problem, make_exact_saddle_problem, perturb_data
from ..reporting import OutputStore, attach_verification_context, summarize_run
from .common import ExperimentProfile


def _collect(
    rows: list[dict[str, Any]],
    logs: list[pd.DataFrame],
    problem,
    result,
    run_key: str,
    x0: np.ndarray,
    **context: Any,
) -> None:
    rows.append(
        summarize_run(
            problem,
            result,
            run_key,
            initial_objective=objective(problem, x0),
            **context,
        )
    )
    log = attach_verification_context(result, run_key, **context)
    if not log.empty:
        logs.append(log)


def run_objective_rescaling(
    profile: ExperimentProfile, store: OutputStore
) -> tuple[pd.DataFrame, pd.DataFrame]:
    solver = scale_solver(max_iter=profile.max_iter)
    homogeneous = scale_four_v(max_iter=profile.max_iter)
    absolute = replace(homogeneous, guard_mode="absolute_max")
    rows: list[dict[str, Any]] = []
    logs: list[pd.DataFrame] = []
    for n in profile.exact_dims:
        for p in profile.exact_p:
            base = make_exact_saddle_problem(n, p=p, lam=0.1, kind="quartic")
            x0 = np.asarray(base.meta["x_saddle"])
            for scale in profile.rescaling_factors:
                problem = rescale_objective(base, scale)
                for label, config in (
                    ("homogeneous", homogeneous),
                    ("absolute", absolute),
                ):
                    result = run_four_v_fb(problem, x0, config)
                    _collect(
                        rows,
                        logs,
                        problem,
                        result,
                        f"rescale_n{n}_p{p}_a{scale:g}_{label}",
                        x0,
                        experiment="objective_rescaling",
                        guard_variant=label,
                        objective_scale=scale,
                        normalized_final_objective=result.final_objective / scale,
                    )
    frame = pd.DataFrame(rows)
    verification = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()
    store.csv("raw", "objective_rescaling_runs", frame)
    store.csv("raw", "objective_rescaling_verifications", verification)
    return frame, verification


def run_correlated_zero_initialization(
    profile: ExperimentProfile, store: OutputStore
) -> tuple[pd.DataFrame, pd.DataFrame]:
    solver = scale_solver(max_iter=profile.max_iter)
    four_v = scale_four_v(max_iter=profile.max_iter)
    rows: list[dict[str, Any]] = []
    logs: list[pd.DataFrame] = []
    ratios = (0.98, 0.995, 1.0, 1.005, 1.02)
    p_values = (0.3, 0.5, 0.8) if profile.name == "paper" else (0.5,)
    for c in profile.correlations:
        for p in p_values:
            for ratio in ratios:
                problem = make_correlated_problem(c=c, p=p, lam=0.1, b_ratio=ratio)
                x0 = np.zeros(problem.n)
                for method in ("FB", "4V-FB"):
                    result = (
                        run_fb(problem, x0, solver)
                        if method == "FB"
                        else run_four_v_fb(problem, x0, four_v)
                    )
                    _collect(
                        rows,
                        logs,
                        problem,
                        result,
                        f"correlated_c{c}_p{p}_r{ratio}_{method}",
                        x0,
                        experiment="correlated_zero_init",
                        c=c,
                        b_ratio=ratio,
                        side=("below" if ratio < 1.0 else "transition" if ratio == 1.0 else "above"),
                    )
    frame = pd.DataFrame(rows)
    verification = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()
    store.csv("raw", "correlated_zero_init_runs", frame)
    store.csv("raw", "correlated_zero_init_verifications", verification)
    return frame, verification


def run_data_perturbations(
    profile: ExperimentProfile, store: OutputStore
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    solver = scale_solver(max_iter=profile.max_iter)
    four_v = scale_four_v(max_iter=profile.max_iter)
    pilot_four_v = replace(four_v, eligibility_tol=1e-7)
    rows: list[dict[str, Any]] = []
    logs: list[pd.DataFrame] = []
    pilot_rows: list[dict[str, Any]] = []
    for c in profile.correlations:
        base = make_correlated_problem(c=c, p=0.5, lam=0.1, b_ratio=1.0)
        for nu in profile.data_nu:
            for seed in profile.data_seeds:
                problem = perturb_data(base, nu=nu, seed=seed)
                x0 = np.zeros(problem.n)
                for method in ("FB", "4V-FB"):
                    result = (
                        run_fb(problem, x0, solver)
                        if method == "FB"
                        else run_four_v_fb(problem, x0, four_v)
                    )
                    _collect(
                        rows,
                        logs,
                        problem,
                        result,
                        f"data_c{c}_nu{nu:g}_seed{seed}_{method}",
                        x0,
                        experiment="data_perturbation",
                        c=c,
                        nu=nu,
                        seed=seed,
                    )
                pilot_seeds = profile.data_seeds[:5] if profile.name == "paper" else profile.data_seeds[:1]
                if nu == 1e-6 and seed in pilot_seeds:
                    pilot = run_four_v_fb(problem, x0, pilot_four_v)
                    pilot_rows.append(
                        summarize_run(
                            problem,
                            pilot,
                            f"pilot_c{c}_nu{nu:g}_seed{seed}",
                            initial_objective=objective(problem, x0),
                            experiment="eligibility_pilot",
                            c=c,
                            nu=nu,
                            seed=seed,
                            eligibility_tol=1e-7,
                        )
                    )
    frame = pd.DataFrame(rows)
    verification = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()
    pilot = pd.DataFrame(pilot_rows)
    store.csv("raw", "data_perturbation_runs", frame)
    store.csv("raw", "data_perturbation_verifications", verification)
    store.csv("raw", "eligibility_pilot_runs", pilot)
    return frame, verification, pilot


def manuscript_summary(
    rescaling: pd.DataFrame,
    correlated: pd.DataFrame,
    perturbations: pd.DataFrame,
    pilot: pd.DataFrame,
) -> dict[str, Any]:
    homogeneous = rescaling[rescaling.guard_variant == "homogeneous"]
    absolute = rescaling[rescaling.guard_variant == "absolute"]
    normalized_spread = (
        homogeneous.groupby(["n", "p"]).normalized_final_objective
        .agg(lambda values: float(values.max() - values.min()))
        .max()
    )
    correlated_fb = correlated[correlated.method == "FB"]
    correlated_4v = correlated[correlated.method == "4V-FB"]
    perturb_fb = perturbations[perturbations.method == "FB"]
    perturb_4v = perturbations[perturbations.method == "4V-FB"]
    per_nu = []
    for nu in sorted(perturbations.nu.unique()):
        # These values come directly from the fixed profile tuple.  Exact
        # equality is intentional: np.isclose's default absolute tolerance
        # would merge the 1e-8 and 1e-12 scale-control rows.
        fb = perturb_fb[perturb_fb.nu == nu]
        v4 = perturb_4v[perturb_4v.nu == nu]
        per_nu.append(
            {
                "nu": float(nu),
                "fb_nlm": int(fb.numerical_locmin.sum()),
                "four_v_nlm": int(v4.numerical_locmin.sum()),
                "four_v_corrected": int((v4.accepted_corrections > 0).sum()),
                "four_v_iteration_min": int(v4.iterations.min()),
                "four_v_iteration_max": int(v4.iterations.max()),
            }
        )
    return {
        "objective_rescaling": {
            "homogeneous_runs": len(homogeneous),
            "homogeneous_nlm": int(homogeneous.numerical_locmin.sum()),
            "homogeneous_corrected": int((homogeneous.accepted_corrections == 1).sum()),
            "iteration_min": int(homogeneous.iterations.min()),
            "iteration_max": int(homogeneous.iterations.max()),
            "max_within_problem_normalized_objective_spread": float(normalized_spread),
            "absolute_nlm": int(absolute.numerical_locmin.sum()),
            "absolute_smallest_scale_corrected": int(
                (
                    absolute[absolute.objective_scale == 1e-12]
                    .accepted_corrections
                    .gt(0)
                    .sum()
                )
            ),
        },
        "correlated_zero_init": {
            "instances": len(correlated_fb),
            "fb_nlm": int(correlated_fb.numerical_locmin.sum()),
            "four_v_nlm": int(correlated_4v.numerical_locmin.sum()),
            "four_v_corrected": int((correlated_4v.accepted_corrections > 0).sum()),
            "four_v_max_corrections": int(correlated_4v.accepted_corrections.max()),
        },
        "data_perturbations": {
            "instances": len(perturb_fb),
            "fb_nlm": int(perturb_fb.numerical_locmin.sum()),
            "fb_capped": int((perturb_fb.status == "max_iter").sum()),
            "four_v_nlm": int(perturb_4v.numerical_locmin.sum()),
            "four_v_corrected": int((perturb_4v.accepted_corrections > 0).sum()),
            "per_nu": per_nu,
        },
        "eligibility_pilot": {
            "runs": len(pilot),
            "corrected": int((pilot.accepted_corrections > 0).sum()) if len(pilot) else 0,
        },
    }


def run_scale_validation(profile: ExperimentProfile, output: Path) -> dict[str, Any]:
    store = OutputStore(output)
    store.write_environment(output.name, profile.name)
    store.json(
        "metadata",
        "protocol",
        {
            "profile": asdict(profile),
            "solver": asdict(scale_solver(max_iter=profile.max_iter)),
            "four_v_fb": asdict(scale_four_v(max_iter=profile.max_iter)),
            "absolute_guard_control": asdict(
                replace(
                    scale_four_v(max_iter=profile.max_iter),
                    guard_mode="absolute_max",
                )
            ),
        },
    )
    rescaling, _ = run_objective_rescaling(profile, store)
    correlated, _ = run_correlated_zero_initialization(profile, store)
    perturbations, _, pilot = run_data_perturbations(profile, store)
    summary = manuscript_summary(rescaling, correlated, perturbations, pilot)
    store.json("metadata", "manuscript_summary", summary)
    table = pd.DataFrame(summary["data_perturbations"]["per_nu"])
    store.csv("tables", "data_perturbation_summary", table)
    return summary
