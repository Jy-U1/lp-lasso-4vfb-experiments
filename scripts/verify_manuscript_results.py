#!/usr/bin/env python3
"""Audit frozen result files against every quantitative manuscript claim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy",
        type=Path,
        default=ROOT / "results" / "reference_20260907T164315Z",
    )
    parser.add_argument(
        "--scale", type=Path, default=ROOT / "results" / "scale_validation"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "audit" / "manuscript_result_audit.json",
    )
    args = parser.parse_args()
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, observed: Any, expected: Any) -> None:
        checks.append(
            {
                "claim": name,
                "passed": bool(passed),
                "observed": observed,
                "expected": expected,
            }
        )

    def raw(name: str) -> pd.DataFrame:
        return pd.read_csv(args.legacy / "raw" / f"{name}.csv")

    ordinary = raw("ordinary_runs")
    fb = ordinary[ordinary.method == "FB"]
    v4 = ordinary[ordinary.method == "4V-FB"]
    ratio_table = ordinary.pivot(
        index="run_key", columns="method", values="runtime_algorithm"
    )
    ratios = ratio_table["4V-FB"] / ratio_table["FB"]
    record("ordinary paired instances", len(v4) == len(fb) == 60, len(v4), 60)
    record("ordinary 4V-FB NLM", int(v4.numerical_locmin.sum()) == 60, int(v4.numerical_locmin.sum()), 60)
    record("ordinary accepted corrections", int(v4.accepted_corrections.sum()) == 0, int(v4.accepted_corrections.sum()), 0)
    record("ordinary iteration range", (int(v4.iterations.min()), int(v4.iterations.max())) == (1228, 26767), [int(v4.iterations.min()), int(v4.iterations.max())], [1228, 26767])
    record("ordinary verification range", (int(v4.verification_calls.min()), int(v4.verification_calls.max())) == (11, 15), [int(v4.verification_calls.min()), int(v4.verification_calls.max())], [11, 15])
    endpoint_match = True
    for field in ("iterations", "final_objective", "final_residual", "support_size", "relative_lambda_min"):
        pivot = ordinary.pivot(index="run_key", columns="method", values=field)
        endpoint_match &= np.array_equal(pivot["FB"].to_numpy(), pivot["4V-FB"].to_numpy())
    record("ordinary paired endpoints identical", endpoint_match, endpoint_match, True)
    timing = [float(ratios.min()), float(ratios.median()), float(ratios.max())]
    record("ordinary timing ratio rounded", [round(x, 3) for x in timing] == [1.031, 1.052, 1.205], timing, [1.031, 1.052, 1.205])
    fractions = [100 * float(v4.verification_time_fraction.median()), 100 * float(v4.verification_time_fraction.max())]
    record("ordinary verification fractions percent", abs(fractions[0] - 0.6856237831644) < 1e-9 and abs(fractions[1] - 3.58109117558221) < 1e-9, fractions, [0.6856237831644, 3.58109117558221])

    ablation = raw("schedule_ablation_runs").set_index("schedule")
    calls = [int(ablation.loc["powers_of_two", "verification_calls"]), int(ablation.loc["every_eligible", "verification_calls"])]
    times = [float(ablation.loc["powers_of_two", "runtime_algorithm"]), float(ablation.loc["every_eligible", "runtime_algorithm"])]
    record("schedule ablation calls", calls == [11, 925], calls, [11, 925])
    record("schedule ablation times", abs(times[0] - 0.199885) < 5e-7 and abs(times[1] - 0.299677) < 5e-7, times, [0.199885, 0.299677])

    exact = raw("exact_saddle_runs")
    exact_fb = exact[exact.method == "FB"]
    exact_v4 = exact[exact.method == "4V-FB"]
    record("exact FB NLM", int(exact_fb.numerical_locmin.sum()) == 0, int(exact_fb.numerical_locmin.sum()), 0)
    record("exact 4V-FB NLM", len(exact_v4) == 12 and int(exact_v4.numerical_locmin.sum()) == 12, int(exact_v4.numerical_locmin.sum()), 12)
    record("exact corrections and verification calls", bool((exact_v4.accepted_corrections == 1).all() and (exact_v4.verification_calls == 6).all()), {"corrections": sorted(exact_v4.accepted_corrections.unique().tolist()), "calls": sorted(exact_v4.verification_calls.unique().tolist())}, {"corrections": [1], "calls": [6]})
    record("exact maximum residual", abs(float(exact_v4.final_residual.max()) - 4.790904047749073e-12) < 1e-24, float(exact_v4.final_residual.max()), 4.790904047749073e-12)
    exact_eigs = [float(exact_v4.relative_lambda_min.min()), float(exact_v4.relative_lambda_min.max())]
    record("exact relative eigenvalue range", abs(exact_eigs[0] - 0.5333011437095794) < 1e-14 and abs(exact_eigs[1] - 0.606491945014569) < 1e-14, exact_eigs, [0.5333011437095794, 0.606491945014569])

    strict = raw("perturbed_strict_runs")
    strict_counts = strict.groupby("method").agg(runs=("method", "size"), escaped=("escaped_radius_0p1", "sum"), nlm=("numerical_locmin", "sum"))
    strict_ok = bool((strict_counts == 10).all().all())
    record("perturbed strict all methods 10/10", strict_ok, strict_counts.to_dict("index"), "runs=escaped=NLM=10 for each method")

    quartic = raw("perturbed_quartic_runs")
    quartic_counts = quartic.groupby("method").agg(
        runs=("method", "size"),
        escaped=("escaped_radius_0p1", "sum"),
        nlm=("numerical_locmin", "sum"),
        corrected=("accepted_corrections", lambda values: int((values > 0).sum())),
        capped=("status", lambda values: int((values == "max_iter").sum())),
    )
    q4 = quartic_counts.loc["4V-FB"]
    q4_ok = tuple(map(int, q4[["runs", "escaped", "nlm", "corrected", "capped"]])) == (120, 120, 120, 120, 0)
    baseline_ok = all(
        tuple(map(int, quartic_counts.loc[method, ["runs", "escaped", "nlm", "corrected", "capped"]])) == (120, 0, 0, 0, 18)
        for method in ("FB", "DIRL1", "DIRL2")
    )
    record("perturbed quartic 4V-FB", q4_ok, q4.to_dict(), {"runs": 120, "escaped": 120, "nlm": 120, "corrected": 120, "capped": 0})
    record("perturbed quartic baselines", baseline_ok, quartic_counts.loc[["FB", "DIRL1", "DIRL2"]].to_dict("index"), "each: 120 runs, 0 escape/NLM/correction, 18 capped")

    long = raw("long_horizon_fb").iloc[-1]
    long_observed = {
        "iteration": int(long.iteration),
        "first_machine_fixed_iteration": int(long.first_machine_fixed_iteration),
        "distance_to_saddle": float(long.distance_to_saddle),
        "fb_residual": float(long.fb_residual),
        "relative_lambda_min": float(long.relative_lambda_min),
        "objective_change_high_precision": float(long.objective_change_high_precision),
    }
    long_ok = (
        long_observed["iteration"] == 1_000_000
        and long_observed["first_machine_fixed_iteration"] == 10
        and abs(long_observed["distance_to_saddle"] - 3.836607879617091e-6) < 1e-18
        and long_observed["fb_residual"] == 0.0
        and abs(long_observed["relative_lambda_min"] + 2.113066666087349e-12) < 1e-24
        and abs(long_observed["objective_change_high_precision"] + 5.085880738919365e-24) < 1e-34
    )
    record("long-horizon control", long_ok, long_observed, "manuscript Table long-horizon values")

    scale_summary = json.loads(
        (args.scale / "metadata" / "manuscript_summary.json").read_text(encoding="utf-8")
    )
    expected_scale = {
        "objective": [42, 42, 42, 39, 56, 36, 0],
        "correlated": [45, 18, 45, 36, 2],
        "perturbed": [180, 47, 133, 180, 155],
        "pilot": [15, 0],
    }
    s1 = scale_summary["objective_rescaling"]
    s2 = scale_summary["correlated_zero_init"]
    s3 = scale_summary["data_perturbations"]
    observed_scale = {
        "objective": [s1["homogeneous_runs"], s1["homogeneous_nlm"], s1["homogeneous_corrected"], s1["iteration_min"], s1["iteration_max"], s1["absolute_nlm"], s1["absolute_smallest_scale_corrected"]],
        "correlated": [s2["instances"], s2["fb_nlm"], s2["four_v_nlm"], s2["four_v_corrected"], s2["four_v_max_corrections"]],
        "perturbed": [s3["instances"], s3["fb_nlm"], s3["fb_capped"], s3["four_v_nlm"], s3["four_v_corrected"]],
        "pilot": [scale_summary["eligibility_pilot"]["runs"], scale_summary["eligibility_pilot"]["corrected"]],
    }
    record("scale-equivariant aggregate claims", observed_scale == expected_scale, observed_scale, expected_scale)
    record("rescaling normalized objective spread", abs(s1["max_within_problem_normalized_objective_spread"] - 2.6645352591003757e-15) < 1e-27, s1["max_within_problem_normalized_objective_spread"], 2.6645352591003757e-15)
    per_nu_expected = [
        {"nu": 1e-8, "fb_nlm": 0, "four_v_nlm": 60, "four_v_corrected": 60, "four_v_iteration_min": 44, "four_v_iteration_max": 1267},
        {"nu": 1e-6, "fb_nlm": 0, "four_v_nlm": 60, "four_v_corrected": 60, "four_v_iteration_min": 44, "four_v_iteration_max": 1267},
        {"nu": 1e-4, "fb_nlm": 47, "four_v_nlm": 60, "four_v_corrected": 35, "four_v_iteration_min": 44, "four_v_iteration_max": 7206},
    ]
    record("data perturbation table", s3["per_nu"] == per_nu_expected, s3["per_nu"], per_nu_expected)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "legacy_results": args.legacy.as_posix(),
        "scale_results": args.scale.as_posix(),
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("manuscript/result audit failed")


if __name__ == "__main__":
    main()

