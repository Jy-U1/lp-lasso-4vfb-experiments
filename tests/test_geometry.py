from lp_lasso_4vfb.config import legacy_solver
from lp_lasso_4vfb.problems import diagnose_constructed_saddle, make_exact_saddle_problem


def test_quartic_construction() -> None:
    problem = make_exact_saddle_problem(20, p=0.5, kind="quartic")
    row = diagnose_constructed_saddle(problem, legacy_solver())
    assert row["stationarity_norm"] < 1e-12
    assert row["fb_fixed_residual"] < 1e-11
    assert abs(row["lambda_min"]) < 1e-12
    assert abs(row["D3_along_escape"]) < 1e-12
    assert row["D4_along_escape"] < 0.0


def test_strict_construction_has_negative_curvature() -> None:
    problem = make_exact_saddle_problem(20, p=0.5, kind="strict")
    row = diagnose_constructed_saddle(problem, legacy_solver())
    assert row["lambda_min"] < 0.0

