import numpy as np

from lp_lasso_4vfb.algorithms import run_fb, run_four_v_fb
from lp_lasso_4vfb.config import legacy_four_v, legacy_solver
from lp_lasso_4vfb.problems import make_exact_saddle_problem


def test_verified_method_rejects_quartic_fixed_point() -> None:
    problem = make_exact_saddle_problem(20, p=0.5, kind="quartic")
    x0 = np.asarray(problem.meta["x_saddle"])
    fb = run_fb(problem, x0, legacy_solver(max_iter=100))
    verified = run_four_v_fb(problem, x0, legacy_four_v(max_iter=100))
    assert not fb.passes_nlm()
    assert verified.passes_nlm()
    assert verified.accepted_corrections == 1
    assert verified.verification_calls == 6

