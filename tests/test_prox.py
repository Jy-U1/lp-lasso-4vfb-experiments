import numpy as np

from lp_lasso_4vfb.prox import lp_prox_threshold, prox_lp, prox_lp_scalar


def test_vector_and_scalar_prox_agree() -> None:
    rng = np.random.default_rng(9173)
    for p in (0.3, 0.5, 0.8):
        for tau in (0.01, 0.1, 1.0):
            _, threshold = lp_prox_threshold(tau, p)
            values = np.r_[
                rng.normal(size=30),
                threshold * np.array([0.99, 1.0, 1.01, -0.99, -1.0, -1.01]),
            ]
            vector = prox_lp(values, tau, p)
            scalar = np.array([prox_lp_scalar(value, tau, p) for value in values])
            assert np.max(np.abs(vector - scalar)) < 1e-11


def test_zero_is_selected_at_threshold() -> None:
    for p in (0.3, 0.5, 0.8):
        _, threshold = lp_prox_threshold(0.1, p)
        assert prox_lp_scalar(threshold, 0.1, p) == 0.0

