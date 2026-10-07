import numpy as np

from temporal_sampling_correlation.sampler import ar1_uniform


def test_ar1_uniform_is_bounded_and_reproducible():
    a = ar1_uniform(0.5, 40, 3)
    b = ar1_uniform(0.5, 40, 3)
    assert np.all((a >= 0) & (a <= 1))
    assert np.allclose(a, b)


def test_rho_zero_has_stationary_uniform_marginals():
    x = ar1_uniform(0.0, 5000, 123)
    assert abs(float(x.mean()) - 0.5) < 0.03
