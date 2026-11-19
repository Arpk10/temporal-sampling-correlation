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


def test_nucleus_keep_mask_uses_minimal_prefix_and_includes_equality():
    import torch

    from temporal_sampling_correlation.sampler import nucleus_keep_mask

    sorted_probs = torch.tensor([[0.5, 0.3, 0.2], [0.9, 0.05, 0.05]])
    keep = nucleus_keep_mask(sorted_probs, 0.7)
    assert keep.tolist() == [[True, True, False], [True, False, False]]


def test_nucleus_keep_mask_top_p_one_keeps_full_support():
    import torch

    from temporal_sampling_correlation.sampler import nucleus_keep_mask

    sorted_probs = torch.tensor([[0.4, 0.3, 0.2, 0.1]])
    assert nucleus_keep_mask(sorted_probs, 1.0).tolist() == [
        [True, True, True, True]
    ]


def test_nucleus_keep_mask_rejects_invalid_threshold():
    import pytest
    import torch

    from temporal_sampling_correlation.sampler import nucleus_keep_mask

    with pytest.raises(ValueError, match="top_p"):
        nucleus_keep_mask(torch.tensor([[0.6, 0.4]]), 0.0)
