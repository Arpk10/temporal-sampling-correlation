import pandas as pd

from temporal_sampling_correlation.metrics import lag1_cdf_slope


def test_slope():
    df = pd.DataFrame({
        "rho": [0.0, 0.0, 0.5, 0.5, 1.0, 1.0],
        "lag1_cdf": [0.0, 0.1, 0.5, 0.6, 1.0, 1.1],
    })
    assert abs(lag1_cdf_slope(df) - 1.0) < 1e-9
