"""Sampler-level nulls and source diagnostics for the temporal-sampling study."""
import math
import numpy as np

def gaussian_copula_uniform_corr(rho: float) -> float:
    return float((6.0 / math.pi) * math.asin(rho / 2.0))

def source_lag1_corr(u: np.ndarray) -> float:
    u = np.asarray(u, dtype=float)
    if u.ndim == 1:
        return float(np.corrcoef(u[:-1], u[1:])[0, 1])
    return float(np.mean([np.corrcoef(x[:-1], x[1:])[0, 1] for x in u]))

def decoupled_uniforms(u: np.ndarray, seed: int = 0) -> np.ndarray:
    """Break temporal alignment within each trajectory while preserving values."""
    u = np.asarray(u, dtype=float)
    rng = np.random.default_rng(seed)
    out = u.copy()
    for i in range(out.shape[0]):
        out[i] = rng.permutation(out[i])
    return out

def decoupled_lag_stat(u: np.ndarray, cdf: np.ndarray, seed: int = 0) -> float:
    u_null = decoupled_uniforms(u, seed=seed)
    vals = [np.corrcoef(u_null[i, :-1], cdf[i, 1:])[0, 1] for i in range(u.shape[0])]
    return float(np.nanmean(vals))
