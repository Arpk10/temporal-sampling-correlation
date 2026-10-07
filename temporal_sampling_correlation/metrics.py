import numpy as np
import pandas as pd


METRICS = [
    "lag1_cdf",
    "lag2_cdf",
    "lag1_normrank",
    "lag2_normrank",
    "lag1_prob",
    "lag2_prob",
    "entropy",
    "distinct2",
    "mean_logp",
]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("rho")[METRICS].agg(["mean", "std"])


def lag1_cdf_slope(df: pd.DataFrame) -> float:
    grouped = df.groupby("rho")["lag1_cdf"].mean()
    x = grouped.index.to_numpy(dtype=float)
    y = grouped.to_numpy(dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def permutation_slope_test(
    df: pd.DataFrame,
    n_perm: int = 5000,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    observed = lag1_cdf_slope(df)

    work = df[["prompt", "seed", "rho", "lag1_cdf"]].copy()
    strata = list(work.groupby(["prompt", "seed"], sort=False).groups.values())

    null = np.empty(n_perm, dtype=float)

    for i in range(n_perm):
        shuffled = work["rho"].to_numpy().copy()
        for idx in strata:
            shuffled[idx] = rng.permutation(shuffled[idx])
        tmp = work.copy()
        tmp["rho"] = shuffled
        null[i] = lag1_cdf_slope(tmp)

    p_two_sided = (1 + np.sum(np.abs(null) >= abs(observed))) / (n_perm + 1)

    return {
        "observed_slope": observed,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)),
        "p_two_sided": float(p_two_sided),
    }
