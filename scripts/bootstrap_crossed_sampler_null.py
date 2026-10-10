"""Two-way prompt/source-seed bootstrap for paired sampler-null slopes.

Input: the trajectory-level CSV written by run_paired_sampler_null.py.
The same sampled prompt IDs and seed IDs are used across all rho values,
preserving the paired full/frozen/excess contrasts and the shared-source design.

This is a sensitivity analysis, not a claim that trajectory rows are independent.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RHO_COL = "rho"
PROMPT_COL = "prompt_id"
SEED_COL = "seed"
VALUE_COLS = ("full_lag1_cdf", "frozen_lag1_cdf", "excess_lag1_cdf")


def crossed_bootstrap(
    df: pd.DataFrame,
    value_col: str,
    n_boot: int = 20_000,
    seed: int = 20261010,
) -> dict:
    required = {RHO_COL, PROMPT_COL, SEED_COL, value_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")

    prompts = np.sort(df[PROMPT_COL].unique())
    seeds = np.sort(df[SEED_COL].unique())
    rhos = np.sort(df[RHO_COL].unique())
    expected = len(prompts) * len(seeds) * len(rhos)
    if len(df) != expected or df.duplicated([PROMPT_COL, SEED_COL, RHO_COL]).any():
        raise ValueError(
            "Expected exactly one row per prompt_id × seed × rho combination."
        )

    pivot = df.pivot(index=[PROMPT_COL, SEED_COL], columns=RHO_COL, values=value_col)
    pivot = pivot.reindex(
        pd.MultiIndex.from_product([prompts, seeds], names=[PROMPT_COL, SEED_COL])
    ).reindex(columns=rhos)
    if pivot.isna().any().any():
        raise ValueError("Input contains missing values in the crossed design.")

    values = pivot.to_numpy().reshape(len(prompts), len(seeds), len(rhos))
    observed_means = values.mean(axis=(0, 1))
    observed_slope = float(np.polyfit(rhos.astype(float), observed_means, 1)[0])

    rng = np.random.default_rng(seed)
    slopes = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        prompt_idx = rng.integers(0, len(prompts), size=len(prompts))
        seed_idx = rng.integers(0, len(seeds), size=len(seeds))
        # Crossed cluster resampling: sampled prompt and seed IDs are shared
        # across rho conditions; preserve the original paired contrasts.
        means = values[prompt_idx[:, None], seed_idx[None, :], :].mean(axis=(0, 1))
        slopes[b] = np.polyfit(rhos.astype(float), means, 1)[0]

    lo, hi = np.quantile(slopes, [0.025, 0.975])
    return {
        "metric": value_col,
        "n_rows": int(len(df)),
        "n_prompts": int(len(prompts)),
        "n_seed_ids": int(len(seeds)),
        "n_rho_conditions": int(len(rhos)),
        "n_boot": int(n_boot),
        "bootstrap_seed": int(seed),
        "slope": observed_slope,
        "crossed_bootstrap_ci95": [float(lo), float(hi)],
        "method": (
            "Percentile interval from independent resampling of prompt IDs and "
            "seed IDs, with sampled IDs reused across rho conditions."
        ),
        "caveat": (
            "Seed IDs index source sequences reused across prompts; this interval "
            "is a crossed prompt/source-seed sensitivity analysis, not a claim "
            "that all trajectory rows are independent."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/paired_sampler_null.csv")
    parser.add_argument("--output", default="results/paired_sampler_null_crossed_bootstrap.json")
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    reports = [
        crossed_bootstrap(df, col, n_boot=args.bootstrap, seed=args.seed + i)
        for i, col in enumerate(VALUE_COLS)
    ]
    payload = {
        "input": str(Path(args.input)),
        "design": "crossed prompt_id × seed × rho",
        "results": reports,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
