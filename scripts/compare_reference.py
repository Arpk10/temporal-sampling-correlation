"""Check structural coverage of a confirmatory CSV against the v1 task grid."""
import argparse
import pandas as pd

from temporal_sampling_correlation.prompts import PROMPTS, RHO_GRID, SEEDS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    expected = {
        (prompt_id, float(rho), int(seed))
        for prompt_id in range(len(PROMPTS))
        for rho in RHO_GRID
        for seed in SEEDS
    }

    if "prompt_id" in df.columns:
        got = {
            (int(r["prompt_id"]), float(r["rho"]), int(r["seed"]))
            for _, r in df.iterrows()
        }
    else:
        got = {
            (int(r["prompt"]), float(r["rho"]), int(r["seed"]))
            for _, r in df.iterrows()
        }

    missing = sorted(expected - got)
    extra = sorted(got - expected)
    print(f"rows={len(df)} unique_conditions={len(got)} expected={len(expected)}")
    print(f"missing={len(missing)} extra={len(extra)}")
    if missing:
        print("first_missing:", missing[:20])
    if extra:
        print("first_extra:", extra[:20])

    required = {"seed", "rho", "lag1_cdf", "lag1_normrank", "lag1_prob", "entropy", "distinct2", "mean_logp"}
    print("missing_columns:", sorted(required - set(df.columns)))


if __name__ == "__main__":
    main()
