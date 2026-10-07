import argparse
import pandas as pd
from temporal_sampling_correlation.metrics import permutation_slope_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--n-perm", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    result = permutation_slope_test(
        df, n_perm=args.n_perm, seed=args.seed
    )

    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
