import argparse
import glob
import os

import pandas as pd

from plots import make_plots


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        required=True,
        help="Directory containing metrics_<method>.csv files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pattern = os.path.join(args.run_dir, "metrics_*.csv")
    metric_files = sorted(
        path
        for path in glob.glob(pattern)
        if os.path.basename(path) != "metrics_all_methods.csv"
    )
    if not metric_files:
        raise FileNotFoundError(f"No per-method metrics files found with pattern: {pattern}")

    combined = pd.concat((pd.read_csv(path) for path in metric_files), ignore_index=True)
    out_csv = os.path.join(args.run_dir, "metrics_all_methods.csv")
    combined.to_csv(out_csv, index=False)

    make_plots(out_csv, os.path.join(args.run_dir, "plots"))
    print(f"Aggregated {len(metric_files)} metrics files into: {out_csv}")
    print(f"Plots saved to: {os.path.join(args.run_dir, 'plots')}")


if __name__ == "__main__":
    main()