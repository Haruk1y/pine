from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_GROUPS = ["dataset", "model", "alpha", "n_bins", "beta"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate PINE result CSV files.")
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/summary.csv"))
    parser.add_argument("--group-by", nargs="*", default=DEFAULT_GROUPS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frames = [pd.read_csv(path) for path in args.csv]
    data = pd.concat(frames, ignore_index=True)
    numeric = data.select_dtypes(include="number").columns.difference(args.group_by)
    summary = data.groupby(args.group_by, dropna=False)[numeric].agg(["mean", "std"])
    summary.columns = [".".join(col).strip(".") for col in summary.columns]
    summary = summary.reset_index()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
