#!/usr/bin/env python3
"""
Concatenates per-shard results CSVs (written by compute_metrics.py, one per
embedding shard) into a single results/<family>_comparison.csv.

Each shard's embedding phase numbers exp_idx independently starting from 0,
so a naive concat would silently conflate rows from different graphs in
different shards that happen to share an exp_idx. This renumbers exp_idx to
be globally unique across the merged file, preserving which rows came from
the same graph (all methods for one graph keep the same new exp_idx).

Usage:
    python merge_results.py --glob "results/sbm_*_comparison.csv" --output results/sbm_comparison.csv
    python merge_results.py --shard-csv results/sbm_a.csv results/sbm_b.csv --output results/sbm_comparison.csv
"""

from __future__ import annotations

import argparse
import glob as _glob

import pandas as pd


def merge_results(shard_csvs: list[str], output_csv: str) -> pd.DataFrame:
    frames = []
    next_exp_idx = 0
    for path in shard_csvs:
        df = pd.read_csv(path)
        if df.empty:
            continue
        old_ids = sorted(df["exp_idx"].unique())
        old_to_new = {old: next_exp_idx + i for i, old in enumerate(old_ids)}
        df = df.copy()
        df["exp_idx"] = df["exp_idx"].map(old_to_new)
        df["source_shard"] = path
        next_exp_idx += len(old_ids)
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    merged.to_csv(output_csv, index=False)
    print(f"Merged {len(shard_csvs)} shard CSVs ({len(merged)} rows) -> {output_csv}")
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge per-shard results CSVs into one file")
    parser.add_argument("--glob", type=str, default=None,
                        help="Glob pattern matching per-shard result CSVs")
    parser.add_argument("--shard-csv", type=str, nargs="*", default=None,
                        help="Explicit list of per-shard result CSV paths (alternative to --glob)")
    parser.add_argument("--output", type=str, required=True,
                        help="Merged output CSV path")
    args = parser.parse_args()

    if args.glob:
        paths = sorted(_glob.glob(args.glob))
    elif args.shard_csv:
        paths = args.shard_csv
    else:
        parser.error("Provide either --glob or --shard-csv")

    if not paths:
        parser.error("No shard CSVs matched")

    merge_results(paths, args.output)
