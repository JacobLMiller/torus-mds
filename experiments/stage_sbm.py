#!/usr/bin/env python3
"""
Pregenerates SBM graphs and caches them as lightweight sparse-adjacency
.npz files (same format stage_suitesparse.py uses for its cache): one
graph_<exp_idx>.npz per graph plus a manifest.csv, under --output-dir.

Run this once per (n_min, n_max) size tier. Point sbm_comparison.py
--cache-dir at the resulting directory (with --shard-index/--num-shards
for SLURM array parallelism) instead of its on-the-fly
--n-graphs/--n-min/--n-max generation -- this fixes the graph set once so
it can be resharded/rerun without regenerating (and re-randomizing) graphs.

Resumable: reruns with the same arguments pick up where they left off
rather than regenerating already-staged graphs.

Usage:
    python stage_sbm.py --n-graphs 336 --n-min 100 --n-max 1000 \
        --output-dir data/sbm_cache/100_1000
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.experiment_runner import save_graph
from sbm_comparison import sbm_graph_iterator


def stage_sbm(
    n_graphs: int,
    n_min: int,
    n_max: int,
    k_min: int,
    k_max: int,
    seed: int,
    output_dir: str,
    checkpoint_every: int = 50,
) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.csv")

    manifest_rows: list[dict] = []
    done: set[int] = set()
    if os.path.exists(manifest_path):
        existing = pd.read_csv(manifest_path)
        manifest_rows = existing.to_dict("records")
        done = set(existing["exp_idx"])
        print(f"Resuming: {len(done)} graphs already staged.")

    rng = np.random.default_rng(seed)
    graph_iterator = sbm_graph_iterator(
        n_graphs=n_graphs, n_min=n_min, n_max=n_max, k_min=k_min, k_max=k_max, rng=rng,
    )

    since_checkpoint = 0
    for rec in graph_iterator:
        if rec.exp_idx in done:
            continue
        save_graph(rec.graph, os.path.join(output_dir, f"graph_{rec.exp_idx}.npz"))
        manifest_rows.append({
            "exp_idx": rec.exp_idx,
            "n": rec.graph.number_of_nodes(),
            "m": rec.graph.number_of_edges(),
            "k": rec.meta["k"],
        })
        since_checkpoint += 1
        if since_checkpoint >= checkpoint_every:
            pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
            since_checkpoint = 0

    df = pd.DataFrame(manifest_rows)
    df.to_csv(manifest_path, index=False)
    print(f"Staged {len(df)} graphs -> {output_dir} (manifest: {manifest_path})")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pregenerate and cache SBM graphs")
    parser.add_argument("--n-graphs", type=int, default=1000,
                        help="Number of SBM graphs to generate (default: 1000)")
    parser.add_argument("--n-min", type=int, default=100,
                        help="Minimum number of nodes (default: 100)")
    parser.add_argument("--n-max", type=int, default=10000,
                        help="Maximum number of nodes (default: 10000)")
    parser.add_argument("--k-min", type=int, default=3,
                        help="Minimum number of SBM blocks (default: 3)")
    parser.add_argument("--k-max", type=int, default=8,
                        help="Maximum number of SBM blocks (default: 8)")
    parser.add_argument("--output-dir", type=str, default="data/sbm_cache",
                        help="Local cache directory (default: data/sbm_cache)")
    parser.add_argument("--checkpoint-every", type=int, default=50,
                        help="Save manifest every N graphs (default: 50)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Global random seed (default: 0)")
    args = parser.parse_args()

    stage_sbm(
        n_graphs=args.n_graphs,
        n_min=args.n_min,
        n_max=args.n_max,
        k_min=args.k_min,
        k_max=args.k_max,
        seed=args.seed,
        output_dir=args.output_dir,
        checkpoint_every=args.checkpoint_every,
    )
