#!/usr/bin/env python3
"""
Embedding phase: TorusMDS vs. s_gd2 vs. Chen wrap_python on the graphs
staged locally by stage_suitesparse.py. Persists layouts + a run manifest
under --output-dir; does not compute metrics.

Reads only the local cache produced by stage_suitesparse.py -- never touches
the network, so this is safe to run on SLURM compute nodes with no internet
access as long as --cache-dir is on shared/accessible storage.

Usage:
    python suitesparse_comparison.py --cache-dir data/suitesparse_cache --output-dir layouts/suitesparse
    # SLURM array sharding: process every --num-shards-th graph
    python suitesparse_comparison.py --shard-index 3 --num-shards 20
"""

from __future__ import annotations

import argparse
import os
import sys

import networkx as nx
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.experiment_runner import GraphRecord, load_graph, run_embeddings


def suitesparse_graph_iterator(cache_dir: str, shard_index: int, num_shards: int):
    manifest = pd.read_csv(os.path.join(cache_dir, "manifest.csv"))
    manifest = manifest.sort_values("matrix_id").reset_index(drop=True)

    for exp_idx, row in manifest.iterrows():
        if exp_idx % num_shards != shard_index:
            continue
        graph_file = os.path.join(cache_dir, f"graph_{row.matrix_id}.npz")
        try:
            G = load_graph(graph_file)
        except Exception as e:
            print(f"[{exp_idx}] Failed to load {graph_file}: {e}")
            continue
        G = nx.convert_node_labels_to_integers(G)
        yield GraphRecord(
            exp_idx=int(exp_idx),
            graph=G,
            meta={"matrix_id": int(row.matrix_id), "group": row.group, "name": row["name"]},
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SuiteSparse embedding phase: TorusMDS vs s_gd2 vs wrap_python (Chen)"
    )
    parser.add_argument("--cache-dir", type=str, default="data/suitesparse_cache",
                        help="Local cache produced by stage_suitesparse.py (default: data/suitesparse_cache)")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="This shard's index in [0, num_shards) for SLURM array parallelism (default: 0)")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of shards (default: 1, i.e. no sharding)")
    parser.add_argument("--torus-max-iters", type=int, default=2000,
                        help="SGD iterations for TorusMDS (default: 2000)")
    parser.add_argument("--wrap-python-max-iters", type=int, default=200,
                        help="Descent iterations for wrap_python -- kept low since its cost is "
                             "O(n^2 * iters); 200 matches the Chen reference's own default (default: 200)")
    parser.add_argument("--wrap-python-max-n", type=int, default=3500,
                        help="Max n for the Chen wrap_python method; skipped above this (default: 3500)")
    parser.add_argument("--output-dir", type=str, default="layouts/suitesparse",
                        help="Directory to persist layouts + run manifest (default: layouts/suitesparse)")
    parser.add_argument("--checkpoint-every", type=int, default=20,
                        help="Save manifest every N graphs (default: 20)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Global random seed (default: 0)")
    args = parser.parse_args()

    if not (0 <= args.shard_index < args.num_shards):
        parser.error("--shard-index must be in [0, --num-shards)")

    graph_iterator = suitesparse_graph_iterator(
        cache_dir=args.cache_dir,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )

    run_embeddings(
        graph_iterator=graph_iterator,
        output_dir=args.output_dir,
        torus_max_iters=args.torus_max_iters,
        wrap_python_max_iters=args.wrap_python_max_iters,
        method_max_n={"wrap_python": args.wrap_python_max_n},
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
    )
