#!/usr/bin/env python3
"""
Embedding phase: TorusMDS vs. s_gd2 vs. Chen wrap_python on stochastic block
model graphs. Persists layouts + a run manifest under --output-dir; does not
compute metrics (see compute_metrics.py for that, run as a separate phase).

Usage:
    python sbm_comparison.py [--n-graphs N] [--seed S] [--output-dir layouts/sbm]
    python sbm_comparison.py --torus-max-iters 1000
"""

from __future__ import annotations

import argparse
import os
import sys

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.experiment_runner import ASPECT_INIT_VARIANTS, METHODS, GraphRecord, run_embeddings


# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------

def _make_block_probs(k: int, p_in: float, p_out: float) -> list[list[float]]:
    return [[p_in if i == j else p_out for j in range(k)] for i in range(k)]


def generate_sbm(
    n: int,
    k: int,
    rng: np.random.Generator,
    p_in: float = 0.3,
    p_out: float = 0.05,
    max_retries: int = 20,
) -> tuple[nx.Graph, list[int]]:
    """Return a connected SBM with k roughly-equal blocks, retrying until connected."""
    sizes = [n // k] * k
    sizes[-1] += n - sum(sizes)
    probs = _make_block_probs(k, p_in, p_out)

    for _ in range(max_retries):
        seed = int(rng.integers(0, 2**31))
        G = nx.stochastic_block_model(sizes, probs, seed=seed)
        if nx.is_connected(G):
            return G, sizes
    G = nx.stochastic_block_model(sizes, probs, seed=int(rng.integers(0, 2**31)))
    comps = list(nx.connected_components(G))
    for i in range(len(comps) - 1):
        G.add_edge(next(iter(comps[i])), next(iter(comps[i + 1])))
    return G, sizes


def sbm_graph_iterator(
    n_graphs: int,
    n_min: int,
    n_max: int,
    k_min: int,
    k_max: int,
    rng: np.random.Generator,
):
    for exp_idx in range(n_graphs):
        n = int(np.exp(rng.uniform(np.log(n_min), np.log(n_max))))
        k = int(rng.integers(k_min, k_max + 1))
        try:
            G, _ = generate_sbm(n, k, rng)
        except Exception as e:
            print(f"[{exp_idx}] Graph generation failed: {e}")
            continue
        G = nx.convert_node_labels_to_integers(G)
        yield GraphRecord(exp_idx=exp_idx, graph=G, meta={"k": k})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SBM embedding phase: TorusMDS vs s_gd2 vs wrap_python (Chen)"
    )
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
    parser.add_argument("--torus-max-iters", type=int, default=2000,
                        help="SGD iterations for TorusMDS (default: 2000)")
    parser.add_argument("--stress-mode", type=str, default="raw", choices=["raw", "normalized"],
                        help="TorusMDS training objective: 'raw' minimizes sum((alpha*r-d)^2), "
                             "'normalized' minimizes sum((alpha*r-d)^2 / d^2) (default: raw)")
    parser.add_argument("--methods", type=str, nargs="+", default=list(METHODS),
                        choices=list(METHODS) + list(ASPECT_INIT_VARIANTS),
                        help=f"Which methods to run, space-separated (default: all of {list(METHODS)}; "
                             f"also available: {list(ASPECT_INIT_VARIANTS)})")
    parser.add_argument("--wrap-python-max-iters", type=int, default=200,
                        help="Descent iterations for wrap_python -- kept low since its cost is "
                             "O(n^2 * iters); 200 matches the Chen reference's own default (default: 200)")
    parser.add_argument("--wrap-python-max-n", type=int, default=3500,
                        help="Max n for the Chen wrap_python method; skipped above this (default: 3500)")
    parser.add_argument("--output-dir", type=str, default="layouts/sbm",
                        help="Directory to persist layouts + run manifest (default: layouts/sbm)")
    parser.add_argument("--checkpoint-every", type=int, default=20,
                        help="Save manifest every N graphs (default: 20)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Global random seed (default: 0)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    graph_iterator = sbm_graph_iterator(
        n_graphs=args.n_graphs,
        n_min=args.n_min,
        n_max=args.n_max,
        k_min=args.k_min,
        k_max=args.k_max,
        rng=rng,
    )

    run_embeddings(
        graph_iterator=graph_iterator,
        output_dir=args.output_dir,
        torus_max_iters=args.torus_max_iters,
        wrap_python_max_iters=args.wrap_python_max_iters,
        method_max_n={"wrap_python": args.wrap_python_max_n},
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
        methods=tuple(args.methods),
        torus_stress_mode=args.stress_mode,
    )
