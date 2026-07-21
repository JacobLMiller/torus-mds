#!/usr/bin/env python3
"""
Embedding phase: TorusMDS vs. s_gd2 vs. Chen wrap_python on geometric random
graphs (GRG) -- euclidean, toroidal, and spherical variants. Persists layouts
+ a run manifest under --output-dir; does not compute metrics (see
compute_metrics.py for that, run as a separate phase).

Usage:
    python grg_comparison.py [--n-graphs N] [--seed S] [--output-dir layouts/grg]
    python grg_comparison.py --graph-type-weights 0.4,0.3,0.3
"""

from __future__ import annotations

import argparse
import os
import sys

import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.experiment_runner import GraphRecord, run_embeddings

GRAPH_TYPES = ("euclidean", "toroidal", "spherical")


# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------

def _torus_cdist(pts: np.ndarray) -> np.ndarray:
    """Pairwise flat-torus distances on [0,1]^2."""
    diff = np.abs(pts[:, None, :] - pts[None, :, :])   # (n, n, 2)
    diff = np.minimum(diff, 1.0 - diff)                 # wrap each axis
    return np.sqrt((diff ** 2).sum(axis=-1))


def _sphere_cdist(pts: np.ndarray) -> np.ndarray:
    """Pairwise great-circle distances for unit vectors on S^2."""
    cos_sim = np.clip(pts @ pts.T, -1.0, 1.0)
    return np.arccos(cos_sim)


def _connect_and_stitch(
    sample_points,
    pairwise_distance,
    n: int,
    epsilon: float,
    rng: np.random.Generator,
    max_retries: int,
) -> tuple[nx.Graph, np.ndarray, float]:
    """Shared connectivity-retry/stitch loop for all three GRG generators."""
    for attempt in range(max_retries):
        eps = epsilon * (1.0 + 0.10 * attempt)
        pts = sample_points(n, rng)
        adj = pairwise_distance(pts) <= eps
        np.fill_diagonal(adj, False)
        G = nx.from_numpy_array(adj.astype(np.uint8))
        if nx.is_connected(G):
            return G, pts, round(eps, 6)

    pts = sample_points(n, rng)
    adj = pairwise_distance(pts) <= epsilon
    np.fill_diagonal(adj, False)
    G = nx.from_numpy_array(adj.astype(np.uint8))
    comps = list(nx.connected_components(G))
    for i in range(len(comps) - 1):
        G.add_edge(next(iter(comps[i])), next(iter(comps[i + 1])))
    return G, pts, round(epsilon, 6)


def generate_grg(n, epsilon, rng, max_retries=20):
    """Connected geometric random graph on n points in [0,1]^2 (Euclidean)."""
    return _connect_and_stitch(
        lambda n, rng: rng.uniform(0.0, 1.0, size=(n, 2)),
        lambda pts: cdist(pts, pts),
        n, epsilon, rng, max_retries,
    )


def generate_torus_grg(n, epsilon, rng, max_retries=20):
    """Connected geometric random graph on the flat unit torus."""
    return _connect_and_stitch(
        lambda n, rng: rng.uniform(0.0, 1.0, size=(n, 2)),
        _torus_cdist,
        n, epsilon, rng, max_retries,
    )


def generate_spherical_grg(n, epsilon, rng, max_retries=20):
    """
    Connected geometric random graph on the unit sphere S^2.

    Points are drawn uniformly on the sphere via normalized Gaussian vectors;
    an edge is added between every pair whose great-circle geodesic distance
    <= epsilon. Connectivity retries/stitching follow the same strategy as
    the other two generators.
    """
    def sample(n, rng):
        vecs = rng.normal(size=(n, 3))
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    return _connect_and_stitch(sample, _sphere_cdist, n, epsilon, rng, max_retries)


_GENERATORS = {
    "euclidean": generate_grg,
    "toroidal": generate_torus_grg,
    "spherical": generate_spherical_grg,
}


def grg_graph_iterator(
    n_graphs: int,
    n_min: int,
    n_max: int,
    eps_min: float,
    eps_max: float,
    n_eps: int,
    graph_type_weights: tuple[float, float, float],
    rng: np.random.Generator,
):
    eps_grid = np.linspace(eps_min, eps_max, n_eps).round(4).tolist()

    for exp_idx in range(n_graphs):
        n = int(np.exp(rng.uniform(np.log(n_min), np.log(n_max))))
        eps_nom = float(rng.choice(eps_grid))
        graph_type = rng.choice(GRAPH_TYPES, p=graph_type_weights)
        generator = _GENERATORS[graph_type]

        try:
            G, _, eps_actual = generator(n, eps_nom, rng)
        except Exception as e:
            print(f"[{exp_idx}] Graph generation failed: {e}")
            continue
        G = nx.convert_node_labels_to_integers(G)
        yield GraphRecord(
            exp_idx=exp_idx,
            graph=G,
            meta={"epsilon": eps_nom, "eps_actual": eps_actual, "graph_type": graph_type},
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GRG embedding phase: TorusMDS vs s_gd2 vs wrap_python (Chen)"
    )
    parser.add_argument("--n-graphs", type=int, default=1000,
                        help="Number of GRG graphs to generate (default: 1000)")
    parser.add_argument("--n-min", type=int, default=100,
                        help="Minimum number of nodes (default: 100)")
    parser.add_argument("--n-max", type=int, default=10000,
                        help="Maximum number of nodes (default: 10000)")
    parser.add_argument("--eps-min", type=float, default=0.10,
                        help="Minimum connection radius epsilon (default: 0.10)")
    parser.add_argument("--eps-max", type=float, default=0.40,
                        help="Maximum connection radius epsilon (default: 0.40)")
    parser.add_argument("--n-eps", type=int, default=6,
                        help="Number of discrete epsilon values in grid (default: 6)")
    parser.add_argument("--graph-type-weights", type=str, default="1,1,1",
                        help="Comma-separated euclidean,toroidal,spherical sampling weights (default: 1,1,1 = equal thirds)")
    parser.add_argument("--torus-max-iters", type=int, default=2000,
                        help="SGD iterations for TorusMDS (default: 2000)")
    parser.add_argument("--wrap-python-max-iters", type=int, default=200,
                        help="Descent iterations for wrap_python -- kept low since its cost is "
                             "O(n^2 * iters); 200 matches the Chen reference's own default (default: 200)")
    parser.add_argument("--wrap-python-max-n", type=int, default=3500,
                        help="Max n for the Chen wrap_python method; skipped above this (default: 3500)")
    parser.add_argument("--output-dir", type=str, default="layouts/grg",
                        help="Directory to persist layouts + run manifest (default: layouts/grg)")
    parser.add_argument("--checkpoint-every", type=int, default=20,
                        help="Save manifest every N graphs (default: 20)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Global random seed (default: 0)")
    args = parser.parse_args()

    weights = np.array([float(w) for w in args.graph_type_weights.split(",")], dtype=float)
    if weights.shape != (3,):
        parser.error("--graph-type-weights must have exactly 3 comma-separated values")
    weights = weights / weights.sum()

    rng = np.random.default_rng(args.seed)
    graph_iterator = grg_graph_iterator(
        n_graphs=args.n_graphs,
        n_min=args.n_min,
        n_max=args.n_max,
        eps_min=args.eps_min,
        eps_max=args.eps_max,
        n_eps=args.n_eps,
        graph_type_weights=tuple(weights),
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
    )
