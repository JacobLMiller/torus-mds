#!/usr/bin/env python3
"""
Sampler comparison for MDSTorusProjector on geometric random graphs.

Runs the same GRG instances through multiple pair-sampling strategies and records
stress, distortion, Spearman rank correlation, neighbourhood preservation,
wall-clock time, and gradient evaluations per epoch.

Sampling strategies
-------------------
Pair-sampling (uniform over all C(n,2) pairs, varying batch size):
  pair_full        -- min(batch_size, C(n,2)) pairs/epoch  [baseline]
  pair_frac10      -- 10% of C(n,2) pairs/epoch
  pair_sqrt_pairs  -- sqrt(C(n,2)) pairs/epoch
  pair_sqrt_n      -- sqrt(n) pairs/epoch

Vertex-centric (k partners sampled per node per epoch):
  vertex_k1        -- 1 partner/node  → n evals/epoch
  vertex_k10       -- 10 partners/node → 10n evals/epoch
  vertex_kN        -- n-1 partners/node (full pass) → n(n-1) evals/epoch

Node-sampling (sample k nodes, pair into k//2 disjoint pairs per epoch):
  node_k_frac05    -- k = 5%  of n → k//2 pairs/epoch
  node_k_frac10    -- k = 10% of n → k//2 pairs/epoch  [default k]
  node_k_frac25    -- k = 25% of n → k//2 pairs/epoch
  node_k_frac50    -- k = 50% of n → k//2 pairs/epoch

Usage
-----
    python sampler_comparison.py
    python sampler_comparison.py --n-graphs 500 --n-min 50 --n-max 300
    python sampler_comparison.py --output results/sampler_comparison.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.geometry import torus_distance
from modules.metrics import (
    SGS,
    estimate_alpha,
    geodesic_distortion,
    geodesic_NP,
    geodesic_stress,
)
from modules.projector import MDSTorusProjector


# ---------------------------------------------------------------------------
# Graph generation  (identical to grg_comparison.py)
# ---------------------------------------------------------------------------

def _torus_cdist(pts: np.ndarray) -> np.ndarray:
    diff = np.abs(pts[:, None, :] - pts[None, :, :])
    diff = np.minimum(diff, 1.0 - diff)
    return np.sqrt((diff ** 2).sum(axis=-1))


def generate_grg(
    n: int,
    epsilon: float,
    rng: np.random.Generator,
    max_retries: int = 20,
) -> tuple[nx.Graph, np.ndarray, float]:
    for attempt in range(max_retries):
        eps = epsilon * (1.0 + 0.10 * attempt)
        pts = rng.uniform(0.0, 1.0, size=(n, 2))
        adj = cdist(pts, pts) <= eps
        np.fill_diagonal(adj, False)
        G = nx.from_numpy_array(adj.astype(np.uint8))
        if nx.is_connected(G):
            return G, pts, round(eps, 6)
    pts = rng.uniform(0.0, 1.0, size=(n, 2))
    adj = cdist(pts, pts) <= epsilon
    np.fill_diagonal(adj, False)
    G = nx.from_numpy_array(adj.astype(np.uint8))
    comps = list(nx.connected_components(G))
    for i in range(len(comps) - 1):
        G.add_edge(next(iter(comps[i])), next(iter(comps[i + 1])))
    return G, pts, round(epsilon, 6)


def generate_torus_grg(
    n: int,
    epsilon: float,
    rng: np.random.Generator,
    max_retries: int = 20,
) -> tuple[nx.Graph, np.ndarray, float]:
    for attempt in range(max_retries):
        eps = epsilon * (1.0 + 0.10 * attempt)
        pts = rng.uniform(0.0, 1.0, size=(n, 2))
        adj = _torus_cdist(pts) <= eps
        np.fill_diagonal(adj, False)
        G = nx.from_numpy_array(adj.astype(np.uint8))
        if nx.is_connected(G):
            return G, pts, round(eps, 6)
    pts = rng.uniform(0.0, 1.0, size=(n, 2))
    adj = _torus_cdist(pts) <= epsilon
    np.fill_diagonal(adj, False)
    G = nx.from_numpy_array(adj.astype(np.uint8))
    comps = list(nx.connected_components(G))
    for i in range(len(comps) - 1):
        G.add_edge(next(iter(comps[i])), next(iter(comps[i + 1])))
    return G, pts, round(epsilon, 6)


# ---------------------------------------------------------------------------
# SPD matrix
# ---------------------------------------------------------------------------

def spd_matrix(G: nx.Graph) -> np.ndarray:
    nodes = sorted(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    D = np.zeros((n, n), dtype=np.float64)
    for u, dists in nx.all_pairs_shortest_path_length(G):
        i = idx[u]
        for v, d in dists.items():
            D[i, idx[v]] = d
    return D


# ---------------------------------------------------------------------------
# Embedding + metrics
# ---------------------------------------------------------------------------

def embed_torus_mds(
    D: np.ndarray,
    vertex_k: int,
    node_k: int,
    batch_fraction,
    max_iters: int,
    batch_size: int,
    tol: float,
    patience: int,
    seed: int,
) -> tuple[np.ndarray, float, int]:
    proj = MDSTorusProjector(projection="wrap")
    proj.fit_transform(
        D,
        max_iters=max_iters,
        batch_size=batch_size,
        seed=seed,
        tol=tol,
        patience=patience,
        vertex_k=vertex_k,
        node_k=node_k,
        batch_fraction=batch_fraction,
    )
    return proj.torus_embedding_, proj.alpha_, proj.actual_epochs_


def compute_metrics(
    X: np.ndarray,
    D: np.ndarray,
    alpha: float,
    max_n: int,
    np_radius: float,
) -> dict:
    if X.shape[0] > max_n:
        sub_rng = np.random.default_rng(42)
        idx = sub_rng.choice(X.shape[0], max_n, replace=False)
        X, D = X[idx], D[np.ix_(idx, idx)]

    geod = lambda p, q, a=alpha: a * torus_distance(p, q)
    return dict(
        stress=geodesic_stress(X, D, geod),
        distortion=geodesic_distortion(X, D, geod),
        sgs=SGS(X, D, geod),
        np_score=geodesic_NP(X, D, geod, rg=np_radius),
    )


# ---------------------------------------------------------------------------
# Sampler configurations
# ---------------------------------------------------------------------------

def build_sampler_configs(n: int, batch_size: int) -> list[dict]:
    """Return all sampler configs to compare for a graph of size n."""
    # node_k values as fractions of n; must be >= 2 and even
    def node_k_for(frac: float) -> int:
        k = max(2, int(frac * n))
        return k - (k % 2)

    return [
        # --- Pair-sampling variants ---
        dict(label="pair_full",       vertex_k=0, node_k=0, batch_fraction=None),
        dict(label="pair_frac10",     vertex_k=0, node_k=0, batch_fraction=0.1),
        dict(label="pair_sqrt_pairs", vertex_k=0, node_k=0, batch_fraction="sqrt_pairs"),
        dict(label="pair_sqrt_n",     vertex_k=0, node_k=0, batch_fraction="sqrt_n"),
        # --- Vertex-centric variants ---
        dict(label="vertex_k1",  vertex_k=1,   node_k=0, batch_fraction=None),
        dict(label="vertex_k10", vertex_k=10,  node_k=0, batch_fraction=None),
        dict(label="vertex_kN",  vertex_k=n-1, node_k=0, batch_fraction=None),
        # --- Node-sampling variants ---
        dict(label="node_k_frac05", vertex_k=0, node_k=node_k_for(0.05), batch_fraction=None),
        dict(label="node_k_frac10", vertex_k=0, node_k=node_k_for(0.10), batch_fraction=None),
        dict(label="node_k_frac25", vertex_k=0, node_k=node_k_for(0.25), batch_fraction=None),
        dict(label="node_k_frac50", vertex_k=0, node_k=node_k_for(0.50), batch_fraction=None),
    ]


def evals_per_epoch(
    n: int,
    vertex_k: int,
    node_k: int,
    batch_fraction,
    batch_size: int,
) -> int:
    """Number of pair evaluations performed per epoch for a given config."""
    if vertex_k >= 1:
        return n * min(vertex_k, n - 1)
    if node_k >= 2:
        k = min(node_k, n)
        k -= k % 2
        return k // 2
    # pair-sampling branch — mirror _resolve_batch_pairs logic
    num_pairs = n * (n - 1) // 2
    if batch_fraction is None:
        return min(batch_size, num_pairs)
    if isinstance(batch_fraction, float):
        return max(1, int(batch_fraction * num_pairs))
    if batch_fraction == "sqrt_n":
        return max(1, int(np.sqrt(n)))
    if batch_fraction == "sqrt_pairs":
        return max(1, int(np.sqrt(num_pairs)))
    return min(batch_size, num_pairs)


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(
    n_graphs: int = 500,
    n_min: int = 50,
    n_max: int = 300,
    eps_min: float = 0.10,
    eps_max: float = 0.40,
    n_eps: int = 6,
    max_iters: int = 5000,
    batch_size: int = 4096,
    tol: float = 1e-4,
    patience: int = 5,
    metric_subsample: int = 100,
    np_radius: float = 2.0,
    output_csv: str = "results/sampler_comparison.csv",
    checkpoint_every: int = 50,
    seed: int = 0,
) -> pd.DataFrame:

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    rng = np.random.default_rng(seed)
    eps_grid = np.linspace(eps_min, eps_max, n_eps).round(4).tolist()

    # Resume from checkpoint
    existing: list[dict] = []
    start_idx = 0
    if os.path.exists(output_csv):
        try:
            existing_df = pd.read_csv(output_csv)
            existing = existing_df.to_dict("records")
            completed = set(existing_df["exp_idx"].unique())
            start_idx = max(completed) + 1 if completed else 0
            print(f"Resuming from exp_idx={start_idx} ({len(existing)} records loaded).")
        except Exception:
            pass

    records: list[dict] = list(existing)

    for exp_idx in tqdm(range(start_idx, n_graphs), desc="sampler experiments"):
        n = int(np.exp(rng.uniform(np.log(n_min), np.log(n_max))))
        eps_nom = float(rng.choice(eps_grid))
        graph_type = "toroidal" if rng.random() < 0.5 else "euclidean"
        generator = generate_torus_grg if graph_type == "toroidal" else generate_grg

        try:
            G, pts, eps_actual = generator(n, eps_nom, rng)
            G = nx.convert_node_labels_to_integers(G)
        except Exception as e:
            print(f"[{exp_idx}] graph generation failed: {e}")
            continue

        n_actual = len(G)

        t0 = time.perf_counter()
        D = spd_matrix(G)
        t_spd = time.perf_counter() - t0

        base = dict(
            exp_idx=exp_idx,
            n=n_actual,
            epsilon=eps_nom,
            eps_actual=eps_actual,
            graph_type=graph_type,
            t_spd=round(t_spd, 4),
        )

        embed_seed = int(rng.integers(0, 2**31))

        for cfg in build_sampler_configs(n_actual, batch_size):
            label          = cfg["label"]
            vertex_k       = cfg["vertex_k"]
            node_k         = cfg["node_k"]
            batch_fraction = cfg["batch_fraction"]
            n_evals        = evals_per_epoch(n_actual, vertex_k, node_k, batch_fraction, batch_size)

            try:
                t0 = time.perf_counter()
                X, alpha_fit, actual_epochs = embed_torus_mds(
                    D,
                    vertex_k=vertex_k,
                    node_k=node_k,
                    batch_fraction=batch_fraction,
                    max_iters=max_iters,
                    batch_size=batch_size,
                    tol=tol,
                    patience=patience,
                    seed=embed_seed,
                )
                t_embed = time.perf_counter() - t0

                alpha = estimate_alpha(X, D)
                m = compute_metrics(X, D, alpha,
                                    max_n=metric_subsample, np_radius=np_radius)

                records.append({
                    **base,
                    "method":          label,
                    "vertex_k":        vertex_k if vertex_k >= 1 else (f"node_{node_k}" if node_k >= 2 else "pair"),
                    "evals_per_epoch": n_evals,
                    "actual_epochs":   actual_epochs,
                    "t_embed":         round(t_embed, 4),
                    "alpha":           round(alpha, 6),
                    **m,
                })

            except Exception:
                print(f"[{exp_idx}] {label} failed:\n{traceback.format_exc(limit=2)}")

        if (exp_idx + 1) % checkpoint_every == 0:
            pd.DataFrame(records).to_csv(output_csv, index=False)

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(df)} records ({n_graphs} experiments) → {output_csv}")
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MDSTorusProjector sampler comparison on GRG graphs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-graphs",        type=int,   default=500,
                        help="Number of GRG graph instances")
    parser.add_argument("--n-min",           type=int,   default=50,
                        help="Minimum graph size (nodes)")
    parser.add_argument("--n-max",           type=int,   default=300,
                        help="Maximum graph size (nodes)")
    parser.add_argument("--eps-min",         type=float, default=0.10,
                        help="Minimum connection radius")
    parser.add_argument("--eps-max",         type=float, default=0.40,
                        help="Maximum connection radius")
    parser.add_argument("--n-eps",           type=int,   default=6,
                        help="Number of discrete epsilon values")
    parser.add_argument("--max-iters",       type=int,   default=5000,
                        help="Max SGD epochs per embedding")
    parser.add_argument("--batch-size",      type=int,   default=4096,
                        help="Pair batch size for pair_full strategy")
    parser.add_argument("--tol",             type=float, default=1e-4,
                        help="Early-stopping relative stress tolerance")
    parser.add_argument("--patience",        type=int,   default=5,
                        help="Early-stopping patience (chunks of 128 epochs)")
    parser.add_argument("--metric-subsample",type=int,   default=100,
                        help="Max nodes used for metric computation")
    parser.add_argument("--np-radius",       type=float, default=2.0,
                        help="Graph-distance radius for neighbourhood preservation")
    parser.add_argument("--output",          type=str,
                        default="results/sampler_comparison.csv",
                        help="Output CSV path")
    parser.add_argument("--checkpoint-every",type=int,   default=50,
                        help="Save CSV every N experiments")
    parser.add_argument("--seed",            type=int,   default=0,
                        help="Global random seed")
    args = parser.parse_args()

    run_experiment(
        n_graphs=args.n_graphs,
        n_min=args.n_min,
        n_max=args.n_max,
        eps_min=args.eps_min,
        eps_max=args.eps_max,
        n_eps=args.n_eps,
        max_iters=args.max_iters,
        batch_size=args.batch_size,
        tol=args.tol,
        patience=args.patience,
        metric_subsample=args.metric_subsample,
        np_radius=args.np_radius,
        output_csv=args.output,
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
    )
