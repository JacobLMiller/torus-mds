#!/usr/bin/env python3
"""
Experimental comparison: Euclidean MDS vs. MDS by SGD vs. Torus-MDS
on 1000 stochastic block model graphs.

Usage:
    python sbm_comparison.py [--n-graphs N] [--seed S] [--output results/sbm_comparison.csv]
    python sbm_comparison.py --torus-max-iters 1000 
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
import s_gd2
from sklearn.manifold import MDS as SklearnMDS
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.geometry import euc_distance, torus_distance
from modules.metrics import (
    SGS,
    estimate_alpha,
    geodesic_distortion,
    geodesic_NP,
    geodesic_stress,
)
from modules.projector import MDSTorusProjector
from standalone_toruslayout import wrap_python


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


# ---------------------------------------------------------------------------
# Distance matrix
# ---------------------------------------------------------------------------

def spd_matrix(G: nx.Graph) -> np.ndarray:
    """Shortest-path distance matrix (hop distances)."""
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
# Embedding methods
# ---------------------------------------------------------------------------

def embed_mds(D: np.ndarray, random_state: int = 42) -> np.ndarray:
    mds = SklearnMDS(
        n_components=2,
        metric=True,
        dissimilarity="precomputed",
        random_state=random_state,
        init="random",
        n_init=1,
        max_iter=300,
        normalized_stress=False,
    )
    return mds.fit_transform(D)


def embed_sgd2(G: nx.Graph, random_seed: int = 42) -> np.ndarray:
    """
    s_gd2 graph layout. 
    """
    nodes = sorted(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    edges = list(G.edges())
    I = np.array([idx[u] for u, _ in edges], dtype=np.int32)
    J = np.array([idx[v] for _, v in edges], dtype=np.int32)
    return s_gd2.layout(I, J, random_seed=random_seed)


def embed_torus_mds(
    D: np.ndarray,
    max_iters: int = 2000,
    seed: int = 42,
) -> tuple[np.ndarray, float, MDSTorusProjector]:
    proj = MDSTorusProjector(projection="robust_wrap")
    X = proj.fit_transform(D, max_iters=max_iters, seed=seed)
    return X, proj.alpha_, proj


def embed_wrap_python(
    G: nx.Graph,
    D: np.ndarray,
    max_iters: int = 2000,
    seed: int = 42,
) -> np.ndarray:
    result = wrap_python(
        G,
        seed=seed,
        max_iters=max_iters,
        shortest_path_lengths=D,
    )
    return result.positions


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    X: np.ndarray,
    D: np.ndarray,
    geod,
    max_n: int = 100,
    rg: float = 2.0,
) -> dict:
    """
    Compute all four metrics from metrics.py, subsampling to max_n nodes if needed.
    """
    if X.shape[0] > max_n:
        rng_sub = np.random.default_rng(42)
        idx = rng_sub.choice(X.shape[0], max_n, replace=False)
        X_s = X[idx]
        D_s = D[np.ix_(idx, idx)]
    else:
        X_s, D_s = X, D

    return dict(
        stress=geodesic_stress(X_s, D_s, geod),
        distortion=geodesic_distortion(X_s, D_s, geod),
        sgs=SGS(X_s, D_s, geod),
        np_score=geodesic_NP(X_s, D_s, geod, rg=rg),
    )


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(
    n_graphs: int = 1000,
    n_min: int = 100,
    n_max: int = 1000,
    k_min: int = 3,
    k_max: int = 8,
    torus_max_iters: int = 2000,
    metric_subsample: int = 100,
    np_radius: float = 2.0,
    output_csv: str = "results/sbm_comparison.csv",
    checkpoint_every: int = 50,
    seed: int = 0,
) -> pd.DataFrame:

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    rng = np.random.default_rng(seed)

    # Resume from checkpoint if CSV already exists
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

    for exp_idx in tqdm(range(start_idx, n_graphs), desc="SBM experiments"):
        # Sample graph size log-uniformly and block count uniformly
        n = int(np.exp(rng.uniform(np.log(n_min), np.log(n_max))))
        k = int(rng.integers(k_min, k_max + 1))

        try:
            G, block_sizes = generate_sbm(n, k, rng)
        except Exception as e:
            print(f"[{exp_idx}] Graph generation failed: {e}")
            continue

        n_actual = len(G)

        t0 = time.perf_counter()
        D = spd_matrix(G)
        t_spd = time.perf_counter() - t0

        base = dict(
            exp_idx=exp_idx,
            n=n_actual,
            k=k,
            t_spd=round(t_spd, 4),
        )

        # ---- Classical MDS ----
        try:
            t0 = time.perf_counter()
            X_mds = embed_mds(D)
            t_embed = time.perf_counter() - t0

            metrics = compute_metrics(
                X_mds, D, euc_distance, max_n=metric_subsample, rg=np_radius
            )
            records.append({**base, "method": "MDS", "t_embed": round(t_embed, 4),
                             **metrics, "alpha": float("nan")})
        except Exception:
            print(f"[{exp_idx}] MDS failed:\n{traceback.format_exc(limit=2)}")

        # ---- s_gd2 ----
        try:
            t0 = time.perf_counter()
            X_sgd2 = embed_sgd2(G)
            t_embed = time.perf_counter() - t0

            metrics = compute_metrics(
                X_sgd2, D, euc_distance, max_n=metric_subsample, rg=np_radius
            )
            records.append({**base, "method": "s_gd2", "t_embed": round(t_embed, 4),
                             **metrics, "alpha": float("nan")})
        except Exception:
            print(f"[{exp_idx}] s_gd2 failed:\n{traceback.format_exc(limit=2)}")

        # ---- standalone_toruslayout.wrap_python ----
        try:
            t0 = time.perf_counter()
            X_wrap_python = embed_wrap_python(G, D, max_iters=torus_max_iters)
            t_embed = time.perf_counter() - t0

            alpha = estimate_alpha(X_wrap_python, D)
            geod_torus = lambda p, q, a=alpha: a * torus_distance(p, q)

            metrics = compute_metrics(
                X_wrap_python, D, geod_torus, max_n=metric_subsample, rg=np_radius
            )
            records.append(
                {
                    **base,
                    "method": "wrap_python",
                    "t_embed": round(t_embed, 4),
                    **metrics,
                    "alpha": round(alpha, 6),
                }
            )
        except Exception:
            print(f"[{exp_idx}] wrap_python failed:\n{traceback.format_exc(limit=2)}")

        # ---- Toroidal MDS ----
        try:
            t0 = time.perf_counter()
            X_torus, alpha_fit, _ = embed_torus_mds(D, max_iters=torus_max_iters)
            t_embed = time.perf_counter() - t0

            # Re-estimate scale on the projected embedding for accurate metric evaluation
            alpha = estimate_alpha(X_torus, D)
            geod_torus = lambda p, q, a=alpha: a * torus_distance(p, q)

            metrics = compute_metrics(
                X_torus, D, geod_torus, max_n=metric_subsample, rg=np_radius
            )
            records.append({**base, "method": "TorusMDS", "t_embed": round(t_embed, 4),
                             **metrics, "alpha": round(alpha, 6)})
        except Exception:
            print(f"[{exp_idx}] TorusMDS failed:\n{traceback.format_exc(limit=2)}")

        # Checkpoint
        if (exp_idx + 1) % checkpoint_every == 0:
            pd.DataFrame(records).to_csv(output_csv, index=False)

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(df)} records ({n_graphs} experiments) -> {output_csv}")
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SBM embedding comparison: MDS vs s_gd2 vs TorusMDS"
    )
    parser.add_argument("--n-graphs", type=int, default=1000,
                        help="Number of SBM graphs to generate (default: 1000)")
    parser.add_argument("--n-min", type=int, default=100,
                        help="Minimum number of nodes (default: 100)")
    parser.add_argument("--n-max", type=int, default=1000,
                        help="Maximum number of nodes (default: 1000)")
    parser.add_argument("--k-min", type=int, default=3,
                        help="Minimum number of SBM blocks (default: 3)")
    parser.add_argument("--k-max", type=int, default=8,
                        help="Maximum number of SBM blocks (default: 8)")
    parser.add_argument("--torus-max-iters", type=int, default=2000,
                        help="SGD iterations for TorusMDS (default: 2000)")
    parser.add_argument("--metric-subsample", type=int, default=100,
                        help="Max nodes used for metric computation (default: 100)")
    parser.add_argument("--np-radius", type=float, default=2.0,
                        help="Graph-distance radius for neighbourhood preservation (default: 2.0)")
    parser.add_argument("--output", type=str, default="results/sbm_comparison.csv",
                        help="Output CSV path (default: results/sbm_comparison.csv)")
    parser.add_argument("--checkpoint-every", type=int, default=50,
                        help="Save CSV every N experiments (default: 50)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Global random seed (default: 0)")
    args = parser.parse_args()

    run_experiment(
        n_graphs=args.n_graphs,
        n_min=args.n_min,
        n_max=args.n_max,
        k_min=args.k_min,
        k_max=args.k_max,
        torus_max_iters=args.torus_max_iters,
        metric_subsample=args.metric_subsample,
        np_radius=args.np_radius,
        output_csv=args.output,
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
    )
