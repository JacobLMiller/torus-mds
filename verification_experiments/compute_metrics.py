#!/usr/bin/env python3
"""
Metrics phase: reads layouts persisted by a driver script's embedding phase
(sbm_comparison.py / grg_comparison.py / suitesparse_comparison.py) and
computes the evaluation metrics, writing a results CSV with the same schema
the old, all-in-one scripts used.

Deliberately decoupled from embedding (see modules/experiment_runner.py):
this only reads graphs/*.npz + coords/*.npy + runs.csv, so it can be re-run
any time metrics.py changes without re-doing the (often much more expensive)
embedding step, and can run anywhere -- it has no dependency on s_gd2's
native extension being present, no numba JIT of the embedding methods, etc.

Usage:
    python compute_metrics.py --layouts-dir layouts/sbm/shard_0 \
        --output results/sbm_shard_0_comparison.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.experiment_runner import coords_path, graph_path, load_graph
from modules.geometry import euc_distance, torus_distance
from modules.graphio import apsp_distance_matrix
from modules.metrics import (
    SGS,
    estimate_alpha,
    geodesic_distortion,
    geodesic_NP,
    geodesic_stress,
    pointwise_geodesic_stress,
)

# Manifest columns that are embedding-phase bookkeeping, not result metadata.
_MANIFEST_ONLY_COLUMNS = {"status", "alpha_fit", "seed"}


def _subsample(X: np.ndarray, D: np.ndarray, max_n: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly subsample to at most max_n nodes, keeping X and D in sync.

    Same fixed seed regardless of method, so every method for a given graph
    (same n) gets the identical subsample -- required for an apples-to-apples
    comparison across methods.
    """
    if X.shape[0] <= max_n:
        return X, D
    rng_sub = np.random.default_rng(seed)
    idx = rng_sub.choice(X.shape[0], max_n, replace=False)
    return X[idx], D[np.ix_(idx, idx)]


def compute_metrics(X: np.ndarray, D: np.ndarray, geod, rg: float = 2.0) -> dict:
    """Compute all four metrics from metrics.py on the given (already max_n-sized) layout/distances."""
    return dict(
        stress=geodesic_stress(X, D, geod),
        pointwise_stress=pointwise_geodesic_stress(X, D, geod),
        distortion=geodesic_distortion(X, D, geod),
        sgs=SGS(X, D, geod),
        np_score=geodesic_NP(X, D, geod, rg=rg),
    )


def _geod_for(method: str, X: np.ndarray, D: np.ndarray):
    # Re-estimate scale on the same (already subsampled) points used for the
    # metrics below, not the full graph -- estimate_alpha is an unvectorized
    # O(k^2) Python loop (like every other metric here), and at n=10,000
    # that's ~50M Python-level calls if run on the full graph instead of the
    # subsample. Fitting alpha for s_gd2/Euclidean too (not just the torus
    # methods) keeps stress comparable across methods -- an unscaled
    # Euclidean embedding's raw distances are an arbitrary size, so without
    # this the stress numbers aren't on the same footing.
    if method == "s_gd2":
        alpha = estimate_alpha(X, D, geod=euc_distance)
        return (lambda p, q, a=alpha: a * euc_distance(p, q)), alpha
    alpha = estimate_alpha(X, D)
    return (lambda p, q, a=alpha: a * torus_distance(p, q)), alpha


def run_metrics(
    layouts_dir: str,
    output_csv: str,
    metric_subsample: int = 100,
    np_radius: float = 2.0,
    checkpoint_every: int = 50,
) -> pd.DataFrame:
    manifest_path = os.path.join(layouts_dir, "runs.csv")
    manifest = pd.read_csv(manifest_path)
    ok = manifest[manifest["status"] == "ok"].copy()

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    records: list[dict] = []
    done: set[tuple[int, str]] = set()
    if os.path.exists(output_csv):
        try:
            existing_df = pd.read_csv(output_csv)
            records = existing_df.to_dict("records")
            done = {(int(r["exp_idx"]), r["method"]) for r in records}
            print(f"Resuming: {len(done)} rows already scored.")
        except Exception:
            pass

    def checkpoint() -> None:
        pd.DataFrame(records).to_csv(output_csv, index=False)

    since_checkpoint = 0
    # Group by exp_idx so the (possibly expensive-ish) graph load + APSP is
    # done once per graph, not once per method.
    for exp_idx, group in tqdm(ok.groupby("exp_idx"), desc="Scoring graphs"):
        rows_needed = [r for _, r in group.iterrows() if (int(exp_idx), r["method"]) not in done]
        if not rows_needed:
            continue

        try:
            G = load_graph(graph_path(layouts_dir, int(exp_idx)))
            t0 = time.perf_counter()
            D, _ = apsp_distance_matrix(G)
            t_spd = time.perf_counter() - t0
        except Exception:
            print(f"[{exp_idx}] failed to load graph / compute APSP:\n{traceback.format_exc(limit=2)}")
            continue

        for row in rows_needed:
            method = row["method"]
            meta = {k: v for k, v in row.items() if k not in _MANIFEST_ONLY_COLUMNS}
            try:
                X = np.load(coords_path(layouts_dir, int(exp_idx), method))
                X_s, D_s = _subsample(X, D, metric_subsample)
                geod, alpha = _geod_for(method, X_s, D_s)
                metrics = compute_metrics(X_s, D_s, geod, rg=np_radius)
                records.append({**meta, "t_spd": round(t_spd, 4), **metrics, "alpha": round(alpha, 6) if np.isfinite(alpha) else alpha})
                done.add((int(exp_idx), method))
            except Exception:
                print(f"[{exp_idx}] {method} scoring failed:\n{traceback.format_exc(limit=2)}")

        since_checkpoint += 1
        if since_checkpoint >= checkpoint_every:
            checkpoint()
            since_checkpoint = 0

    checkpoint()
    df = pd.DataFrame(records)
    print(f"\nSaved {len(df)} scored rows -> {output_csv}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute metrics from persisted embedding layouts")
    parser.add_argument("--layouts-dir", type=str, required=True,
                        help="Directory written by the embedding phase (contains runs.csv, graphs/, coords/)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output results CSV path")
    parser.add_argument("--metric-subsample", type=int, default=100,
                        help="Max nodes used for metric computation (default: 100)")
    parser.add_argument("--np-radius", type=float, default=2.0,
                        help="Graph-distance radius for neighbourhood preservation (default: 2.0)")
    parser.add_argument("--checkpoint-every", type=int, default=50,
                        help="Save CSV every N graphs (default: 50)")
    args = parser.parse_args()

    run_metrics(
        layouts_dir=args.layouts_dir,
        output_csv=args.output,
        metric_subsample=args.metric_subsample,
        np_radius=args.np_radius,
        checkpoint_every=args.checkpoint_every,
    )
