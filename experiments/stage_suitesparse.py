#!/usr/bin/env python3
"""
Stages the SuiteSparse Matrix Collection locally: downloads candidate square
matrices, symmetrizes their sparsity pattern into an unweighted graph, takes
the largest connected component, and keeps the smallest (by nnz) --limit
graphs whose LCC node count falls in [--n-min, --n-max].

Run this ONCE, locally or on a login node with internet access. SLURM
compute jobs never touch the network -- they only read the cache this script
produces (--output-dir, default data/suitesparse_cache/): one graph_<id>.npz
per graph (same sparse-adjacency format modules.experiment_runner.save_graph
uses) plus a manifest.csv.

Usage:
    python stage_suitesparse.py --limit 2000 --n-min 100 --n-max 10000
    python stage_suitesparse.py --limit 20   # smoke test
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

import networkx as nx
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
import ssgetpy
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.experiment_runner import save_graph


def _load_matrix_pattern(mm_dir: str, name: str) -> sp.csr_matrix:
    """Load a downloaded MatrixMarket bundle's .mtx file as a sparse matrix."""
    mtx_path = os.path.join(mm_dir, f"{name}.mtx")
    if not os.path.exists(mtx_path):
        candidates = [f for f in os.listdir(mm_dir) if f.endswith(".mtx")]
        if not candidates:
            raise FileNotFoundError(f"No .mtx file found in {mm_dir}")
        mtx_path = os.path.join(mm_dir, candidates[0])
    return sp.csr_matrix(sio.mmread(mtx_path))


def to_simple_graph(A: sp.spmatrix) -> nx.Graph:
    """Symmetrized, unweighted, self-loop-free graph from a matrix's sparsity pattern."""
    A = sp.csr_matrix(A)
    A = ((A + A.T) != 0).astype(np.uint8)
    A.setdiag(0)
    A.eliminate_zeros()
    return nx.from_scipy_sparse_array(A)


def stage_suitesparse(
    limit: int,
    n_min: int,
    n_max: int,
    output_dir: str,
    prefilter_limit: int = 20000,
) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.csv")

    already: set[int] = set()
    manifest_rows: list[dict] = []
    if os.path.exists(manifest_path):
        existing = pd.read_csv(manifest_path)
        manifest_rows = existing.to_dict("records")
        already = set(existing["matrix_id"])
        print(f"Resuming: {len(already)} matrices already staged.")

    n_needed = limit - len(already)
    if n_needed <= 0:
        print(f"Already have {len(already)} >= --limit {limit}; nothing to do.")
        return pd.DataFrame(manifest_rows)

    print("Querying SuiteSparse Matrix Collection index...")
    candidates = ssgetpy.search(rowbounds=(n_min, n_max), colbounds=(n_min, n_max), limit=prefilter_limit)
    candidates = [m for m in candidates if m.rows == m.cols and m.id not in already]
    candidates.sort(key=lambda m: m.nnz)
    print(f"{len(candidates)} square candidate matrices in range after prefilter.")

    tmp = tempfile.mkdtemp(prefix="suitesparse_stage_")
    pbar = tqdm(total=n_needed, desc="Staging SuiteSparse graphs")
    try:
        for m in candidates:
            if n_needed <= 0:
                break
            mm_dir, archive_path = None, None
            try:
                mm_dir, archive_path = m.download(format="MM", destpath=tmp, extract=True)
                A = _load_matrix_pattern(mm_dir, m.name)
                G = to_simple_graph(A)
                largest_cc = max(nx.connected_components(G), key=len)
                G = nx.convert_node_labels_to_integers(G.subgraph(largest_cc).copy())
                n = G.number_of_nodes()
                if not (n_min <= n <= n_max):
                    continue

                save_graph(G, os.path.join(output_dir, f"graph_{m.id}.npz"))
                manifest_rows.append({
                    "matrix_id": m.id, "group": m.group, "name": m.name,
                    "raw_rows": m.rows, "raw_nnz": m.nnz,
                    "n": n, "m": G.number_of_edges(),
                })
                n_needed -= 1
                pbar.update(1)
                if len(manifest_rows) % 20 == 0:
                    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
            except Exception as e:
                print(f"[{m.group}/{m.name}] skipped: {e}")
            finally:
                if mm_dir and os.path.exists(mm_dir):
                    shutil.rmtree(mm_dir, ignore_errors=True)
                if archive_path and os.path.exists(archive_path):
                    os.remove(archive_path)
    finally:
        pbar.close()
        shutil.rmtree(tmp, ignore_errors=True)

    df = pd.DataFrame(manifest_rows)
    df.to_csv(manifest_path, index=False)
    print(f"\nStaged {len(df)} graphs -> {output_dir} (manifest: {manifest_path})")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage SuiteSparse Matrix Collection graphs locally")
    parser.add_argument("--limit", type=int, default=2000,
                        help="Number of qualifying graphs to stage (default: 2000)")
    parser.add_argument("--n-min", type=int, default=100,
                        help="Minimum LCC node count (default: 100)")
    parser.add_argument("--n-max", type=int, default=10000,
                        help="Maximum LCC node count (default: 10000)")
    parser.add_argument("--output-dir", type=str, default="data/suitesparse_cache",
                        help="Local cache directory (default: data/suitesparse_cache)")
    parser.add_argument("--prefilter-limit", type=int, default=20000,
                        help="Max candidates to query from the SuiteSparse index before downloading (default: 20000)")
    args = parser.parse_args()

    stage_suitesparse(
        limit=args.limit,
        n_min=args.n_min,
        n_max=args.n_max,
        output_dir=args.output_dir,
        prefilter_limit=args.prefilter_limit,
    )
