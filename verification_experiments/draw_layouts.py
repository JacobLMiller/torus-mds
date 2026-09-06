#!/usr/bin/env python3
"""
Renders a PNG for every persisted (graph, method) layout under a layouts/
tree (as written by run_embeddings -- see modules/experiment_runner.py),
using the drawing routines in modules/visualization.

Walks --layouts-root for shard directories (any directory containing a
runs.csv, same convention run_metrics_array.sbatch uses), and for each
status=="ok" row loads the graph + coords it points at and renders one PNG,
mirroring the shard's path under --output-root so files from different
shards/tiers/families never collide.

`s_gd2` is the one method whose output isn't confined to the periodic
[0,1)^2 torus domain (it's a plain force-directed layout in an arbitrary
scale/offset) -- it's min-max normalized to [0,1]^2 and drawn with straight
(non-wrapped) edges via plot_embedding. Every other method (TorusMDS and all
its aspect-init variants, wrap_python/wrap_typescript/wrap_python_newdist)
already lives in [0,1)^2 and is drawn wrap-aware via
plot_embedding_with_torus_edges, using alpha_fit/r0_fit/r1_fit from that
row when present (NaN r0/r1 -- e.g. plain TorusMDS's fixed unit-square torus,
or wrap_python which fits neither -- default to 1.0).

This can produce a LOT of files (shards x graphs x methods) -- use
--methods/--max-graphs-per-shard to scope a run down before doing the full
tree. Resumable: existing PNGs are skipped unless --overwrite is passed.

Usage:
    python draw_layouts.py --layouts-root layouts --output-root layout_drawings
    python draw_layouts.py --methods TorusMDS s_gd2 --max-graphs-per-shard 5
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import Counter
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.experiment_runner import coords_path, graph_path, load_graph
from modules.visualization import plot_embedding, plot_embedding_with_torus_edges

NON_PERIODIC_METHODS = {"s_gd2"}


def find_shard_dirs(layouts_root: str) -> list[str]:
    pattern = os.path.join(layouts_root, "**", "runs.csv")
    return sorted(os.path.dirname(p) for p in glob.glob(pattern, recursive=True))


def draw_one(G, X: np.ndarray, method: str, row: pd.Series, outpath: str, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    try:
        if method in NON_PERIODIC_METHODS:
            lo, hi = X.min(axis=0), X.max(axis=0)
            span = np.where(hi > lo, hi - lo, 1.0)
            X_norm = (X - lo) / span
            plot_embedding(X_norm, G, ax=ax)
        else:
            alpha = row.get("alpha_fit")
            r0 = row.get("r0_fit")
            r1 = row.get("r1_fit")
            torus = SimpleNamespace(
                torus_embedding_=None,
                alpha_=float(alpha) if pd.notna(alpha) else 1.0,
                r0_=float(r0) if pd.notna(r0) else 1.0,
                r1_=float(r1) if pd.notna(r1) else 1.0,
                theta_=np.pi / 2,
            )
            plot_embedding_with_torus_edges(X=X, G=G, torus=torus, ax=ax)
        ax.set_title(f"exp_idx={row['exp_idx']} n={G.number_of_nodes()} method={method}", fontsize=9)
        fig.tight_layout()
        fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)


def draw_shard(
    shard_dir: str,
    layouts_root: str,
    output_root: str,
    methods: set[str] | None,
    max_graphs: int | None,
    dpi: int,
    overwrite: bool,
) -> tuple[int, int, Counter, set[int]]:
    runs_path = os.path.join(shard_dir, "runs.csv")
    try:
        all_runs = pd.read_csv(runs_path)
    except Exception as e:
        print(f"[{shard_dir}] failed to read runs.csv: {e}")
        return 0, 0, Counter(), set()

    # (exp_idx, method) pairs whose embedding never produced coords to draw --
    # counted from the full, unfiltered runs.csv so this reflects the whole
    # experiment's completeness, not just whatever --methods/--max-graphs-
    # per-shard narrows this particular drawing pass down to.
    unfinished = all_runs[all_runs["status"] != "ok"]
    status_counts = Counter(unfinished["status"].fillna("unknown"))
    unfinished_exp_idx = set(unfinished["exp_idx"].unique().tolist())

    runs = all_runs[all_runs["status"] == "ok"]
    if methods is not None:
        runs = runs[runs["method"].isin(methods)]
    if runs.empty:
        return 0, 0, status_counts, unfinished_exp_idx

    if max_graphs is not None:
        keep_idx = sorted(runs["exp_idx"].unique())[:max_graphs]
        runs = runs[runs["exp_idx"].isin(keep_idx)]

    rel = os.path.relpath(shard_dir, layouts_root)
    out_dir = os.path.join(output_root, rel)
    os.makedirs(out_dir, exist_ok=True)

    graph_cache: dict[int, object] = {}
    n_ok, n_failed = 0, 0
    for _, row in runs.iterrows():
        exp_idx, method = int(row["exp_idx"]), row["method"]
        outpath = os.path.join(out_dir, f"exp_{exp_idx}_{method}.png")
        if os.path.exists(outpath) and not overwrite:
            continue
        try:
            if exp_idx not in graph_cache:
                graph_cache[exp_idx] = load_graph(graph_path(shard_dir, exp_idx))
            G = graph_cache[exp_idx]
            X = np.load(coords_path(shard_dir, exp_idx, method))
            draw_one(G, X, method, row, outpath, dpi)
            n_ok += 1
        except Exception as e:
            print(f"[{shard_dir}] exp_idx={exp_idx} method={method} failed: {e}")
            n_failed += 1
    return n_ok, n_failed, status_counts, unfinished_exp_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render PNGs for every persisted layout under a layouts/ tree")
    parser.add_argument("--layouts-root", type=str, default="layouts",
                        help="Root directory to search for shard dirs (default: layouts)")
    parser.add_argument("--output-root", type=str, default="layout_drawings",
                        help="Root directory to mirror shard structure into (default: layout_drawings)")
    parser.add_argument("--methods", type=str, nargs="+", default=None,
                        help="Only draw these methods (default: all methods present in each runs.csv)")
    parser.add_argument("--max-graphs-per-shard", type=int, default=None,
                        help="Cap on distinct exp_idx values drawn per shard (default: no cap)")
    parser.add_argument("--dpi", type=int, default=110, help="PNG resolution (default: 110)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Redraw PNGs that already exist (default: skip them)")
    parser.add_argument("--only-shard-dir", type=str, default=None,
                        help="Process just this one shard directory instead of discovering every "
                             "shard under --layouts-root -- for splitting the work across a SLURM "
                             "array (see slurm/draw_layouts_array.sbatch), one task per shard.")
    args = parser.parse_args()

    methods = set(args.methods) if args.methods else None
    if args.only_shard_dir:
        shard_dirs = [args.only_shard_dir]
    else:
        shard_dirs = find_shard_dirs(args.layouts_root)
    if not shard_dirs:
        print(f"No runs.csv found anywhere under {args.layouts_root}")
        sys.exit(1)

    total_ok, total_failed = 0, 0
    total_status_counts: Counter = Counter()
    total_unfinished_graphs = 0
    for i, shard_dir in enumerate(shard_dirs):
        n_ok, n_failed, status_counts, unfinished_exp_idx = draw_shard(
            shard_dir, args.layouts_root, args.output_root,
            methods, args.max_graphs_per_shard, args.dpi, args.overwrite,
        )
        total_ok += n_ok
        total_failed += n_failed
        total_status_counts += status_counts
        total_unfinished_graphs += len(unfinished_exp_idx)
        n_unfinished_rows = sum(status_counts.values())
        print(f"[{i + 1}/{len(shard_dirs)}] {shard_dir}: drew {n_ok} ({n_failed} failed to draw, "
              f"{n_unfinished_rows} rows never finished embedding)")

    print(f"\nDone: {total_ok} PNGs written under {args.output_root} ({total_failed} failed to draw)")
    n_unfinished_rows = sum(total_status_counts.values())
    print(f"Embedding runs that never finished (status != 'ok'): {n_unfinished_rows} rows across "
          f"{total_unfinished_graphs} distinct graphs")
    for status, count in total_status_counts.most_common():
        print(f"  {status}: {count}")
