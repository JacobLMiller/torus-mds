#!/usr/bin/env python3
"""
Renders one PNG per (graph, method) under a layouts/ tree (as written by
run_embeddings -- see modules/experiment_runner.py), grouping each graph's
method PNGs into their own folder. Companion to draw_layouts.py (same one-
PNG-per-method output, but flat) and draw_layout_grids.py (combines a
graph's methods into a single grid PNG instead of separate files). Reuses
draw_layout_grids.py's per-panel drawing helper, which always draws
tsNET-style (colored edges by length, small black vertices) with no axes
ticks/labels on the torus drawings.

Walks --layouts-root for shard directories (any directory containing a
runs.csv, same convention run_metrics_array.sbatch uses). For each exp_idx
with at least one status=="ok" row, writes one PNG per method present for
that graph into --output-root/<mirrored shard path>/exp_<idx>/<method>.png.

Resumable: existing PNGs are skipped unless --overwrite is passed.

Usage:
    python draw_layout_singles.py
    python draw_layout_singles.py --layouts-root ../layouts_converged/suitesparse_normalized \\
        --output-root ../layout_drawings_singles/suitesparse_normalized
    python draw_layout_singles.py --max-graphs-per-shard 5
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.experiment_runner import coords_path, graph_path, load_graph
from draw_layout_grids import find_shard_dirs, method_order, draw_panel


def draw_graph_singles(shard_dir: str, exp_idx: int, group: pd.DataFrame, out_dir: str,
                        dpi: int, overwrite: bool) -> tuple[int, int]:
    """group: this exp_idx's status=='ok' rows. Returns (n_drawn, n_failed)."""
    G = load_graph(graph_path(shard_dir, exp_idx))
    rows = {row["method"]: row for _, row in group.iterrows()}
    methods = method_order(list(rows.keys()))

    os.makedirs(out_dir, exist_ok=True)
    n_ok, n_failed = 0, 0
    for method in methods:
        outpath = os.path.join(out_dir, f"{method}.png")
        if os.path.exists(outpath) and not overwrite:
            continue
        row = rows[method]
        fig, ax = plt.subplots(figsize=(6, 6))
        try:
            X = np.load(coords_path(shard_dir, exp_idx, method))
            draw_panel(ax, G, X, method, row, tsnet_style=True)
            fig.tight_layout()
            fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
            n_ok += 1
        except Exception as e:
            print(f"  [{shard_dir}] exp_idx={exp_idx} method={method} failed: {e}")
            n_failed += 1
        finally:
            plt.close(fig)
    return n_ok, n_failed


def draw_shard(shard_dir: str, layouts_root: str, output_root: str,
                max_graphs: int | None, dpi: int, overwrite: bool) -> tuple[int, int]:
    runs_path = os.path.join(shard_dir, "runs.csv")
    try:
        runs = pd.read_csv(runs_path)
    except Exception as e:
        print(f"[{shard_dir}] failed to read runs.csv: {e}")
        return 0, 0

    runs = runs[runs["status"] == "ok"]
    if runs.empty:
        return 0, 0

    exp_idxs = sorted(runs["exp_idx"].unique())
    if max_graphs is not None:
        exp_idxs = exp_idxs[:max_graphs]

    rel = os.path.relpath(shard_dir, layouts_root)
    shard_out = os.path.join(output_root, rel)

    n_ok, n_failed = 0, 0
    for exp_idx in exp_idxs:
        group = runs[runs["exp_idx"] == exp_idx]
        out_dir = os.path.join(shard_out, f"exp_{exp_idx}")
        ok, failed = draw_graph_singles(shard_dir, int(exp_idx), group, out_dir, dpi, overwrite)
        n_ok += ok
        n_failed += failed
    return n_ok, n_failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render one PNG per (graph, method) under a layouts/ tree, "
                     "grouped into one folder per graph")
    parser.add_argument("--layouts-root", type=str,
                        default="../layouts/suitesparse_normalized",
                        help="Root directory to search for shard dirs "
                             "(default: ../layouts/suitesparse_normalized)")
    parser.add_argument("--output-root", type=str,
                        default="../layout_drawings_singles/suitesparse_normalized",
                        help="Root directory to mirror shard structure into "
                             "(default: ../layout_drawings_singles/suitesparse_normalized)")
    parser.add_argument("--max-graphs-per-shard", type=int, default=None,
                        help="Cap on distinct exp_idx values drawn per shard (default: no cap)")
    parser.add_argument("--dpi", type=int, default=110, help="PNG resolution (default: 110)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Redraw PNGs that already exist (default: skip them)")
    args = parser.parse_args()

    shard_dirs = find_shard_dirs(args.layouts_root)
    if not shard_dirs:
        print(f"No runs.csv found anywhere under {args.layouts_root}")
        sys.exit(1)

    total_ok, total_failed = 0, 0
    for i, shard_dir in enumerate(shard_dirs):
        n_ok, n_failed = draw_shard(
            shard_dir, args.layouts_root, args.output_root,
            args.max_graphs_per_shard, args.dpi, args.overwrite,
        )
        total_ok += n_ok
        total_failed += n_failed
        print(f"[{i + 1}/{len(shard_dirs)}] {shard_dir}: drew {n_ok} PNGs ({n_failed} failed)")

    print(f"\nDone: {total_ok} PNGs written under {args.output_root} ({total_failed} failed)")
