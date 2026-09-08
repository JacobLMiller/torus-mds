#!/usr/bin/env python3
"""
Renders one combined-methods PNG per graph under a layouts/ tree (as written
by run_embeddings -- see modules/experiment_runner.py), using the drawing
routines in modules/visualization. Companion to draw_layouts.py, which
renders one PNG per (graph, method) instead of grouping them.

Walks --layouts-root for shard directories (any directory containing a
runs.csv, same convention run_metrics_array.sbatch uses). For each exp_idx
with at least one status=="ok" row, draws every method present for that
graph into a single figure with METHOD_GRID_SHAPE panels (default 2x3, so up
to 6 methods; unused panels are left blank), one panel per method in
canonical METHODS order, and saves it as one PNG mirroring the shard's path
under --output-root.

`s_gd2` is the one method whose output isn't confined to the periodic
[0,1)^2 torus domain (it's a plain force-directed layout in an arbitrary
scale/offset) -- it's min-max normalized to [0,1]^2 and drawn with straight
(non-wrapped) edges via plot_embedding. Every other method (TorusMDS and all
its aspect-init variants, wrap_python/wrap_typescript/wrap_python_newdist)
already lives in [0,1)^2 and is drawn wrap-aware via
plot_embedding_with_torus_edges, using alpha_fit/r0_fit/r1_fit from that
row when present (NaN r0/r1 -- e.g. plain TorusMDS's fixed unit-square torus,
or wrap_python which fits neither -- default to 1.0).

Resumable: existing PNGs are skipped unless --overwrite is passed.

Usage:
    python draw_layout_grids.py
    python draw_layout_grids.py --layouts-root ../layouts_converged/suitesparse_normalized \\
        --output-root ../layout_drawings_grids/suitesparse_normalized
    python draw_layout_grids.py --max-graphs-per-shard 5 --no-tsnet-style
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
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

# Canonical panel order -- matches METHODS in combined_results_normalized.ipynb.
METHODS = [
    "s_gd2", "wrap_python",
    "TorusMDS_spectral_init", "TorusMDS_smart_1x_scale2", "TorusMDS_random_1x_scale2",
]
GRID_SHAPE = (2, 3)   # rows, cols -- up to 6 panels


def find_shard_dirs(layouts_root: str) -> list[str]:
    pattern = os.path.join(layouts_root, "**", "runs.csv")
    return sorted(os.path.dirname(p) for p in glob.glob(pattern, recursive=True))


def method_order(present: list[str]) -> list[str]:
    """Canonical methods first (in METHODS order), then any unrecognized ones."""
    ordered = [m for m in METHODS if m in present]
    ordered += sorted(m for m in present if m not in METHODS)
    return ordered


def draw_panel(ax, G, X: np.ndarray, method: str, row: pd.Series, tsnet_style: bool) -> None:
    if method in NON_PERIODIC_METHODS:
        lo, hi = X.min(axis=0), X.max(axis=0)
        span = np.where(hi > lo, hi - lo, 1.0)
        X_norm = (X - lo) / span
        plot_embedding(X_norm, G, ax=ax, tsnet_style=tsnet_style)
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
        plot_embedding_with_torus_edges(X=X, G=G, torus=torus, ax=ax,
                                         tsnet_style=tsnet_style, show_ticks=False)
    t_embed = row.get("t_embed")
    t_str = f", t={t_embed:.2g}s" if pd.notna(t_embed) else ""
    ax.set_title(f"{method}{t_str}", fontsize=9)


def draw_graph_grid(shard_dir: str, exp_idx: int, group: pd.DataFrame, outpath: str,
                     dpi: int, tsnet_style: bool) -> int:
    """group: this exp_idx's status=='ok' rows. Returns the number of panels drawn."""
    G = load_graph(graph_path(shard_dir, exp_idx))
    rows = {row["method"]: row for _, row in group.iterrows()}
    methods = method_order(list(rows.keys()))

    nrows, ncols = GRID_SHAPE
    if len(methods) > nrows * ncols:
        print(f"  [{shard_dir}] exp_idx={exp_idx}: {len(methods)} methods > "
              f"{nrows * ncols} panels, dropping {methods[nrows * ncols:]}")
        methods = methods[: nrows * ncols]

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    n_drawn = 0
    try:
        for ax, method in zip(axes, methods):
            row = rows[method]
            X = np.load(coords_path(shard_dir, exp_idx, method))
            draw_panel(ax, G, X, method, row, tsnet_style)
            n_drawn += 1
        for ax in axes[len(methods):]:
            ax.axis("off")

        sample_row = next(iter(rows.values()))
        title = f"{sample_row.get('group', '')}/{sample_row.get('name', '')} " \
                f"(n={G.number_of_nodes()}, exp_idx={exp_idx})"
        fig.suptitle(title, fontsize=12)
        fig.tight_layout()
        fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)
    return n_drawn


def draw_shard(shard_dir: str, layouts_root: str, output_root: str,
                max_graphs: int | None, dpi: int, overwrite: bool,
                tsnet_style: bool) -> tuple[int, int]:
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
    out_dir = os.path.join(output_root, rel)
    os.makedirs(out_dir, exist_ok=True)

    n_ok, n_failed = 0, 0
    for exp_idx in exp_idxs:
        outpath = os.path.join(out_dir, f"exp_{exp_idx}.png")
        if os.path.exists(outpath) and not overwrite:
            continue
        group = runs[runs["exp_idx"] == exp_idx]
        try:
            draw_graph_grid(shard_dir, int(exp_idx), group, outpath, dpi, tsnet_style)
            n_ok += 1
        except Exception as e:
            print(f"[{shard_dir}] exp_idx={exp_idx} failed: {e}")
            n_failed += 1
    return n_ok, n_failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render one combined-methods grid PNG per graph under a layouts/ tree")
    parser.add_argument("--layouts-root", type=str,
                        default="../layouts_converged/suitesparse_normalized",
                        help="Root directory to search for shard dirs "
                             "(default: ../layouts_converged/suitesparse_normalized)")
    parser.add_argument("--output-root", type=str,
                        default="../layout_drawings_grids/suitesparse_normalized",
                        help="Root directory to mirror shard structure into "
                             "(default: ../layout_drawings_grids/suitesparse_normalized)")
    parser.add_argument("--max-graphs-per-shard", type=int, default=None,
                        help="Cap on distinct exp_idx values drawn per shard (default: no cap)")
    parser.add_argument("--dpi", type=int, default=110, help="PNG resolution (default: 110)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Redraw PNGs that already exist (default: skip them)")
    parser.add_argument("--tsnet-style", action=argparse.BooleanOptionalAction, default=True,
                        help="Draw tsNET-like: edges colored by length, small black vertices "
                             "(default: on; pass --no-tsnet-style for the plain style)")
    args = parser.parse_args()

    shard_dirs = find_shard_dirs(args.layouts_root)
    if not shard_dirs:
        print(f"No runs.csv found anywhere under {args.layouts_root}")
        sys.exit(1)

    total_ok, total_failed = 0, 0
    for i, shard_dir in enumerate(shard_dirs):
        n_ok, n_failed = draw_shard(
            shard_dir, args.layouts_root, args.output_root,
            args.max_graphs_per_shard, args.dpi, args.overwrite, args.tsnet_style,
        )
        total_ok += n_ok
        total_failed += n_failed
        print(f"[{i + 1}/{len(shard_dirs)}] {shard_dir}: drew {n_ok} grids ({n_failed} failed)")

    print(f"\nDone: {total_ok} grid PNGs written under {args.output_root} ({total_failed} failed)")
