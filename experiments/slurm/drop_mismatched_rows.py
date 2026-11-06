#!/usr/bin/env python3
"""
Removes manifest rows that describe a different graph than the shard's
graph_<exp_idx>.npz, plus redundant repeated skip rows.

Follow-up to drop_offstride_rows.py. The accidental generator-mode run also
touched ON-stride ids whose only row was status "skipped_too_large":
run_embeddings' resume logic only treats "ok" rows as done, so it re-ran
wrap_python on the freshly generated graph sharing that exp_idx and appended
a row (and, when that graph had n <= 3500, an exp_<id>_wrap_python.npy) for
the wrong graph. graph_<id>.npz itself was never overwritten, so it is the
ground truth: any row whose n differs from it belongs to a foreign graph.
n alone isn't enough -- the GRG generator reproduces the staged pool's node
counts but not its graphs -- so the graph metadata columns (epsilon, k,
graph_type, ...) must also match the exp_idx's first row in the manifest,
which is always the original cache-mode run's (later runs only append).

Per shard this
  1. drops every row whose n != the node count of graphs/graph_<id>.npz or
     whose metadata differs from that exp_idx's first row, moving the coords
     file of such an "ok" row into quarantine when the file's row count also
     disagrees with the graph;
  2. collapses exact repeats of "skipped_too_large" rows (one gets appended
     on every resume) down to the first copy.
A shard is left untouched if the cleanup would leave any (exp_idx, method)
pair without a row, or if a kept "ok" row's coords don't match its graph.

Nothing is deleted: the original manifest and moved coords go to

    shard_<s>/_mismatch_quarantine/<timestamp>/
        runs.csv.orig       original manifest, byte-for-byte (not named
                            runs.csv, so draw_layouts*/find globs skip it)
        coords/...          coords computed on the foreign graph

Restore a shard by copying runs.csv.orig back over runs.csv and moving
coords back.

Dry run by default; pass --apply to modify anything. Do not run while embed
or metrics jobs are using the same shard dirs.

Usage:
    python drop_mismatched_rows.py layouts/sbm_normalized layouts/grg_normalized layouts/suitesparse_normalized
    python drop_mismatched_rows.py layouts/sbm_normalized layouts/grg_normalized layouts/suitesparse_normalized --apply
"""

from __future__ import annotations

import argparse
import glob as _glob
import os
import shutil
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

QUARANTINE = "_mismatch_quarantine"
# Per-run columns written by run_embeddings; everything else describes the graph.
RUN_COLUMNS = {"exp_idx", "n", "method", "seed", "t_embed", "alpha_fit", "r0_fit", "r1_fit",
               "theta_fit", "n_iter", "termination_reason", "status"}


def _meta_mismatch(df: pd.DataFrame) -> pd.Series:
    meta = [c for c in df.columns if c not in RUN_COLUMNS]
    if not meta:
        return pd.Series(False, index=df.index)
    first = df.groupby("exp_idx")[meta].transform("first")
    # transform("first") skips NaN, so treat NaN as equal only when both sides are NaN.
    a, b = df[meta], first
    differs = ~((a == b) | (a.isna() & b.isna()))
    return differs.any(axis=1)


def _graph_sizes(shard_dir: str, exp_ids) -> dict[int, int | None]:
    sizes = {}
    for i in exp_ids:
        path = os.path.join(shard_dir, "graphs", f"graph_{i}.npz")
        sizes[i] = sp.load_npz(path).shape[0] if os.path.exists(path) else None
    return sizes


def _coords_rows(shard_dir: str, exp_idx: int, method: str) -> int | None:
    path = os.path.join(shard_dir, "coords", f"exp_{exp_idx}_{method}.npy")
    return np.load(path, mmap_mode="r").shape[0] if os.path.exists(path) else None


def clean_shard(shard_dir: str, apply: bool, stamp: str) -> tuple[int, int, bool]:
    manifest = os.path.join(shard_dir, "runs.csv")
    df = pd.read_csv(manifest)
    exp_ids = [int(i) for i in df["exp_idx"].unique()]
    sizes = _graph_sizes(shard_dir, exp_ids)

    no_graph = sorted(i for i, n in sizes.items() if n is None)
    if no_graph:
        print(f"  {shard_dir}: graph files missing for ids {no_graph[:10]}; skipping shard")
        return 0, 0, False

    graph_n = df["exp_idx"].astype(int).map(sizes)
    mismatched = (df["n"].astype(int) != graph_n) | _meta_mismatch(df)
    repeated_skip = (df["status"] == "skipped_too_large") & df.duplicated(keep="first") & ~mismatched
    keep = ~mismatched & ~repeated_skip
    kept = df[keep]

    # Coords of dropped "ok" rows: only move a file when it is itself foreign
    # to the graph, so a correct layout sharing the filename is never touched.
    to_move = []
    for r in df[mismatched & (df["status"] == "ok")].itertuples():
        rows = _coords_rows(shard_dir, int(r.exp_idx), r.method)
        if rows is not None and rows != sizes[int(r.exp_idx)]:
            to_move.append(f"exp_{int(r.exp_idx)}_{r.method}.npy")

    # Safety checks on the result before touching anything.
    before = set(map(tuple, df[["exp_idx", "method"]].values.tolist()))
    after = set(map(tuple, kept[["exp_idx", "method"]].values.tolist()))
    orphaned = sorted(before - after)
    bad_kept = []
    moved = set(to_move)
    for r in kept[kept["status"] == "ok"].itertuples():
        name = f"exp_{int(r.exp_idx)}_{r.method}.npy"
        rows = None if name in moved else _coords_rows(shard_dir, int(r.exp_idx), r.method)
        if rows != sizes[int(r.exp_idx)]:
            bad_kept.append((int(r.exp_idx), r.method, rows, sizes[int(r.exp_idx)]))
    if orphaned or bad_kept:
        print(f"  {shard_dir}: UNSAFE, skipping shard -- pairs left without a row: {orphaned[:5]}, "
              f"kept ok rows with mismatched coords (id, method, coords n, graph n): {bad_kept[:5]}")
        return 0, 0, False

    n_mis, n_rep = int(mismatched.sum()), int(repeated_skip.sum())
    if n_mis == 0 and n_rep == 0:
        print(f"  {shard_dir}: clean ({len(df)} rows)")
        return 0, 0, True

    mis = df[mismatched]
    detail = mis.groupby(["method", "status"]).size().to_dict()
    print(f"  {shard_dir}: keep {len(kept)}/{len(df)} rows; drop {n_mis} foreign-graph rows {detail}"
          f"{' ' + str(sorted(f[4:-4] for f in to_move)) if to_move else ''}, {n_rep} repeated skip rows; "
          f"move {len(to_move)} coords")
    if not apply:
        return n_mis, n_rep, True

    quarantine = os.path.join(shard_dir, QUARANTINE, stamp)
    os.makedirs(os.path.join(quarantine, "coords"), exist_ok=True)
    shutil.copy2(manifest, os.path.join(quarantine, "runs.csv.orig"))

    # Atomic replace so an interrupted run never leaves a truncated manifest.
    tmp = manifest + ".tmp"
    kept.to_csv(tmp, index=False)
    os.replace(tmp, manifest)

    for name in to_move:
        shutil.move(os.path.join(shard_dir, "coords", name), os.path.join(quarantine, "coords", name))
    return n_mis, n_rep, True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("family_dirs", nargs="+",
                        help="Family dirs (e.g. layouts/sbm_normalized); every shard_<i>/runs.csv below is checked")
    parser.add_argument("--apply", action="store_true",
                        help="Actually rewrite manifests and move files (default: dry run)")
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%d-%H%M%S")
    print("APPLYING" if args.apply else "DRY RUN (pass --apply to modify)")
    tot_mis = tot_rep = 0
    skipped = []
    for family_dir in args.family_dirs:
        manifests = sorted(p for p in _glob.glob(os.path.join(family_dir, "**", "shard_*", "runs.csv"), recursive=True)
                           if "_quarantine" not in p)
        if not manifests:
            print(f"{family_dir}: no shard_*/runs.csv found")
        for p in manifests:
            n_mis, n_rep, ok = clean_shard(os.path.dirname(p), args.apply, stamp)
            tot_mis += n_mis
            tot_rep += n_rep
            if not ok:
                skipped.append(os.path.dirname(p))
    verb = "Dropped" if args.apply else "Would drop"
    print(f"{verb} {tot_mis} foreign-graph rows and {tot_rep} repeated skip rows in total.")
    if skipped:
        print(f"{len(skipped)} shard(s) skipped and left unchanged -- see messages above: {skipped}")


if __name__ == "__main__":
    main()
