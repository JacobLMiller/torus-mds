#!/usr/bin/env python3
"""
Removes off-stride rows from cache-mode embedding manifests.

In cache mode, shard_<s> of an N-shard array only owns exp_idx with
exp_idx % N == s. A submission that accidentally ran in generator mode
(GRAPHS_PER_SHARD set, CACHE_DIR/NUM_SHARDS missing) into the same shard
dirs appended rows for exp_idx 0..k on EVERY shard: the on-stride ids were
skipped as "already done", the off-stride ids were embedded on freshly
generated graphs that are either duplicates of another shard's graphs
(seed 0) or graphs outside the staged pool (seed != 0).

This keeps every on-stride row untouched and moves -- never deletes --
everything else into a quarantine dir per shard:

    shard_<s>/_offstride_quarantine/<timestamp>/
        runs.csv.orig       original manifest, byte-for-byte (not named
                            runs.csv, so draw_layouts*/find globs skip it)
        coords/...          off-stride exp_<id>_<method>.npy
        graphs/...          off-stride graph_<id>.npz

Restore a shard by copying runs.csv.orig back over runs.csv and moving coords/graphs back.

Dry run by default; pass --apply to modify anything. Do not run while embed
jobs are writing to the same shard dirs.

Usage:
    python drop_offstride_rows.py layouts/sbm_normalized layouts/grg_normalized
    python drop_offstride_rows.py layouts/sbm_normalized layouts/grg_normalized --apply
"""

from __future__ import annotations

import argparse
import glob as _glob
import os
import re
import shutil
import time

import pandas as pd

SHARD_RE = re.compile(r"^shard_(\d+)$")
COORDS_RE = re.compile(r"^exp_(\d+)_.+\.npy$")
GRAPH_RE = re.compile(r"^graph_(\d+)\.npz$")


def _shard_dirs(tier_dir: str) -> dict[int, str]:
    out = {}
    for name in os.listdir(tier_dir):
        m = SHARD_RE.match(name)
        path = os.path.join(tier_dir, name)
        if m and os.path.isfile(os.path.join(path, "runs.csv")):
            out[int(m.group(1))] = path
    return out


def _offstride_files(subdir: str, pattern: re.Pattern, shard: int, num_shards: int) -> list[str]:
    if not os.path.isdir(subdir):
        return []
    return sorted(
        name for name in os.listdir(subdir)
        if (m := pattern.match(name)) and int(m.group(1)) % num_shards != shard
    )


def clean_shard(shard_dir: str, shard: int, num_shards: int, apply: bool, stamp: str) -> int:
    manifest = os.path.join(shard_dir, "runs.csv")
    df = pd.read_csv(manifest)
    on_stride = df["exp_idx"].astype(int) % num_shards == shard
    bad = df[~on_stride]
    coords = _offstride_files(os.path.join(shard_dir, "coords"), COORDS_RE, shard, num_shards)
    graphs = _offstride_files(os.path.join(shard_dir, "graphs"), GRAPH_RE, shard, num_shards)

    if bad.empty and not coords and not graphs:
        print(f"  {shard_dir}: clean ({len(df)} rows)")
        return 0

    per_method = bad.groupby("method").size().to_dict()
    print(f"  {shard_dir}: keep {int(on_stride.sum())}/{len(df)} rows, "
          f"drop {len(bad)} {per_method}, move {len(coords)} coords + {len(graphs)} graphs")
    if not apply:
        return len(bad)

    quarantine = os.path.join(shard_dir, "_offstride_quarantine", stamp)
    os.makedirs(os.path.join(quarantine, "coords"), exist_ok=True)
    os.makedirs(os.path.join(quarantine, "graphs"), exist_ok=True)
    shutil.copy2(manifest, os.path.join(quarantine, "runs.csv.orig"))

    # Atomic replace so an interrupted run never leaves a truncated manifest.
    tmp = manifest + ".tmp"
    df[on_stride].to_csv(tmp, index=False)
    os.replace(tmp, manifest)

    for sub, names in (("coords", coords), ("graphs", graphs)):
        for name in names:
            shutil.move(os.path.join(shard_dir, sub, name), os.path.join(quarantine, sub, name))
    return len(bad)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("family_dirs", nargs="+",
                        help="Family dirs (e.g. layouts/sbm_normalized); tiers are the subdirs holding shard_<i>/")
    parser.add_argument("--num-shards", type=int, default=None,
                        help="NUM_SHARDS the cache-mode run used (default: number of shard_<i> dirs in each tier)")
    parser.add_argument("--apply", action="store_true",
                        help="Actually rewrite manifests and move files (default: dry run)")
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%d-%H%M%S")
    print("APPLYING" if args.apply else "DRY RUN (pass --apply to modify)")
    total = 0
    for family_dir in args.family_dirs:
        tier_dirs = sorted({os.path.dirname(os.path.dirname(p))
                            for p in _glob.glob(os.path.join(family_dir, "**", "shard_*", "runs.csv"), recursive=True)
                            if "_offstride_quarantine" not in p})
        for tier_dir in tier_dirs:
            shards = _shard_dirs(tier_dir)
            num_shards = args.num_shards or len(shards)
            if sorted(shards) != list(range(num_shards)):
                print(f"{tier_dir}: shard dirs {sorted(shards)} don't match num_shards={num_shards}; skipping")
                continue
            print(f"{tier_dir} (num_shards={num_shards})")
            for shard in sorted(shards):
                total += clean_shard(shards[shard], shard, num_shards, args.apply, stamp)
    print(f"{'Dropped' if args.apply else 'Would drop'} {total} rows in total.")


if __name__ == "__main__":
    main()
