#!/usr/bin/env python3
"""
Read-only checks before (re)scoring layouts; run from the repo root.

Layouts, per tier/shard of each *_normalized family:
  * no off-stride or duplicate (exp_idx, method) rows,
  * row n == node count of graphs/graph_<id>.npz, consistent graph metadata,
  * every "ok" row has coords of shape (n, 2),
  * every method has a row ("ok", or wrap_python "skipped_too_large" above
    its n cap) for every graph in the tier -- the staged pool's
    data/<family>_cache/<tier>/manifest.csv when present, else ids 0..max.

Existing compute_metrics.py results (results/<FAMILY_SUBDIR>_<tier>_shard_<s>_comparison.csv):
  * every scored (exp_idx, method) must still be an "ok" layout with the same
    n and metadata -- otherwise resuming would silently keep a stale score,
  * reports how many pairs a resumed run will add.

Exits nonzero on any error. Reads only small files plus .npz/.npy headers.

Usage:
    python experiments/slurm/preflight_metrics.py [--layouts-root layouts] [--results-root results]
"""

from __future__ import annotations

import argparse
import glob as _glob
import os
import re
import zipfile

import numpy as np
import pandas as pd

FAMILIES = ("sbm_normalized", "grg_normalized", "suitesparse_normalized")
METHODS = ("s_gd2", "wrap_python", "TorusMDS_smart_1x_scale2", "TorusMDS_random_1x_scale2",
           "TorusMDS_spectral_init", "TorusMDS_parallelogram")
RUN_COLUMNS = {"exp_idx", "n", "method", "seed", "t_embed", "alpha_fit", "r0_fit", "r1_fit",
               "theta_fit", "n_iter", "termination_reason", "status"}
WRAP_MAX_N = 3500


def _graph_n(path: str) -> int:
    # Read only the 'shape' member of the sparse .npz, not the edge arrays.
    with zipfile.ZipFile(path) as z, z.open("shape.npy") as f:
        return int(np.lib.format.read_array(f)[0])


def _coords_rows(path: str) -> int | None:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        shape, _, _ = np.lib.format._read_array_header(f, version)
    return shape if len(shape) == 2 and shape[1] == 2 else None


def _shards(tier_dir: str) -> dict[int, str]:
    out = {}
    for p in _glob.glob(os.path.join(tier_dir, "shard_*", "runs.csv")):
        out[int(re.search(r"shard_(\d+)", p).group(1))] = os.path.dirname(p)
    return out


def check_tier(tier_dir: str, family: str, tier: str | None, results_root: str, errors: list, warnings: list):
    shards = _shards(tier_dir)
    num = len(shards)
    label = os.path.relpath(tier_dir)
    if sorted(shards) != list(range(num)):
        errors.append(f"{label}: shard dirs {sorted(shards)} are not 0..{num - 1}")
        return
    cache = f"data/{family.split('_')[0]}_cache" + (f"/{tier}" if tier else "")
    manifest = os.path.join(cache, "manifest.csv")
    expected, matrix_ids = None, None
    if os.path.exists(manifest):
        pool = pd.read_csv(manifest)
        if "exp_idx" in pool:
            expected = set(pool["exp_idx"].astype(int))
        elif "matrix_id" in pool:
            # suitesparse_comparison.py numbers the (unfiltered) staged pool by
            # its position after sorting on matrix_id.
            matrix_ids = pool.sort_values("matrix_id")["matrix_id"].astype(int).tolist()
            expected = set(range(len(matrix_ids)))

    seen, to_score, existing_rows = set(), 0, 0
    per_method = {m: 0 for m in METHODS}
    for s, d in sorted(shards.items()):
        df = pd.read_csv(os.path.join(d, "runs.csv"))
        where = f"{label}/shard_{s}"
        if (df.exp_idx % num != s).any():
            errors.append(f"{where}: {(df.exp_idx % num != s).sum()} off-stride rows (run drop_offstride_rows.py)")
        if df.duplicated(["exp_idx", "method"]).any():
            errors.append(f"{where}: {df.duplicated(['exp_idx', 'method']).sum()} duplicate rows (run drop_mismatched_rows.py)")
        gn = {}
        for i in df.exp_idx.unique():
            path = os.path.join(d, "graphs", f"graph_{i}.npz")
            gn[i] = _graph_n(path) if os.path.exists(path) else None
        if any(v is None for v in gn.values()):
            errors.append(f"{where}: missing graph files for ids {[i for i, v in gn.items() if v is None][:10]}")
            continue
        if (df.n != df.exp_idx.map(gn)).any():
            errors.append(f"{where}: {(df.n != df.exp_idx.map(gn)).sum()} rows whose n != graph file (run drop_mismatched_rows.py)")
        if matrix_ids is not None and "matrix_id" in df:
            wrong = df[[i >= len(matrix_ids) or matrix_ids[i] != m
                        for i, m in zip(df.exp_idx.astype(int), df.matrix_id.astype(int))]]
            if len(wrong):
                errors.append(f"{where}: {len(wrong)} rows whose matrix_id doesn't match the staged pool's "
                              f"exp_idx order, e.g. {wrong[['exp_idx', 'matrix_id']].head(3).values.tolist()}")
        meta = [c for c in df.columns if c not in RUN_COLUMNS]
        if meta and df.groupby("exp_idx")[meta].nunique(dropna=False).gt(1).any().any():
            errors.append(f"{where}: conflicting graph metadata within an exp_idx (run drop_mismatched_rows.py)")
        ok = df[df.status == "ok"]
        bad = [(r.exp_idx, r.method) for r in ok.itertuples()
               if _coords_rows(os.path.join(d, "coords", f"exp_{r.exp_idx}_{r.method}.npy")) != (gn[r.exp_idx], 2)]
        if bad:
            errors.append(f"{where}: {len(bad)} ok rows with missing/mis-shaped coords, e.g. {bad[:5]}")
        valid = (df.status == "ok") | ((df.status == "skipped_too_large") & (df.method == "wrap_python") & (df.n > WRAP_MAX_N))
        if (~valid).any():
            errors.append(f"{where}: {(~valid).sum()} rows that are neither ok nor an expected wrap skip: "
                          f"{df[~valid][['exp_idx', 'method', 'status']].head(5).values.tolist()}")
        for m in METHODS:
            per_method[m] += int(valid[df.method == m].sum())
        seen |= set(int(i) for i in df.exp_idx)

        # Existing compute_metrics.py results for this shard.
        # Same naming as run_metrics_array.sbatch's SHARD_NAME.
        out = os.path.join(results_root, f"{family}_{tier + '_' if tier else ''}shard_{s}_comparison.csv")
        layout_ok = {(int(r.exp_idx), r.method): r for r in ok.itertuples()}
        if os.path.exists(out):
            res = pd.read_csv(out)
            existing_rows += len(res)
            # Run fields compute_metrics.py copies from runs.csv: a mismatch means
            # the layout was re-embedded after it was scored.
            run_fields = [c for c in ("t_embed", "n_iter", "r0_fit", "r1_fit") if c in res.columns and c in df.columns]
            stale, rerun = [], []
            for r in res.itertuples():
                lay = layout_ok.get((int(r.exp_idx), r.method))
                if lay is None or int(r.n) != int(lay.n) or any(
                        c in res.columns and not (pd.isna(getattr(r, c)) and pd.isna(getattr(lay, c)))
                        and getattr(r, c) != getattr(lay, c) for c in meta):
                    stale.append((int(r.exp_idx), r.method))
                elif not all(np.isclose(getattr(r, c), getattr(lay, c), rtol=1e-6, atol=1e-4, equal_nan=True)
                             for c in run_fields):
                    rerun.append((int(r.exp_idx), r.method))
            if stale:
                errors.append(f"{out}: {len(stale)} scored rows no longer match an ok layout, e.g. {stale[:5]} "
                              "-- resuming would keep these stale scores")
            if rerun:
                errors.append(f"{out}: {len(rerun)} scored rows were computed on an earlier embedding of the same "
                              f"graph ({'/'.join(run_fields)} differ from runs.csv), e.g. {rerun[:3]} "
                              "-- resuming would keep scores for layouts that no longer exist")
            if res.duplicated(["exp_idx", "method"]).any():
                errors.append(f"{out}: duplicate (exp_idx, method) rows")
            n_para = int((res.method == "TorusMDS_parallelogram").sum())
            if n_para:
                warnings.append(f"{out}: already has {n_para} TorusMDS_parallelogram rows (last modified "
                                f"{pd.Timestamp(os.path.getmtime(out), unit='s'):%Y-%m-%d}); make sure they were "
                                "scored with the theta_fit-aware compute_metrics.py")
            scored = set(zip(res.exp_idx.astype(int), res.method))
            to_score += len(set(layout_ok) - scored)
        else:
            to_score += len(layout_ok)

    want = expected if expected is not None else set(range(max(seen) + 1))
    if seen != want:
        errors.append(f"{label}: graphs missing from layouts: {sorted(want - seen)[:10]} "
                      f"({len(want - seen)}){'' if expected is not None else ' (pool size inferred, no cache manifest)'}")
    short = {m: c for m, c in per_method.items() if c != len(want)}
    if short:
        errors.append(f"{label}: methods without a result for every graph ({len(want)}): {short}")
    print(f"  {label:34} {len(want):>5} graphs | existing old-metric rows {existing_rows:>5} | "
          f"old-metric pairs to compute {to_score:>5}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--layouts-root", default="layouts")
    parser.add_argument("--results-root", default="results")
    args = parser.parse_args()

    errors, warnings = [], []
    for family in FAMILIES:
        fam_dir = os.path.join(args.layouts_root, family)
        if not os.path.isdir(fam_dir):
            errors.append(f"{fam_dir}: missing")
            continue
        print(family)
        if family.startswith("suitesparse"):
            check_tier(fam_dir, family, None, args.results_root, errors, warnings)
        else:
            for tier_dir in sorted(_glob.glob(os.path.join(fam_dir, "*_*"))):
                if _shards(tier_dir):
                    check_tier(tier_dir, family, os.path.basename(tier_dir), args.results_root, errors, warnings)
    for w in warnings:
        print("WARNING:", w)
    for e in errors:
        print("ERROR:", e)
    print("PREFLIGHT OK" if not errors else f"PREFLIGHT FAILED ({len(errors)} errors)")
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
