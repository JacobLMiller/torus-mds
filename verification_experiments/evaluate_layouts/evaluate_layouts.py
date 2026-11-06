#!/usr/bin/env python3
"""Evaluate persisted layouts without running embedding methods.

See evaluate_layouts.md for input conventions, metrics and examples.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numba
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from verification_experiments.evaluate_layouts.layout_graph import load_layout_graph
from verification_experiments.evaluate_layouts.layout_metrics import (
    layout_distance_matrix, distance_metrics, edge_metrics, silhouette_score, fitted_distance_scale,
    graph_distance_ranks, sampled_query_metrics, node_resolution,
)
from verification_experiments.evaluate_layouts.cluster_distance import cluster_distance, sbm_labels

METRIC_COLUMNS = ["stress", "pointwise_stress", "distortion", "sgs", "np_score",
                  "edge_crossings", "incidence_angle", "cluster_distance",
                  "mean_edge_length", "cluster_distance_normalized", "silhouette_score",
                  "node_resolution"]


def _parameter(row, name, default):
    value = row.get(name, np.nan)
    value = default if pd.isna(value) else float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _source_shard(shard):
    """Stable identity across single-shard and recursive invocations.

    Use family/tier/shard relative to the nearest layouts/ ancestor. For custom
    input trees without that ancestor, use the resolved absolute directory.
    """
    shard = shard.resolve()
    for ancestor in shard.parents:
        if ancestor.name == "layouts":
            return shard.relative_to(ancestor).as_posix()
    return shard.as_posix()


def prepare_metric_sample(D, metric_subsample):
    if metric_subsample != 0 and metric_subsample < 2:
        raise ValueError("metric_subsample must be at least 2, or 0 for all nodes")
    n = len(D)
    indices = (np.random.default_rng(42).choice(n, metric_subsample, replace=False)
               if metric_subsample and n > metric_subsample else np.arange(n))
    target = D[np.ix_(indices, indices)] if len(indices) < n else D
    return dict(indices=indices, target=target, graph_ranks=graph_distance_ranks(target),
                diameter=float(D.max()))


def score_layout(X, D, edges, method, row, np_radius=2., threads=1,
                 metric_subsample=1000, sample=None):
    """Sample distance metrics/queries; retain full cluster hulls and edge geometry."""
    if X.shape != (len(D), 2) or not np.isfinite(X).all():
        raise ValueError("Coordinates must match graph node order and have shape (n, 2)")
    if method.startswith("TorusMDS") and method != "TorusMDS_spectral_init":
        if pd.isna(row.get("alpha_fit", np.nan)):
            raise ValueError("TorusMDS layout has no saved alpha_fit")
    alpha = _parameter(row, "alpha_fit", 1.)
    periodic = method != "s_gd2"
    r0 = _parameter(row, "r0_fit", 1.) if periodic else 1.
    r1 = _parameter(row, "r1_fit", 1.) if periodic else 1.
    theta = _parameter(row, "theta_fit", np.pi/2) if periodic else np.pi/2
    if theta >= np.pi:
        raise ValueError("theta_fit must lie in (0, pi)")
    if periodic:
        X = X % 1.
    geometry = dict(periodic=periodic, r0=r0, r1=r1, theta=theta)
    sample = prepare_metric_sample(D, metric_subsample) if sample is None else sample
    indices, target = sample["indices"], sample["target"]
    distances = layout_distance_matrix(X[indices], alpha=alpha, **geometry)
    if not np.isfinite(distances).all():
        raise ValueError("Nonfinite embedding distances")
    if method.startswith("TorusMDS_smart_"):
        # Smart trains on D / D.max(); restore raw graph-distance units.
        evaluation_scale = sample["diameter"]
    elif method.startswith("TorusMDS") and method != "TorusMDS_spectral_init":
        evaluation_scale = 1.
    else:
        # Spectral initialization saves no learned alpha, like the baselines.
        evaluation_scale = fitted_distance_scale(distances, target)
    exact_crossings = not metric_subsample or len(X) <= metric_subsample
    edge_results = edge_metrics(X, edges, threads=threads, compute_crossings=exact_crossings,
                                alpha=alpha, **geometry)
    mean_edge_length = edge_results["mean_edge_length"]
    # Only SBM manifests carry the planted-block count k in this project.
    # Other graph families retain blank cluster-dependent CSV columns.
    separation = np.nan
    silhouette = np.nan
    labels = None
    k = row.get("k", np.nan)
    if pd.notna(k):
        labels = sbm_labels(len(X), float(k))
        separation = cluster_distance(X, labels, alpha=alpha, **geometry)
        if len(indices) == len(X):
            silhouette = silhouette_score(distances, labels)
    normalized = (separation / mean_edge_length
                  if np.isfinite(mean_edge_length) and mean_edge_length > 0 else np.nan)
    result = {**distance_metrics(distances, target, rg=np_radius,
                                scale=evaluation_scale if np.isfinite(evaluation_scale) else 1.,
                                graph_ranks=sample["graph_ranks"], include_np=len(indices) == len(X)),
            **edge_results, "alpha": alpha,
            "alpha_eval": alpha*evaluation_scale,
            "cluster_distance": separation, "mean_edge_length": mean_edge_length,
            "cluster_distance_normalized": normalized, "silhouette_score": silhouette,
            "node_resolution": node_resolution(X, **geometry),
            "metric_sample_n": len(indices),
            "crossings_status": "exact" if exact_crossings else "skipped_vertex_limit"}
    if len(indices) < len(X):
        result.update(sampled_query_metrics(X, D, indices, labels=labels, rg=np_radius,
                                           alpha=alpha, **geometry))
    return result


def evaluate_layouts(*, output, layouts_dir=None, layouts_root=None, methods=None,
                     np_radius=2., checkpoint_every=20, overwrite=False, threads=1,
                     metric_subsample=1000):
    """Write/resume one row per selected successful manifest instance.

    Completed ok/partial rows are reused; error rows are retried and replaced.
    --overwrite starts fresh. Input layouts must remain unchanged when resuming.
    """
    if (layouts_dir is None) == (layouts_root is None):
        raise ValueError("Specify exactly one of layouts_dir and layouts_root")
    if not np.isfinite(np_radius) or np_radius <= 0 or checkpoint_every < 1:
        raise ValueError("np_radius and checkpoint_every must be positive")
    if not 1 <= threads <= numba.config.NUMBA_NUM_THREADS:
        raise ValueError(f"threads must be between 1 and {numba.config.NUMBA_NUM_THREADS}")
    if metric_subsample != 0 and metric_subsample < 2:
        raise ValueError("metric_subsample must be at least 2, or 0 for all nodes")
    output = Path(output)
    root = Path(layouts_dir or layouts_root).resolve()
    manifests = [root / "runs.csv"] if layouts_dir else sorted(root.rglob("runs.csv"))
    if not manifests or any(not p.is_file() for p in manifests):
        raise FileNotFoundError(f"No runs.csv found under {root}")
    if output.resolve() in manifests:
        raise ValueError("Output must not overwrite an input manifest")
    records = {}
    done = set()
    if output.exists() and not overwrite:
        existing = pd.read_csv(output)
        required = {"source_shard", "exp_idx", "method", "metric_status", "np_radius",
                    *METRIC_COLUMNS, "alpha", "alpha_eval", "metric_subsample",
                    "metric_sample_n", "metric_sample_seed", "crossings_status"}
        if not required.issubset(existing.columns):
            raise ValueError("Existing CSV is incompatible with resume (missing columns); "
                             "use --overwrite to recompute it")
        if (existing["source_shard"].isna().any()
                or existing["source_shard"].isin([".", ""]).any()):
            raise ValueError("Existing CSV lacks unambiguous shard identities; use --overwrite")
        if existing.duplicated(["source_shard", "exp_idx", "method"]).any():
            raise ValueError("Existing CSV has duplicate instance keys; use --overwrite")
        if not existing["np_radius"].eq(np_radius).all():
            raise ValueError("Existing CSV uses a different np_radius; use another output or --overwrite")
        if (not existing["metric_subsample"].eq(metric_subsample).all()
                or not existing["metric_sample_seed"].eq(42).all()):
            raise ValueError("Existing CSV uses different sampling settings; use another output or --overwrite")
        for record in existing.to_dict("records"):
            key = (record["source_shard"], int(record["exp_idx"]), record["method"])
            records[key] = record
            if record["metric_status"] in {"ok", "partial"}:
                done.add(key)
        print(f"Resuming: {len(done)} completed rows; failed rows will be retried.", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint():
        frame = pd.DataFrame(list(records.values()))
        for column in ["source_shard", "exp_idx", "method", *METRIC_COLUMNS,
                       "alpha", "alpha_eval", "np_radius", "metric_subsample", "metric_sample_n",
                       "metric_sample_seed", "crossings_status", "metric_status", "metric_error"]:
            if column not in frame:
                frame[column] = pd.Series(dtype=object)
        temporary = output.with_name(output.name + ".tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(output)
        return frame

    since_checkpoint = 0
    for manifest_path in manifests:
        shard = manifest_path.parent
        source = _source_shard(shard)
        manifest = pd.read_csv(manifest_path)
        required = {"exp_idx", "method", "status"}
        if not required.issubset(manifest.columns):
            raise ValueError(f"{manifest_path}: missing {required - set(manifest.columns)}")
        selected = manifest.loc[manifest.status.eq("ok")]
        if methods is not None:
            selected = selected.loc[selected.method.isin(methods)]
        if selected.duplicated(["exp_idx", "method"]).any():
            raise ValueError(f"{manifest_path}: duplicate successful instances")
        for exp_idx, group in selected.groupby("exp_idx", sort=True):
            pending = [row for _, row in group.iterrows()
                       if (source, int(exp_idx), row["method"]) not in done]
            if not pending:
                continue  # Avoid loading/APSP when every selected layout is complete.
            graph_error = None
            D = edges = sample = None  # Release the previous graph before loading the next.
            try:
                D, edges = load_layout_graph(
                    shard / "graphs" / f"graph_{int(exp_idx)}.npz", threads=threads)
                sample = prepare_metric_sample(D, metric_subsample)
            except Exception as exc:
                graph_error = str(exc)
            for row in pending:
                key = (source, int(exp_idx), row["method"])
                record = {**row.to_dict(), "source_shard": source, "np_radius": np_radius,
                          "metric_subsample": metric_subsample, "metric_sample_seed": 42,
                          "metric_sample_n": np.nan, "crossings_status": "not_computed",
                          **dict.fromkeys(METRIC_COLUMNS, np.nan), "alpha": np.nan, "alpha_eval": np.nan,
                          "metric_status": "error", "metric_error": ""}
                try:
                    if graph_error is not None:
                        raise ValueError(graph_error)
                    if pd.notna(row.get("n")) and int(row["n"]) != len(D):
                        raise ValueError("Manifest node count does not match graph")
                    X = np.load(shard / "coords" / f"exp_{int(exp_idx)}_{row['method']}.npy", allow_pickle=False)
                    record.update(score_layout(X, D, edges, row["method"], row, np_radius, threads,
                                               metric_subsample=metric_subsample, sample=sample))
                    record.update(metric_status="ok")
                    if np.isnan(record["incidence_angle"]):
                        record.update(metric_status="partial", metric_error="Coincident linked endpoints: incidence undefined")
                except Exception as exc:
                    record["metric_error"] = str(exc)
                records[key] = record
                if record["metric_status"] in {"ok", "partial"}:
                    done.add(key)
                print(f"{record['source_shard']} {exp_idx} {row['method']}: {record['metric_status']}", flush=True)
                since_checkpoint += 1
                if since_checkpoint >= checkpoint_every:
                    checkpoint()
                    since_checkpoint = 0
    return checkpoint()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--layouts-dir", type=Path)
    inputs.add_argument("--layouts-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--np-radius", type=float, default=2.)
    parser.add_argument("--metric-subsample", type=int, default=1000,
                        help="Max sampled vertices and exact-crossing vertex limit (default: 1000; 0: all nodes)")
    parser.add_argument("--checkpoint-every", type=int, default=20,
                        help="Save every N newly evaluated rows (default: 20)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute from scratch instead of resuming an existing CSV")
    parser.add_argument("--threads", type=int, default=1,
                        help="CPU threads for unweighted shortest paths and crossings (default: 1; try 4 or 8)")
    args = parser.parse_args()
    try:
        result = evaluate_layouts(**vars(args))
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(f"Saved {len(result)} rows to {args.output}")
    return 1 if result.metric_status.eq("error").any() else 0


if __name__ == "__main__":
    raise SystemExit(main())
