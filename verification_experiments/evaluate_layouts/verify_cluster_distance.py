"""Reproducible saved-SBM and analytic-layout checks for cluster_distance.

Run from the repository root; writes verification CSVs without changing inputs.
An independent separating-axis / point-to-segment oracle checks signed hull
distances. The oracle shares only the documented hull-lifting convention.
"""
import argparse
from pathlib import Path
import shutil
import sys
import tempfile
import time

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, triu

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from modules.geometry import gauss_reduce_basis
from verification_experiments.evaluate_layouts.cluster_distance import (
    cluster_distance, sbm_labels, _compact_lift, _hull,
)
from verification_experiments.evaluate_layouts.evaluate_layouts import evaluate_layouts

METHODS = ["s_gd2", "TorusMDS_random_1x_scale2", "TorusMDS_smart_1x_scale2",
           "TorusMDS_parallelogram", "TorusMDS_spectral_init", "wrap_python"]


def oracle_hulls(a, b):
    """Independent polygon signed distance: SAT penetration or boundary distance."""
    axes = []
    for polygon in (a, b):
        edges = np.roll(polygon, -1, axis=0) - polygon
        lengths = np.linalg.norm(edges, axis=1)
        axes.extend(np.column_stack([-edges[:, 1], edges[:, 0]]) / lengths[:, None])
    penetration = np.inf
    separated = False
    for axis in axes:
        pa, pb = a @ axis, b @ axis
        if pa.max() < pb.min() or pb.max() < pa.min():
            separated = True
        penetration = min(penetration, pa.max()-pb.min(), pb.max()-pa.min())
    if not separated:
        return -penetration
    best = np.inf
    for vertices, polygon in [(a, b), (b, a)]:
        ends = np.roll(polygon, -1, axis=0)
        for start, end in zip(polygon, ends):
            edge = end - start
            t = np.clip((vertices-start) @ edge / (edge @ edge), 0., 1.)
            best = min(best, np.linalg.norm(vertices-start-t[:, None]*edge, axis=1).min())
    return best


def oracle_metric(X, labels, periodic, alpha, r0, r1, theta, radius=2):
    """Use a wider 5x5 image neighbourhood to also check the production 3x3 rule."""
    basis = alpha*np.eye(2)
    if periodic:
        if np.isclose(theta, np.pi/2):
            basis = alpha*np.diag([r0, r1])
        else:
            x, y, scale, transform = gauss_reduce_basis(r1/r0*np.cos(theta), r1/r0*np.sin(theta))
            X = X @ transform.T
            basis = alpha*r0*scale*np.array([[1., x], [0., y]])
        hulls = [_hull(_compact_lift(X[labels == g]) @ basis.T) for g in np.unique(labels)]
        shifts = np.array([(a, b) for a in range(-radius, radius+1)
                           for b in range(-radius, radius+1)]) @ basis.T
    else:
        hulls = [_hull(alpha*X[labels == g]) for g in np.unique(labels)]
        shifts = np.zeros((1, 2))
    if any(len(h) < 3 for h in hulls):
        raise ValueError("The independent polygon oracle requires nondegenerate hulls")
    return np.mean([min(oracle_hulls(a, b + shift) for shift in shifts)
                    for i, a in enumerate(hulls) for b in hulls[i+1:]])


def square(x, y, size):
    return np.array([[x, y], [x+size, y], [x+size, y+size], [x, y+size]])


def verify(layouts_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = sbm_labels(8, 2)
    cases = [
        ("separated", square(0, 0, 1), square(2, 0, 1), False, 1.),
        ("diagonal", square(0, 0, 1), square(2, 2, 1), False, np.sqrt(2)),
        ("touching", square(0, 0, 1), square(1, 0, 1), False, 0.),
        ("overlapping", square(0, 0, 1), square(.75, 0, 1), False, -.25),
        ("contained", square(0, 0, 4), square(1, 1, 1), False, -2.),
        ("torus_seam_separated", square(.95, .45, .1) % 1, square(.25, .45, .1), True, .2),
        ("torus_seam_overlap", square(.95, .45, .1) % 1, square(.98, .48, .04) % 1, True, -.07),
    ]
    records = []
    for name, a, b, periodic, expected in cases:
        X = np.vstack([a, b])
        geometry = dict(periodic=periodic, alpha=1., r0=1., r1=1., theta=np.pi/2)
        measured = cluster_distance(X, labels, **geometry)
        independent = oracle_metric(X, labels, **geometry)
        np.testing.assert_allclose([measured, independent], expected, atol=1e-11)
        records.append(dict(case=name, expected=expected, measured=measured,
                            oracle=independent, passed=True))
    pd.DataFrame(records).to_csv(output_dir / "analytic_layouts.csv", index=False)

    manifest = pd.read_csv(layouts_dir / "runs.csv")
    selected = manifest.loc[manifest.status.eq("ok") & manifest.method.isin(METHODS)]
    complete = [idx for idx, group in selected.groupby("exp_idx")
                if set(group.method) == set(METHODS) and group.n.nunique() == 1]
    if len(complete) < 3:
        raise ValueError("Need at least three graphs with all six selected methods")
    sizes = selected.loc[selected.exp_idx.isin(complete)].groupby("exp_idx").n.first().sort_values()
    graph_ids = sizes.iloc[[0, len(sizes)//2, -1]].index.tolist()
    selected = selected.loc[selected.exp_idx.isin(graph_ids)]
    graph_checks = []
    for idx in graph_ids:
        row = selected.loc[selected.exp_idx.eq(idx)].iloc[0]
        adjacency = load_npz(layouts_dir / "graphs" / f"graph_{idx}.npz")
        n, k = adjacency.shape[0], int(row.k)
        labels = sbm_labels(n, k)
        upper = triu(adjacency, k=1, format="coo")
        inside = int(np.count_nonzero(labels[upper.row] == labels[upper.col]))
        block_sizes = np.bincount(labels)
        possible_inside = int(np.sum(block_sizes*(block_sizes-1)//2))
        possible_outside = n*(n-1)//2 - possible_inside
        p_in = inside / possible_inside
        p_out = (upper.nnz-inside) / possible_outside
        assert p_in > p_out, "Reconstructed blocks do not match the planted dense communities"
        graph_checks.append(dict(exp_idx=idx, n=n, k=k, m=upper.nnz,
                                 observed_p_in=p_in, observed_p_out=p_out))
    pd.DataFrame(graph_checks).to_csv(output_dir / "sbm_graph_checks.csv", index=False)
    records = []
    for _, row in selected.iterrows():
        idx, method = int(row.exp_idx), row.method
        n = load_npz(layouts_dir / "graphs" / f"graph_{idx}.npz").shape[0]
        assert n == int(row.n)
        X = np.load(layouts_dir / "coords" / f"exp_{idx}_{method}.npy")
        assert X.shape == (n, 2)
        labels = sbm_labels(n, row.k)
        def parameter(name, default):
            return default if pd.isna(row.get(name, np.nan)) else float(row[name])
        geometry = dict(periodic=method != "s_gd2", alpha=parameter("alpha_fit", 1.),
                        r0=parameter("r0_fit", 1.), r1=parameter("r1_fit", 1.),
                        theta=parameter("theta_fit", np.pi/2))
        start = time.perf_counter()
        measured = cluster_distance(X, labels, **geometry)
        elapsed = time.perf_counter() - start
        independent = oracle_metric(X, labels, **geometry)
        shifted = cluster_distance(X + [.137, .293], labels, **geometry)
        scaled = cluster_distance(X, labels, **{**geometry, "alpha": 3*geometry["alpha"]})
        permutation = np.random.default_rng(4).permutation(n)
        permuted = cluster_distance(X[permutation], labels[permutation], **geometry)
        np.testing.assert_allclose([independent, shifted, scaled/3, permuted], measured, atol=1e-10)
        records.append(dict(exp_idx=idx, n=n, k=int(row.k), method=method,
                            cluster_distance=measured, oracle=independent,
                            oracle_error=abs(independent-measured),
                            translation_error=abs(shifted-measured),
                            scale_error=abs(scaled/3-measured),
                            permutation_error=abs(permuted-measured), seconds=elapsed, passed=True))
        print(f"{idx} n={n} {method}: {measured:.8g}, oracle error={abs(independent-measured):.2g}", flush=True)
    pd.DataFrame(records).to_csv(output_dir / "saved_sbm_layouts.csv", index=False)

    # Full production evaluator on one actual saved graph, all six methods.
    # Temporary input view avoids running every other graph in the shard.
    idx = graph_ids[0]
    with tempfile.TemporaryDirectory() as temporary:
        shard = Path(temporary)
        (shard / "graphs").mkdir()
        (shard / "coords").mkdir()
        subset = selected.loc[selected.exp_idx.eq(idx)]
        subset.to_csv(shard / "runs.csv", index=False)
        shutil.copy2(layouts_dir / "graphs" / f"graph_{idx}.npz", shard / "graphs")
        for method in subset.method:
            shutil.copy2(layouts_dir / "coords" / f"exp_{idx}_{method}.npy", shard / "coords")
        result = evaluate_layouts(layouts_dir=shard, output=output_dir / "evaluation.csv", overwrite=True)
    assert result.metric_status.eq("ok").all()
    for _, row in result.iterrows():
        expected = next(r["cluster_distance"] for r in records
                        if r["exp_idx"] == idx and r["method"] == row.method)
        np.testing.assert_allclose(row.cluster_distance, expected, atol=1e-12)
    print(f"PASS: {len(cases)} analytic cases, {len(records)} saved layouts, {len(result)} full CSV rows.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layouts-dir", type=Path,
                        default=Path("verification_experiments/layouts/sbm_normalized/100_1000/shard_0"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("verification_experiments/evaluate_layouts/validation"))
    args = parser.parse_args()
    verify(args.layouts_dir, args.output_dir)
