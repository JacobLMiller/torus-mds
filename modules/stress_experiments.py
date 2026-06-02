from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence
import math
import time

import matplotlib.pyplot as plt
import networkx as nx
import numba
from numba.typed import List as NumbaList
import numpy as np
import pandas as pd

from . import graphio, metrics, visualization
from .geometry import euclidean_grad, make_torus_geod, stress_and_grad_rect_torus
from .projector import sgd_minibatch_njit


@dataclass(frozen=True)
class OptimizerVariant:
    name: str
    description: str
    sampling: str
    order: str
    updates: str
    full_sweep: bool = False
    deterministic: bool = False
    max_n: int | None = None


@dataclass
class BenchmarkSpec:
    name: str
    group: str
    graph: nx.Graph
    distances: np.ndarray
    colors: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingRun:
    dataset: str
    group: str
    variant: str
    seed: int
    coords: np.ndarray
    alpha_: float
    r0_: float
    r1_: float
    theta_: float
    runtime_s: float
    pair_evals: int
    epochs: int
    description: str

    @property
    def torus_embedding_(self) -> np.ndarray:
        return self.coords

    def geod(self) -> Callable[[np.ndarray, np.ndarray], float]:
        return make_torus_geod(self.alpha_, self.r0_, self.r1_, self.theta_)


VARIANTS: tuple[OptimizerVariant, ...] = (
    OptimizerVariant(
        name="baseline_random_minibatch",
        description="Current baseline: random unordered pairs with replacement and immediate pairwise updates.",
        sampling="random pairs with replacement",
        order="fully random",
        updates="online pairwise",
    ),
    OptimizerVariant(
        name="sampled_unique_online",
        description="Minibatch sampled without replacement from unique unordered pairs; updates applied immediately.",
        sampling="unique unordered minibatch",
        order="random minibatch",
        updates="online pairwise",
    ),
    OptimizerVariant(
        name="sampled_unique_sync_batchavg",
        description="Unique unordered minibatch with synchronous node updates from accumulated pair gradients.",
        sampling="unique unordered minibatch",
        order="random minibatch",
        updates="synchronous batch average",
    ),
    OptimizerVariant(
        name="sampled_unique_sync_countnorm",
        description="Synchronous minibatch update with per-node averaging by visit count.",
        sampling="unique unordered minibatch",
        order="random minibatch",
        updates="synchronous per-node average",
    ),
    OptimizerVariant(
        name="sampled_unique_sync_sqrtnorm",
        description="Synchronous minibatch update with inverse-sqrt visit normalization to keep highly visited nodes stable while preserving movement scale.",
        sampling="unique unordered minibatch",
        order="random minibatch",
        updates="synchronous inverse-sqrt visit normalization",
    ),
    OptimizerVariant(
        name="sampled_unique_sync_hybridnorm",
        description="Two-stage torus optimizer: early inverse-sqrt normalized synchronous updates, followed by per-node average normalization for refinement.",
        sampling="unique unordered minibatch",
        order="random minibatch, staged schedule",
        updates="synchronous sqrtnorm then synchronous countnorm",
    ),
    OptimizerVariant(
        name="stratified_distance_sync_sqrtnorm",
        description="Inverse-sqrt normalized synchronous updates with minibatches balanced across short, medium, and long target graph distances.",
        sampling="distance-stratified unique minibatch",
        order="random within distance strata",
        updates="synchronous inverse-sqrt visit normalization",
    ),
    OptimizerVariant(
        name="coverage_unique_sync_sqrtnorm",
        description="Inverse-sqrt normalized synchronous updates with minibatches assembled from repeated near-perfect node-covering rounds.",
        sampling="coverage-constrained unique minibatch",
        order="random node-cover rounds",
        updates="synchronous inverse-sqrt visit normalization",
    ),
    OptimizerVariant(
        name="anchor_unique_online",
        description="Anchor-node sweep: random node order, a few unique partners per anchor, immediate pairwise updates.",
        sampling="anchor-driven unique partners",
        order="random anchor sweep",
        updates="online pairwise",
    ),
    OptimizerVariant(
        name="full_sweep_sync_sqrtnorm",
        description="Small-graph control: shuffled full sweep over all unique pairs with synchronous inverse-sqrt normalized updates.",
        sampling="all unique unordered pairs",
        order="random full sweep",
        updates="synchronous inverse-sqrt visit normalization",
        full_sweep=True,
        max_n=120,
    ),
)

VARIANT_MAP = {variant.name: variant for variant in VARIANTS}


def variant_catalog_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "variant": variant.name,
                "sampling": variant.sampling,
                "order": variant.order,
                "updates": variant.updates,
                "full_sweep": variant.full_sweep,
                "max_n": variant.max_n,
                "description": variant.description,
            }
            for variant in VARIANTS
        ]
    )


def benchmark_catalog_df(specs: Sequence[BenchmarkSpec]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        row = {
            "dataset": spec.name,
            "group": spec.group,
            "n_nodes": spec.graph.number_of_nodes(),
            "n_edges": spec.graph.number_of_edges(),
        }
        row.update(spec.metadata)
        rows.append(row)
    return pd.DataFrame(rows)


def _largest_connected_component(graph: nx.Graph) -> nx.Graph:
    if graph.number_of_nodes() == 0 or nx.is_connected(graph):
        return graph.copy()
    nodes = max(nx.connected_components(graph), key=len)
    return graph.subgraph(nodes).copy()


def _set_unit_weights(graph: nx.Graph) -> nx.Graph:
    nx.set_edge_attributes(graph, 1.0, "weight")
    return graph


def _build_knn_graph(points: np.ndarray, k: int) -> nx.Graph:
    n = points.shape[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    dists = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    for i in range(n):
        nbrs = np.argpartition(dists[i], k)[:k]
        for j in nbrs:
            graph.add_edge(int(i), int(j))
    if not nx.is_connected(graph):
        components = [np.array(sorted(component), dtype=np.int32) for component in nx.connected_components(graph)]
        while len(components) > 1:
            base = components[0]
            best_dist = np.inf
            best_pair: tuple[int, int] | None = None
            best_component_index = -1
            for comp_index in range(1, len(components)):
                other = components[comp_index]
                local = dists[np.ix_(base, other)]
                flat_index = int(np.argmin(local))
                i_idx, j_idx = np.unravel_index(flat_index, local.shape)
                candidate_dist = float(local[i_idx, j_idx])
                if candidate_dist < best_dist:
                    best_dist = candidate_dist
                    best_pair = (int(base[i_idx]), int(other[j_idx]))
                    best_component_index = comp_index
            assert best_pair is not None
            graph.add_edge(*best_pair)
            merged = np.concatenate([base, components[best_component_index]])
            components = [merged] + [components[i] for i in range(1, len(components)) if i != best_component_index]
    return _set_unit_weights(graph)


def build_representative_benchmarks(
    *,
    chen_indices: Sequence[int] | None = None,
    include_large_grid: bool = True,
    planted_seed: int = 0,
) -> list[BenchmarkSpec]:
    specs: list[BenchmarkSpec] = []

    grid_sizes = [(10, 10), (20, 20), (10, 30)]
    if include_large_grid:
        grid_sizes += [(30, 30), (40, 40), (20,60), (15, 75), (25, 35)]
    for nx_size, ny_size in grid_sizes:
        graph, distances = graphio.get_periodic_lattice(nx_size, ny_size)
        specs.append(
            BenchmarkSpec(
                name=f"grid_{nx_size}x{ny_size}",
                group="grid_graphs",
                graph=graph,
                distances=distances,
                metadata={"nx_size": nx_size, "ny_size": ny_size},
            )
        )

    chen_graphs = sorted(graphio.load_chen_graphs("chengraphs/*.json"), key=lambda item: int(item[0]))
    if chen_indices is None:
        chen_indices = tuple(range(1, len(chen_graphs) + 1))
    for chen_index in chen_indices:
        name, graph, given_layout, distances = chen_graphs[chen_index - 1]
        specs.append(
            BenchmarkSpec(
                name=f"chen_{name}",
                group="chengraphs",
                graph=graph,
                distances=distances,
                metadata={"source_layout_available": given_layout is not None},
            )
        )

    block_specs = ((4, 40, 0.2, 0.01), (4, 80, 0.2, 0.01), (5, 40, 0.2, 0.01), (5, 80, 0.2, 0.01))
    if include_large_grid:
        block_specs += ((6, 40, 0.2, 0.01), (6, 80, 0.2, 0.01), (7, 40, 0.2, 0.01), (7, 80, 0.2, 0.01))
    for k, nodes_per_block, p_in, p_out in block_specs:
        graph = nx.planted_partition_graph(k, nodes_per_block, p_in, p_out, seed=planted_seed)
        distances, _ = graphio.apsp_distance_matrix(graph)
        colors = np.repeat(np.arange(k), nodes_per_block)
        specs.append(
            BenchmarkSpec(
                name=f"planted_k{k}_n{nodes_per_block}",
                group="planted_partition",
                graph=graph,
                distances=distances,
                colors=colors,
                metadata={"k": k, "nodes_per_block": nodes_per_block, "p_in": p_in, "p_out": p_out},
            )
        )
    return specs


def build_diverse_control_benchmarks(*, seed: int = 0) -> list[BenchmarkSpec]:
    rng = np.random.default_rng(seed)
    specs: list[BenchmarkSpec] = []

    # Torus-native graphs
    for name, nx_size, ny_size, diagonal in (
        ("torus_grid_10x10", 10, 10, False),
        ("torus_grid_12x24", 12, 24, False),
        ("torus_diag_10x10", 10, 10, True),
        ("torus_diag_16x16", 16, 16, True),
    ):
        graph = graphio.periodic_lattice_graph(nx_size, ny_size, diagonal=diagonal)
        distances, _ = graphio.apsp_distance_matrix(graph)
        specs.append(
            BenchmarkSpec(
                name=name,
                group="torus_native",
                graph=graph,
                distances=distances,
                metadata={"nx_size": nx_size, "ny_size": ny_size, "diagonal": diagonal},
            )
        )

    # Euclidean-native graphs
    for name, nx_size, ny_size, diagonal in (
        ("planar_grid_10x10", 10, 10, False),
        ("planar_diag_grid_12x12", 12, 12, True),
    ):
        graph = nx.grid_2d_graph(nx_size, ny_size, periodic=False)
        if diagonal:
            for i in range(nx_size - 1):
                for j in range(ny_size - 1):
                    graph.add_edge((i, j), (i + 1, j + 1))
                    graph.add_edge((i + 1, j), (i, j + 1))
        graph = _set_unit_weights(nx.convert_node_labels_to_integers(graph))
        distances, _ = graphio.apsp_distance_matrix(graph)
        specs.append(
            BenchmarkSpec(
                name=name,
                group="euclidean_native",
                graph=graph,
                distances=distances,
                metadata={"nx_size": nx_size, "ny_size": ny_size, "diagonal": diagonal},
            )
        )

    geometric = nx.random_geometric_graph(140, 0.19, seed=seed)
    geometric = _set_unit_weights(_largest_connected_component(nx.convert_node_labels_to_integers(geometric)))
    distances, _ = graphio.apsp_distance_matrix(geometric)
    specs.append(
        BenchmarkSpec(
            name="random_geometric_140",
            group="euclidean_native",
            graph=geometric,
            distances=distances,
            metadata={"generator": "random_geometric_graph", "n": 140, "radius": 0.19},
        )
    )

    cluster_a = rng.normal(loc=(-1.2, -0.2), scale=(0.35, 0.18), size=(70, 2))
    cluster_b = rng.normal(loc=(1.0, 0.4), scale=(0.30, 0.22), size=(70, 2))
    knn_points = np.vstack([cluster_a, cluster_b])
    knn_graph = _build_knn_graph(knn_points, k=8)
    distances, _ = graphio.apsp_distance_matrix(knn_graph)
    specs.append(
        BenchmarkSpec(
            name="knn_gaussian_clusters_140",
            group="euclidean_native",
            graph=knn_graph,
            distances=distances,
            metadata={"generator": "knn_gaussian_clusters", "n": 140, "k": 8},
        )
    )

    # Random / structured graphs
    er_graph = nx.erdos_renyi_graph(120, 0.06, seed=seed)
    er_graph = _set_unit_weights(_largest_connected_component(nx.convert_node_labels_to_integers(er_graph)))
    distances, _ = graphio.apsp_distance_matrix(er_graph)
    specs.append(
        BenchmarkSpec(
            name="erdos_renyi_120",
            group="random_structured",
            graph=er_graph,
            distances=distances,
            metadata={"generator": "erdos_renyi", "n": 120, "p": 0.06},
        )
    )

    ws_graph = nx.watts_strogatz_graph(120, 6, 0.15, seed=seed)
    ws_graph = _set_unit_weights(_largest_connected_component(nx.convert_node_labels_to_integers(ws_graph)))
    distances, _ = graphio.apsp_distance_matrix(ws_graph)
    specs.append(
        BenchmarkSpec(
            name="watts_strogatz_120",
            group="random_structured",
            graph=ws_graph,
            distances=distances,
            metadata={"generator": "watts_strogatz", "n": 120, "k": 6, "p": 0.15},
        )
    )

    ba_graph = _set_unit_weights(nx.barabasi_albert_graph(140, 3, seed=seed))
    distances, _ = graphio.apsp_distance_matrix(ba_graph)
    specs.append(
        BenchmarkSpec(
            name="barabasi_albert_140",
            group="random_structured",
            graph=ba_graph,
            distances=distances,
            metadata={"generator": "barabasi_albert", "n": 140, "m": 3},
        )
    )

    hypercube = _set_unit_weights(nx.convert_node_labels_to_integers(nx.hypercube_graph(7)))
    distances, _ = graphio.apsp_distance_matrix(hypercube)
    specs.append(
        BenchmarkSpec(
            name="hypercube_7",
            group="random_structured",
            graph=hypercube,
            distances=distances,
            metadata={"generator": "hypercube", "dimension": 7},
        )
    )

    tree = _set_unit_weights(nx.convert_node_labels_to_integers(nx.balanced_tree(3, 4)))
    distances, _ = graphio.apsp_distance_matrix(tree)
    specs.append(
        BenchmarkSpec(
            name="balanced_tree_3_4",
            group="random_structured",
            graph=tree,
            distances=distances,
            metadata={"generator": "balanced_tree", "branching": 3, "height": 4},
        )
    )

    # Keep a few previous families for continuity.
    chen_graphs = sorted(graphio.load_chen_graphs("chengraphs/*.json"), key=lambda item: int(item[0]))
    for name, graph, given_layout, distances in chen_graphs:
        specs.append(
            BenchmarkSpec(
                name=f"chen_{name}",
                group="chengraphs",
                graph=graph,
                distances=distances,
                metadata={"source_layout_available": given_layout is not None},
            )
        )

    for k, nodes_per_block, p_in, p_out in ((4, 40, 0.2, 0.01), (4, 80, 0.2, 0.01), (6, 40, 0.2, 0.01)):
        graph = nx.planted_partition_graph(k, nodes_per_block, p_in, p_out, seed=seed)
        graph = _set_unit_weights(nx.convert_node_labels_to_integers(graph))
        distances, _ = graphio.apsp_distance_matrix(graph)
        specs.append(
            BenchmarkSpec(
                name=f"planted_k{k}_n{nodes_per_block}",
                group="planted_partition",
                graph=graph,
                distances=distances,
                colors=np.repeat(np.arange(k), nodes_per_block),
                metadata={"k": k, "nodes_per_block": nodes_per_block, "p_in": p_in, "p_out": p_out},
            )
        )

    return specs


def euclidean_distance(
    p1: np.ndarray,
    p2: np.ndarray,
    *,
    alpha: float = 1.0,
) -> float:
    return alpha * float(np.linalg.norm(np.asarray(p2) - np.asarray(p1)))


def make_euclidean_geod(alpha: float) -> Callable[[np.ndarray, np.ndarray], float]:
    return lambda p, q: euclidean_distance(p, q, alpha=alpha)


def _legacy_random_init(n: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.rand(n, 2).astype(np.float64, copy=False)


def _normalize_learn_mode(learn_mode: int | str) -> int:
    if isinstance(learn_mode, str):
        mapping = {"fixed": 0, "alpha": 1, "square": 2, "rectangular": 3}
        return mapping[learn_mode]
    return int(learn_mode)


def _all_unique_pairs(n: int) -> np.ndarray:
    ii, jj = np.triu_indices(n, 1)
    return np.column_stack([ii, jj]).astype(np.int32, copy=False)


def _to_numba_list(items: Sequence[np.ndarray]) -> NumbaList:
    result = NumbaList()
    for item in items:
        result.append(np.asarray(item, dtype=np.int32))
    return result


def _build_sampled_unique_pair_sequence(n: int, epochs: int, batch_pairs: int, seed: int) -> NumbaList:
    rng = np.random.default_rng(seed)
    pairs = _all_unique_pairs(n)
    total_pairs = len(pairs)
    items = []
    for _ in range(epochs):
        take = min(batch_pairs, total_pairs)
        choice = rng.choice(total_pairs, size=take, replace=False)
        items.append(pairs[choice])
    return _to_numba_list(items)


def _build_distance_stratified_pair_sequence(data: np.ndarray, epochs: int, batch_pairs: int, seed: int) -> NumbaList:
    rng = np.random.default_rng(seed)
    pairs = _all_unique_pairs(data.shape[0])
    total_pairs = len(pairs)
    take = min(batch_pairs, total_pairs)
    if total_pairs == 0:
        return _to_numba_list([np.empty((0, 2), dtype=np.int32) for _ in range(epochs)])

    pair_distances = data[pairs[:, 0], pairs[:, 1]]
    positive = pair_distances[pair_distances > 0]
    if positive.size == 0:
        return _build_sampled_unique_pair_sequence(data.shape[0], epochs, batch_pairs, seed)

    q1, q2 = np.quantile(positive, [1.0 / 3.0, 2.0 / 3.0])
    bins = [
        np.flatnonzero(pair_distances <= q1),
        np.flatnonzero((pair_distances > q1) & (pair_distances <= q2)),
        np.flatnonzero(pair_distances > q2),
    ]
    bins = [b for b in bins if b.size > 0]
    if len(bins) < 2:
        return _build_sampled_unique_pair_sequence(data.shape[0], epochs, batch_pairs, seed)

    items: list[np.ndarray] = []
    base = take // len(bins)
    remainder = take % len(bins)
    for _ in range(epochs):
        selected: list[int] = []
        used = np.zeros(total_pairs, dtype=bool)
        for bin_index, bin_ids in enumerate(bins):
            target = base + (1 if bin_index < remainder else 0)
            if target <= 0:
                continue
            local_take = min(target, bin_ids.size)
            choice = rng.choice(bin_ids, size=local_take, replace=False)
            selected.extend(choice.tolist())
            used[choice] = True

        if len(selected) < take:
            remaining = np.flatnonzero(~used)
            extra_take = min(take - len(selected), remaining.size)
            if extra_take > 0:
                extra = rng.choice(remaining, size=extra_take, replace=False)
                selected.extend(extra.tolist())
        items.append(pairs[np.asarray(selected, dtype=np.int32)])
    return _to_numba_list(items)


def _build_coverage_constrained_pair_sequence(n: int, epochs: int, batch_pairs: int, seed: int) -> NumbaList:
    rng = np.random.default_rng(seed)
    total_pairs = n * (n - 1) // 2
    take = min(batch_pairs, total_pairs)
    items: list[np.ndarray] = []
    nodes = np.arange(n, dtype=np.int32)
    for _ in range(epochs):
        selected: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()

        while len(selected) < take:
            perm = rng.permutation(nodes)
            for start in range(0, max(len(perm) - 1, 0), 2):
                i = int(perm[start])
                j = int(perm[start + 1])
                if i == j:
                    continue
                pair = (i, j) if i < j else (j, i)
                if pair in seen:
                    continue
                seen.add(pair)
                selected.append(pair)
                if len(selected) >= take:
                    break
            if len(selected) >= take:
                break

            if len(seen) >= total_pairs:
                break

            # Fill gaps when repeated matching rounds stall.
            anchors = rng.permutation(nodes)
            for i in anchors:
                if len(selected) >= take:
                    break
                candidates = rng.permutation(nodes)
                for j in candidates:
                    if i == j:
                        continue
                    pair = (int(i), int(j)) if i < j else (int(j), int(i))
                    if pair in seen:
                        continue
                    seen.add(pair)
                    selected.append(pair)
                    break

        items.append(np.asarray(selected, dtype=np.int32))
    return _to_numba_list(items)


def _build_random_replacement_pair_sequence(n: int, epochs: int, batch_pairs: int, seed: int) -> NumbaList:
    rng = np.random.default_rng(seed)
    items = []
    for _ in range(epochs):
        epoch_pairs = np.empty((batch_pairs, 2), dtype=np.int32)
        drawn = 0
        while drawn < batch_pairs:
            ij = rng.integers(0, n, size=2, endpoint=False)
            if ij[0] == ij[1]:
                continue
            i = int(ij[0])
            j = int(ij[1])
            if i < j:
                epoch_pairs[drawn, 0] = i
                epoch_pairs[drawn, 1] = j
            else:
                epoch_pairs[drawn, 0] = j
                epoch_pairs[drawn, 1] = i
            drawn += 1
        items.append(epoch_pairs)
    return _to_numba_list(items)


def _build_anchor_unique_pair_sequence(n: int, epochs: int, batch_pairs: int, seed: int) -> NumbaList:
    rng = np.random.default_rng(seed)
    per_anchor = max(1, batch_pairs // max(n, 1))
    items = []
    nodes = np.arange(n)
    for _ in range(epochs):
        epoch_chunks = []
        for i in rng.permutation(nodes):
            partner_count = min(per_anchor, n - 1)
            if partner_count <= 0:
                continue
            partners = rng.choice(n - 1, size=partner_count, replace=False)
            partners = np.where(partners >= i, partners + 1, partners)
            lo = np.minimum(i, partners)
            hi = np.maximum(i, partners)
            epoch_chunks.append(np.column_stack([lo, hi]).astype(np.int32, copy=False))
        epoch_pairs = np.vstack(epoch_chunks) if epoch_chunks else np.empty((0, 2), dtype=np.int32)
        items.append(epoch_pairs)
    return _to_numba_list(items)


def _build_full_sweep_pair_sequence(n: int, epochs: int, seed: int) -> NumbaList:
    rng = np.random.default_rng(seed)
    pairs = _all_unique_pairs(n)
    items = []
    for _ in range(epochs):
        items.append(pairs[rng.permutation(len(pairs))])
    return _to_numba_list(items)


@numba.njit(cache=True, fastmath=True)
def _run_pair_sequence_online_njit(
    data,
    pair_sequence,
    learning_rate,
    init_params,
    alpha_init=1.0,
    alpha_ema=0.05,
    eps=1e-12,
    alpha_min=1e-6,
    alpha_max=1e6,
    r0_init=1.0,
    r1_init=1.0,
    learn_mode=1,
    geom_lr=0.01,
    geom_min=1e-6,
    theta=np.pi / 2,
):
    n = data.shape[0]
    params = init_params.copy()
    alpha = alpha_init
    r0 = r0_init
    r1 = r1_init
    epochs = len(pair_sequence)

    for it in range(epochs):
        step_pos = max(1.0 / (learning_rate + it), 1e-4)
        seq = pair_sequence[it]
        used = seq.shape[0]
        num = 0.0
        den = 0.0
        gr0_sum = 0.0
        gr1_sum = 0.0

        for k in range(used):
            i = seq[k, 0]
            j = seq[k, 1]
            d = data[i, j]
            _, grad, dist, gr0_k, gr1_k = stress_and_grad_rect_torus(
                params[i], params[j], d, alpha, r0, r1, eps, theta
            )
            params[i, 0] -= step_pos * grad[0]
            params[i, 1] -= step_pos * grad[1]
            params[j, 0] += step_pos * grad[0]
            params[j, 1] += step_pos * grad[1]
            num += d * dist
            den += dist * dist
            gr0_sum += gr0_k
            gr1_sum += gr1_k

        params %= 1.0
        if used > 0:
            if learn_mode == 1:
                alpha_hat = num / (den + eps)
                alpha_hat = max(alpha_min, min(alpha_max, alpha_hat))
                alpha = (1.0 - alpha_ema) * alpha + alpha_ema * alpha_hat
            elif learn_mode == 2:
                r = max(geom_min, r0 - geom_lr * (gr0_sum + gr1_sum) / used)
                r0 = r
                r1 = r
            elif learn_mode == 3:
                r0 = max(geom_min, r0 - geom_lr * gr0_sum / used)
                r1 = max(geom_min, r1 - geom_lr * gr1_sum / used)
    return params, alpha, r0, r1


@numba.njit(cache=True, fastmath=True)
def _run_pair_sequence_sync_njit(
    data,
    pair_sequence,
    learning_rate,
    init_params,
    norm_mode=0,
    alpha_init=1.0,
    alpha_ema=0.05,
    eps=1e-12,
    alpha_min=1e-6,
    alpha_max=1e6,
    r0_init=1.0,
    r1_init=1.0,
    learn_mode=1,
    geom_lr=0.01,
    geom_min=1e-6,
    theta=np.pi / 2,
):
    n = data.shape[0]
    params = init_params.copy()
    alpha = alpha_init
    r0 = r0_init
    r1 = r1_init
    epochs = len(pair_sequence)

    for it in range(epochs):
        step_pos = max(1.0 / (learning_rate + it), 1e-4)
        seq = pair_sequence[it]
        used = seq.shape[0]
        grad_acc = np.zeros((n, 2))
        counts = np.zeros(n, dtype=np.int32)
        num = 0.0
        den = 0.0
        gr0_sum = 0.0
        gr1_sum = 0.0

        for k in range(used):
            i = seq[k, 0]
            j = seq[k, 1]
            d = data[i, j]
            _, grad, dist, gr0_k, gr1_k = stress_and_grad_rect_torus(
                params[i], params[j], d, alpha, r0, r1, eps, theta
            )
            grad_acc[i, 0] += grad[0]
            grad_acc[i, 1] += grad[1]
            grad_acc[j, 0] -= grad[0]
            grad_acc[j, 1] -= grad[1]
            counts[i] += 1
            counts[j] += 1
            num += d * dist
            den += dist * dist
            gr0_sum += gr0_k
            gr1_sum += gr1_k

        if used > 0:
            avg_count = 2.0 * used / n
            for i in range(n):
                if counts[i] == 0:
                    continue
                if norm_mode == 0:
                    scale = step_pos / used
                elif norm_mode == 1:
                    scale = step_pos / counts[i]
                else:
                    scale = step_pos * avg_count * math.sqrt(avg_count / counts[i]) / used
                params[i, 0] -= scale * grad_acc[i, 0]
                params[i, 1] -= scale * grad_acc[i, 1]

        params %= 1.0
        if used > 0:
            if learn_mode == 1:
                alpha_hat = num / (den + eps)
                alpha_hat = max(alpha_min, min(alpha_max, alpha_hat))
                alpha = (1.0 - alpha_ema) * alpha + alpha_ema * alpha_hat
            elif learn_mode == 2:
                r = max(geom_min, r0 - geom_lr * (gr0_sum + gr1_sum) / used)
                r0 = r
                r1 = r
            elif learn_mode == 3:
                r0 = max(geom_min, r0 - geom_lr * gr0_sum / used)
                r1 = max(geom_min, r1 - geom_lr * gr1_sum / used)
    return params, alpha, r0, r1


@numba.njit(cache=True, fastmath=True)
def _stress_and_grad_euclidean_scaled(p1, p2, d, alpha, eps=1e-12):
    dist, direction = euclidean_grad(p1, p2)
    dist = dist + eps
    diff = alpha * dist - d
    grad = 2.0 * diff * alpha * direction
    return diff * diff, grad, dist


@numba.njit(cache=True, fastmath=True)
def _run_pair_sequence_online_euclidean_njit(
    data,
    pair_sequence,
    learning_rate,
    init_params,
    alpha_init=1.0,
    alpha_ema=0.05,
    eps=1e-12,
    alpha_min=1e-6,
    alpha_max=1e6,
):
    n = data.shape[0]
    params = init_params.copy()
    alpha = alpha_init
    epochs = len(pair_sequence)

    for it in range(epochs):
        step_pos = max(1.0 / (learning_rate + it), 1e-4)
        seq = pair_sequence[it]
        used = seq.shape[0]
        num = 0.0
        den = 0.0

        for k in range(used):
            i = seq[k, 0]
            j = seq[k, 1]
            d = data[i, j]
            _, grad, dist = _stress_and_grad_euclidean_scaled(params[i], params[j], d, alpha, eps)
            params[i, 0] -= step_pos * grad[0]
            params[i, 1] -= step_pos * grad[1]
            params[j, 0] += step_pos * grad[0]
            params[j, 1] += step_pos * grad[1]
            num += d * dist
            den += dist * dist

        if used > 0:
            alpha_hat = num / (den + eps)
            alpha_hat = max(alpha_min, min(alpha_max, alpha_hat))
            alpha = (1.0 - alpha_ema) * alpha + alpha_ema * alpha_hat
    return params, alpha


@numba.njit(cache=True, fastmath=True)
def _run_pair_sequence_sync_euclidean_njit(
    data,
    pair_sequence,
    learning_rate,
    init_params,
    norm_mode=0,
    alpha_init=1.0,
    alpha_ema=0.05,
    eps=1e-12,
    alpha_min=1e-6,
    alpha_max=1e6,
):
    n = data.shape[0]
    params = init_params.copy()
    alpha = alpha_init
    epochs = len(pair_sequence)

    for it in range(epochs):
        step_pos = max(1.0 / (learning_rate + it), 1e-4)
        seq = pair_sequence[it]
        used = seq.shape[0]
        grad_acc = np.zeros((n, 2))
        counts = np.zeros(n, dtype=np.int32)
        num = 0.0
        den = 0.0

        for k in range(used):
            i = seq[k, 0]
            j = seq[k, 1]
            d = data[i, j]
            _, grad, dist = _stress_and_grad_euclidean_scaled(params[i], params[j], d, alpha, eps)
            grad_acc[i, 0] += grad[0]
            grad_acc[i, 1] += grad[1]
            grad_acc[j, 0] -= grad[0]
            grad_acc[j, 1] -= grad[1]
            counts[i] += 1
            counts[j] += 1
            num += d * dist
            den += dist * dist

        if used > 0:
            avg_count = 2.0 * used / n
            for i in range(n):
                if counts[i] == 0:
                    continue
                if norm_mode == 0:
                    scale = step_pos / used
                elif norm_mode == 1:
                    scale = step_pos / counts[i]
                else:
                    scale = step_pos * avg_count * math.sqrt(avg_count / counts[i]) / used
                params[i, 0] -= scale * grad_acc[i, 0]
                params[i, 1] -= scale * grad_acc[i, 1]

        if used > 0:
            alpha_hat = num / (den + eps)
            alpha_hat = max(alpha_min, min(alpha_max, alpha_hat))
            alpha = (1.0 - alpha_ema) * alpha + alpha_ema * alpha_hat
    return params, alpha


def _baseline_random_minibatch(
    data: np.ndarray,
    learning_rate: float,
    *,
    max_iters: int,
    batch_pairs: int,
    seed: int,
    alpha_init: float,
    alpha_ema: float,
    eps: float,
    alpha_min: float,
    alpha_max: float,
    r0_init: float,
    r1_init: float,
    learn_mode: int,
    geom_lr: float,
    geom_min: float,
    theta: float,
) -> tuple[np.ndarray, float, float, float, int, int]:
    coords, alpha, r0, r1 = sgd_minibatch_njit(
        data=data,
        learning_rate=learning_rate,
        max_iters=max_iters,
        batch_pairs=batch_pairs,
        seed=seed,
        alpha_init=alpha_init,
        alpha_ema=alpha_ema,
        eps=eps,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        r0_init=r0_init,
        r1_init=r1_init,
        learn_mode=learn_mode,
        geom_lr=geom_lr,
        geom_min=geom_min,
        theta=theta,
    )
    return coords, float(alpha), float(r0), float(r1), int(max_iters * batch_pairs), int(max_iters)


def _sampled_unique_online(
    data: np.ndarray,
    learning_rate: float,
    *,
    max_iters: int,
    batch_pairs: int,
    seed: int,
    alpha_init: float,
    alpha_ema: float,
    eps: float,
    alpha_min: float,
    alpha_max: float,
    r0_init: float,
    r1_init: float,
    learn_mode: int,
    geom_lr: float,
    geom_min: float,
    theta: float,
) -> tuple[np.ndarray, float, float, float, int, int]:
    sequence = _build_sampled_unique_pair_sequence(data.shape[0], max_iters, batch_pairs, seed)
    init_params = _legacy_random_init(data.shape[0], seed)
    coords, alpha, r0, r1 = _run_pair_sequence_online_njit(
        data,
        sequence,
        learning_rate,
        init_params,
        alpha_init,
        alpha_ema,
        eps,
        alpha_min,
        alpha_max,
        r0_init,
        r1_init,
        learn_mode,
        geom_lr,
        geom_min,
        theta,
    )
    pair_evals = sum(int(seq.shape[0]) for seq in sequence)
    return coords, float(alpha), float(r0), float(r1), pair_evals, len(sequence)


def _sampled_unique_sync(
    data: np.ndarray,
    learning_rate: float,
    *,
    max_iters: int,
    batch_pairs: int,
    seed: int,
    alpha_init: float,
    alpha_ema: float,
    eps: float,
    alpha_min: float,
    alpha_max: float,
    r0_init: float,
    r1_init: float,
    learn_mode: int,
    geom_lr: float,
    geom_min: float,
    theta: float,
    norm_mode: int,
) -> tuple[np.ndarray, float, float, float, int, int]:
    sequence = _build_sampled_unique_pair_sequence(data.shape[0], max_iters, batch_pairs, seed)
    init_params = _legacy_random_init(data.shape[0], seed)
    coords, alpha, r0, r1 = _run_pair_sequence_sync_njit(
        data,
        sequence,
        learning_rate,
        init_params,
        norm_mode,
        alpha_init,
        alpha_ema,
        eps,
        alpha_min,
        alpha_max,
        r0_init,
        r1_init,
        learn_mode,
        geom_lr,
        geom_min,
        theta,
    )
    pair_evals = sum(int(seq.shape[0]) for seq in sequence)
    return coords, float(alpha), float(r0), float(r1), pair_evals, len(sequence)


def _sampled_unique_sync_hybridnorm(
    data: np.ndarray,
    learning_rate: float,
    *,
    max_iters: int,
    batch_pairs: int,
    seed: int,
    alpha_init: float,
    alpha_ema: float,
    eps: float,
    alpha_min: float,
    alpha_max: float,
    r0_init: float,
    r1_init: float,
    learn_mode: int,
    geom_lr: float,
    geom_min: float,
    theta: float,
) -> tuple[np.ndarray, float, float, float, int, int]:
    first_stage = max(1, int(math.ceil(0.65 * max_iters)))
    second_stage = max(1, max_iters - first_stage)

    seq_first = _build_sampled_unique_pair_sequence(data.shape[0], first_stage, batch_pairs, seed)
    init_params = _legacy_random_init(data.shape[0], seed)
    coords, alpha, r0, r1 = _run_pair_sequence_sync_njit(
        data,
        seq_first,
        learning_rate,
        init_params,
        2,
        alpha_init,
        alpha_ema,
        eps,
        alpha_min,
        alpha_max,
        r0_init,
        r1_init,
        learn_mode,
        geom_lr,
        geom_min,
        theta,
    )

    seq_second = _build_sampled_unique_pair_sequence(data.shape[0], second_stage, batch_pairs, seed + 1009)
    coords, alpha, r0, r1 = _run_pair_sequence_sync_njit(
        data,
        seq_second,
        learning_rate + first_stage,
        coords,
        1,
        alpha,
        alpha_ema,
        eps,
        alpha_min,
        alpha_max,
        r0,
        r1,
        learn_mode,
        geom_lr,
        geom_min,
        theta,
    )
    pair_evals = sum(int(seq.shape[0]) for seq in seq_first) + sum(int(seq.shape[0]) for seq in seq_second)
    return coords, float(alpha), float(r0), float(r1), pair_evals, first_stage + second_stage


def _stratified_distance_sync_sqrtnorm(
    data: np.ndarray,
    learning_rate: float,
    *,
    max_iters: int,
    batch_pairs: int,
    seed: int,
    alpha_init: float,
    alpha_ema: float,
    eps: float,
    alpha_min: float,
    alpha_max: float,
    r0_init: float,
    r1_init: float,
    learn_mode: int,
    geom_lr: float,
    geom_min: float,
    theta: float,
) -> tuple[np.ndarray, float, float, float, int, int]:
    sequence = _build_distance_stratified_pair_sequence(data, max_iters, batch_pairs, seed)
    init_params = _legacy_random_init(data.shape[0], seed)
    coords, alpha, r0, r1 = _run_pair_sequence_sync_njit(
        data,
        sequence,
        learning_rate,
        init_params,
        2,
        alpha_init,
        alpha_ema,
        eps,
        alpha_min,
        alpha_max,
        r0_init,
        r1_init,
        learn_mode,
        geom_lr,
        geom_min,
        theta,
    )
    pair_evals = sum(int(seq.shape[0]) for seq in sequence)
    return coords, float(alpha), float(r0), float(r1), pair_evals, len(sequence)


def _coverage_unique_sync_sqrtnorm(
    data: np.ndarray,
    learning_rate: float,
    *,
    max_iters: int,
    batch_pairs: int,
    seed: int,
    alpha_init: float,
    alpha_ema: float,
    eps: float,
    alpha_min: float,
    alpha_max: float,
    r0_init: float,
    r1_init: float,
    learn_mode: int,
    geom_lr: float,
    geom_min: float,
    theta: float,
) -> tuple[np.ndarray, float, float, float, int, int]:
    sequence = _build_coverage_constrained_pair_sequence(data.shape[0], max_iters, batch_pairs, seed)
    init_params = _legacy_random_init(data.shape[0], seed)
    coords, alpha, r0, r1 = _run_pair_sequence_sync_njit(
        data,
        sequence,
        learning_rate,
        init_params,
        2,
        alpha_init,
        alpha_ema,
        eps,
        alpha_min,
        alpha_max,
        r0_init,
        r1_init,
        learn_mode,
        geom_lr,
        geom_min,
        theta,
    )
    pair_evals = sum(int(seq.shape[0]) for seq in sequence)
    return coords, float(alpha), float(r0), float(r1), pair_evals, len(sequence)


def _anchor_unique_online(
    data: np.ndarray,
    learning_rate: float,
    *,
    max_iters: int,
    batch_pairs: int,
    seed: int,
    alpha_init: float,
    alpha_ema: float,
    eps: float,
    alpha_min: float,
    alpha_max: float,
    r0_init: float,
    r1_init: float,
    learn_mode: int,
    geom_lr: float,
    geom_min: float,
    theta: float,
) -> tuple[np.ndarray, float, float, float, int, int]:
    sequence = _build_anchor_unique_pair_sequence(data.shape[0], max_iters, batch_pairs, seed)
    init_params = _legacy_random_init(data.shape[0], seed)
    coords, alpha, r0, r1 = _run_pair_sequence_online_njit(
        data,
        sequence,
        learning_rate,
        init_params,
        alpha_init,
        alpha_ema,
        eps,
        alpha_min,
        alpha_max,
        r0_init,
        r1_init,
        learn_mode,
        geom_lr,
        geom_min,
        theta,
    )
    pair_evals = sum(int(seq.shape[0]) for seq in sequence)
    return coords, float(alpha), float(r0), float(r1), pair_evals, len(sequence)


def _full_sweep_sync_sqrtnorm(
    data: np.ndarray,
    learning_rate: float,
    *,
    max_iters: int,
    batch_pairs: int,
    seed: int,
    alpha_init: float,
    alpha_ema: float,
    eps: float,
    alpha_min: float,
    alpha_max: float,
    r0_init: float,
    r1_init: float,
    learn_mode: int,
    geom_lr: float,
    geom_min: float,
    theta: float,
) -> tuple[np.ndarray, float, float, float, int, int]:
    total_pairs = data.shape[0] * (data.shape[0] - 1) // 2
    pair_budget = max_iters * batch_pairs
    epochs = max(1, int(math.ceil(pair_budget / max(total_pairs, 1))))
    epochs = min(epochs, 24)
    sequence = _build_full_sweep_pair_sequence(data.shape[0], epochs, seed)
    init_params = _legacy_random_init(data.shape[0], seed)
    coords, alpha, r0, r1 = _run_pair_sequence_sync_njit(
        data,
        sequence,
        learning_rate,
        init_params,
        2,
        alpha_init,
        alpha_ema,
        eps,
        alpha_min,
        alpha_max,
        r0_init,
        r1_init,
        learn_mode,
        geom_lr,
        geom_min,
        theta,
    )
    pair_evals = sum(int(seq.shape[0]) for seq in sequence)
    return coords, float(alpha), float(r0), float(r1), pair_evals, len(sequence)


def run_optimizer_variant_euclidean_control(
    data: np.ndarray,
    variant_name: str,
    *,
    learning_rate: float = 1.0,
    max_iters: int = 120,
    batch_pairs: int = 2048,
    seed: int = 0,
    alpha_init: float = 1.0,
    alpha_ema: float = 0.05,
    eps: float = 1e-12,
    alpha_min: float = 1e-6,
    alpha_max: float = 1e6,
) -> tuple[np.ndarray, float, int, int]:
    data = np.asarray(data, dtype=np.float64)
    n = data.shape[0]
    init_params = _legacy_random_init(n, seed)

    if variant_name == "baseline_random_minibatch":
        sequence = _build_random_replacement_pair_sequence(n, max_iters, batch_pairs, seed)
        coords, alpha = _run_pair_sequence_online_euclidean_njit(
            data,
            sequence,
            learning_rate,
            init_params,
            alpha_init,
            alpha_ema,
            eps,
            alpha_min,
            alpha_max,
        )
    elif variant_name == "sampled_unique_online":
        sequence = _build_sampled_unique_pair_sequence(n, max_iters, batch_pairs, seed)
        coords, alpha = _run_pair_sequence_online_euclidean_njit(
            data,
            sequence,
            learning_rate,
            init_params,
            alpha_init,
            alpha_ema,
            eps,
            alpha_min,
            alpha_max,
        )
    elif variant_name == "sampled_unique_sync_batchavg":
        sequence = _build_sampled_unique_pair_sequence(n, max_iters, batch_pairs, seed)
        coords, alpha = _run_pair_sequence_sync_euclidean_njit(
            data,
            sequence,
            learning_rate,
            init_params,
            0,
            alpha_init,
            alpha_ema,
            eps,
            alpha_min,
            alpha_max,
        )
    elif variant_name == "sampled_unique_sync_countnorm":
        sequence = _build_sampled_unique_pair_sequence(n, max_iters, batch_pairs, seed)
        coords, alpha = _run_pair_sequence_sync_euclidean_njit(
            data,
            sequence,
            learning_rate,
            init_params,
            1,
            alpha_init,
            alpha_ema,
            eps,
            alpha_min,
            alpha_max,
        )
    elif variant_name == "sampled_unique_sync_sqrtnorm":
        sequence = _build_sampled_unique_pair_sequence(n, max_iters, batch_pairs, seed)
        coords, alpha = _run_pair_sequence_sync_euclidean_njit(
            data,
            sequence,
            learning_rate,
            init_params,
            2,
            alpha_init,
            alpha_ema,
            eps,
            alpha_min,
            alpha_max,
        )
    elif variant_name == "sampled_unique_sync_hybridnorm":
        first_stage = max(1, int(math.ceil(0.65 * max_iters)))
        second_stage = max(1, max_iters - first_stage)

        seq_first = _build_sampled_unique_pair_sequence(n, first_stage, batch_pairs, seed)
        coords, alpha = _run_pair_sequence_sync_euclidean_njit(
            data,
            seq_first,
            learning_rate,
            init_params,
            2,
            alpha_init,
            alpha_ema,
            eps,
            alpha_min,
            alpha_max,
        )

        seq_second = _build_sampled_unique_pair_sequence(n, second_stage, batch_pairs, seed + 1009)
        coords, alpha = _run_pair_sequence_sync_euclidean_njit(
            data,
            seq_second,
            learning_rate + first_stage,
            coords,
            1,
            alpha,
            alpha_ema,
            eps,
            alpha_min,
            alpha_max,
        )
        sequence = list(seq_first) + list(seq_second)
    elif variant_name == "stratified_distance_sync_sqrtnorm":
        sequence = _build_distance_stratified_pair_sequence(data, max_iters, batch_pairs, seed)
        coords, alpha = _run_pair_sequence_sync_euclidean_njit(
            data,
            sequence,
            learning_rate,
            init_params,
            2,
            alpha_init,
            alpha_ema,
            eps,
            alpha_min,
            alpha_max,
        )
    elif variant_name == "anchor_unique_online":
        sequence = _build_anchor_unique_pair_sequence(n, max_iters, batch_pairs, seed)
        coords, alpha = _run_pair_sequence_online_euclidean_njit(
            data,
            sequence,
            learning_rate,
            init_params,
            alpha_init,
            alpha_ema,
            eps,
            alpha_min,
            alpha_max,
        )
    elif variant_name == "full_sweep_sync_sqrtnorm":
        total_pairs = n * (n - 1) // 2
        pair_budget = max_iters * batch_pairs
        epochs = max(1, int(math.ceil(pair_budget / max(total_pairs, 1))))
        epochs = min(epochs, 24)
        sequence = _build_full_sweep_pair_sequence(n, epochs, seed)
        coords, alpha = _run_pair_sequence_sync_euclidean_njit(
            data,
            sequence,
            learning_rate,
            init_params,
            2,
            alpha_init,
            alpha_ema,
            eps,
            alpha_min,
            alpha_max,
        )
    else:
        raise KeyError(variant_name)

    pair_evals = sum(int(seq.shape[0]) for seq in sequence)
    return coords, float(alpha), pair_evals, len(sequence)


RUNNERS: dict[str, Callable[..., tuple[np.ndarray, float, float, float, int, int]]] = {
    "baseline_random_minibatch": _baseline_random_minibatch,
    "sampled_unique_online": _sampled_unique_online,
    "sampled_unique_sync_batchavg": lambda *args, **kwargs: _sampled_unique_sync(*args, norm_mode=0, **kwargs),
    "sampled_unique_sync_countnorm": lambda *args, **kwargs: _sampled_unique_sync(*args, norm_mode=1, **kwargs),
    "sampled_unique_sync_sqrtnorm": lambda *args, **kwargs: _sampled_unique_sync(*args, norm_mode=2, **kwargs),
    "sampled_unique_sync_hybridnorm": _sampled_unique_sync_hybridnorm,
    "stratified_distance_sync_sqrtnorm": _stratified_distance_sync_sqrtnorm,
    "coverage_unique_sync_sqrtnorm": _coverage_unique_sync_sqrtnorm,
    "anchor_unique_online": _anchor_unique_online,
    "full_sweep_sync_sqrtnorm": _full_sweep_sync_sqrtnorm,
}


def warmup_variants() -> None:
    graph, distances = graphio.get_periodic_lattice(5, 5)
    del graph
    kwargs = {
        "max_iters": 4,
        "batch_pairs": 32,
        "seed": 0,
        "alpha_init": 1.0,
        "alpha_ema": 0.05,
        "eps": 1e-12,
        "alpha_min": 1e-6,
        "alpha_max": 1e6,
        "r0_init": 1.0,
        "r1_init": 1.0,
        "learn_mode": 1,
        "geom_lr": 0.01,
        "geom_min": 1e-6,
        "theta": np.pi / 2,
    }
    for variant_name in RUNNERS:
        variant = VARIANT_MAP[variant_name]
        if variant.max_n is not None and distances.shape[0] > variant.max_n:
            continue
        RUNNERS[variant_name](distances, 1.0, **kwargs)
    for variant_name in (
        "baseline_random_minibatch",
        "sampled_unique_online",
        "sampled_unique_sync_batchavg",
        "sampled_unique_sync_countnorm",
        "sampled_unique_sync_sqrtnorm",
        "anchor_unique_online",
        "full_sweep_sync_sqrtnorm",
    ):
        run_optimizer_variant_euclidean_control(
            distances,
            variant_name,
            learning_rate=1.0,
            max_iters=4,
            batch_pairs=32,
            seed=0,
            alpha_init=1.0,
            alpha_ema=0.05,
        )


def run_optimizer_variant(
    data: np.ndarray,
    variant_name: str,
    *,
    learning_rate: float = 1.0,
    max_iters: int = 120,
    batch_pairs: int = 2048,
    seed: int = 0,
    alpha_init: float = 1.0,
    alpha_ema: float = 0.05,
    eps: float = 1e-12,
    alpha_min: float = 1e-6,
    alpha_max: float = 1e6,
    r0_init: float = 1.0,
    r1_init: float = 1.0,
    learn_mode: int | str = "alpha",
    geom_lr: float = 0.01,
    geom_min: float = 1e-6,
    theta: float = np.pi / 2,
) -> tuple[np.ndarray, float, float, float, int, int]:
    runner = RUNNERS[variant_name]
    return runner(
        np.asarray(data, dtype=np.float64),
        float(learning_rate),
        max_iters=int(max_iters),
        batch_pairs=int(batch_pairs),
        seed=int(seed),
        alpha_init=float(alpha_init),
        alpha_ema=float(alpha_ema),
        eps=float(eps),
        alpha_min=float(alpha_min),
        alpha_max=float(alpha_max),
        r0_init=float(r0_init),
        r1_init=float(r1_init),
        learn_mode=_normalize_learn_mode(learn_mode),
        geom_lr=float(geom_lr),
        geom_min=float(geom_min),
        theta=float(theta),
    )


def benchmark_variants(
    specs: Sequence[BenchmarkSpec],
    *,
    variant_names: Sequence[str] | None = None,
    seeds: Sequence[int] = (0, 1, 2),
    learning_rate: float = 1.0,
    max_iters: int = 120,
    batch_pairs: int = 2048,
    alpha_init: float = 1.0,
    alpha_ema: float = 0.05,
    learn_mode: int | str = "alpha",
    geom_lr: float = 0.01,
    theta: float = np.pi / 2,
    compute_extra_metrics: bool = True,
    max_extra_metric_n: int = 400,
    store_embeddings: bool = True,
) -> tuple[pd.DataFrame, dict[tuple[str, str, int], EmbeddingRun]]:
    if variant_names is None:
        variant_names = [variant.name for variant in VARIANTS]

    rows: list[dict[str, Any]] = []
    embeddings: dict[tuple[str, str, int], EmbeddingRun] = {}

    for spec in specs:
        n_nodes = spec.graph.number_of_nodes()
        n_edges = spec.graph.number_of_edges()
        for seed in seeds:
            for variant_name in variant_names:
                variant = VARIANT_MAP[variant_name]
                if variant.max_n is not None and n_nodes > variant.max_n:
                    rows.append(
                        {
                            "dataset": spec.name,
                            "group": spec.group,
                            "variant": variant_name,
                            "seed": seed,
                            "status": "skipped",
                            "skip_reason": f"n_nodes={n_nodes} exceeds max_n={variant.max_n}",
                            "n_nodes": n_nodes,
                            "n_edges": n_edges,
                            "runtime_s": np.nan,
                            "pair_evals": np.nan,
                            "epochs": np.nan,
                            "stress": np.nan,
                            "distortion": np.nan,
                            "sgs": np.nan,
                            "neighborhood_precision": np.nan,
                            "alpha": np.nan,
                            "r0": np.nan,
                            "r1": np.nan,
                            "theta_deg": np.degrees(theta),
                            "sampling": variant.sampling,
                            "order": variant.order,
                            "updates": variant.updates,
                            "description": variant.description,
                        }
                    )
                    continue

                started = time.perf_counter()
                coords, alpha, r0, r1, pair_evals, epochs = run_optimizer_variant(
                    spec.distances,
                    variant_name,
                    learning_rate=learning_rate,
                    max_iters=max_iters,
                    batch_pairs=batch_pairs,
                    seed=seed,
                    alpha_init=alpha_init,
                    alpha_ema=alpha_ema,
                    learn_mode=learn_mode,
                    geom_lr=geom_lr,
                    theta=theta,
                )
                runtime_s = time.perf_counter() - started
                geod = make_torus_geod(alpha, r0, r1, theta)
                stress = metrics.geodesic_stress(coords, spec.distances, geod)
                distortion = metrics.geodesic_distortion(coords, spec.distances, geod)

                sgs = np.nan
                neighborhood_precision = np.nan
                if compute_extra_metrics and n_nodes <= max_extra_metric_n:
                    sgs = metrics.SGS(coords, spec.distances, geod)
                    neighborhood_precision = metrics.geodesic_NP(coords, spec.distances, geod, rg=2)

                row = {
                    "dataset": spec.name,
                    "group": spec.group,
                    "variant": variant_name,
                    "seed": seed,
                    "status": "ok",
                    "skip_reason": "",
                    "n_nodes": n_nodes,
                    "n_edges": n_edges,
                    "runtime_s": runtime_s,
                    "pair_evals": pair_evals,
                    "epochs": epochs,
                    "stress": stress,
                    "distortion": distortion,
                    "sgs": sgs,
                    "neighborhood_precision": neighborhood_precision,
                    "alpha": alpha,
                    "r0": r0,
                    "r1": r1,
                    "theta_deg": np.degrees(theta),
                    "sampling": variant.sampling,
                    "order": variant.order,
                    "updates": variant.updates,
                    "description": variant.description,
                }
                row.update(spec.metadata)
                rows.append(row)

                if store_embeddings:
                    embeddings[(spec.name, variant_name, seed)] = EmbeddingRun(
                        dataset=spec.name,
                        group=spec.group,
                        variant=variant_name,
                        seed=seed,
                        coords=coords,
                        alpha_=alpha,
                        r0_=r0,
                        r1_=r1,
                        theta_=theta,
                        runtime_s=runtime_s,
                        pair_evals=pair_evals,
                        epochs=epochs,
                        description=variant.description,
                    )

    return pd.DataFrame(rows), embeddings


def benchmark_objective_controls(
    specs: Sequence[BenchmarkSpec],
    *,
    variant_names: Sequence[str] | None = None,
    objectives: Sequence[str] = ("torus", "euclidean"),
    seeds: Sequence[int] = (0, 1, 2),
    learning_rate: float = 1.0,
    max_iters: int = 120,
    batch_pairs: int = 2048,
    alpha_init: float = 1.0,
    alpha_ema: float = 0.05,
    theta: float = np.pi / 2,
) -> pd.DataFrame:
    if variant_names is None:
        variant_names = [variant.name for variant in VARIANTS]

    rows: list[dict[str, Any]] = []
    for objective in objectives:
        for spec in specs:
            n_nodes = spec.graph.number_of_nodes()
            for seed in seeds:
                for variant_name in variant_names:
                    variant = VARIANT_MAP[variant_name]
                    if variant.max_n is not None and n_nodes > variant.max_n:
                        rows.append(
                            {
                                "objective": objective,
                                "dataset": spec.name,
                                "group": spec.group,
                                "variant": variant_name,
                                "seed": seed,
                                "status": "skipped",
                                "stress": np.nan,
                                "runtime_s": np.nan,
                                "pair_evals": np.nan,
                                "epochs": np.nan,
                            }
                        )
                        continue

                    started = time.perf_counter()
                    if objective == "torus":
                        coords, alpha, r0, r1, pair_evals, epochs = run_optimizer_variant(
                            spec.distances,
                            variant_name,
                            learning_rate=learning_rate,
                            max_iters=max_iters,
                            batch_pairs=batch_pairs,
                            seed=seed,
                            alpha_init=alpha_init,
                            alpha_ema=alpha_ema,
                            learn_mode="alpha",
                            theta=theta,
                        )
                        geod = make_torus_geod(alpha, r0, r1, theta)
                    else:
                        coords, alpha, pair_evals, epochs = run_optimizer_variant_euclidean_control(
                            spec.distances,
                            variant_name,
                            learning_rate=learning_rate,
                            max_iters=max_iters,
                            batch_pairs=batch_pairs,
                            seed=seed,
                            alpha_init=alpha_init,
                            alpha_ema=alpha_ema,
                        )
                        geod = make_euclidean_geod(alpha)
                    runtime_s = time.perf_counter() - started
                    stress = metrics.geodesic_stress(coords, spec.distances, geod)
                    rows.append(
                        {
                            "objective": objective,
                            "dataset": spec.name,
                            "group": spec.group,
                            "variant": variant_name,
                            "seed": seed,
                            "status": "ok",
                            "stress": stress,
                            "runtime_s": runtime_s,
                            "pair_evals": pair_evals,
                            "epochs": epochs,
                        }
                    )
    return pd.DataFrame(rows)


def summarize_objective_controls(control_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ok = control_df.loc[control_df["status"] == "ok"].copy()
    baseline = ok.loc[
        ok["variant"] == "baseline_random_minibatch",
        ["objective", "dataset", "seed", "stress"],
    ].rename(columns={"stress": "baseline_stress"})
    ok = ok.merge(baseline, on=["objective", "dataset", "seed"], how="left")
    ok["stress_delta_vs_baseline"] = ok["stress"] - ok["baseline_stress"]

    overall = (
        ok.groupby(["objective", "variant"], dropna=False)
        .agg(
            runs=("stress", "size"),
            mean_stress=("stress", "mean"),
            median_stress=("stress", "median"),
            mean_runtime_s=("runtime_s", "mean"),
            mean_stress_delta=("stress_delta_vs_baseline", "mean"),
        )
        .sort_values(["objective", "mean_stress", "mean_runtime_s"])
        .reset_index()
    )

    by_group = (
        ok.groupby(["objective", "group", "variant"], dropna=False)
        .agg(
            runs=("stress", "size"),
            mean_stress=("stress", "mean"),
            mean_runtime_s=("runtime_s", "mean"),
            mean_stress_delta=("stress_delta_vs_baseline", "mean"),
        )
        .sort_values(["objective", "group", "mean_stress", "mean_runtime_s"])
        .reset_index()
    )
    return {"ok": ok, "overall": overall, "by_group": by_group}


def diagnose_torus_optimization(
    data: np.ndarray,
    variant_name: str,
    *,
    learning_rate: float = 1.0,
    max_iters: int = 120,
    batch_pairs: int = 2048,
    seed: int = 0,
    alpha_init: float = 1.0,
    alpha_ema: float = 0.05,
    theta: float = np.pi / 2,
) -> pd.DataFrame:
    data = np.asarray(data, dtype=np.float64)
    n = data.shape[0]
    if variant_name == "baseline_random_minibatch":
        sequence = _build_random_replacement_pair_sequence(n, max_iters, batch_pairs, seed)
        stages = [(sequence, -1, float(learning_rate))]
    elif variant_name == "sampled_unique_sync_batchavg":
        sequence = _build_sampled_unique_pair_sequence(n, max_iters, batch_pairs, seed)
        stages = [(sequence, 0, float(learning_rate))]
    elif variant_name == "sampled_unique_sync_countnorm":
        sequence = _build_sampled_unique_pair_sequence(n, max_iters, batch_pairs, seed)
        stages = [(sequence, 1, float(learning_rate))]
    elif variant_name == "sampled_unique_sync_sqrtnorm":
        sequence = _build_sampled_unique_pair_sequence(n, max_iters, batch_pairs, seed)
        stages = [(sequence, 2, float(learning_rate))]
    elif variant_name == "sampled_unique_sync_hybridnorm":
        first_stage = max(1, int(math.ceil(0.65 * max_iters)))
        second_stage = max(1, max_iters - first_stage)
        seq_first = _build_sampled_unique_pair_sequence(n, first_stage, batch_pairs, seed)
        seq_second = _build_sampled_unique_pair_sequence(n, second_stage, batch_pairs, seed + 1009)
        stages = [
            (seq_first, 2, float(learning_rate)),
            (seq_second, 1, float(learning_rate + first_stage)),
        ]
    elif variant_name == "stratified_distance_sync_sqrtnorm":
        sequence = _build_distance_stratified_pair_sequence(data, max_iters, batch_pairs, seed)
        stages = [(sequence, 2, float(learning_rate))]
    else:
        raise ValueError(
            "Diagnostics currently implemented for baseline, sync variants, hybridnorm, "
            f"and stratified sqrtnorm only, got {variant_name!r}"
        )

    params = _legacy_random_init(n, seed)
    alpha = float(alpha_init)
    r0 = 1.0
    r1 = 1.0
    rows: list[dict[str, Any]] = []
    epoch = 0
    for sequence, norm_mode, stage_learning_rate in stages:
        for seq in sequence:
            step_pos = max(1.0 / (stage_learning_rate + epoch), 1e-4)
            counts = np.zeros(n, dtype=np.int32)
            grad_acc = np.zeros((n, 2), dtype=np.float64)
            move_sum = np.zeros(n, dtype=np.float64)
            pair_grad_norms = np.zeros(seq.shape[0], dtype=np.float64)
            num = 0.0
            den = 0.0

            for k in range(seq.shape[0]):
                i = int(seq[k, 0])
                j = int(seq[k, 1])
                d = float(data[i, j])
                _, grad, dist, _, _ = stress_and_grad_rect_torus(params[i], params[j], d, alpha, r0, r1, 1e-12, theta)
                grad_norm = float(np.linalg.norm(grad))
                pair_grad_norms[k] = grad_norm
                counts[i] += 1
                counts[j] += 1
                num += d * dist
                den += dist * dist

                if norm_mode < 0:
                    delta = step_pos * grad
                    params[i] -= delta
                    params[j] += delta
                    move_sum[i] += float(np.linalg.norm(delta))
                    move_sum[j] += float(np.linalg.norm(delta))
                else:
                    grad_acc[i] += grad
                    grad_acc[j] -= grad

            if norm_mode >= 0:
                avg_count = 2.0 * seq.shape[0] / n
                for i in range(n):
                    if counts[i] == 0:
                        continue
                    if norm_mode == 0:
                        scale = step_pos / seq.shape[0]
                    elif norm_mode == 1:
                        scale = step_pos / counts[i]
                    else:
                        scale = step_pos * avg_count * math.sqrt(avg_count / counts[i]) / seq.shape[0]
                    delta = scale * grad_acc[i]
                    params[i] -= delta
                    move_sum[i] += float(np.linalg.norm(delta))

            params %= 1.0

            alpha_hat = num / (den + 1e-12)
            alpha_hat = max(1e-6, min(1e6, alpha_hat))
            alpha = (1.0 - alpha_ema) * alpha + alpha_ema * alpha_hat

            mean_visits = float(counts.mean())
            visit_std = float(counts.std())
            rows.append(
                {
                    "epoch": epoch,
                    "variant": variant_name,
                    "seed": seed,
                    "visit_mean": mean_visits,
                    "visit_std": visit_std,
                    "visit_cv": visit_std / mean_visits if mean_visits > 0 else 0.0,
                    "visit_min": int(counts.min()),
                    "visit_max": int(counts.max()),
                    "mean_pair_grad_norm": float(pair_grad_norms.mean()) if pair_grad_norms.size else 0.0,
                    "median_pair_grad_norm": float(np.median(pair_grad_norms)) if pair_grad_norms.size else 0.0,
                    "mean_node_move_norm": float(move_sum.mean()),
                    "median_node_move_norm": float(np.median(move_sum)),
                    "max_node_move_norm": float(move_sum.max()),
                    "alpha": alpha,
                }
            )
            epoch += 1
    return pd.DataFrame(rows)


def add_baseline_deltas(results_df: pd.DataFrame, baseline_variant: str = "baseline_random_minibatch") -> pd.DataFrame:
    ok = results_df.loc[results_df["status"] == "ok"].copy()
    baseline = ok.loc[ok["variant"] == baseline_variant, ["dataset", "seed", "stress", "runtime_s"]].rename(
        columns={"stress": "baseline_stress", "runtime_s": "baseline_runtime_s"}
    )
    merged = ok.merge(baseline, on=["dataset", "seed"], how="left")
    merged["stress_delta_vs_baseline"] = merged["stress"] - merged["baseline_stress"]
    merged["stress_ratio_vs_baseline"] = merged["stress"] / merged["baseline_stress"]
    merged["runtime_delta_vs_baseline"] = merged["runtime_s"] - merged["baseline_runtime_s"]
    merged["runtime_ratio_vs_baseline"] = merged["runtime_s"] / merged["baseline_runtime_s"]
    return merged


def summarize_results(results_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ok = add_baseline_deltas(results_df)
    overall = (
        ok.groupby("variant", dropna=False)
        .agg(
            runs=("stress", "size"),
            mean_stress=("stress", "mean"),
            median_stress=("stress", "median"),
            mean_runtime_s=("runtime_s", "mean"),
            median_runtime_s=("runtime_s", "median"),
            mean_stress_delta=("stress_delta_vs_baseline", "mean"),
            mean_runtime_ratio=("runtime_ratio_vs_baseline", "mean"),
        )
        .sort_values(["mean_stress", "mean_runtime_s"])
        .reset_index()
    )

    by_group = (
        ok.groupby(["group", "variant"], dropna=False)
        .agg(
            runs=("stress", "size"),
            mean_stress=("stress", "mean"),
            median_stress=("stress", "median"),
            mean_runtime_s=("runtime_s", "mean"),
            mean_stress_delta=("stress_delta_vs_baseline", "mean"),
        )
        .sort_values(["group", "mean_stress", "mean_runtime_s"])
        .reset_index()
    )

    dataset_means = (
        ok.groupby(["dataset", "group", "variant"], dropna=False)
        .agg(
            mean_stress=("stress", "mean"),
            mean_runtime_s=("runtime_s", "mean"),
            mean_stress_delta=("stress_delta_vs_baseline", "mean"),
        )
        .reset_index()
    )

    dataset_ranked = dataset_means.sort_values(["dataset", "mean_stress", "mean_runtime_s"])
    best_by_dataset = dataset_ranked.groupby("dataset", as_index=False).first()

    return {
        "ok": ok,
        "overall": overall,
        "by_group": by_group,
        "dataset_means": dataset_means,
        "best_by_dataset": best_by_dataset,
    }


def recommendation_table(results_df: pd.DataFrame) -> pd.DataFrame:
    summary = summarize_results(results_df)
    overall = summary["overall"].copy()
    wins = summary["best_by_dataset"].groupby("variant").size().rename("dataset_wins")
    overall = overall.merge(wins, on="variant", how="left")
    overall["dataset_wins"] = overall["dataset_wins"].fillna(0).astype(int)
    overall["stress_rank"] = overall["mean_stress"].rank(method="dense")
    overall["runtime_rank"] = overall["mean_runtime_s"].rank(method="dense")
    overall["overall_score"] = overall["stress_rank"] + 0.35 * overall["runtime_rank"]
    return overall.sort_values(["overall_score", "mean_stress", "mean_runtime_s"]).reset_index(drop=True)


def plot_representative_embeddings(
    spec: BenchmarkSpec,
    embeddings: dict[tuple[str, str, int], EmbeddingRun],
    variant_names: Sequence[str],
    *,
    seed: int = 0,
    figsize_per_col: tuple[float, float] = (4.2, 4.1),
) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(1, len(variant_names), figsize=(figsize_per_col[0] * len(variant_names), figsize_per_col[1]))
    if len(variant_names) == 1:
        axes = np.array([axes])

    for ax, variant_name in zip(axes, variant_names):
        run = embeddings[(spec.name, variant_name, seed)]
        visualization.plot_embedding_with_torus_edges(
            torus=run,
            G=spec.graph,
            colors=spec.colors,
            ax=ax,
            s=10 if spec.graph.number_of_nodes() <= 120 else 6,
            edge_alpha=0.08 if spec.graph.number_of_edges() > 400 else 0.14,
        )
        ax.set_title(f"{variant_name}\nstress={metrics.geodesic_stress(run.coords, spec.distances, run.geod()):.3f}")
    fig.suptitle(spec.name, y=1.02)
    fig.tight_layout()
    return fig, axes
