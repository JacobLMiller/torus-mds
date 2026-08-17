from __future__ import annotations

import os
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

import networkx as nx
import numpy as np
import pandas as pd
import s_gd2
import scipy.sparse as sp

from standalone_toruslayout import wrap_python

from .graphio import apsp_distance_matrix
from .initialization import find_fundamental_torus_directions
from .projector import MDSTorusProjector

# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

METHODS: tuple[str, ...] = ("TorusMDS", "s_gd2", "wrap_python")

DEFAULT_METHOD_MAX_N: dict[str, int | None] = {
    "TorusMDS": None,
    "s_gd2": None,
    "wrap_python": 3500,
}

# learn_mode='alpha_aspect' variants: {smart spectral init, random init} x
# {default batch size, 100x batch size}. Kept separate from METHODS (not
# merged into it) so existing scripts' default `--methods` behavior is
# unchanged -- these are opt-in via an explicit --methods list.
ASPECT_INIT_VARIANTS: dict[str, tuple[str, int]] = {
    "TorusMDS_smart_1x": ("smart", 1),
    "TorusMDS_smart_100x": ("smart", 100),
    "TorusMDS_random_1x": ("random", 1),
    "TorusMDS_random_100x": ("random", 100),
}

DEFAULT_ASPECT_MAX_N: dict[str, int | None] = {name: None for name in ASPECT_INIT_VARIANTS}


@dataclass
class GraphRecord:
    """One graph to embed: a unique id, family-specific metadata, and the graph itself."""

    exp_idx: int
    graph: nx.Graph
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Embedding methods
# ---------------------------------------------------------------------------

def embed_sgd2(G: nx.Graph, random_seed: int = 42) -> np.ndarray:
    nodes = sorted(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    edges = list(G.edges())
    I = np.array([idx[u] for u, _ in edges], dtype=np.int32)
    J = np.array([idx[v] for _, v in edges], dtype=np.int32)
    return s_gd2.layout(I, J, random_seed=random_seed)


def embed_torus_mds(
    D: np.ndarray,
    max_iters: int = 2000,
    seed: int = 42,
    stress_mode: str = "raw",
) -> tuple[np.ndarray, float]:
    proj = MDSTorusProjector(projection="wrap")
    X = proj.fit_transform(D, max_iters=max_iters, seed=seed, stress_mode=stress_mode)
    return X, float(proj.alpha_)


def embed_torus_mds_aspect(
    D: np.ndarray,
    init_mode: str,
    batch_multiplier: int,
    spectral_result: dict | None = None,
    max_iters: int = 2000,
    seed: int = 42,
    stress_mode: str = "raw",
) -> tuple[np.ndarray, float, float, float]:
    """
    learn_mode='alpha_aspect' TorusMDS: jointly learns alpha and the aspect
    ratio (r0, r1), instead of the fixed unit-square torus of embed_torus_mds.

    init_mode="smart" requires D normalized to max 1 (the projector's
    spectral-dict init check) and uses spectral_result (from
    find_fundamental_torus_directions(D / D.max()), passed in so callers can
    compute/cache it once per graph and share it across batch-size variants).
    init_mode="random" uses D as-is (matching embed_torus_mds) and init=None.

    learning_rate is left at "auto" so the projector's has-init-dependent
    schedule applies: a smaller tail step with no warmup for the supplied
    spectral init, vs. the default tail step with a warmup for random init.
    """
    if init_mode == "smart":
        data = D / D.max()
        init = spectral_result
    elif init_mode == "random":
        data = D
        init = None
    else:
        raise ValueError(f"Unknown init_mode {init_mode!r}")

    proj = MDSTorusProjector(projection="wrap")
    X = proj.fit_transform(
        data,
        max_iters=max_iters,
        seed=seed,
        stress_mode=stress_mode,
        learn_mode="alpha_aspect",
        batch_size=4096 * batch_multiplier,
        init=init,
    )
    return X, float(proj.alpha_), float(proj.r0_), float(proj.r1_)


def embed_wrap_python(
    G: nx.Graph,
    max_iters: int = 2000,
    seed: int = 42,
) -> np.ndarray:
    """
    Chen "it's-a-wrap" layout via the pure-Python/numba port.

    Deliberately does NOT pass shortest_path_lengths: for unweighted graphs
    (all these are) the runner computes BFS distances internally from the
    edge list, which avoids ever materializing a dense n x n matrix here.
    """
    result = wrap_python(G, seed=seed, max_iters=max_iters)
    return result.positions


# ---------------------------------------------------------------------------
# On-disk layout persistence
# ---------------------------------------------------------------------------

def _graphs_dir(output_dir: str) -> str:
    return os.path.join(output_dir, "graphs")


def _coords_dir(output_dir: str) -> str:
    return os.path.join(output_dir, "coords")


def graph_path(output_dir: str, exp_idx: int) -> str:
    return os.path.join(_graphs_dir(output_dir), f"graph_{exp_idx}.npz")


def coords_path(output_dir: str, exp_idx: int, method: str) -> str:
    return os.path.join(_coords_dir(output_dir), f"exp_{exp_idx}_{method}.npy")


def save_graph(G: nx.Graph, path: str) -> None:
    nodes = sorted(G.nodes())
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, format="csr", dtype=np.uint8)
    sp.save_npz(path, A)


def load_graph(path: str) -> nx.Graph:
    A = sp.load_npz(path)
    return nx.from_scipy_sparse_array(A)


# ---------------------------------------------------------------------------
# Embedding phase: run methods on a stream of graphs, persist layouts, and
# checkpoint a manifest CSV. No metrics are computed here.
# ---------------------------------------------------------------------------

def run_embeddings(
    graph_iterator: Iterable[GraphRecord],
    output_dir: str,
    torus_max_iters: int = 2000,
    wrap_python_max_iters: int = 200,
    method_max_n: dict[str, int | None] | None = None,
    checkpoint_every: int = 20,
    seed: int = 0,
    methods: tuple[str, ...] = METHODS,
    torus_stress_mode: str = "raw",
) -> pd.DataFrame:
    method_max_n = {**DEFAULT_METHOD_MAX_N, **DEFAULT_ASPECT_MAX_N, **(method_max_n or {})}

    os.makedirs(_graphs_dir(output_dir), exist_ok=True)
    os.makedirs(_coords_dir(output_dir), exist_ok=True)
    manifest_path = os.path.join(output_dir, "runs.csv")

    records: list[dict] = []
    done: set[tuple[int, str]] = set()
    if os.path.exists(manifest_path):
        try:
            existing_df = pd.read_csv(manifest_path)
            records = existing_df.to_dict("records")
            for r in records:
                if r.get("status") == "ok":
                    done.add((int(r["exp_idx"]), r["method"]))
            print(f"Resuming: {len(done)} completed (exp_idx, method) pairs loaded.")
        except Exception:
            pass

    def checkpoint() -> None:
        pd.DataFrame(records).to_csv(manifest_path, index=False)

    since_checkpoint = 0
    for rec in graph_iterator:
        exp_idx, meta, G = rec.exp_idx, rec.meta, rec.graph
        n = G.number_of_nodes()

        methods_needed = [m for m in methods if (exp_idx, m) not in done]
        if not methods_needed:
            continue

        gpath = graph_path(output_dir, exp_idx)
        if not os.path.exists(gpath):
            save_graph(G, gpath)

        # Shared, lazily-computed per-graph state: D (shortest-path distances)
        # is needed by TorusMDS and every aspect-init variant, and the spectral
        # decomposition is needed by both "smart" variants -- compute each at
        # most once per graph instead of once per method that needs it.
        D: np.ndarray | None = None
        spectral_result: dict | None = None
        spectral_failed = False

        for method in methods_needed:
            base_row = {**meta, "exp_idx": exp_idx, "n": n, "method": method, "seed": seed}
            max_n = method_max_n.get(method)
            if max_n is not None and n > max_n:
                records.append({**base_row, "t_embed": float("nan"), "alpha_fit": float("nan"),
                                 "r0_fit": float("nan"), "r1_fit": float("nan"),
                                 "status": "skipped_too_large"})
                continue

            try:
                t0 = time.perf_counter()
                if method == "s_gd2":
                    X = embed_sgd2(G, random_seed=seed)
                    alpha_fit = r0_fit = r1_fit = float("nan")
                elif method == "TorusMDS":
                    if D is None:
                        D, _ = apsp_distance_matrix(G)
                    X, alpha_fit = embed_torus_mds(
                        D, max_iters=torus_max_iters, seed=seed, stress_mode=torus_stress_mode
                    )
                    r0_fit = r1_fit = float("nan")
                elif method in ASPECT_INIT_VARIANTS:
                    if D is None:
                        D, _ = apsp_distance_matrix(G)
                    init_mode, batch_multiplier = ASPECT_INIT_VARIANTS[method]
                    if init_mode == "smart":
                        if spectral_result is None and not spectral_failed:
                            try:
                                spectral_result = find_fundamental_torus_directions(D / D.max())
                            except Exception:
                                spectral_failed = True
                        if spectral_failed:
                            raise RuntimeError("spectral init failed for this graph")
                    X, alpha_fit, r0_fit, r1_fit = embed_torus_mds_aspect(
                        D, init_mode=init_mode, batch_multiplier=batch_multiplier,
                        spectral_result=spectral_result, max_iters=torus_max_iters,
                        seed=seed, stress_mode=torus_stress_mode,
                    )
                elif method == "wrap_python":
                    X = embed_wrap_python(G, max_iters=wrap_python_max_iters, seed=seed)
                    alpha_fit = r0_fit = r1_fit = float("nan")
                else:
                    raise ValueError(f"Unknown method {method!r}")
                t_embed = time.perf_counter() - t0

                np.save(coords_path(output_dir, exp_idx, method), X)
                records.append({**base_row, "t_embed": round(t_embed, 4), "alpha_fit": alpha_fit,
                                 "r0_fit": r0_fit, "r1_fit": r1_fit, "status": "ok"})
                done.add((exp_idx, method))
            except Exception:
                print(f"[{exp_idx}] {method} failed:\n{traceback.format_exc(limit=2)}")
                records.append({**base_row, "t_embed": float("nan"), "alpha_fit": float("nan"),
                                 "r0_fit": float("nan"), "r1_fit": float("nan"), "status": "failed"})

        since_checkpoint += 1
        if since_checkpoint >= checkpoint_every:
            checkpoint()
            since_checkpoint = 0

    checkpoint()
    df = pd.DataFrame(records)
    print(f"\nSaved {len(df)} manifest rows -> {manifest_path}")
    return df
