from __future__ import annotations

import json
import glob as _glob
from pathlib import Path

import numpy as np
import networkx as nx


def apsp_distance_matrix(G: nx.Graph, weight: str = "weight") -> tuple[np.ndarray, list]:
    """
    All-pairs shortest path distance matrix for graph G.

    Returns
    -------
    D     : (N, N) float64 distance matrix
    nodes : list of nodes in row/column order
    """
    nodes = list(G.nodes())
    idx = {n: k for k, n in enumerate(nodes)}
    n = len(nodes)
    D = np.zeros((n, n), dtype=np.float64)
    for src, dist_dict in nx.all_pairs_dijkstra_path_length(G, weight=weight):
        i = idx[src]
        for dst, d in dist_dict.items():
            D[i, idx[dst]] = float(d)
    return D, nodes


def periodic_lattice_graph(nx_size: int, ny_size: int, diagonal: bool = False) -> nx.Graph:
    """
    2D periodic lattice (torus topology) graph. Nodes are (i, j) tuples.
    If diagonal=True, adds 4 diagonal neighbours (8-neighbourhood total).
    """
    G = nx.grid_2d_graph(nx_size, ny_size, periodic=True)
    if diagonal:
        for i in range(nx_size):
            for j in range(ny_size):
                G.add_edge((i, j), ((i + 1) % nx_size, (j + 1) % ny_size))
                G.add_edge((i, j), ((i + 1) % nx_size, (j - 1) % ny_size))
    nx.set_edge_attributes(G, 1.0, "weight")
    return G


def get_periodic_lattice(nx_size: int = 20, ny_size: int = 20) -> tuple[nx.Graph, np.ndarray]:
    """Convenience wrapper: build a periodic lattice and return (G, D)."""
    G = periodic_lattice_graph(nx_size, ny_size, diagonal=False)
    D, _ = apsp_distance_matrix(G)
    return G, D


def _normalise_axis(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    span = float(values.max() - values.min())
    if span == 0.0:
        return np.zeros_like(values, dtype=float)
    return (values - values.min()) / span


def _parse_shortest_paths(raw_paths: object, nodes: list[int]) -> np.ndarray | None:
    if raw_paths is None:
        return None
    if isinstance(raw_paths, str):
        raw_paths = json.loads(raw_paths)
    if not isinstance(raw_paths, dict):
        return None

    node_to_index = {node: index for index, node in enumerate(nodes)}
    matrix = np.full((len(nodes), len(nodes)), np.inf, dtype=np.float64)
    np.fill_diagonal(matrix, 0.0)
    for source_key, destinations in raw_paths.items():
        source = int(source_key)
        if source not in node_to_index or not isinstance(destinations, dict):
            continue
        source_index = node_to_index[source]
        for target_key, path_nodes in destinations.items():
            target = int(target_key)
            if target not in node_to_index or not isinstance(path_nodes, list):
                continue
            matrix[source_index, node_to_index[target]] = float(max(len(path_nodes) - 1, 0))
    return matrix


def parse_chen_json(path: str) -> tuple[nx.Graph, np.ndarray | None, np.ndarray]:
    """
    Parse a Chen et al. toroidal graph layout JSON file.

    Node (x, y) pixel coordinates are normalised to [0, 1] per axis.
    APSP distances are computed from the graph topology.

    Returns
    -------
    G : nx.Graph          nodes indexed by ``id`` or ``index`` field
    X : (N, 2) or None    unit-torus coordinates if present in the file
    D : (N, N)            APSP distance matrix
    """
    with open(path) as f:
        data = json.load(f)

    graph_data = data.get("graph")
    nodes_data = data.get("nodes")
    links_data = data.get("links")
    if isinstance(graph_data, dict):
        nodes_data = graph_data.get("nodes", nodes_data)
        links_data = graph_data.get("links", links_data)
    if not isinstance(nodes_data, list) or not isinstance(links_data, list):
        raise KeyError("Could not find graph nodes/links in Chen JSON")

    G = nx.Graph()
    has_positions = all("x" in node and "y" in node for node in nodes_data)
    xs = _normalise_axis(np.array([node["x"] for node in nodes_data], dtype=float)) if has_positions else None
    ys = _normalise_axis(np.array([node["y"] for node in nodes_data], dtype=float)) if has_positions else None

    for index, node in enumerate(nodes_data):
        node_id = node.get("index", node.get("id"))
        if node_id is None:
            raise KeyError(f"Node at position {index} is missing both 'id' and 'index'")
        attributes = dict(node)
        if has_positions and xs is not None and ys is not None:
            attributes["x"] = float(xs[index])
            attributes["y"] = float(ys[index])
        G.add_node(int(node_id), **attributes)

    for edge in links_data:
        source = edge.get("source")
        target = edge.get("target")
        if isinstance(source, dict):
            source = source.get("index", source.get("id"))
        if isinstance(target, dict):
            target = target.get("index", target.get("id"))
        if source is None or target is None:
            raise KeyError(f"Edge is missing source/target information: {edge!r}")
        G.add_edge(int(source), int(target))

    node_order = list(G.nodes())
    X = None
    if has_positions:
        X = np.array([[G.nodes[node]["x"], G.nodes[node]["y"]] for node in node_order], dtype=float)

    D = _parse_shortest_paths(data.get("path"), node_order)
    if D is None:
        D, _ = apsp_distance_matrix(G)
    return G, X, D


def load_chen_graphs(pattern: str = "chengraphs/*.json") -> list[tuple[str, nx.Graph, np.ndarray | None, np.ndarray]]:
    """
    Load all Chen et al. JSON files matching a glob pattern.

    Returns a list of ``(name, G, X, D)`` tuples sorted by graph name.
    ``X`` is ``None`` when the source JSON does not provide node coordinates.
    """
    paths = sorted(_glob.glob(pattern), key=lambda p: Path(p).stem)
    return [(Path(path).stem, *parse_chen_json(path)) for path in paths]
