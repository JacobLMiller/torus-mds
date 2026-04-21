from __future__ import annotations

import json
import glob as _glob

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


def parse_chen_json(path: str) -> tuple[nx.Graph, np.ndarray, np.ndarray]:
    """
    Parse a Chen et al. toroidal graph layout JSON file.

    Node (x, y) pixel coordinates are normalised to [0, 1] per axis.
    APSP distances are computed from the graph topology.

    Returns
    -------
    G : nx.Graph    nodes indexed by 'index' field
    X : (N, 2)      unit-torus coordinates, row order matching G.nodes()
    D : (N, N)      APSP distance matrix
    """
    with open(path) as f:
        data = json.load(f)

    nodes = data['graph']['nodes']
    xs = np.array([n['x'] for n in nodes], dtype=float)
    ys = np.array([n['y'] for n in nodes], dtype=float)
    xs = (xs - xs.min()) / (xs.max() - xs.min())
    ys = (ys - ys.min()) / (ys.max() - ys.min())

    G = nx.Graph()
    for n, x, y in zip(nodes, xs, ys):
        G.add_node(n['index'], x=x, y=y)
    for edge in data['graph']['links']:
        G.add_edge(edge['source']['index'], edge['target']['index'])

    X = np.array([[G.nodes[n]['x'], G.nodes[n]['y']] for n in G.nodes()])
    D, _ = apsp_distance_matrix(G)
    return G, X, D


def load_chen_graphs(pattern: str = "chengraphs/*.json") -> list[tuple[str, nx.Graph, np.ndarray, np.ndarray]]:
    """
    Load all Chen et al. JSON files matching a glob pattern.

    Returns a list of (name, G, X, D) tuples sorted by graph name.
    """
    paths = sorted(_glob.glob(pattern), key=lambda p: p.split("/")[-1].replace(".json", ""))
    return [(p.split("/")[-1].replace(".json", ""), *parse_chen_json(p)) for p in paths]
