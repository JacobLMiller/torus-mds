"""Evaluator graph loading directly from persisted sparse adjacency matrices."""
import numba
import numpy as np
from scipy.sparse import load_npz, triu
from scipy.sparse.csgraph import connected_components, dijkstra


@numba.njit(cache=True, parallel=True)
def _bfs_all_pairs(indptr, indices):
    n = len(indptr) - 1
    distances = np.full((n, n), np.inf)
    for source in numba.prange(n):
        queue = np.empty(n, dtype=np.int64)
        queue[0] = source
        distances[source, source] = 0.
        head, tail = 0, 1
        # Once all vertices are discovered their BFS distances are final.
        # Dense, small-diameter graphs often need only a small part of the queue.
        while head < tail and tail < n:
            node = queue[head]
            head += 1
            distance = distances[source, node] + 1.
            for pos in range(indptr[node], indptr[node+1]):
                neighbour = indices[pos]
                if distances[source, neighbour] == np.inf:
                    distances[source, neighbour] = distance
                    queue[tail] = neighbour
                    tail += 1
            # Finishing this adjacency list also keeps the hot loop branch-free
            # with respect to the global discovery count.
    return distances


def load_layout_graph(path, threads=1):
    """Return (shortest-path matrix, indexed edges) without a NetworkX round trip.

    Preserves adjacency row order. Unit-weight graphs use exact BFS; positive
    weighted graphs retain SciPy Dijkstra. Input must be symmetric and loop-free,
    as written by the experiment runner.
    """
    if not 1 <= threads <= numba.config.NUMBA_NUM_THREADS:
        raise ValueError(f"threads must be between 1 and {numba.config.NUMBA_NUM_THREADS}")
    adjacency = load_npz(path).tocsr().astype(np.float64)
    if adjacency.shape[0] != adjacency.shape[1] or adjacency.shape[0] < 2:
        raise ValueError("Graph must be square with at least two nodes")
    if not np.isfinite(adjacency.data).all() or np.any(adjacency.data <= 0):
        raise ValueError("Graph edge weights must be finite and positive")
    if np.any(adjacency.diagonal() != 0):
        raise ValueError("Self-loops are unsupported")
    if (adjacency != adjacency.T).nnz:
        raise ValueError("Saved undirected adjacency must be symmetric")
    adjacency.sort_indices()
    adjacency.indices = adjacency.indices.astype(np.int32, copy=False)
    adjacency.indptr = adjacency.indptr.astype(np.int32, copy=False)
    if connected_components(adjacency, directed=False, return_labels=False) != 1:
        raise ValueError("Graph must be connected with at least two nodes")
    upper = triu(adjacency, k=1, format="coo")
    edges = np.column_stack([upper.row, upper.col]).astype(np.int64, copy=False)
    if np.all(adjacency.data == 1.):
        previous = numba.get_num_threads()
        try:
            numba.set_num_threads(threads)
            distances = _bfs_all_pairs(adjacency.indptr, adjacency.indices)
        finally:
            numba.set_num_threads(previous)
    else:
        distances = dijkstra(adjacency, directed=False)
    return distances, edges
