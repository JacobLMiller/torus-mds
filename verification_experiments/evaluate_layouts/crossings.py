"""Evaluator crossing counts using a bounding-volume hierarchy of lifted edges."""
import numba
import numpy as np


def _edge_tree(starts, delta, leaf_size=24):
    lo = np.minimum(starts, starts + delta)
    hi = np.maximum(starts, starts + delta)
    centers = (lo + hi) * .5
    order = np.arange(len(starts), dtype=np.int64)
    # A binary tree has fewer than 2*m nodes (including empty input).
    bounds = np.empty((max(1, 2*len(order)), 4))
    children = np.full((len(bounds), 2), -1, dtype=np.int64)
    spans = np.empty((len(bounds), 3), dtype=np.int64)
    used = 0

    def build(begin, end):
        nonlocal used
        node = used
        used += 1
        ids = order[begin:end]
        bounds[node] = [lo[ids, 0].min(), lo[ids, 1].min(),
                        hi[ids, 0].max(), hi[ids, 1].max()]
        spans[node] = [begin, end, ids.max()]
        if end - begin > leaf_size:
            axis = int(np.argmax(np.ptp(centers[ids], axis=0)))
            middle = (end - begin)//2
            order[begin:end] = ids[np.argpartition(centers[ids, axis], middle)]
            left = build(begin, begin + middle)
            right = build(begin + middle, end)
            children[node] = [left, right]
        return node

    if len(order):
        build(0, len(order))
    return lo, hi, order, bounds[:used], children[:used], spans[:used]


@numba.njit(cache=True, parallel=True)
def _parallel_indexed_crossings(starts, delta, edges, periodic, eps, lo, hi,
                  order, bounds, children, spans):
    counts = np.zeros(len(starts), dtype=np.int64)
    for i in numba.prange(len(starts)):
        # The median-split tree is balanced; 64 slots exceed its possible depth
        # for arrays that fit in memory. Each worker has its own stack.
        stack = np.empty(64, dtype=np.int64)
        xmin = int(np.ceil(lo[i, 0] - bounds[0, 2] - eps)) if periodic else 0
        xmax = int(np.floor(hi[i, 0] - bounds[0, 0] + eps)) if periodic else 0
        ymin = int(np.ceil(lo[i, 1] - bounds[0, 3] - eps)) if periodic else 0
        ymax = int(np.floor(hi[i, 1] - bounds[0, 1] + eps)) if periodic else 0
        a, b = delta[i, 0], delta[i, 1]
        count = 0
        for sx in range(xmin, xmax + 1):
            for sy in range(ymin, ymax + 1):
                qlo0, qlo1 = lo[i, 0]-sx-eps, lo[i, 1]-sy-eps
                qhi0, qhi1 = hi[i, 0]-sx+eps, hi[i, 1]-sy+eps
                stack[0] = 0
                size = 1
                while size:
                    size -= 1
                    node = stack[size]
                    if (spans[node, 2] <= i or bounds[node, 0] > qhi0
                            or bounds[node, 1] > qhi1 or bounds[node, 2] < qlo0
                            or bounds[node, 3] < qlo1):
                        continue
                    if children[node, 0] >= 0:
                        stack[size] = children[node, 0]
                        stack[size+1] = children[node, 1]
                        size += 2
                        continue
                    for pos in range(spans[node, 0], spans[node, 1]):
                        j = order[pos]
                        if (j <= i or lo[j, 0] > qhi0 or lo[j, 1] > qhi1
                                or hi[j, 0] < qlo0 or hi[j, 1] < qlo1):
                            continue
                        if (edges[i, 0] == edges[j, 0] or edges[i, 0] == edges[j, 1]
                                or edges[i, 1] == edges[j, 0] or edges[i, 1] == edges[j, 1]):
                            continue
                        c, d = delta[j, 0], delta[j, 1]
                        det = a*d - b*c
                        if abs(det) <= eps * np.sqrt((a*a+b*b)*(c*c+d*d)):
                            continue
                        qx = starts[j, 0] + sx - starts[i, 0]
                        qy = starts[j, 1] + sy - starts[i, 1]
                        t = (qx*d - qy*c)/det
                        u = (qx*b - qy*a)/det
                        if eps < t < 1-eps and eps < u < 1-eps:
                            count += 1
        counts[i] = count
    return counts.sum()


def indexed_crossings(starts, delta, edges, periodic, eps=1e-12, threads=1):
    """Count original intersection events exactly, using 1 or more CPU threads.

    The tree prunes pairs with disjoint bounding boxes. Translated torus images
    are queried separately, preserving multiple crossings of the same pair.
    Memory is linear in edge count; dense overlapping boxes remain expensive.
    """
    if not len(starts):
        return 0
    tree = _edge_tree(starts, delta)
    args = (starts, delta, edges, periodic, eps, *tree)
    previous = numba.get_num_threads()
    try:
        numba.set_num_threads(threads)
        return int(_parallel_indexed_crossings(*args))
    finally:
        numba.set_num_threads(previous)
