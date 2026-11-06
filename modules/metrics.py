from __future__ import annotations

import numpy as np
import numba
from scipy.stats import spearmanr

from .geometry import torus_distance, _min_image_offset as _layout_min_image_offset


def subsample(max_n: int, *arrays, seed: int = 42):
    """
    Uniformly subsample arrays to at most max_n rows, keeping them in sync.
    Returns a single array if one argument is given, otherwise a tuple.
    """
    n = arrays[0].shape[0]
    if n <= max_n:
        return arrays[0] if len(arrays) == 1 else arrays
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, max_n, replace=False)
    result = tuple(a[idx] for a in arrays)
    return result[0] if len(arrays) == 1 else result


def geodesic_matrix(X: np.ndarray, geod) -> np.ndarray:
    """Full pairwise distance matrix under embedding metric geod(p, q)."""
    n = X.shape[0]
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            M[i, j] = geod(X[i], X[j])
            M[j, i] = M[i, j]
    return M


def geodesic_stress(X: np.ndarray, D: np.ndarray, geod) -> float:
    """
    Normalised stress between input distances D and embedding distances under geod.

        stress = sqrt( sum_{i<j} (geod(x_i,x_j) - D_ij)^2 / sum_{i<j} D_ij^2 )
    """
    n = X.shape[0]
    num, denom = 0.0, 0.0
    for i in range(n):
        for j in range(i):
            num   += (geod(X[i], X[j]) - D[i, j]) ** 2
            denom += D[i, j] ** 2
    return float(np.sqrt(num / denom))


def scaled_geodesic_stress(
    X: np.ndarray, D: np.ndarray, geod, alpha: float | None = None
) -> tuple[float, float]:
    """
    Scale-invariant counterpart to ``geodesic_stress``: fits a uniform scale
    alpha for the embedding distances r_ij = geod(x_i, x_j) before computing
    stress, so the result is unchanged if X is globally rescaled (e.g. an
    unconstrained Euclidean embedding, where ``geodesic_stress`` would
    otherwise change with the embedding's size). Same objective as
    ``rect_torus_stress``, but for an arbitrary geod:

        alpha = sum_{i<j}[D_ij * r_ij] / sum_{i<j}[r_ij^2]
        stress = sqrt( sum_{i<j}(alpha*r_ij - D_ij)^2 / sum_{i<j} D_ij^2 )

    alpha=None -> closed-form best-fit alpha; pass a fixed value to evaluate
    stress at that scale instead. Returns (stress, alpha).
    """
    n = X.shape[0]
    sdr, sr2, sd2 = 0.0, 0.0, 0.0
    for i in range(n):
        for j in range(i):
            r = geod(X[i], X[j])
            d = D[i, j]
            sdr += d * r
            sr2 += r * r
            sd2 += d * d
    if alpha is None:
        alpha = sdr / sr2
    num = alpha * alpha * sr2 - 2.0 * alpha * sdr + sd2   # = sum (alpha*r - D)^2
    return float(np.sqrt(num / sd2)), float(alpha)


def pointwise_geodesic_stress(X: np.ndarray, D: np.ndarray, geod) -> float:
    """
    Per-pair-normalised counterpart to ``geodesic_stress``: each pair's squared
    residual is divided by its own D_ij^2 before summing (rather than once, by
    the total sum(D^2), at the end), then averaged over all pairs -- an RMS
    relative-distance error. Matches the per-pair weighting used for
    ``stress_mode='normalized'`` training in ``TorusProjector.fit``.

        stress = sqrt( mean_{i<j}[ (geod(x_i,x_j) - D_ij)^2 / D_ij^2 ] )
    """
    n = X.shape[0]
    nchoose2 = (n * (n-1) // 2)
    total = 0.0
    for i in range(n):
        for j in range(i):
            d = D[i, j]
            total += ((geod(X[i], X[j]) - d) / d) ** 2

    return float(np.sqrt(total / nchoose2))


def scaled_pointwise_geodesic_stress(
    X: np.ndarray, D: np.ndarray, geod, alpha: float | None = None
) -> tuple[float, float]:
    """
    Scale-invariant counterpart to ``pointwise_geodesic_stress``: fits a
    uniform scale alpha for the embedding distances r_ij = geod(x_i, x_j)
    before computing stress, so the result is unchanged if X is globally
    rescaled. This minimises a differently-weighted objective than
    ``scaled_geodesic_stress``, so the closed-form alpha differs:

        alpha = sum_{i<j}[r_ij / D_ij] / sum_{i<j}[(r_ij / D_ij)^2]
        stress = sqrt( mean_{i<j}[ (alpha*r_ij - D_ij)^2 / D_ij^2 ] )

    alpha=None -> closed-form best-fit alpha; pass a fixed value to evaluate
    stress at that scale instead. Returns (stress, alpha).
    """
    n = X.shape[0]
    sum_s, sum_s2 = 0.0, 0.0
    count = 0
    for i in range(n):
        for j in range(i):
            s = geod(X[i], X[j]) / D[i, j]
            sum_s += s
            sum_s2 += s * s
            count += 1
    if alpha is None:
        alpha = sum_s / sum_s2
    loss = alpha * alpha * sum_s2 - 2.0 * alpha * sum_s + count   # = sum (alpha*s - 1)^2
    return float(np.sqrt(loss / count)), float(alpha)


@numba.njit(parallel=True, fastmath=True, cache=True)
def _rect_torus_stress_accum(X, D, r0, r1):
    """Accumulate sum_{i<j} of d*r, r*r, d*d for a rectangular (theta=90) torus,
    with r = sqrt((r0*du0)^2 + (r1*du1)^2) and du the per-coordinate min-image."""
    n = X.shape[0]
    sdr = 0.0
    sr2 = 0.0
    sd2 = 0.0
    for i in numba.prange(n):
        for j in range(i):
            a = ((X[j, 0] - X[i, 0] + 0.5) % 1.0) - 0.5
            b = ((X[j, 1] - X[i, 1] + 0.5) % 1.0) - 0.5
            r = np.sqrt((r0 * a) ** 2 + (r1 * b) ** 2)
            d = D[i, j]
            sdr += d * r
            sr2 += r * r
            sd2 += d * d
    return sdr, sr2, sd2


def rect_torus_stress(X: np.ndarray, D: np.ndarray, r0: float = 1.0, r1: float = 1.0,
                      alpha: float | None = None) -> tuple[float, float]:
    """
    Normalised stress on a rectangular (theta=90) torus, numba-accelerated for
    large N. Same objective as ``geodesic_stress`` with a rectangular geodesic:

        stress = sqrt( sum_{i<j}(alpha*r_ij - D_ij)^2 / sum_{i<j} D_ij^2 )

    alpha=None -> closed-form best-fit alpha = sum(D*r) / sum(r*r).
    Returns (stress, alpha).
    """
    sdr, sr2, sd2 = _rect_torus_stress_accum(
        np.asarray(X, np.float64), np.asarray(D, np.float64), float(r0), float(r1))
    if alpha is None:
        alpha = sdr / sr2
    num = alpha * alpha * sr2 - 2.0 * alpha * sdr + sd2   # = sum (alpha*r - D)^2
    return float(np.sqrt(num / sd2)), float(alpha)


def geodesic_distortion(X: np.ndarray, D: np.ndarray, geod) -> float:
    """
    Mean absolute relative distortion:

        distortion = sum_{i<j} |geod(x_i,x_j) - D_ij| / sum_{i<j} D_ij
    """
    n = X.shape[0]
    num, denom = 0.0, 0.0
    for i in range(n):
        for j in range(i):
            num   += abs(geod(X[i], X[j]) - D[i, j])
            denom += D[i, j]
    return float(num / denom)


def SGS(X: np.ndarray, D: np.ndarray, geod) -> float:
    """
    Spearman rank correlation between input distances and embedding distances
    (Spearman's Goodness Score). Higher is better; max 1.
    """
    hd = D[np.triu_indices_from(D, k=1)]
    ld = geodesic_matrix(X, geod)[np.triu_indices_from(D, k=1)]
    return float(spearmanr(hd, ld).statistic)


def geodesic_NP(X: np.ndarray, D: np.ndarray, geod, rg: float = 2) -> float:
    """
    Neighbourhood precision: mean Jaccard similarity between the theoretical
    neighbourhood (nodes within radius rg in D) and the embedded neighbourhood
    of the same size under geod. Closer to 1 is better.
    """
    n = X.shape[0]
    dist_mat = geodesic_matrix(X, geod)
    k_theory   = [np.where((D[i] <= rg) & (D[i] > 0))[0] for i in range(n)]
    k_embedded = [np.argsort(dist_mat[i])[1:len(k_theory[i]) + 1] for i in range(n)]

    score = 0.0
    count = 0
    for i in range(n):
        if len(k_theory[i]) == 0:
            continue
        intersect = np.intersect1d(k_theory[i], k_embedded[i]).size
        jaccard = intersect / (2 * k_theory[i].size - intersect)
        score += jaccard
        count += 1
    return float(score / count) if count > 0 else 0.0


def estimate_alpha(
    X: np.ndarray,
    D: np.ndarray,
    geod=None,
    *,
    r0: float = 1.0,
    r1: float = 1.0,
    theta: float = np.pi / 2,
) -> float:
    """
    Closed-form least-squares estimate of the torus scale factor alpha for a fixed layout X and target distances D.

    By default this is the old unit-square torus behavior.
    For rectangular/rhombic tori, either pass an unscaled `geod`, or pass r0, r1, and theta:

        estimate_alpha(X, D, r0=proj.r0_, r1=proj.r1_, theta=proj.theta_)

    The fitted scale is:

        alpha_hat = sum(D_ij * r_ij) / sum(r_ij^2)

    where `geod` returns the unscaled base distance r_ij.
    """
    if geod is None:
        geod = lambda p, q: torus_distance(p, q, r0=r0, r1=r1, theta=theta)

    n = X.shape[0]
    num, denom = 0.0, 0.0
    for i in range(n):
        for j in range(i):
            r = geod(X[i], X[j])
            num   += D[i, j] * r
            denom += r * r
    return float(num / denom)


@numba.njit(cache=True)
def _edge_offsets(delta, gram):
    for i in range(len(delta)):
        delta[i, 0], delta[i, 1] = _layout_min_image_offset(
            delta[i, 0], delta[i, 1], gram[0, 0], gram[0, 1], gram[1, 1])
    return delta


def layout_edge_geometry(X, edges, *, periodic=True, r0=1.0, r1=1.0,
                         theta=np.pi / 2):
    """Return starts, shortest displacements and physical basis for indexed edges.

    Edges are oriented by ascending node index. Tied shortest images follow the
    deterministic candidate order of the project's minimum-image primitive.
    Periodic coordinates are fractional coordinates, never min-max normalized.
    """
    from .geometry import gauss_reduce_basis

    X = np.asarray(X, dtype=float)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if X.ndim != 2 or X.shape[1] != 2 or not np.isfinite(X).all():
        raise ValueError("Coordinates must be a finite (n, 2) array")
    if edges.size and (edges.min() < 0 or edges.max() >= len(X)):
        raise ValueError("Edge index outside coordinate array")
    edges = np.sort(edges, axis=1)
    if np.any(edges[:, 0] == edges[:, 1]):
        raise ValueError("Self-loops are unsupported")
    basis = np.eye(2)
    if periodic:
        if not (np.isfinite([r0, r1, theta]).all() and r0 > 0 and r1 > 0
                and 0 < theta < np.pi):
            raise ValueError("Invalid torus side lengths or angle")
        if np.isclose(theta, np.pi / 2):
            basis = np.diag([r0, r1])
        else:
            x, y, scale, transform = gauss_reduce_basis(
                r1 / r0 * np.cos(theta), r1 / r0 * np.sin(theta))
            X = X @ transform.T
            basis = r0 * scale * np.array([[1., x], [0., y]])
        X = X % 1.
    starts = X[edges[:, 0]]
    delta = X[edges[:, 1]] - starts
    if periodic:
        delta = (delta + .5) % 1. - .5
        gram = basis.T @ basis
        delta = _edge_offsets(delta, gram)
    return starts, delta, basis


@numba.njit(cache=True)
def _count_layout_crossings(starts, delta, edges, periodic, eps):
    """Sweep bounding boxes of lifted edges; enumerate overlapping lattice images."""
    lo = np.minimum(starts, starts + delta)
    hi = np.maximum(starts, starts + delta)
    order = np.argsort(lo[:, 0])
    count = 0
    for oi in range(len(order)):
        i = order[oi]
        for oj in range(oi + 1, len(order)):
            j = order[oj]
            if not periodic and lo[j, 0] > hi[i, 0]:
                break
            if (edges[i, 0] == edges[j, 0] or edges[i, 0] == edges[j, 1]
                    or edges[i, 1] == edges[j, 0] or edges[i, 1] == edges[j, 1]):
                continue
            xmin = int(np.ceil(lo[i, 0] - hi[j, 0] - eps)) if periodic else 0
            xmax = int(np.floor(hi[i, 0] - lo[j, 0] + eps)) if periodic else 0
            if periodic and xmin > xmax:
                continue
            if not periodic and (lo[i, 1] > hi[j, 1] or lo[j, 1] > hi[i, 1]):
                continue
            ymin = int(np.ceil(lo[i, 1] - hi[j, 1] - eps)) if periodic else 0
            ymax = int(np.floor(hi[i, 1] - lo[j, 1] + eps)) if periodic else 0
            if periodic and ymin > ymax:
                continue
            a, b = delta[i, 0], delta[i, 1]
            c, d = delta[j, 0], delta[j, 1]
            determinant = a * d - b * c
            if abs(determinant) <= eps * np.sqrt((a*a+b*b)*(c*c+d*d)):
                continue
            for sx in range(xmin, xmax + 1):
                for sy in range(ymin, ymax + 1):
                    qx = starts[j, 0] + sx - starts[i, 0]
                    qy = starts[j, 1] + sy - starts[i, 1]
                    t = (qx * d - qy * c) / determinant
                    u = (qx * b - qy * a) / determinant
                    if eps < t < 1. - eps and eps < u < 1. - eps:
                        count += 1
    return count


def edge_crossings(X, edges, *, periodic=True, r0=1., r1=1., theta=np.pi/2, threads=1):
    """Number of proper intersections of nonadjacent shortest edges.

    Counts intersection events (a pair may cross multiple times on a torus).
    Endpoint contacts and collinear overlaps are excluded; seams are not endpoints.
    Candidate pruning avoids segment tests for disjoint bounding boxes; worst-case
    running time remains quadratic in the number of edges.
    Large edge sets use a spatial hierarchy; threads controls its CPU parallelism.
    """
    edges = np.sort(np.asarray(edges, dtype=np.int64).reshape(-1, 2), axis=1)
    starts, delta, _ = layout_edge_geometry(
        X, edges, periodic=periodic, r0=r0, r1=r1, theta=theta)
    if len(edges) >= 2048 or threads != 1:
        from verification_experiments.evaluate_layouts.crossings import indexed_crossings
        return indexed_crossings(starts, delta, edges, periodic, threads=threads)
    return int(_count_layout_crossings(starts, delta, edges, periodic, 1e-12))


def incidence_angle(X, edges, *, periodic=True, r0=1., r1=1., theta=np.pi/2):
    """Mean relative minimum-angle deviation, including all nodes in the mean.

    Degree <= 1 contributes zero. Coincident linked endpoints yield NaN because
    the incident tangent is undefined. Smaller scores are better.
    """
    edges = np.sort(np.asarray(edges, dtype=np.int64).reshape(-1, 2), axis=1)
    _, delta, basis = layout_edge_geometry(
        X, edges, periodic=periodic, r0=r0, r1=r1, theta=theta)
    return _incidence_from_geometry(len(X), edges, delta, basis)


def _incidence_from_geometry(n, edges, delta, basis):
    """Evaluate existing incidence definition from already prepared edges."""
    vectors = delta @ basis.T
    if np.any(np.linalg.norm(vectors, axis=1) <= 1e-12 * np.linalg.norm(basis)):
        return float("nan")
    angles = [[] for _ in range(n)]
    for (u, v), (dx, dy) in zip(edges, vectors):
        angle = np.arctan2(dy, dx)
        angles[u].append(angle % (2*np.pi))
        angles[v].append((angle + np.pi) % (2*np.pi))
    total = 0.
    for incident in angles:
        if len(incident) < 2:
            continue
        ordered = np.sort(incident)
        gap = np.min(np.diff(np.append(ordered, ordered[0] + 2*np.pi)))
        ideal = 2*np.pi / len(incident)
        total += abs(ideal - gap) / ideal
    return float(total / n) if n else float("nan")
