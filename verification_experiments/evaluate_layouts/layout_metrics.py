"""Accelerated full-layout evaluation, retaining the existing metric formulas."""
import numba
import numpy as np
from scipy.stats import rankdata

from modules.geometry import gauss_reduce_basis, _min_image_offset, _gram_quad, _wrap_to_half
from modules.metrics import layout_edge_geometry, _count_layout_crossings, _incidence_from_geometry
from .crossings import indexed_crossings


@numba.njit(cache=True)
def _distance_matrix(X, periodic, alpha, g00, g01, g11, transform):
    n = len(X)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            a = X[j, 0] - X[i, 0]
            b = X[j, 1] - X[i, 1]
            if periodic:
                # Transform displacements, as parallelogram_distance does,
                # rather than subtracting already transformed positions.
                a, b = (transform[0, 0]*a + transform[0, 1]*b,
                        transform[1, 0]*a + transform[1, 1]*b)
                a = _wrap_to_half(a)
                b = _wrap_to_half(b)
                a, b = _min_image_offset(a, b, g00, g01, g11)
                squared = _gram_quad(a, b, g00, g01, g11)
            else:
                squared = a*a + b*b
            distances[i, j] = alpha * np.sqrt(max(0., squared))
            distances[j, i] = distances[i, j]
    return distances


def _geometry_parameters(periodic, alpha, r0, r1, theta):
    g00, g01, g11 = 1., 0., 1.
    transform = np.eye(2)
    if periodic:
        if np.isclose(theta, np.pi/2):
            g00, g11 = r0*r0, r1*r1
        else:
            x = r1/r0 * np.cos(theta)
            y = r1/r0 * np.sin(theta)
            # Match parallelogram_distance's reduction decision and arithmetic.
            scale = 1.
            if not (abs(x) <= .5 and x*x + y*y >= 1.):
                x, y, scale, transform = gauss_reduce_basis(x, y)
            s = scale*scale
            g00, g01, g11 = s, s*x, s*(x*x + y*y)
            alpha *= r0
    return alpha, g00, g01, g11, transform


def layout_distance_matrix(X, *, periodic=True, alpha=1., r0=1., r1=1., theta=np.pi/2):
    """Compiled equivalent of geodesic_matrix(X, make_torus_geod(...))."""
    return _distance_matrix(np.ascontiguousarray(X, dtype=np.float64), periodic,
                            *_geometry_parameters(periodic, alpha, r0, r1, theta))


@numba.njit(cache=True)
def _query_distances(queries, X, periodic, alpha, g00, g01, g11, transform):
    result = np.empty((len(queries), len(X)))
    for i in range(len(queries)):
        for j in range(len(X)):
            a = X[j, 0] - queries[i, 0]
            b = X[j, 1] - queries[i, 1]
            if periodic:
                a, b = (transform[0, 0]*a + transform[0, 1]*b,
                        transform[1, 0]*a + transform[1, 1]*b)
                a, b = _wrap_to_half(a), _wrap_to_half(b)
                a, b = _min_image_offset(a, b, g00, g01, g11)
                squared = _gram_quad(a, b, g00, g01, g11)
            else:
                squared = a*a+b*b
            result[i, j] = alpha*np.sqrt(max(0., squared))
    return result


def layout_query_distances(queries, X, *, periodic=True, alpha=1., r0=1., r1=1., theta=np.pi/2):
    return _query_distances(np.ascontiguousarray(queries, dtype=np.float64),
                           np.ascontiguousarray(X, dtype=np.float64), periodic,
                           *_geometry_parameters(periodic, alpha, r0, r1, theta))


@numba.njit(cache=True)
def fitted_distance_scale(distances, D):
    """Best multiplier sum(D*r)/sum(r^2), matching estimate_alpha's objective."""
    dr, rr = 0., 0.
    for i in range(len(D)):
        for j in range(i):
            r = distances[i, j]
            dr += D[i, j]*r
            rr += r*r
    return dr/rr if rr > 0 else np.nan


@numba.njit(cache=True)
def _distance_errors(distances, D, scale):
    squared_error, squared_target, relative_error = 0., 0., 0.
    absolute_error, target_sum = 0., 0.
    for i in range(len(D)):
        for j in range(i):
            d = D[i, j]
            error = scale*distances[i, j] - d
            squared_error += error*error
            squared_target += d*d
            relative_error += (error/d)**2
            absolute_error += abs(error)
            target_sum += d
    pairs = len(D)*(len(D)-1)//2
    return (np.sqrt(squared_error/squared_target), np.sqrt(relative_error/pairs),
            absolute_error/target_sum)


def _condensed(distances):
    return np.concatenate([distances[i, i+1:] for i in range(len(distances)-1)])


def graph_distance_ranks(D):
    """Cache centered average ranks and squared norm once per graph/sample."""
    ranks = rankdata(_condensed(D), method="average")
    ranks -= ranks.mean()
    return ranks, float(ranks @ ranks)


def cached_sgs(distances, graph_ranks):
    target, target_norm = graph_ranks
    embedded = rankdata(_condensed(distances), method="average")
    if len(embedded) != len(target):
        raise ValueError("SGS graph ranks must match the layout sample")
    embedded -= embedded.mean()
    norm = float(embedded @ embedded)
    if target_norm == 0 or norm == 0:
        return float("nan")
    return float(np.clip((target @ embedded) / np.sqrt(target_norm*norm), -1., 1.))


def distance_metrics(distances, D, rg=2., scale=1., graph_ranks=None, include_np=True):
    """Reuse a distance matrix for stress, distortion, SGS and neighbourhoods.

    Neighbourhood sorting intentionally uses NumPy's existing default argsort,
    then drops its first entry, exactly as geodesic_NP currently does.
    """
    # Apply the evaluation scale only to error metrics; rank/neighbourhood
    # calculations retain their original matrix and tie behavior.
    stress, pointwise, distortion = _distance_errors(distances, D, scale)
    sgs = cached_sgs(distances, graph_distance_ranks(D) if graph_ranks is None else graph_ranks)
    score, count = 0., 0
    for i in range(len(D) if include_np else 0):
        theory = (D[i] <= rg) & (D[i] > 0)
        k = np.count_nonzero(theory)
        if not k:
            continue
        closest = np.argsort(distances[i])[1:k+1]
        intersection = np.count_nonzero(theory[closest])
        score += intersection / (2*k - intersection)
        count += 1
    return dict(stress=float(stress), pointwise_stress=float(pointwise),
                distortion=float(distortion), sgs=sgs,
                np_score=float(score/count) if count else 0.)


@numba.njit(cache=True)
def _silhouette_query_total(distances, labels, counts, queries):
    total = 0.
    for row in range(len(queries)):
        i = queries[row]
        own = labels[i]
        if counts[own] == 1:
            continue  # Standard convention: a singleton has silhouette zero.
        sums = np.zeros(len(counts))
        for j in range(len(labels)):
            if i != j:
                sums[labels[j]] += distances[row, j]
        a = sums[own] / (counts[own] - 1)
        b = np.inf
        for group in range(len(counts)):
            if group != own:
                b = min(b, sums[group] / counts[group])
        denominator = max(a, b)
        if denominator > 0:
            total += (b - a) / denominator
    return total


def silhouette_score(distances, labels):
    """Mean silhouette from existing layout distances and fixed cluster labels.

    Returns NaN unless 2 <= number of clusters < number of nodes. Singletons
    and nodes with a=b=0 contribute zero. Runs in O(n^2) time with O(k) working
    storage beyond the existing distance matrix; no new dependency is needed.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1 or distances.shape != (len(labels), len(labels)):
        raise ValueError("Silhouette labels must match the square distance matrix")
    _, encoded, counts = np.unique(labels, return_inverse=True, return_counts=True)
    if not 2 <= len(counts) < len(labels):
        return float("nan")
    return float(_silhouette_query_total(distances, encoded, counts,
                                        np.arange(len(labels))) / len(labels))


def sampled_query_metrics(X, D, queries, *, labels=None, rg=2., **geometry):
    """Sample node averages with full candidate neighbourhoods and cluster means.

    Bound workspace to 64-by-n rows rather than a full n-by-n layout matrix.
    Retains geodesic_NP's existing default argsort/drop-first tie convention.
    """
    score, count, silhouette_total = 0., 0, 0.
    valid_labels = False
    if labels is not None:
        _, encoded, counts = np.unique(labels, return_inverse=True, return_counts=True)
        valid_labels = 2 <= len(counts) < len(labels)
    for begin in range(0, len(queries), 64):
        block = queries[begin:begin+64]
        distances = layout_query_distances(X[block], X, **geometry)
        for row, i in enumerate(block):
            theory = (D[i] <= rg) & (D[i] > 0)
            k = np.count_nonzero(theory)
            if not k:
                continue
            closest = np.argsort(distances[row])[1:k+1]
            intersection = np.count_nonzero(theory[closest])
            score += intersection/(2*k-intersection)
            count += 1
        if valid_labels:
            silhouette_total += _silhouette_query_total(distances, encoded, counts, block)
    return dict(np_score=float(score/count) if count else 0.,
                silhouette_score=float(silhouette_total/len(queries)) if valid_labels else np.nan)


def edge_metrics(X, edges, *, threads=1, compute_crossings=True, alpha=1., **geometry):
    """Share full shortest-edge geometry for incidence, mean length, and optional crossings."""
    edges = np.sort(np.asarray(edges, dtype=np.int64).reshape(-1, 2), axis=1)
    starts, delta, basis = layout_edge_geometry(X, edges, **geometry)
    periodic = geometry.get("periodic", True)
    crossings = np.nan
    if compute_crossings:
        crossings = (int(_count_layout_crossings(starts, delta, edges, periodic, 1e-12))
                     if len(edges) < 2048 and threads == 1 else
                     indexed_crossings(starts, delta, edges, periodic, threads=threads))
    return dict(edge_crossings=crossings,
                mean_edge_length=float(alpha*np.linalg.norm(delta @ basis.T, axis=1).mean())
                                 if len(edges) else np.nan,
                incidence_angle=_incidence_from_geometry(len(X), edges, delta, basis))
