from __future__ import annotations

import numpy as np
import numba
from scipy.stats import spearmanr

from .geometry import torus_distance


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
