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


def estimate_alpha(X: np.ndarray, D: np.ndarray) -> float:
    """
    Closed-form least-squares estimate of the torus scale factor alpha for a
    fixed layout X and target distances D:

        alpha_hat = sum(D_ij * r_ij) / sum(r_ij^2),  r_ij = torus_distance(X[i], X[j])

    Used for evaluating external layouts.
    """
    n = X.shape[0]
    num, denom = 0.0, 0.0
    for i in range(n):
        for j in range(i):
            r = torus_distance(X[i], X[j])
            num   += D[i, j] * r
            denom += r * r
    return float(num / denom)
