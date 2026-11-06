"""Signed convex-hull cluster separation from Chen et al., CHI 2021, §4.3.2.

https://marvl.infotech.monash.edu/~dwyer/papers/itsawrap.pdf

The signed distance is evaluated directly on the 2D Minkowski difference:
outside-origin distance equals GJK separation; inside-origin boundary distance
equals EPA penetration depth. No iterative GJK/EPA dependency is required.
The paper does not specify a unique toroidal hull lift. Here each cluster is
cut at its largest empty gap in each lattice coordinate, giving a minimum-span
axis-aligned lift, then centered in the central tile. Skew bases are reduced
first. The minimum signed distance over the surrounding 3x3 images is used.
"""
import numpy as np
from scipy.spatial import ConvexHull, QhullError

from modules.geometry import gauss_reduce_basis


def sbm_labels(n, k):
    """Recover this project's consecutive, roughly equal planted SBM blocks."""
    if not np.isfinite(k) or int(k) != k or not 2 <= int(k) <= n:
        raise ValueError("SBM k must be an integer between 2 and n")
    k = int(k)
    sizes = np.full(k, n // k, dtype=int)
    sizes[-1] += n % k
    return np.repeat(np.arange(k), sizes)


def _hull(points):
    """Counterclockwise vertices; retain point and segment degeneracies."""
    points = np.unique(np.asarray(points, dtype=float), axis=0)
    if len(points) <= 2:
        return points
    try:
        return points[ConvexHull(points).vertices]
    except QhullError:
        # A collinear hull is a segment. Project onto its longest coordinate.
        axis = np.argmax(np.ptp(points, axis=0))
        return points[[np.argmin(points[:, axis]), np.argmax(points[:, axis])]]


def _signed_origin_distance(hull):
    if len(hull) == 1:
        return float(np.linalg.norm(hull[0]))
    ends = np.roll(hull, -1, axis=0)
    edges = ends - hull
    lengths2 = np.einsum("ij,ij->i", edges, edges)
    t = np.clip(-np.einsum("ij,ij->i", hull, edges) / lengths2, 0., 1.)
    distance = float(np.min(np.linalg.norm(hull + t[:, None]*edges, axis=1)))
    # CCW edge halfplanes contain the origin iff every cross product is >= 0.
    side = edges[:, 0]*(-hull[:, 1]) - edges[:, 1]*(-hull[:, 0])
    inside = len(hull) >= 3 and np.all(side >= 0.)
    return -distance if inside else distance


def signed_hull_distance(a, b):
    """Positive separation, negative minimum translation to separate, zero contact."""
    difference = _hull((a[:, None, :] - b[None, :, :]).reshape(-1, 2))
    return _signed_origin_distance(difference)


def _compact_lift(points):
    points = points % 1.
    lifted = np.empty_like(points)
    for axis in range(2):
        values = np.sort(points[:, axis])
        gaps = np.diff(np.append(values, values[0] + 1.))
        cut = values[(np.argmax(gaps) + 1) % len(values)]
        lifted[:, axis] = (points[:, axis] - cut) % 1. + cut
    # Choose the representative whose bounding-box center is in [0,1)^2.
    lifted -= np.floor((lifted.min(axis=0) + lifted.max(axis=0)) * .5)
    return lifted


def cluster_distance(X, labels, *, periodic=True, alpha=1., r0=1., r1=1.,
                     theta=np.pi/2):
    """Average minimum signed convex-hull distance over unordered cluster pairs.

    Uses physical distances at the saved scale; higher is better. Coordinates
    and node-order labels must correspond. Euclidean layouts use ordinary hulls.
    Degenerate hulls are supported. Requires at least two nonempty clusters.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    if X.ndim != 2 or X.shape[1] != 2 or labels.shape != (len(X),):
        raise ValueError("Cluster labels must match (n, 2) coordinates")
    if not np.isfinite(X).all() or not np.isfinite([alpha, r0, r1, theta]).all():
        raise ValueError("Cluster coordinates and geometry must be finite")
    if min(alpha, r0, r1) <= 0 or not 0 < theta < np.pi:
        raise ValueError("Invalid cluster geometry")
    groups = np.unique(labels)
    if len(groups) < 2:
        raise ValueError("Cluster distance requires at least two clusters")
    basis = np.eye(2) * alpha
    if periodic:
        if np.isclose(theta, np.pi/2):
            basis = alpha * np.diag([r0, r1])
        else:
            x, y, scale, transform = gauss_reduce_basis(
                r1/r0*np.cos(theta), r1/r0*np.sin(theta))
            X = X @ transform.T
            basis = alpha*r0*scale*np.array([[1., x], [0., y]])
        hulls = [_hull(_compact_lift(X[labels == group]) @ basis.T) for group in groups]
        shifts = np.array([(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)]) @ basis.T
    else:
        hulls = [_hull(alpha*X[labels == group]) for group in groups]
        shifts = np.zeros((1, 2))
    distances = []
    for i, a in enumerate(hulls):
        for b in hulls[i+1:]:
            # Build the Minkowski hull once, then translate it for each image.
            difference = _hull((a[:, None, :] - b[None, :, :]).reshape(-1, 2))
            distances.append(min(_signed_origin_distance(difference - shift) for shift in shifts))
    return float(np.mean(distances))
