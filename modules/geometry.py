from __future__ import annotations

import numpy as np
import numba

def torus_distance(p1, p2, r0=1.0, r1=1.0, theta=np.pi / 2):
    """
    Shortest 2D flat-torus distance under a rectangular/rhombic fundamental
    domain.

    Parameters are side lengths r0, r1 and angle theta between sides.
    The default is the unit square torus.

    The 4-candidate image search is exact for the supported cases used by the projector:
    rectangular tori, or equal-side rhombic tori with theta in [pi/3, 2*pi/3].
    """
    if theta != 90.0 and not np.isclose(r0, r1):
        raise ValueError(
            "theta != pi/2 requires equal side lengths: set r0 == r1 "
            "or use theta=pi/2 for a rectangular torus"
        )

    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    cos_theta = np.cos(theta)
    du = p2 - p1
    a = (du[0] + 0.5) - np.floor(du[0] + 0.5) - 0.5
    b = (du[1] + 0.5) - np.floor(du[1] + 0.5) - 0.5
    a1 = a - 1.0 if a >= 0.0 else a + 1.0
    b1 = b - 1.0 if b >= 0.0 else b + 1.0

    r0r0 = r0 * r0
    r1r1 = r1 * r1
    r0r1c = r0 * r1 * cos_theta

    q_best = r0r0 * a * a + 2.0 * r0r1c * a * b + r1r1 * b * b
    du0 = a
    du1 = b
    for u, v in ((a1, b), (a, b1), (a1, b1)):
        q = r0r0 * u * u + 2.0 * r0r1c * u * v + r1r1 * v * v
        if q < q_best:
            q_best = q
            du0 = u
            du1 = v

    r2 = r0r0 * du0 * du0 + 2.0 * r0r1c * du0 * du1 + r1r1 * du1 * du1
    return float(np.sqrt(max(0.0, r2)))


def make_torus_geod(alpha=1.0, r0=1.0, r1=1.0, theta=np.pi / 2):
    """Return geod(p, q) = alpha * torus_distance(p, q, r0, r1, theta)."""
    return lambda p, q: alpha * torus_distance(p, q, r0=r0, r1=r1, theta=theta)

def euc_distance(p1,p2):
    return np.linalg.norm(p2-p1)


@numba.njit(fastmath=True)
def torus_grad(x, y):
    delta = ((x - y + 0.5) % 1.0 - 0.5)
    result = 0.0
    for i in range(x.shape[0]):
        result += delta[i] ** 2
    norm = np.sqrt(result)
    return norm, delta / (1e-6 + norm)


@numba.njit(fastmath=True)
def euclidean_grad(x, y):
    result = 0.0
    for i in range(x.shape[0]):
        result += (x[i] - y[i]) ** 2
    d = np.sqrt(result)
    grad = (x - y) / (1e-6 + d)
    return d, grad


@numba.njit(fastmath=True, cache=True)
def stress_and_grad_rect_torus(p1, p2, d, alpha, r0, r1, eps=1e-12, theta=np.pi / 2):
    """
    Stress and gradient for flat rectangular/rhombic torus MDS.

    u in [0,1)^2 (parameter space); du is min-image on unit torus wrapped to [-0.5, 0.5).
    Geodesic offset on a torus with side lengths r0, r1 and angle theta:
        r = ||du0 * e0 + du1 * e1||
    with e0=(r0, 0) and e1=(r1*cos(theta), r1*sin(theta)).
    For theta != pi/2 (rhombic), min-image is selected via 4-way check on the same metric. Exact for theta in [pi/3, 2*pi/3].

    Loss: (alpha * r_rect - d)^2

    Returns: (loss, grad_p1, r_rect, grad_r0, grad_r1)
    """
    cos_theta = np.cos(theta)
    du = p2 - p1
    a = (du[0] + 0.5) - np.floor(du[0] + 0.5) - 0.5
    b = (du[1] + 0.5) - np.floor(du[1] + 0.5) - 0.5
    a1 = a - 1.0 if a >= 0.0 else a + 1.0
    b1 = b - 1.0 if b >= 0.0 else b + 1.0
    r0r0 = r0 * r0
    r1r1 = r1 * r1
    r0r1c = r0 * r1 * cos_theta

    q_best = r0r0 * a * a + 2.0 * r0r1c * a * b + r1r1 * b * b
    du0 = a
    du1 = b
    q = r0r0 * a1 * a1 + 2.0 * r0r1c * a1 * b + r1r1 * b * b
    if q < q_best:
        q_best = q
        du0 = a1
        du1 = b
    q = r0r0 * a * a + 2.0 * r0r1c * a * b1 + r1r1 * b1 * b1
    if q < q_best:
        q_best = q
        du0 = a
        du1 = b1
    q = r0r0 * a1 * a1 + 2.0 * r0r1c * a1 * b1 + r1r1 * b1 * b1
    if q < q_best:
        du0 = a1
        du1 = b1

    r2 = r0r0 * du0 * du0 + 2.0 * r0r1c * du0 * du1 + r1r1 * du1 * du1
    if r2 < 0.0:
        r2 = 0.0
    r_rect = np.sqrt(r2) + eps
    norm = alpha * r_rect
    diff = norm - d

    scale = -2.0 * diff * alpha / r_rect
    g0 = scale * (r0r0 * du0 + r0r1c * du1)
    g1 = scale * (r1r1 * du1 + r0r1c * du0)

    gr0 = 2.0 * diff * alpha * (r0 * du0 * du0 + r1 * cos_theta * du0 * du1) / r_rect
    gr1 = 2.0 * diff * alpha * (r1 * du1 * du1 + r0 * cos_theta * du0 * du1) / r_rect

    return diff * diff, np.array((g0, g1), dtype=p1.dtype), r_rect, gr0, gr1


@numba.njit(fastmath=True, cache=True)
def grad_rect_torus(p1, p2, d, alpha, r0, r1, eps=1e-12, theta=np.pi / 2):
    """
    Same as stress_and_grad_rect_torus but returns only the gradient.

    Different return pattern: no stress and g0,g1 are returned directly and not as tuple
    """
    cos_theta = np.cos(theta)
    du = p2 - p1
    a = (du[0] + 0.5) - np.floor(du[0] + 0.5) - 0.5
    b = (du[1] + 0.5) - np.floor(du[1] + 0.5) - 0.5
    a1 = a - 1.0 if a >= 0.0 else a + 1.0
    b1 = b - 1.0 if b >= 0.0 else b + 1.0
    r0r0 = r0 * r0
    r1r1 = r1 * r1
    r0r1c = r0 * r1 * cos_theta

    q_best = r0r0 * a * a + 2.0 * r0r1c * a * b + r1r1 * b * b
    du0 = a
    du1 = b
    q = r0r0 * a1 * a1 + 2.0 * r0r1c * a1 * b + r1r1 * b * b
    if q < q_best:
        q_best = q
        du0 = a1
        du1 = b
    q = r0r0 * a * a + 2.0 * r0r1c * a * b1 + r1r1 * b1 * b1
    if q < q_best:
        q_best = q
        du0 = a
        du1 = b1
    q = r0r0 * a1 * a1 + 2.0 * r0r1c * a1 * b1 + r1r1 * b1 * b1
    if q < q_best:
        du0 = a1
        du1 = b1

    r2 = r0r0 * du0 * du0 + 2.0 * r0r1c * du0 * du1 + r1r1 * du1 * du1
    if r2 < 0.0:
        r2 = 0.0
    r_rect = np.sqrt(r2) + eps
    diff = alpha * r_rect - d

    scale = -2.0 * diff * alpha / r_rect
    g0 = scale * (r0r0 * du0 + r0r1c * du1)
    g1 = scale * (r1r1 * du1 + r0r1c * du0)

    gr0 = 2.0 * diff * alpha * (r0 * du0 * du0 + r1 * cos_theta * du0 * du1) / r_rect
    gr1 = 2.0 * diff * alpha * (r1 * du1 * du1 + r0 * cos_theta * du0 * du1) / r_rect

    return g0, g1, r_rect, gr0, gr1


def torus_delta(p, q):
    """Shortest displacement from p to q on unit square torus (per-coordinate in [-0.5, 0.5))."""
    return ((q - p + 0.5) % 1.0) - 0.5


def torus_edge_segments(p, q, eps=1e-12):
    """
    Return list of (a_plot, b_plot) segments for the shortest torus geodesic between p and q,
    split at boundary crossings for plotting in the fundamental domain [0,1]x[0,1].
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    d = torus_delta(p, q)
    r = p + d

    ts = [0.0, 1.0]
    for k in (0, 1):
        dk = d[k]
        if abs(dk) < eps:
            continue
        if dk > 0 and r[k] > 1.0 + eps:
            b = 1.0
        elif dk < 0 and r[k] < 0.0 - eps:
            b = 0.0
        else:
            continue
        t = (b - p[k]) / dk
        if eps < t < 1.0 - eps:
            ts.append(float(t))

    ts = sorted(set(ts))
    segs = []
    for t0, t1 in zip(ts[:-1], ts[1:]):
        a = p + t0 * d
        b = p + t1 * d
        mid = 0.5 * (a + b)
        tile = np.floor(mid)
        a_plot = np.clip(a - tile, 0.0, 1.0)
        b_plot = np.clip(b - tile, 0.0, 1.0)
        segs.append((a_plot, b_plot))

    return segs
