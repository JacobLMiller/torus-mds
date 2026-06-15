from __future__ import annotations

import numpy as np
import numba

def torus_distance(p1, p2, r0=1.0, r1=1.0, theta=np.pi / 2):
    """
    Legacy shortest 2D flat-torus distance under a rectangular/rhombic fundamental
    domain.

    Parameters are side lengths r0, r1 and angle theta between sides.
    The default is the unit square torus.

    The 4-candidate image search is exact for the supported cases used by the projector:
    rectangular tori, or equal-side rhombic tori with theta in [pi/3, 2*pi/3].
    For arbitrary parallelograms use ``parallelogram_distance``.
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
    """
    Return geod(p, q), the alpha-scaled flat-torus distance.

    Rectangular tori (theta == pi/2) use the fast legacy ``torus_distance`` path
    (orthogonal basis, no Gauss reduction needed); non-rectangular tori use
    ``parallelogram_distance`` (Gauss-reduced only when necessary).
    """
    if np.isclose(theta, np.pi / 2):
        return lambda p, q: alpha * torus_distance(p, q, r0=r0, r1=r1, theta=theta)
    # alpha * r0 is the overall scale of the (1, 0)-canonical shape basis.
    _, x, y = lengths_angle_to_xy(r0, r1, theta)
    return lambda p, q: parallelogram_distance(p, q, alpha=alpha * r0, x=x, y=y)


# ---------------------------------------------------------------------------
# Parallelogram-torus geometry: (alpha, x, y) parametrization
#
# The flat torus is R^2 / L with basis  b0 = alpha*(1, 0),  b1 = alpha*(x, y),
# y > 0.  alpha is the overall scale (= |b0|), x is the shear and y the height
# of the second basis vector relative to the first.  This is the parametrization
# used for *learning* the geometry; (r0, r1, theta) (side lengths + angle) is the
# equivalent, more intuitive view used for reporting and visualization.
# ---------------------------------------------------------------------------

def lengths_angle_to_xy(r0, r1, theta):
    """
    Convert side lengths + angle (r0, r1, theta) to scale/shear/height (alpha, x, y).

        alpha = r0,  x = (r1 / r0) * cos(theta),  y = (r1 / r0) * sin(theta)

    Here r0 = |b0|, r1 = |b1| and theta is the angle between the two basis
    vectors.  Note x = 0 (theta = pi/2) makes y = r1 / r0 the aspect ratio.
    """
    rho = r1 / r0
    return float(r0), float(rho * np.cos(theta)), float(rho * np.sin(theta))


def xy_to_lengths_angle(alpha, x, y):
    """
    Convert scale/shear/height (alpha, x, y) to side lengths + angle (r0, r1, theta).

        r0 = alpha,  r1 = alpha * sqrt(x**2 + y**2),  theta = atan2(y, x)

    theta is the angle of b1 = (x, y) from b0 = (1, 0); atan2 keeps it exact and
    in (0, pi) for y > 0 (no arccos domain issues).
    """
    r0 = float(alpha)
    r1 = float(alpha * np.hypot(x, y))
    theta = float(np.arctan2(y, x))
    return r0, r1, theta


def softplus(z):
    """Numerically stable softplus, log(1 + exp(z))."""
    return np.logaddexp(0.0, z)


def height_from_raw(x, y_raw):
    """
    Learnable height parametrization:  y = sqrt(max(0, 1 - x**2)) + softplus(y_raw).

    Guarantees x**2 + y**2 >= 1, so together with |x| <= 1/2 (kept by the shear
    "tidy" step) the basis is always Gauss-reduced and the 4-candidate min-image
    search is exact. y_raw -> -inf recovers the rhombic boundary y = sqrt(1 - x**2).
    """
    base = np.sqrt(max(0.0, 1.0 - x * x))
    return float(base + softplus(y_raw))


def raw_from_height(x, y):
    """Inverse of height_from_raw: recover y_raw such that height_from_raw(x, y_raw) == y."""
    base = np.sqrt(max(0.0, 1.0 - x * x))
    sp = max(y - base, 1e-12)            # softplus output (> 0)
    return float(np.log(np.expm1(sp)))   # inverse softplus: log(exp(sp) - 1)


def gauss_reduce_basis(x, y):
    """
    Gauss (Lagrange) reduction of the 2D lattice basis {(1, 0), (x, y)}.

    The overall scale alpha is irrelevant to reduction, so it is dropped.  Returns

        (x_red, y_red, scale, U_inv)

    where {(1, 0), (x_red, y_red)} (in canonical form, first vector along the
    x-axis) generates the *same* lattice rescaled by ``scale`` (= length of the
    reduced shorter vector, in units of the original |b0| = 1), and ``U_inv`` is
    the integer (det = +/-1) matrix mapping original basis coordinates to reduced
    canonical coordinates:  c_red = U_inv @ c.

    A reduced basis satisfies ``|x_red| <= 1/2`` and ``x_red**2 + y_red**2 >= 1``,
    which is exactly the condition under which the 4-candidate min-image search is
    exact.  Reduction is a unimodular change of basis: it does not change the
    lattice (hence not the torus or any distance).
    """
    b0 = np.array([1.0, 0.0])
    b1 = np.array([float(x), float(y)])
    # U_inv maps original coords -> current-basis coords; start as identity.
    U_inv = np.array([[1.0, 0.0], [0.0, 1.0]])

    for _ in range(1000):  # terminates in O(log) steps; bound guards against NaN
        if b1 @ b1 < b0 @ b0:
            b0, b1 = b1.copy(), b0.copy()
            U_inv = np.array([[0.0, 1.0], [1.0, 0.0]]) @ U_inv
        m = round((b0 @ b1) / (b0 @ b0))
        if m == 0:
            break
        b1 = b1 - m * b0
        U_inv = np.array([[1.0, float(m)], [0.0, 1.0]]) @ U_inv

    # Re-express the reduced basis in canonical form b0 = scale*(1,0),
    # b1 = scale*(x_red, y_red). Distances only depend on the Gram matrix, so the
    # rotation that aligns b0 with the x-axis is irrelevant to coordinates.
    scale = float(np.hypot(b0[0], b0[1]))
    cross = float(b0[0] * b1[1] - b0[1] * b1[0])
    x_red = float((b0 @ b1) / (scale * scale))
    y_red = float(abs(cross) / (scale * scale))
    return x_red, y_red, scale, U_inv


@numba.njit(fastmath=True, cache=True)
def _gauss_reduce_njit(x, y):
    """
    Numba scalar version of ``gauss_reduce_basis`` for the projector tidy step.
    Returns (x_red, y_red, scale, u00, u01, u10, u11), where (u00, u01, u10, u11)
    are the entries of the integer transform U_inv mapping original basis
    coordinates to reduced canonical coordinates: c_red = U_inv @ c.
    """
    b0x, b0y = 1.0, 0.0
    b1x, b1y = x, y
    u00, u01, u10, u11 = 1.0, 0.0, 0.0, 1.0
    for _ in range(1000):
        if (b1x * b1x + b1y * b1y) < (b0x * b0x + b0y * b0y):
            b0x, b1x = b1x, b0x
            b0y, b1y = b1y, b0y
            u00, u10 = u10, u00
            u01, u11 = u11, u01
        m = np.round((b0x * b1x + b0y * b1y) / (b0x * b0x + b0y * b0y))
        if m == 0.0:
            break
        b1x -= m * b0x
        b1y -= m * b0y
        u00 += m * u10
        u01 += m * u11

    scale = np.sqrt(b0x * b0x + b0y * b0y)
    s2 = scale * scale
    cross = b0x * b1y - b0y * b1x
    x_red = (b0x * b1x + b0y * b1y) / s2
    y_red = abs(cross) / s2
    return x_red, y_red, scale, u00, u01, u10, u11


def parallelogram_distance(p1, p2, alpha=1.0, x=0.0, y=1.0):
    """
    Exact shortest flat-torus distance for an arbitrary parallelogram fundamental
    domain with basis b0 = alpha*(1, 0), b1 = alpha*(x, y).

    The basis is Gauss-reduced first (so the 4-candidate search is exact for any
    x, y), then the closest of the 4 surrounding lattice points is taken.
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    du = p2 - p1

    # Only Gauss-reduce when the basis is not already reduced; otherwise the
    # 4-candidate search is exact as-is and we skip the work.
    if abs(x) <= 0.5 and (x * x + y * y) >= 1.0:
        x_red, y_red, scale = float(x), float(y), 1.0
        du_red = du
    else:
        x_red, y_red, scale, U_inv = gauss_reduce_basis(x, y)
        du_red = U_inv @ du  # offset expressed in the reduced basis coordinates

    a = (du_red[0] + 0.5) - np.floor(du_red[0] + 0.5) - 0.5
    b = (du_red[1] + 0.5) - np.floor(du_red[1] + 0.5) - 0.5
    a1 = a - 1.0 if a >= 0.0 else a + 1.0
    b1 = b - 1.0 if b >= 0.0 else b + 1.0

    s = scale * scale
    g00 = s
    g01 = s * x_red
    g11 = s * (x_red * x_red + y_red * y_red)

    r2_best = g00 * a * a + 2.0 * g01 * a * b + g11 * b * b
    for u, v in ((a1, b), (a, b1), (a1, b1)):
        r2 = g00 * u * u + 2.0 * g01 * u * v + g11 * v * v
        if r2 < r2_best:
            r2_best = r2
    return float(alpha * np.sqrt(max(0.0, r2_best)))

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


@numba.njit(fastmath=True, cache=True)
def grad_parallelogram_torus(p1, p2, d, alpha, x, y, eps=1e-12):
    """
    Stress gradient for a flat parallelogram torus in the (alpha, x, y)
    parametrization. Shape metric (alpha factored out as overall scale):

        r_shape^2 = u^2 + 2*x*u*v + (x^2 + y^2)*v^2,   D = alpha * r_shape

    Assumes the basis is Gauss-reduced (|x| <= 1/2 and x^2 + y^2 >= 1), which the
    projector's tidy step maintains, so the plain 4-candidate min-image search is
    exact (no per-call reduction). g0, g1 are returned directly (not as a tuple)
    to avoid per-pair array allocation in the hot SGD loop; the per-pair loss, if
    needed, is (alpha * r_shape - d)**2.

    Returns: (g0, g1, r_shape, grad_x, grad_y)
    """
    du = p2 - p1
    a = (du[0] + 0.5) - np.floor(du[0] + 0.5) - 0.5
    b = (du[1] + 0.5) - np.floor(du[1] + 0.5) - 0.5
    a1 = a - 1.0 if a >= 0.0 else a + 1.0
    b1 = b - 1.0 if b >= 0.0 else b + 1.0

    g11 = x * x + y * y

    q_best = a * a + 2.0 * x * a * b + g11 * b * b
    du0 = a
    du1 = b
    q = a1 * a1 + 2.0 * x * a1 * b + g11 * b * b
    if q < q_best:
        q_best = q
        du0 = a1
        du1 = b
    q = a * a + 2.0 * x * a * b1 + g11 * b1 * b1
    if q < q_best:
        q_best = q
        du0 = a
        du1 = b1
    q = a1 * a1 + 2.0 * x * a1 * b1 + g11 * b1 * b1
    if q < q_best:
        du0 = a1
        du1 = b1

    r2 = du0 * du0 + 2.0 * x * du0 * du1 + g11 * du1 * du1
    if r2 < 0.0:
        r2 = 0.0
    r_shape = np.sqrt(r2) + eps
    diff = alpha * r_shape - d

    scale = -2.0 * diff * alpha / r_shape
    g0 = scale * (du0 + x * du1)
    g1 = scale * (x * du0 + g11 * du1)

    gx = 2.0 * diff * alpha * du1 * (du0 + x * du1) / r_shape
    gy = 2.0 * diff * alpha * y * du1 * du1 / r_shape

    return g0, g1, r_shape, gx, gy


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
