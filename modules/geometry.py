from __future__ import annotations

import numpy as np
import numba


# ---------------------------------------------------------------------------
# Shared min-image primitives.
#
# Every flat-torus distance/gradient here reduces to the same problem: given a
# displacement wrapped to [-1/2, 1/2)^2 and a Gram metric [[g00, g01], [g01, g11]],
# find the closest lattice image and its squared length. The min-image math has
# only two regimes -- orthogonal (g01 == 0, closed form) and non-orthogonal
# (4-candidate search) -- so the wrapping, candidate search, quadratic form and
# position gradient are factored out here and reused by all three parametrizations
# (rect, rhombic, parallelogram), which differ only in how they build the Gram and
# in their shape gradient.
# ---------------------------------------------------------------------------

@numba.njit(fastmath=True, cache=True)
def _wrap_to_half(t):
    """Wrap a torus displacement coordinate to [-1/2, 1/2)."""
    return ((t + 0.5) % 1.0) - 0.5


@numba.njit(fastmath=True, cache=True)
def _gram_quad(u, v, g00, g01, g11):
    """Squared length of (u, v) under the Gram metric [[g00, g01], [g01, g11]]."""
    return g00 * u * u + 2.0 * g01 * u * v + g11 * v * v


@numba.njit(fastmath=True, cache=True)
def _min_image_offset(a, b, g00, g01, g11):
    """
    Closest-image lattice offset for a displacement already wrapped to
    (a, b) in [-1/2, 1/2)^2, under the Gram metric [[g00, g01], [g01, g11]].

    g01 == 0 (orthogonal basis): the metric is separable, so the per-coordinate
    wrap (a, b) is the exact closest image -- closed form, no candidate search.
    Otherwise a 4-candidate search over {a, a-/+1} x {b, b-/+1} is used, which is
    exact for a Gauss-reduced basis (|x| <= 1/2, x^2 + y^2 >= 1) or an equal-side
    rhombus (theta in [pi/3, 2*pi/3]). Returns the selected offset (du0, du1).
    """
    if g01 == 0.0:
        return a, b
    a1 = a - 1.0 if a >= 0.0 else a + 1.0
    b1 = b - 1.0 if b >= 0.0 else b + 1.0
    du0, du1 = a, b
    q_best = _gram_quad(a, b, g00, g01, g11)
    q = _gram_quad(a1, b, g00, g01, g11)
    if q < q_best:
        q_best = q
        du0, du1 = a1, b
    q = _gram_quad(a, b1, g00, g01, g11)
    if q < q_best:
        q_best = q
        du0, du1 = a, b1
    q = _gram_quad(a1, b1, g00, g01, g11)
    if q < q_best:
        du0, du1 = a1, b1
    return du0, du1


@numba.njit(fastmath=True, cache=True)
def _position_grad(du0, du1, g00, g01, g11, scale):
    """
    Gradient of the scaled torus distance wrt the first point's coordinates for
    the selected image offset (du0, du1) under Gram (g00, g01, g11). ``scale``
    folds in -2*diff*alpha/r. Returns (g0, g1).
    """
    g0 = scale * (g00 * du0 + g01 * du1)
    g1 = scale * (g01 * du0 + g11 * du1)
    return g0, g1


@numba.njit(fastmath=True, cache=True)
def _torus_dist_grad_core(p1, p2, d, alpha, g00, g01, g11, eps):
    """
    Geometry-agnostic per-pair step shared by the rect and parallelogram kernels:
    wrap the displacement, pick the closest image under Gram (g00, g01, g11), and
    form the (alpha-free) shape length r, the position gradient and the residual
    diff = alpha*r - d.

    Returns (g0, g1, r, du0, du1, diff). The shape-parameter gradient (side lengths
    or shear/height) is computed separately by the parametrization-specific helper.
    """
    du = p2 - p1
    a = _wrap_to_half(du[0])
    b = _wrap_to_half(du[1])
    du0, du1 = _min_image_offset(a, b, g00, g01, g11)
    r2 = _gram_quad(du0, du1, g00, g01, g11)
    if r2 < 0.0:
        r2 = 0.0
    r = np.sqrt(r2) + eps
    diff = alpha * r - d
    scale = -2.0 * diff * alpha / r
    g0, g1 = _position_grad(du0, du1, g00, g01, g11, scale)
    return g0, g1, r, du0, du1, diff


@numba.njit(fastmath=True, cache=True)
def _shape_grad_rect(du0, du1, diff, r, alpha, r0, r1):
    """Gradient wrt the two side lengths (r0, r1) for the rect (orthogonal) basis."""
    gr0 = 2.0 * diff * alpha * r0 * du0 * du0 / r
    gr1 = 2.0 * diff * alpha * r1 * du1 * du1 / r
    return gr0, gr1


@numba.njit(fastmath=True, cache=True)
def _shape_grad_parallelogram(du0, du1, diff, r, alpha, x, y):
    """Gradient wrt the shear x and height y for the (alpha, x, y) parametrization."""
    gx = 2.0 * diff * alpha * du1 * (du0 + x * du1) / r
    gy = 2.0 * diff * alpha * y * du1 * du1 / r
    return gx, gy


def rect_distance(p1, p2, r0=1.0, r1=1.0):
    """
    Shortest 2D flat-torus distance for a *rectangular* (orthogonal) fundamental
    domain with side lengths r0, r1. Default: the unit square torus.

    The metric is separable, so the per-coordinate wrap is the exact closest image
    -- closed form, no candidate search (the core's g01 == 0 path). For an equal-side
    rhombic torus use ``rhombic_distance``; for a general parallelogram use
    ``parallelogram_distance``.
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    du = p2 - p1
    a = _wrap_to_half(du[0])
    b = _wrap_to_half(du[1])
    g00 = r0 * r0
    g11 = r1 * r1
    du0, du1 = _min_image_offset(a, b, g00, 0.0, g11)
    r2 = _gram_quad(du0, du1, g00, 0.0, g11)
    return float(np.sqrt(max(0.0, r2)))


def torus_distance(p1, p2, r0=1.0, r1=1.0, theta=np.pi / 2):
    """
    Legacy entry point for the rectangular/rhombic flat-torus distance, kept for
    backward compatibility (e.g. ``metrics.estimate_alpha``'s default geod).

    Dispatches on the angle: theta ~ pi/2 uses the orthogonal ``rect_distance``;
    otherwise an equal-side rhombic torus is assumed (r0 must equal r1) and
    ``rhombic_distance`` is used. For r0 != r1 at theta != pi/2 (a general
    parallelogram) use ``parallelogram_distance`` / ``make_torus_geod`` directly.
    """
    if np.isclose(theta, np.pi / 2):
        return rect_distance(p1, p2, r0=r0, r1=r1)
    if not np.isclose(r0, r1):
        raise ValueError(
            f"theta != pi/2 (theta={theta}) requires equal side lengths (r0 == r1) for "
            "rhombic_distance; use parallelogram_distance for a general parallelogram"
        )
    return rhombic_distance(p1, p2, alpha=r0, theta=theta)


def make_torus_geod(alpha=1.0, r0=1.0, r1=1.0, theta=np.pi / 2):
    """
    Return geod(p, q), the alpha-scaled flat-torus distance.

    Rectangular tori (theta == pi/2) use the orthogonal closed-form ``rect_distance``;
    non-rectangular tori use ``parallelogram_distance`` (Gauss-reduced only when
    necessary).
    """
    if np.isclose(theta, np.pi / 2):
        return lambda p, q: alpha * rect_distance(p, q, r0=r0, r1=r1)
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


def rhombic_distance(p1, p2, alpha=1.0, theta=np.pi / 2):
    """
    Exact shortest flat-torus distance for an equal-side *rhombic* fundamental
    domain: basis b0 = alpha*(1, 0), b1 = alpha*(cos theta, sin theta) (both sides
    length alpha). theta is parametrized directly as the angle between the sides.

    For theta in [pi/3, 2*pi/3] the basis is Gauss-reduced (the rhombic locus
    |tau| = 1 is the reduced boundary), so the 4-candidate min-image search is exact.
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    du = p2 - p1
    a = _wrap_to_half(du[0])
    b = _wrap_to_half(du[1])
    x = np.cos(theta)
    # Gram for unit-side rhombus (alpha factored out): (1, x, 1).
    du0, du1 = _min_image_offset(a, b, 1.0, x, 1.0)
    r2 = _gram_quad(du0, du1, 1.0, x, 1.0)
    return float(alpha * np.sqrt(max(0.0, r2)))


def parallelogram_distance(p1, p2, alpha=1.0, x=0.0, y=1.0):
    """
    Exact shortest flat-torus distance for an arbitrary parallelogram fundamental
    domain with basis b0 = alpha*(1, 0), b1 = alpha*(x, y).

    The basis is Gauss-reduced first (so the 4-candidate search is exact for any
    x, y), then the closest of the surrounding lattice points is taken via the
    shared ``_min_image_offset``.
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

    a = _wrap_to_half(du_red[0])
    b = _wrap_to_half(du_red[1])

    s = scale * scale
    g00 = s
    g01 = s * x_red
    g11 = s * (x_red * x_red + y_red * y_red)
    du0, du1 = _min_image_offset(a, b, g00, g01, g11)
    r2 = _gram_quad(du0, du1, g00, g01, g11)
    return float(alpha * np.sqrt(max(0.0, r2)))

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
def rect_stress_and_grad(p1, p2, d, alpha, r0, r1, eps=1e-12):
    """
    Stress and gradient for a flat *rectangular* (orthogonal) torus with side
    lengths r0, r1. The metric is separable, so the per-coordinate wrap is the exact
    closest image (the core's g01 == 0 closed form -- no candidate search).

    Loss: (alpha * r_rect - d)^2.  Returns: (loss, grad_p1, r_rect, grad_r0, grad_r1).

    g00, g01(=0), g11 depend only on (r0, r1) and are constant across an epoch's pair
    loop; rebuilt per call for simplicity but hoistable into the SGD loop (computed
    once per epoch) if this becomes a bottleneck.
    """
    g00 = r0 * r0
    g11 = r1 * r1
    g0, g1, r_rect, du0, du1, diff = _torus_dist_grad_core(p1, p2, d, alpha, g00, 0.0, g11, eps)
    gr0, gr1 = _shape_grad_rect(du0, du1, diff, r_rect, alpha, r0, r1)
    return diff * diff, np.array((g0, g1), dtype=p1.dtype), r_rect, gr0, gr1


@numba.njit(fastmath=True, cache=True)
def rect_grad(p1, p2, d, alpha, r0, r1, eps=1e-12):
    """
    Same as rect_stress_and_grad but returns only the gradient (no stress, and g0,g1
    returned directly rather than as an array, to avoid per-pair allocation in the
    hot SGD loop). Returns: (g0, g1, r_rect, grad_r0, grad_r1).

    Orthogonal only (theta = pi/2): for equal-side rhombic or general parallelogram
    geometry use parallelogram_grad. See rect_stress_and_grad re: the per-call Gram
    build being hoistable.
    """
    g00 = r0 * r0
    g11 = r1 * r1
    g0, g1, r_rect, du0, du1, diff = _torus_dist_grad_core(p1, p2, d, alpha, g00, 0.0, g11, eps)
    gr0, gr1 = _shape_grad_rect(du0, du1, diff, r_rect, alpha, r0, r1)
    return g0, g1, r_rect, gr0, gr1


@numba.njit(fastmath=True, cache=True)
def parallelogram_grad(p1, p2, d, alpha, x, y, eps=1e-12):
    """
    Stress gradient for a flat parallelogram torus in the (alpha, x, y)
    parametrization. Shape metric (alpha factored out as overall scale):

        r_shape^2 = u^2 + 2*x*u*v + (x^2 + y^2)*v^2,   D = alpha * r_shape

    Assumes the basis is Gauss-reduced (|x| <= 1/2 and x^2 + y^2 >= 1), which the
    projector's tidy step maintains, so the plain 4-candidate min-image search is
    exact (no per-call reduction). If x == 0 the orthogonal closed form is taken
    automatically. Also serves the rhombic learn mode, whose single angle gradient
    is the equal-sides combination of (gx, gy), applied to the aggregated sums in the
    projector's geometry update.

    The Gram (1, x, x^2+y^2) depends only on the geometry and is constant across an
    epoch's pair loop; rebuilt per call for simplicity but hoistable into the SGD loop
    if needed. Returns: (g0, g1, r_shape, grad_x, grad_y).
    """
    g11 = x * x + y * y
    g0, g1, r_shape, du0, du1, diff = _torus_dist_grad_core(p1, p2, d, alpha, 1.0, x, g11, eps)
    gx, gy = _shape_grad_parallelogram(du0, du1, diff, r_shape, alpha, x, y)
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
