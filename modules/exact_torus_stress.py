from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional
import warnings

import numpy as np
import numba
from scipy.optimize import minimize

from .geometry import raw_from_height, xy_to_lengths_angle


EmbeddingMode = Literal["torus", "euclidean"]
ShapeMode = Literal["fixed", "rectangular", "rhombic", "parallelogram"]

_SHAPE_MODE_TO_INT = {"fixed": 0, "rectangular": 1, "rhombic": 2, "parallelogram": 3}


@dataclass
class TorusPolishResult:
    """Result of exact profiled L-BFGS polishing of an existing layout."""

    y: np.ndarray # coordinates
    s: float # aspect-ratio parameter
    R: float # aspect ratio exp(s)
    alpha: float
    stress_sq: float
    F: float
    n_iter: int
    x: float = 0.0
    y_shape: float = 1.0
    r0: float = 1.0
    r1: float = 1.0
    theta: float = float(np.pi / 2)
    convergence_info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TorusPairData:
    """Precomputed finite weighted pairs for dense all-pairs stress."""

    n: int
    i: np.ndarray
    j: np.ndarray
    d: np.ndarray # target-distance
    w: np.ndarray # weights, always 1
    Dsum: float


def _normalize_layout(y: np.ndarray, embedding: EmbeddingMode) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if embedding == "torus":
        return y % 1.0
    if embedding == "euclidean":
        return y.copy()
    raise ValueError(f"Unsupported embedding mode: {embedding!r}.")


def _resolve_optimize_aspect(
    embedding: EmbeddingMode,
    optimize_aspect: Optional[bool],
    warn: bool = False,
) -> bool:
    if optimize_aspect is None:
        return embedding == "torus"
    flag = bool(optimize_aspect)
    if warn and embedding == "euclidean" and flag:
        warnings.warn(
            "optimize_aspect=True with embedding='euclidean' enables anisotropic rectangular "
            "Euclidean stress. Use optimize_aspect=False for ordinary isotropic Euclidean stress.",
            stacklevel=3,
        )
    return flag


def _shape_mode_from_legacy(embedding: EmbeddingMode, optimize_aspect: Optional[bool]) -> ShapeMode:
    return "rectangular" if _resolve_optimize_aspect(embedding, optimize_aspect, warn=False) else "fixed"


def prepare_pair_data(distances: np.ndarray, weights: Optional[np.ndarray] = None) -> TorusPairData:
    """Precompute all finite positive weighted pairs from dense distances."""
    D = np.asarray(distances, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"distances must be a square matrix, got {D.shape}.")
    if not np.allclose(np.diag(D), 0.0, atol=1e-12):
        raise ValueError("distance matrix diagonal must be zero.")
    n = D.shape[0]
    i_all, j_all = np.triu_indices(n, 1)
    d_all = D[i_all, j_all]
    mask = np.isfinite(d_all) & (d_all > 0.0)
    if weights is None:
        i = i_all[mask]
        j = j_all[mask]
        d = d_all[mask].astype(np.float64, copy=True)
        w = np.ones_like(d)
    else:
        W = np.asarray(weights, dtype=np.float64)
        if W.shape != (n, n):
            raise ValueError(f"weights must have shape {(n, n)}, got {W.shape}.")
        if np.any(W < 0.0):
            raise ValueError("weights must be nonnegative.")
        w_all = W[i_all, j_all]
        mask = mask & (w_all > 0.0)
        i = i_all[mask]
        j = j_all[mask]
        d = d_all[mask].astype(np.float64, copy=True)
        w = w_all[mask].astype(np.float64, copy=True)
    Dsum = float(np.sum(w * d * d))
    if Dsum <= 0.0:
        raise ValueError("No positive finite weighted vertex pairs are available.")
    return TorusPairData(n=n, i=i.astype(np.intp), j=j.astype(np.intp), d=d, w=w, Dsum=Dsum)


def _shape_info(
    shape_mode: ShapeMode,
    shape_vars: np.ndarray,
    *,
    r0: float,
    r1: float,
    theta: float,
) -> tuple[float, float, float, float]:
    if shape_mode == "fixed":
        _, x, y = _fixed_shape_xy(r0, r1, theta)
        return 0.0, 1.0, x, y
    if shape_mode == "rectangular":
        s = float(shape_vars[0])
        R = float(np.exp(s))
        return s, R, 0.0, 1.0 / R
    if shape_mode == "rhombic":
        x = float(np.clip(shape_vars[0], -0.5, 0.5))
        return 0.0, 1.0, x, float(np.sqrt(max(0.0, 1.0 - x * x)))
    if shape_mode == "parallelogram":
        x = float(np.clip(shape_vars[0], -0.5, 0.5))
        y_raw = float(shape_vars[1])
        base = float(np.sqrt(max(0.0, 1.0 - x * x)))
        softplus = float(np.logaddexp(0.0, y_raw))
        y = base + softplus
        return 0.0, 1.0, x, y
    raise ValueError(f"Unsupported shape_mode: {shape_mode!r}.")


@numba.njit(cache=True, fastmath=True)
def _metric_from_shape_njit(shape_mode, shape_vars, r0, r1, theta):
    if shape_mode == 0:
        c = np.cos(theta)
        return r0 * r0, r0 * r1 * c, r1 * r1, 0.0, 1.0, 0.0, 1.0
    if shape_mode == 1:
        s = shape_vars[0]
        R = np.exp(s)
        return R, 0.0, 1.0 / R, s, R, 0.0, 1.0 / R
    if shape_mode == 2:
        x = min(0.5, max(-0.5, shape_vars[0]))
        y = np.sqrt(max(0.0, 1.0 - x * x))
        return 1.0, x, 1.0, 0.0, 1.0, x, y

    x = min(0.5, max(-0.5, shape_vars[0]))
    y_raw = shape_vars[1]
    base = np.sqrt(max(0.0, 1.0 - x * x))
    softplus = max(y_raw, 0.0) + np.log1p(np.exp(-abs(y_raw)))
    y = base + softplus
    return 1.0, x, x * x + y * y, 0.0, 1.0, x, y


@numba.njit(cache=True, fastmath=True)
def _wrap_delta_scalar(t):
    return t - np.floor(t + 0.5)


@numba.njit(cache=True, fastmath=True)
def _selected_pair_terms_njit(y, i, j, g00, g01, g11, embedding):
    u = y[i, 0] - y[j, 0]
    v = y[i, 1] - y[j, 1]
    if embedding == 0:
        u = _wrap_delta_scalar(u)
        v = _wrap_delta_scalar(v)
        if abs(g01) > 1e-15:
            u_alt = u - 1.0 if u >= 0.0 else u + 1.0
            v_alt = v - 1.0 if v >= 0.0 else v + 1.0
            best_u = u
            best_v = v
            best_h = g00 * u * u + 2.0 * g01 * u * v + g11 * v * v

            h = g00 * u_alt * u_alt + 2.0 * g01 * u_alt * v + g11 * v * v
            if h < best_h:
                best_h = h
                best_u = u_alt
                best_v = v

            h = g00 * u * u + 2.0 * g01 * u * v_alt + g11 * v_alt * v_alt
            if h < best_h:
                best_h = h
                best_u = u
                best_v = v_alt

            h = g00 * u_alt * u_alt + 2.0 * g01 * u_alt * v_alt + g11 * v_alt * v_alt
            if h < best_h:
                best_u = u_alt
                best_v = v_alt

            u = best_u
            v = best_v

    h = g00 * u * u + 2.0 * g01 * u * v + g11 * v * v
    if h < 0.0:
        h = 0.0
    return u, v, h


@numba.njit(cache=True, fastmath=True)
def _evaluate_profiled_njit(
    y,
    pair_i,
    pair_j,
    d,
    w,
    Dsum,
    shape_vars,
    shape_mode,
    embedding,
    r0,
    r1,
    theta,
    eps_q,
):
    n = y.shape[0]
    g00, g01, g11, _s, R, x_shape, y_shape = _metric_from_shape_njit(
        shape_mode, shape_vars, r0, r1, theta
    )

    A = 0.0
    B = 0.0
    for k in range(d.shape[0]):
        u, v, h = _selected_pair_terms_njit(y, pair_i[k], pair_j[k], g00, g01, g11, embedding)
        q = np.sqrt(h + eps_q * eps_q)
        A += w[k] * d[k] * q
        B += w[k] * h

    F = -2.0 * np.log(A) + np.log(B)
    alpha = A / B
    stress_sq = 1.0 - A * A / (B * Dsum)

    grad_y = np.zeros((n, 2), dtype=np.float64)
    grad_shape = np.zeros(2, dtype=np.float64)
    for k in range(d.shape[0]):
        i = pair_i[k]
        j = pair_j[k]
        u, v, h = _selected_pair_terms_njit(y, i, j, g00, g01, g11, embedding)
        q = np.sqrt(h + eps_q * eps_q)
        coeff = w[k] * (1.0 / B - d[k] / (A * q))

        dh0 = 2.0 * (g00 * u + g01 * v)
        dh1 = 2.0 * (g01 * u + g11 * v)
        g0 = coeff * dh0
        g1 = coeff * dh1
        grad_y[i, 0] += g0
        grad_y[i, 1] += g1
        grad_y[j, 0] -= g0
        grad_y[j, 1] -= g1

        if shape_mode == 1:
            grad_shape[0] += coeff * (R * u * u - (1.0 / R) * v * v)
        elif shape_mode == 2:
            grad_shape[0] += coeff * (2.0 * u * v)
        elif shape_mode == 3:
            gx = 2.0 * v * (u + x_shape * v)
            gy = 2.0 * y_shape * v * v
            base = np.sqrt(max(0.0, 1.0 - x_shape * x_shape))
            dx_base = -x_shape / base if base > 1e-12 else 0.0
            y_raw = shape_vars[1]
            sig = 1.0 / (1.0 + np.exp(-y_raw))
            grad_shape[0] += coeff * (gx + gy * dx_base)
            grad_shape[1] += coeff * (gy * sig)

    return F, grad_y, grad_shape, alpha, stress_sq, A, B


@numba.njit(cache=True, fastmath=True)
def _evaluate_fixed_alpha_stress_njit(
    y,
    pair_i,
    pair_j,
    d,
    w,
    Dsum,
    alpha,
    embedding,
    r0,
    r1,
    theta,
    eps_q,
):
    c = np.cos(theta)
    g00 = r0 * r0
    g01 = r0 * r1 * c
    g11 = r1 * r1

    num = 0.0
    for k in range(d.shape[0]):
        _u, _v, h = _selected_pair_terms_njit(y, pair_i[k], pair_j[k], g00, g01, g11, embedding)
        q = np.sqrt(h + eps_q * eps_q)
        residual = alpha * q - d[k]
        num += w[k] * residual * residual
    return np.sqrt(num / Dsum)


def _evaluate_general_objective_and_gradient(
    y: np.ndarray,
    shape_vars: np.ndarray,
    pair_data: TorusPairData,
    *,
    eps_q: float,
    embedding: EmbeddingMode,
    shape_mode: ShapeMode,
    r0: float,
    r1: float,
    theta: float,
) -> tuple[float, np.ndarray, np.ndarray, float, float, float, float, dict[str, float]]:
    y = _normalize_layout(y, embedding)
    if y.shape != (pair_data.n, 2):
        raise ValueError(f"y must have shape {(pair_data.n, 2)}, got {y.shape}.")
    if embedding == "euclidean" and shape_mode not in ("fixed", "rectangular"):
        raise ValueError("Euclidean polishing supports only fixed or rectangular shape modes.")

    shape_vars = np.asarray(shape_vars, dtype=np.float64)
    shape_mode_int = _SHAPE_MODE_TO_INT[shape_mode]
    embedding_int = 0 if embedding == "torus" else 1
    F, grad_y, grad_shape_full, alpha, stress_sq, A, B = _evaluate_profiled_njit(
        y,
        pair_data.i,
        pair_data.j,
        pair_data.d,
        pair_data.w,
        pair_data.Dsum,
        shape_vars,
        shape_mode_int,
        embedding_int,
        float(r0),
        float(r1),
        float(theta),
        float(eps_q),
    )
    s, R, x_shape, y_shape = _shape_info(shape_mode, shape_vars, r0=r0, r1=r1, theta=theta)
    if A <= 0.0 or B <= 0.0:
        raise FloatingPointError("Degenerate embedding: A and B must be positive.")
    grad_shape = grad_shape_full[: shape_vars.size].copy()
    info = {"s": float(s), "R": float(R), "x": float(x_shape), "y": float(y_shape)}
    return float(F), grad_y, grad_shape, float(alpha), float(stress_sq), float(A), float(B), info


def evaluate_profiled_objective_and_gradient(
    y: np.ndarray,
    s: float,
    pair_data: TorusPairData,
    eps_q: float = 1e-12,
    embedding: EmbeddingMode = "torus",
    optimize_aspect: Optional[bool] = None,
) -> tuple[float, np.ndarray, float, float, float, float, float]:
    """Evaluate the legacy profiled rectangular objective and gradients."""
    shape_mode = _shape_mode_from_legacy(embedding, optimize_aspect)
    shape_vars = np.array([float(s)], dtype=np.float64) if shape_mode == "rectangular" else np.zeros(0)
    F, grad_y, grad_shape, alpha, stress_sq, A, B, _info = _evaluate_general_objective_and_gradient(
        y,
        shape_vars,
        pair_data,
        eps_q=eps_q,
        embedding=embedding,
        shape_mode=shape_mode,
        r0=1.0,
        r1=1.0,
        theta=float(np.pi / 2),
    )
    grad_s = float(grad_shape[0]) if grad_shape.size else 0.0
    return F, grad_y, grad_s, alpha, stress_sq, A, B


def evaluate_torus_stress(
    y: np.ndarray,
    distances: np.ndarray,
    *,
    alpha: float = 1.0,
    r0: float = 1.0,
    r1: float = 1.0,
    theta: float = float(np.pi / 2),
    weights: Optional[np.ndarray] = None,
    embedding: EmbeddingMode = "torus",
    eps_q: float = 0.0,
) -> float:
    """Evaluate fixed-alpha normalized stress with the fast pairwise torus kernel."""
    yy = _normalize_layout(y, embedding)
    pair_data = prepare_pair_data(distances, weights)
    if yy.shape != (pair_data.n, 2):
        raise ValueError(f"y must have shape {(pair_data.n, 2)}, got {yy.shape}.")
    if embedding == "euclidean" and abs(np.cos(theta)) > 1e-12:
        raise ValueError("Non-rectangular fixed-alpha stress is only supported for torus embeddings.")
    embedding_int = 0 if embedding == "torus" else 1
    return float(
        _evaluate_fixed_alpha_stress_njit(
            yy,
            pair_data.i,
            pair_data.j,
            pair_data.d,
            pair_data.w,
            pair_data.Dsum,
            float(alpha),
            embedding_int,
            float(r0),
            float(r1),
            float(theta),
            float(eps_q),
        )
    )


def _pack(y: np.ndarray, shape_vars: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(y[1:], dtype=np.float64).ravel(), np.asarray(shape_vars, dtype=np.float64)])


def _unpack(
    x: np.ndarray,
    n: int,
    shape_size: int,
    embedding: EmbeddingMode,
) -> tuple[np.ndarray, np.ndarray]:
    y_size = max(0, 2 * (n - 1))
    y = np.zeros((n, 2), dtype=np.float64)
    if n > 1:
        y[1:] = np.asarray(x[:y_size], dtype=np.float64).reshape(n - 1, 2)
    y = _normalize_layout(y, embedding)
    shape_vars = np.asarray(x[y_size : y_size + shape_size], dtype=np.float64)
    return y, shape_vars


def _initial_shape_vars(
    shape_mode: ShapeMode,
    *,
    s0: float,
    x0: float,
    y0: float,
    s_bounds: tuple[float, float],
) -> np.ndarray:
    if shape_mode == "fixed":
        return np.zeros(0, dtype=np.float64)
    if shape_mode == "rectangular":
        return np.array([float(np.clip(s0, s_bounds[0], s_bounds[1]))], dtype=np.float64)
    if shape_mode == "rhombic":
        return np.array([float(np.clip(x0, -0.5, 0.5))], dtype=np.float64)
    if shape_mode == "parallelogram":
        x = float(np.clip(x0, -0.5, 0.5))
        y_raw = raw_from_height(x, max(float(y0), np.sqrt(max(0.0, 1.0 - x * x)) + 1e-12))
        return np.array([x, y_raw], dtype=np.float64)
    raise ValueError(f"Unsupported shape_mode: {shape_mode!r}.")


def _shape_bounds(shape_mode: ShapeMode, s_bounds: tuple[float, float]) -> list[tuple[float | None, float | None]]:
    if shape_mode == "fixed":
        return []
    if shape_mode == "rectangular":
        return [s_bounds]
    if shape_mode == "rhombic":
        return [(-0.5, 0.5)]
    if shape_mode == "parallelogram":
        return [(-0.5, 0.5), (None, None)]
    raise ValueError(f"Unsupported shape_mode: {shape_mode!r}.")


def _result_shape(
    shape_mode: ShapeMode,
    shape_vars: np.ndarray,
    *,
    r0: float,
    r1: float,
    theta: float,
) -> tuple[float, float, float, float, float, float, float]:
    s, R, x, y = _shape_info(shape_mode, shape_vars, r0=r0, r1=r1, theta=theta)
    if shape_mode == "fixed":
        _, x_fixed, y_fixed = _fixed_shape_xy(r0, r1, theta)
        return 0.0, 1.0, float(x_fixed), float(y_fixed), float(r0), float(r1), float(theta)
    if shape_mode == "rectangular":
        rr0 = float(np.sqrt(R))
        rr1 = float(1.0 / np.sqrt(R))
        return float(s), float(R), 0.0, rr1 / rr0, rr0, rr1, float(np.pi / 2)
    if shape_mode in ("rhombic", "parallelogram"):
        rr0, rr1, th = xy_to_lengths_angle(1.0, x, y)
        return float(s), float(R), float(x), float(y), float(rr0), float(rr1), float(th)
    raise ValueError(f"Unsupported shape_mode: {shape_mode!r}.")


def _fixed_shape_xy(r0: float, r1: float, theta: float) -> tuple[float, float, float]:
    if r0 <= 0.0:
        raise ValueError("r0 must be positive.")
    rho = r1 / r0
    return float(r0), float(rho * np.cos(theta)), float(rho * np.sin(theta))


def polish_torus_layout(
    y0: np.ndarray,
    distances: np.ndarray,
    s0: float = 0.0,
    weights: Optional[np.ndarray] = None,
    embedding: EmbeddingMode = "torus",
    optimize_aspect: Optional[bool] = None,
    shape_mode: Optional[ShapeMode] = None,
    r0: float = 1.0,
    r1: float = 1.0,
    theta: float = float(np.pi / 2),
    x0: float = 0.0,
    y0_shape: float = 1.0,
    max_iter: int = 50,
    eps_q: float = 1e-12,
    s_bounds: tuple[float, float] = (-10.0, 10.0),
    ftol: float = 1e-9,
    gtol: float = 1e-7,
    maxls: int = 80,
) -> TorusPolishResult:
    """Polish an existing layout with exact profiled L-BFGS.

    The global scale alpha is profiled out exactly. ``shape_mode`` controls which
    torus-shape variables are optimized: no shape, rectangular aspect, rhombic
    angle, or free reduced parallelogram.
    """
    y0 = np.asarray(y0, dtype=np.float64)
    if y0.ndim != 2 or y0.shape[1] != 2:
        raise ValueError(f"y0 must have shape (N,2), got {y0.shape}.")
    if shape_mode is None:
        optimize_aspect = _resolve_optimize_aspect(embedding, optimize_aspect, warn=True)
        shape_mode = "rectangular" if optimize_aspect else "fixed"
    elif embedding == "euclidean" and optimize_aspect:
        _resolve_optimize_aspect(embedding, optimize_aspect, warn=True)

    n = y0.shape[0]
    y = _normalize_layout(y0, embedding)
    if n:
        y = y - y[0]
        y = _normalize_layout(y, embedding)
        y[0] = 0.0
    pair_data = prepare_pair_data(distances, weights)
    if pair_data.n != n:
        raise ValueError("distance matrix size must match y0.")

    shape_vars0 = _initial_shape_vars(shape_mode, s0=s0, x0=x0, y0=y0_shape, s_bounds=s_bounds)
    shape_size = int(shape_vars0.size)
    bounds = [(None, None)] * (2 * max(0, n - 1)) + _shape_bounds(shape_mode, s_bounds)
    history: list[dict[str, float]] = []

    def fun_and_grad(x: np.ndarray):
        yy, shape_vars = _unpack(x, n, shape_size, embedding)
        F, grad_y, grad_shape, alpha, stress_sq, _A, _B, info = _evaluate_general_objective_and_gradient(
            yy,
            shape_vars,
            pair_data,
            eps_q=eps_q,
            embedding=embedding,
            shape_mode=shape_mode,
            r0=float(r0),
            r1=float(r1),
            theta=float(theta),
        )
        grad_y[0] = 0.0
        history.append({"F": F, "alpha": alpha, "stress_sq": stress_sq, **info})
        return F, _pack(grad_y, grad_shape)

    res = minimize(
        fun_and_grad,
        _pack(y, shape_vars0),
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(max_iter), "ftol": ftol, "gtol": gtol, "maxls": int(maxls)},
    )
    y_final, shape_vars_final = _unpack(res.x, n, shape_size, embedding)
    F, grad_y, grad_shape, alpha, stress_sq, A, B, _info = _evaluate_general_objective_and_gradient(
        y_final,
        shape_vars_final,
        pair_data,
        eps_q=eps_q,
        embedding=embedding,
        shape_mode=shape_mode,
        r0=float(r0),
        r1=float(r1),
        theta=float(theta),
    )
    grad_y[0] = 0.0
    grad_inf = float(np.max(np.abs(grad_y[1:]))) if n > 1 else 0.0
    shape_grad_inf = float(np.max(np.abs(grad_shape))) if grad_shape.size else 0.0
    s, R, x, y_shape, rr0, rr1, th = _result_shape(
        shape_mode, shape_vars_final, r0=float(r0), r1=float(r1), theta=float(theta)
    )
    info = {
        "mode": "lbfgs",
        "embedding": embedding,
        "shape_mode": shape_mode,
        "optimize_aspect": shape_mode == "rectangular",
        "success": bool(res.success or (grad_inf < gtol and shape_grad_inf < gtol)),
        "scipy_success": bool(res.success),
        "message": str(res.message),
        "grad_inf": grad_inf,
        "shape_grad_inf": shape_grad_inf,
        "grad_s_abs": abs(float(grad_shape[0])) if shape_mode == "rectangular" and grad_shape.size else 0.0,
        "A": float(A),
        "B": float(B),
        "history": history,
    }
    return TorusPolishResult(
        y=y_final,
        s=float(s),
        R=float(R),
        alpha=float(alpha),
        stress_sq=float(max(0.0, stress_sq) if stress_sq > -1e-12 else stress_sq),
        F=float(F),
        n_iter=int(res.nit),
        x=float(x),
        y_shape=float(y_shape),
        r0=float(rr0),
        r1=float(rr1),
        theta=float(th),
        convergence_info=info,
    )
