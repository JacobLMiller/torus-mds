from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal, Tuple

import numpy as np
import numba
from numba.typed import List as NumbaList

from .geometry import (
    rect_grad,
    parallelogram_grad,
    _torus_dist_grad_core,
    _wrap_to_half,
    _min_image_offset,
    _gram_quad,
    _gauss_reduce_njit,
    lengths_angle_to_xy,
    xy_to_lengths_angle,
    torus_grad,
)

DEFAULT_SEQUENCE_CHUNK_EPOCHS = 128


def _check_distance_matrix(D: np.ndarray) -> np.ndarray:
    D = np.asarray(D, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"Distance matrix must be square (N,N). Got {D.shape}.")
    if np.any(D < 0):
        raise ValueError("Distance matrix has negative entries.")
    if not np.allclose(np.diag(D), 0.0, atol=1e-10):
        raise ValueError("Distance matrix diagonal must be ~0.")
    return D


def _robust_affine_to_unit_square(
    X: np.ndarray,
    q: Tuple[float, float] = (0.01, 0.99),
    eps: float = 1e-12,
) -> np.ndarray:
    """Robustly affine-map each axis to [0,1] using quantiles, then clip."""
    lo = np.quantile(X, q[0], axis=0)
    hi = np.quantile(X, q[1], axis=0)
    span = np.maximum(hi - lo, eps)
    Y = (X - lo) / span
    return np.clip(Y, 0.0, 1.0)


def _wrap_unit_square(X: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=np.float64) % 1.0


def _all_unique_pairs(n: int) -> np.ndarray:
    ii, jj = np.triu_indices(n, 1)
    return np.column_stack([ii, jj]).astype(np.int32, copy=False)


def _lbfgs_polish_options(finalize_iters: int, finalize_options: Optional[dict]) -> dict:
    opts = {
        "max_iter": int(finalize_iters),
        "ftol": 1e-9,
        "gtol": 1e-7,
        "maxls": 80,
    }
    if finalize_options:
        opts.update(finalize_options)
    return opts


def _polish_shape_kwargs(mode_int: int, r0: float, r1: float, x: float, y: float, theta: float) -> dict:
    if mode_int in (3, 4):
        return {
            "shape_mode": "rectangular",
            "s0": float(np.log(max(r0, 1e-300) / max(r1, 1e-300))),
        }
    if mode_int == 5:
        return {"shape_mode": "rhombic", "x0": float(x), "y0_shape": float(y)}
    if mode_int == 6:
        return {"shape_mode": "parallelogram", "x0": float(x), "y0_shape": float(y)}
    return {
        "shape_mode": "fixed",
        "r0": float(r0),
        "r1": float(r1),
        "theta": float(theta),
    }


def _build_sampled_unique_pair_sequence_from_pairs(
    pairs: np.ndarray,
    rng: np.random.Generator,
    epochs: int,
    batch_pairs: int,
) -> NumbaList:
    total_pairs = len(pairs)
    take = min(batch_pairs, total_pairs)
    result = NumbaList()
    for _ in range(epochs):
        choice = rng.choice(total_pairs, size=take, replace=False)
        result.append(np.asarray(pairs[choice], dtype=np.int32))
    return result


def _build_weighted_pair_alias(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Walker alias table for O(1) draws from a fixed pair distribution."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    n = len(probabilities)
    scaled = probabilities * n
    accept = np.empty(n, dtype=np.float64)
    alias = np.empty(n, dtype=np.int32)
    small = [int(i) for i in np.flatnonzero(scaled < 1.0)]
    large = [int(i) for i in np.flatnonzero(scaled >= 1.0)]
    while small and large:
        lo = small.pop()
        hi = large.pop()
        accept[lo] = scaled[lo]
        alias[lo] = hi
        scaled[hi] -= 1.0 - scaled[lo]
        if scaled[hi] < 1.0:
            small.append(hi)
        else:
            large.append(hi)
    for i in small + large:
        accept[i] = 1.0
        alias[i] = i
    return accept, alias


def _build_weighted_pair_sequence_from_pairs(
    pairs: np.ndarray,
    rng: np.random.Generator,
    epochs: int,
    batch_pairs: int,
    accept: np.ndarray,
    alias: np.ndarray,
) -> NumbaList:
    """Sample pair batches with replacement using a precomputed alias table."""
    total_pairs = len(pairs)
    result = NumbaList()
    for _ in range(epochs):
        column = rng.integers(total_pairs, size=batch_pairs)
        choose_alias = rng.random(batch_pairs) >= accept[column]
        column[choose_alias] = alias[column[choose_alias]]
        result.append(np.asarray(pairs[column], dtype=np.int32))
    return result


@numba.njit(cache=True, fastmath=True)
def _batch_stress_njit(data, params, pairs, alpha, r0, r1, x, y, use_rect, eps=1e-12, normalize=False):
    """
    Scale-normalized value of the selected training objective on a fixed sample.

    Raw stress is divided by sum(d^2). Pair-normalized stress is divided by the
    number of sampled pairs, since its per-pair weights are 1/d^2. In either
    case the denominator is fixed across convergence checks, so relative change
    tracks the objective being optimized.
    """
    total_loss = 0.0
    total_scale = 0.0
    for k in range(pairs.shape[0]):
        i = pairs[k, 0]
        j = pairs[k, 1]
        d = data[i, j]
        if use_rect:
            _, _, r, _, _, w = rect_grad(params[i], params[j], d, alpha, r0, r1, eps, normalize)
        else:
            _, _, r, _, _, w = parallelogram_grad(params[i], params[j], d, alpha, x, y, eps, normalize)
        diff = alpha * r - d
        total_loss += w * diff * diff
        total_scale += w * d * d
    return total_loss / (total_scale + eps)


@numba.njit(cache=True, fastmath=True)
def _profile_alpha_njit(data, params, r0, r1, x, y, use_rect, eps=1e-12, normalize=False):
    """Exact post-update least-squares scale for the current geometry."""
    n = data.shape[0]
    num = 0.0
    den = 0.0
    if use_rect:
        g00 = r0 * r0
        g11 = r1 * r1
        for i in range(n):
            for j in range(i + 1, n):
                du0 = _wrap_to_half(params[j, 0] - params[i, 0])
                du1 = _wrap_to_half(params[j, 1] - params[i, 1])
                r = np.sqrt(max(0.0, g00 * du0 * du0 + g11 * du1 * du1)) + eps
                d = data[i, j]
                w = 1.0 / (d * d + eps) if normalize else 1.0
                num += w * d * r
                den += w * r * r
    else:
        g11 = x * x + y * y
        for i in range(n):
            for j in range(i + 1, n):
                a = _wrap_to_half(params[j, 0] - params[i, 0])
                b = _wrap_to_half(params[j, 1] - params[i, 1])
                du0, du1 = _min_image_offset(a, b, 1.0, x, g11)
                r = np.sqrt(max(0.0, _gram_quad(du0, du1, 1.0, x, g11))) + eps
                d = data[i, j]
                w = 1.0 / (d * d + eps) if normalize else 1.0
                num += w * d * r
                den += w * r * r
    return num / (den + eps)


@numba.njit(cache=True, fastmath=True)
def _profile_alpha_pairs_njit(
    data, params, pairs, r0, r1, x, y, use_rect, eps=1e-12, normalize=False,
    sampling_gamma=0.0, sampling_correction=1.0,
):
    """Exact least-squares scale on one sampled pair batch."""
    num = 0.0
    den = 0.0
    if use_rect:
        g00 = r0 * r0
        g11 = r1 * r1
        for k in range(pairs.shape[0]):
            i = pairs[k, 0]
            j = pairs[k, 1]
            du0 = _wrap_to_half(params[j, 0] - params[i, 0])
            du1 = _wrap_to_half(params[j, 1] - params[i, 1])
            r = np.sqrt(max(0.0, g00 * du0 * du0 + g11 * du1 * du1)) + eps
            d = data[i, j]
            w = 1.0 / (d * d + eps) if normalize else 1.0
            if sampling_gamma > 0.0:
                w *= sampling_correction * d ** (2.0 * sampling_gamma)
            num += w * d * r
            den += w * r * r
    else:
        g11 = x * x + y * y
        for k in range(pairs.shape[0]):
            i = pairs[k, 0]
            j = pairs[k, 1]
            a = _wrap_to_half(params[j, 0] - params[i, 0])
            b = _wrap_to_half(params[j, 1] - params[i, 1])
            du0, du1 = _min_image_offset(a, b, 1.0, x, g11)
            r = np.sqrt(max(0.0, _gram_quad(du0, du1, 1.0, x, g11))) + eps
            d = data[i, j]
            w = 1.0 / (d * d + eps) if normalize else 1.0
            if sampling_gamma > 0.0:
                w *= sampling_correction * d ** (2.0 * sampling_gamma)
            num += w * d * r
            den += w * r * r
    return num / (den + eps)


@numba.njit(cache=True, fastmath=True)
def _update_parallelogram_geom(
    params, learn_mode, gx_sum, gy_sum, used,
    alpha, x, y, y_raw, geom_lr,
):
    """
    One batch geometry update for the parallelogram learn modes (alpha, x, y).

    Both modes reuse the SAME gradient kernel (parallelogram_grad), which treats x and
    y as independent and accumulates gx_sum = sum_pairs dL/dx and gy_sum = sum_pairs
    dL/dy. The rhombic-vs-parallelogram distinction is applied HERE, as a constraint on
    how (x, y) may move -- there is no separate rhombic gradient kernel.

    learn_mode == 5 ('rhombic'):       equal sides, so b1 = (x, y) is forced onto the
        unit circle y = sqrt(1 - x^2). That leaves a single free parameter (x = cos of
        the angle); the constraint is folded into the gradient by the chain rule below.
        x is clamped to [-1/2, 1/2] (angle in [60, 120] deg), where the basis stays
        Gauss-reduced, so no tidy is needed.
    learn_mode == 6 ('parallelogram'): free shape via (x, y_raw) with
        y = sqrt(1 - x^2) + softplus(y_raw); a tidy step Gauss-reduces and shears
        the positions whenever |x| > 1/2.

    Returns the updated (alpha, x, y, y_raw); alpha is rescaled only when the
    basis tidy changes coordinate units, then profiled on the completed batch.
    """
    base = np.sqrt(max(0.0, 1.0 - x * x))
    inv_base = x / base if base > 1e-9 else 0.0  # d(sqrt(1-x^2))/dx = -x/base

    if learn_mode == 5:
        # Rhombic: y is tied to x by y = sqrt(1 - x^2), so the only free parameter is x.
        # By the chain rule the total derivative wrt x is
        #     dL/dx = dL/dx|_y + (dL/dy)*(dy/dx) = gx - gy * x/base.
        # dy/dx depends only on the current x (not on the pair), so it factors out of the
        # sum over pairs -- we apply it once to the aggregated gx_sum, gy_sum here rather
        # than per pair in the kernel (hence no dedicated rhombic gradient kernel).
        dx = gx_sum - gy_sum * inv_base
        x = x - geom_lr * dx / used
        if x < -0.5:
            x = -0.5
        elif x > 0.5:
            x = 0.5
        # Re-project b1 back onto the unit circle (restore the equal-sides constraint).
        y = np.sqrt(max(0.0, 1.0 - x * x))
    else:
        # Parallelogram: x and the height move independently. The learnable height is
        # y = sqrt(1 - x^2) + softplus(y_raw); chain-rule each parameter separately --
        # dL/dx still carries the sqrt(1 - x^2) term (gx - gy * x/base), while the
        # softplus part of the height is driven by dL/dy_raw = gy * sigmoid(y_raw).
        sig = 1.0 / (1.0 + np.exp(-y_raw))  # d softplus / d y_raw
        dx = gx_sum - gy_sum * inv_base
        dyr = gy_sum * sig
        x = x - geom_lr * dx / used
        y_raw = y_raw - geom_lr * dyr / used
        base = np.sqrt(max(0.0, 1.0 - x * x))
        # stable softplus = log(1 + exp(y_raw)) (numba lacks np.logaddexp)
        softplus = max(y_raw, 0.0) + np.log1p(np.exp(-abs(y_raw)))
        y = base + softplus

        if abs(x) > 0.5:  # tidy: reduce, shear positions, fold scale, translate back
            xr, yr, scale, u00, u01, u10, u11 = _gauss_reduce_njit(x, y)
            for p in range(params.shape[0]):
                s0 = params[p, 0]
                s1 = params[p, 1]
                params[p, 0] = (u00 * s0 + u01 * s1) % 1.0
                params[p, 1] = (u10 * s0 + u11 * s1) % 1.0
            alpha = alpha * scale
            x = xr
            y = yr
            base = np.sqrt(max(0.0, 1.0 - x * x))
            spv = y - base
            if spv < 1e-12:
                spv = 1e-12
            y_raw = np.log(np.expm1(spv))

    return alpha, x, y, y_raw


@numba.njit(cache=True, fastmath=True)
def _init_parallelogram_yraw(x, y):
    """Inverse-softplus initialization of y_raw so sqrt(1-x^2)+softplus(y_raw)==y."""
    base = np.sqrt(max(0.0, 1.0 - x * x))
    spv = y - base
    if spv < 1e-12:
        spv = 1e-12
    return np.log(np.expm1(spv))


@numba.njit(cache=True, fastmath=True)
def _sgd_minibatch_njit_legacy(
    data,
    learning_rate,
    max_iters=500,
    batch_pairs=4096,
    seed=0,
    alpha_init=1.0,
    eps=1e-12,
    alpha_min=1e-6,
    alpha_max=1e6,
    r0_init=1.0,
    r1_init=1.0,
    learn_mode=0,   # 0=fixed,1=alpha,2=square,3=rectangular,4=alpha_aspect,5=rhombic,6=parallelogram
    geom_lr=0.01,
    geom_min=1e-6,
    theta=np.pi / 2,
    x_init=0.0,
    y_init=1.0,
    lr_warmup_init=10.0,
    lr_warmup_decay=25.0,
    normalize=False,
):
    """
    Minibatch SGD for torus MDS on a flat parallelogram torus.

    learn_mode=0 ('fixed'):       positions only; geometry fixed.
    learn_mode=1 ('alpha'):       positions + alpha via per-batch closed-form profiling.
    learn_mode=2 ('square'):      positions + shared r = r0 = r1 via SGD; alpha fixed.
    learn_mode=3 ('rectangular'): positions + independent r0, r1 via SGD; alpha fixed.
    learn_mode=4 ('alpha_aspect'): positions + alpha + log-aspect.
    learn_mode=5 ('rhombic'):     positions + alpha + rhombus angle (shear x), equal sides.
    learn_mode=6 ('parallelogram'): positions + alpha + free shear x and height y.

    Modes 5/6 use the (alpha, x, y) parametrization (x_init, y_init); modes 0-4 use
    (r0_init, r1_init, theta). Returns: (params, alpha, r0, r1, x, y).

    Position step size follows an exponentially-decaying warmup on top of the
    harmonic tail: step = lr_warmup_init * exp(-it/lr_warmup_decay) + 1/(learning_rate+it),
    floored at 1e-4. The warmup term vanishes after a few multiples of
    lr_warmup_decay, leaving the plain harmonic decay of the tail. Set
    lr_warmup_init=0 to recover the old pure-harmonic schedule.

    normalize: False optimizes raw stress sum((alpha*r-d)^2); True optimizes
    normalized stress sum((alpha*r-d)^2 / d^2), i.e. each pair's term is divided
    by its own d^2 before summing. rect_grad/parallelogram_grad return the
    per-pair weight (1, or 1/(d^2+eps)) already folded into their gradients.
    """
    n = data.shape[0]
    np.random.seed(seed)

    params = np.random.rand(n, 2)
    alpha = alpha_init
    r0 = r0_init
    r1 = r1_init
    x = x_init
    y = y_init
    y_raw = _init_parallelogram_yraw(x, y) if learn_mode == 6 else 0.0
    # Orthogonal closed-form rect path only when theta == pi/2. Equal-side rhombic
    # (legacy modes 0/1 at theta != 90, seeded x=cos t, y=sin t) and modes 5/6 use
    # the parallelogram kernel. theta and learn_mode are constant, so decide once.
    use_rect = learn_mode <= 4 and abs(np.cos(theta)) < 1e-12
    if learn_mode == 4:
        aspect_scale = np.sqrt(max(geom_min, r0 * r1))
        alpha *= aspect_scale
        log_aspect = np.log(max(geom_min, r1)) - np.log(max(geom_min, r0))
        if log_aspect < -6.0:
            log_aspect = -6.0
        elif log_aspect > 6.0:
            log_aspect = 6.0
        r0 = np.exp(-0.5 * log_aspect)
        r1 = np.exp(0.5 * log_aspect)
    else:
        log_aspect = 0.0

    for it in range(max_iters):
        step_pos = max(
            lr_warmup_init * np.exp(-it / lr_warmup_decay) + 1.0 / (learning_rate + it),
            1e-4,
        )
        if learn_mode == 4:
            r0 = np.exp(-0.5 * log_aspect)
            r1 = np.exp(0.5 * log_aspect)

        gr0_sum = 0.0
        gr1_sum = 0.0
        gx_sum = 0.0
        gy_sum = 0.0
        used = 0
        drawn = 0

        while drawn < batch_pairs:
            i = np.random.randint(0, n)
            j = np.random.randint(0, n)
            if i == j:
                continue

            d = data[i, j]
            if use_rect:
                g0, g1, dist, gr0_k, gr1_k, w = rect_grad(
                    params[i], params[j], d, alpha, r0, r1, eps, normalize
                )
                gr0_sum += gr0_k
                gr1_sum += gr1_k
            else:
                g0, g1, dist, ga, gb, w = parallelogram_grad(
                    params[i], params[j], d, alpha, x, y, eps, normalize
                )
                gx_sum += ga
                gy_sum += gb

            params[i, 0] -= step_pos * g0
            params[i, 1] -= step_pos * g1
            params[j, 0] += step_pos * g0
            params[j, 1] += step_pos * g1

            used += 1
            drawn += 1

        params %= 1.0

        if used > 0:
            if learn_mode == 2:
                r = max(geom_min, r0 - geom_lr * (gr0_sum + gr1_sum) / used)
                r0 = r
                r1 = r
            elif learn_mode == 3:
                r0 = max(geom_min, r0 - geom_lr * gr0_sum / used)
                r1 = max(geom_min, r1 - geom_lr * gr1_sum / used)
            elif learn_mode == 4:
                grad_s = (-0.5 * gr0_sum * r0 + 0.5 * gr1_sum * r1) / used
                log_aspect -= geom_lr * grad_s
                if log_aspect < -6.0:
                    log_aspect = -6.0
                elif log_aspect > 6.0:
                    log_aspect = 6.0
                r0 = np.exp(-0.5 * log_aspect)
                r1 = np.exp(0.5 * log_aspect)
            elif learn_mode >= 5:
                alpha, x, y, y_raw = _update_parallelogram_geom(
                    params, learn_mode, gx_sum, gy_sum, used,
                    alpha, x, y, y_raw, geom_lr,
                )

    return params, alpha, r0, r1, x, y


@numba.njit(cache=True, fastmath=True)
def _run_pair_sequence_online_njit(
    data,
    pair_sequence,
    learning_rate,
    init,
    start_iter=0,
    alpha_init=1.0,
    eps=1e-12,
    alpha_min=1e-6,
    alpha_max=1e6,
    r0_init=1.0,
    r1_init=1.0,
    learn_mode=0,   # 0=fixed,1=alpha,2=square,3=rectangular,4=alpha_aspect,5=rhombic,6=parallelogram
    geom_lr=0.01,
    geom_min=1e-6,
    theta=np.pi / 2,
    x_init=0.0,
    y_init=1.0,
    lr_warmup_init=10.0,
    lr_warmup_decay=25.0,
    normalize=False,
    bounded=True,
    pair_step_cap=0.10,       # Max fractional-coordinate displacement of either endpoint per pair.
    batch_profile_alpha=False,
    sampling_gamma=0.0,
    sampling_correction=1.0,
):
    params = init.copy()
    alpha = alpha_init
    r0 = r0_init
    r1 = r1_init
    x = x_init
    y = y_init
    y_raw = _init_parallelogram_yraw(x, y) if learn_mode == 6 else 0.0
    # Orthogonal closed-form rect path only when theta == pi/2. Equal-side rhombic
    # (legacy modes 0/1 at theta != 90, seeded x=cos t, y=sin t) and modes 5/6 use
    # the parallelogram kernel. theta and learn_mode are constant, so decide once.
    use_rect = learn_mode <= 4 and abs(np.cos(theta)) < 1e-12
    if learn_mode == 4:
        aspect_scale = np.sqrt(max(geom_min, r0 * r1))
        alpha *= aspect_scale
        log_aspect = np.log(max(geom_min, r1)) - np.log(max(geom_min, r0))
        if log_aspect < -6.0:
            log_aspect = -6.0
        elif log_aspect > 6.0:
            log_aspect = 6.0
        r0 = np.exp(-0.5 * log_aspect)
        r1 = np.exp(0.5 * log_aspect)
    else:
        log_aspect = 0.0
    epochs = len(pair_sequence)
    total_pairs = data.shape[0] * (data.shape[0] - 1) // 2
    pairs_seen = start_iter * pair_sequence[0].shape[0]

    for it in range(epochs):
        global_it = start_iter + it
        tail_progress = global_it
        if bounded:
            # Decay proportional to number of full pair passes, not per iteration
            tail_progress = pairs_seen / total_pairs
        step_pos = max(
            lr_warmup_init * np.exp(-global_it / lr_warmup_decay)
            + 1.0 / (learning_rate + tail_progress),
            1e-4,
        )
        if learn_mode == 4:
            r0 = np.exp(-0.5 * log_aspect)
            r1 = np.exp(0.5 * log_aspect)

        seq = pair_sequence[it]
        used = seq.shape[0]

        gr0_sum = 0.0
        gr1_sum = 0.0
        gx_sum = 0.0
        gy_sum = 0.0
        for k in range(used):
            i = seq[k, 0]
            j = seq[k, 1]

            d = data[i, j]
            importance = 1.0
            if sampling_gamma > 0.0:
                importance = sampling_correction * d ** (2.0 * sampling_gamma)
            if use_rect:
                g0, g1, dist, gr0_k, gr1_k, w = rect_grad(
                    params[i], params[j], d, alpha, r0, r1, eps, normalize
                )
                gr0_sum += gr0_k
                gr1_sum += gr1_k
            else:
                g0, g1, dist, ga, gb, w = parallelogram_grad(
                    params[i], params[j], d, alpha, x, y, eps, normalize
                )
                gx_sum += ga
                gy_sum += gb

            if importance != 1.0:
                g0 *= importance
                g1 *= importance
                if use_rect:
                    gr0_sum += (importance - 1.0) * gr0_k
                    gr1_sum += (importance - 1.0) * gr1_k
                else:
                    gx_sum += (importance - 1.0) * ga
                    gy_sum += (importance - 1.0) * gb

            if bounded:
                if normalize:
                    # The raw target-distance relaxation is not scale-equivalent to the normalized gradient.
                    # Clip the actual metric gradient step instead, preserving its alpha and 1/d^2 factors for every torus geometry.
                    di0 = -step_pos * g0
                    di1 = -step_pos * g1
                    step_norm = np.sqrt(di0 * di0 + di1 * di1)
                    if step_norm > pair_step_cap:
                        scale = pair_step_cap / step_norm
                        di0 *= scale
                        di1 *= scale
                    params[i, 0] += di0
                    params[i, 1] += di1
                    params[j, 0] -= di0
                    params[j, 1] -= di1
                else:
                    if use_rect:
                        du0 = params[j, 0] - params[i, 0]
                        du1 = params[j, 1] - params[i, 1]
                        du0 -= np.round(du0)
                        du1 -= np.round(du1)
                    else:
                        _, _, _, du0, du1, _, _, _ = _torus_dist_grad_core(
                            params[i], params[j], d, alpha, 1.0, x, x * x + y * y, eps, normalize
                        )
                    mu = step_pos * w * (d - alpha * dist) / (2.0 * alpha * (dist + eps))
                    mu = min(1.0, max(-1.0, mu))
                    di0 = -mu * du0
                    di1 = -mu * du1
                    step_norm = np.sqrt(di0 * di0 + di1 * di1)
                    if step_norm > pair_step_cap:
                        scale = pair_step_cap / step_norm
                        di0 *= scale
                        di1 *= scale
                    params[i, 0] += di0
                    params[i, 1] += di1
                    params[j, 0] -= di0
                    params[j, 1] -= di1
            else:
                params[i, 0] -= step_pos * g0
                params[i, 1] -= step_pos * g1
                params[j, 0] += step_pos * g0
                params[j, 1] += step_pos * g1

        params %= 1.0
        pairs_seen += used

        if used > 0:
            if learn_mode == 2:
                r = max(geom_min, r0 - geom_lr * (gr0_sum + gr1_sum) / used)
                r0 = r
                r1 = r
            elif learn_mode == 3:
                r0 = max(geom_min, r0 - geom_lr * gr0_sum / used)
                r1 = max(geom_min, r1 - geom_lr * gr1_sum / used)
            elif learn_mode == 4:
                grad_s = (-0.5 * gr0_sum * r0 + 0.5 * gr1_sum * r1) / used
                log_aspect -= geom_lr * grad_s
                if log_aspect < -6.0:
                    log_aspect = -6.0
                elif log_aspect > 6.0:
                    log_aspect = 6.0
                r0 = np.exp(-0.5 * log_aspect)
                r1 = np.exp(0.5 * log_aspect)
            elif learn_mode >= 5:
                alpha, x, y, y_raw = _update_parallelogram_geom(
                    params, learn_mode, gx_sum, gy_sum, used,
                    alpha, x, y, y_raw, geom_lr,
                )
            if batch_profile_alpha and learn_mode in (1, 4, 5, 6):
                # Profile on the completed, wrapped batch layout.
                alpha = _profile_alpha_pairs_njit(
                    data, params, seq, r0, r1, x, y, use_rect, eps, normalize,
                    sampling_gamma, sampling_correction,
                )
                alpha = max(alpha_min, min(alpha_max, alpha))

    return params, alpha, r0, r1, x, y


def sgd_minibatch_njit(
    data,
    learning_rate,
    max_iters=500,
    batch_pairs=4096,
    seed=0,
    alpha_init=1.0,
    eps=1e-12,
    alpha_min=1e-6,
    alpha_max=1e6,
    r0_init=1.0,
    r1_init=1.0,
    learn_mode=0,   # 0=fixed,1=alpha,2=square,3=rectangular,4=alpha_aspect,5=rhombic,6=parallelogram
    geom_lr=0.01,
    geom_min=1e-6,
    theta=np.pi / 2,
    x_init=0.0,
    y_init=1.0,
    chunk_epochs=DEFAULT_SEQUENCE_CHUNK_EPOCHS,
    lr_warmup_init=10.0,
    lr_warmup_decay=25.0,
    tol: float = 0.0,     # relative stress improvement threshold (0 = disabled)
    patience: int = 5,    # non-improving chunks before stopping
    init=None,
    normalize=False,       # False: optimize raw stress; True: optimize normalized stress (per-pair /d^2)
    bounded=True,
    pair_step_cap=0.10,
    normalized_pair_sampling="auto",
):
    """
    Minibatch SGD for torus MDS using online pair updates.

    Raw and uniform-normalized batches sample pairs without replacement. Direct
    normalized importance batches sample with replacement from their fixed
    target-distance distribution, with an unbiased correction.

    init: optional (N, 2) initial positions; wrapped to [0,1). Random if None.

    Returns: (params, alpha, r0, r1, x, y)
    """
    data = np.asarray(data, dtype=np.float64)
    if pair_step_cap <= 0.0:
        raise ValueError("pair_step_cap must be positive")
    n = data.shape[0]
    if init is None:
        params = np.random.RandomState(seed).rand(n, 2).astype(np.float64, copy=False)
    else:
        init = np.asarray(init, dtype=np.float64)
        if init.shape != (n, 2):
            raise ValueError(f"init must have shape ({n}, 2), got {init.shape}.")
        params = np.mod(init, 1.0)
    alpha = float(alpha_init)
    r0 = float(r0_init)
    r1 = float(r1_init)
    x = float(x_init)
    y = float(y_init)
    if max_iters <= 0 or n < 2:
        return params, alpha, r0, r1, x, y

    rng = np.random.default_rng(seed)
    pairs = _all_unique_pairs(n)
    use_rect = learn_mode <= 4 and abs(np.cos(theta)) < 1e-12
    # Importance sampling is calibrated for the direct metric-gradient path
    weighted_sampling = normalize and not bounded and normalized_pair_sampling == "importance"
    sampling_gamma = 0.0
    sampling_correction = 1.0
    alias_accept = None
    alias_index = None
    if weighted_sampling:
        pair_distances = data[pairs[:, 0], pairs[:, 1]]
        sampling_gamma = 1.0
        sampling_weights = pair_distances ** (-2.0 * sampling_gamma)
        sampling_correction = float(sampling_weights.mean())
        alias_accept, alias_index = _build_weighted_pair_alias(sampling_weights / sampling_weights.sum())
    elif normalize and not bounded and normalized_pair_sampling == "auto":
        pair_weights = 1.0 / (data[pairs[:, 0], pairs[:, 1]] ** 2 + eps)
        effective_pair_fraction = (pair_weights.sum() ** 2 / np.sum(pair_weights * pair_weights)) / len(pair_weights)
        # Empirical separation: the SuiteSparse cases are below 2.5%, whereas
        # the current synthetic suites are above 25%; 10% selects only the
        # strongly concentrated normalized objectives.
        if effective_pair_fraction < 0.10:
            weighted_sampling = True
            sampling_gamma = 1.0
            sampling_correction = float(pair_weights.mean())
            alias_accept, alias_index = _build_weighted_pair_alias(pair_weights / pair_weights.sum())

    stress_sample = None
    bad_chunks = 0
    prev_stress = np.inf
    if tol > 0.0:
        num_pairs = n * (n - 1) // 2
        n_stress = min(max(256, batch_pairs), num_pairs)
        stress_rng = np.random.default_rng(seed)
        idx = stress_rng.choice(num_pairs, size=n_stress, replace=False)
        stress_sample = pairs[idx].astype(np.int32)

    next_check = chunk_epochs
    start_iter = 0
    while start_iter < max_iters:
        until_check = next_check - start_iter if tol > 0.0 else max_iters - start_iter
        epochs = min(chunk_epochs, until_check, max_iters - start_iter)
        if weighted_sampling:
            sequence = _build_weighted_pair_sequence_from_pairs(
                pairs, rng, epochs, batch_pairs, alias_accept, alias_index
            )
        else:
            sequence = _build_sampled_unique_pair_sequence_from_pairs(pairs, rng, epochs, batch_pairs)
        # The first bounded update uses a scale-consistent profile of its
        # sampled pair batch; every later batch is profiled after its update.
        if start_iter == 0 and bounded and learn_mode in (1, 4, 5, 6):
            alpha = _profile_alpha_pairs_njit(
                data, params, sequence[0], r0, r1, x, y, use_rect, eps, normalize
            )
            alpha = max(alpha_min, min(alpha_max, alpha))
        params, alpha, r0, r1, x, y = _run_pair_sequence_online_njit(
            data=data,
            pair_sequence=sequence,
            learning_rate=learning_rate,
            init=params,
            start_iter=start_iter,
            alpha_init=alpha,
            eps=eps,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            r0_init=r0,
            r1_init=r1,
            learn_mode=learn_mode,
            geom_lr=geom_lr,
            geom_min=geom_min,
            theta=theta,
            x_init=x,
            y_init=y,
            lr_warmup_init=lr_warmup_init,
            lr_warmup_decay=lr_warmup_decay,
            normalize=normalize,
            bounded=bounded,
            pair_step_cap=pair_step_cap,
            batch_profile_alpha=True,
            sampling_gamma=sampling_gamma,
            sampling_correction=sampling_correction,
        )
        start_iter += epochs

        if tol > 0.0 and start_iter == next_check:
            curr_stress = _batch_stress_njit(
                data, params, stress_sample, alpha, r0, r1, x, y, use_rect, eps, normalize
            )
            rel_change = abs(prev_stress - curr_stress) / max(prev_stress, 1e-12)
            if rel_change < tol:
                bad_chunks += 1
                if bad_chunks >= patience:
                    break
            else:
                bad_chunks = 0
            prev_stress = curr_stress
            next_check += chunk_epochs

    return params, alpha, r0, r1, x, y


@dataclass
class TorusProjector:
    """
    Base class: accepts a precomputed distance matrix D (NxN), runs an underlying
    embedding algorithm, then projects/wraps the output to a square torus [0,1)^2.

    Subclasses implement _embed(D) -> (N, n_components).

    Projection modes:
      - "wrap":        X % 1  (assumes embedding already on unit scale)
      - "robust_wrap": robust affine to [0,1] per axis, then wrap (recommended)
    """
    n_components: int = 2
    projection: Literal["wrap", "robust_wrap"] = "robust_wrap"
    random_state: Optional[int] = 42

    embedding_: Optional[np.ndarray] = field(default=None, init=False)
    torus_embedding_: Optional[np.ndarray] = field(default=None, init=False)
    alpha_: Optional[float] = field(default=None, init=False)
    r0_: Optional[float] = field(default=None, init=False)
    r1_: Optional[float] = field(default=None, init=False)
    theta_: Optional[float] = field(default=None, init=False)
    x_: Optional[float] = field(default=None, init=False)
    y_: Optional[float] = field(default=None, init=False)
    finalization_info_: dict = field(default_factory=lambda: {"mode": None}, init=False)

    def fit_transform(self, D: np.ndarray, **kwargs) -> np.ndarray:
        D = _check_distance_matrix(D)
        X = self._embed(D, **kwargs)
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] != self.n_components:
            raise ValueError(f"Expected embedding with {self.n_components} dims, got {X.shape}.")
        self.embedding_ = X
        self.torus_embedding_ = self._project_to_torus(X)
        return self.torus_embedding_

    def fit(self, D: np.ndarray, **kwargs) -> "TorusProjector":
        self.fit_transform(D, **kwargs)
        return self

    def transform(self, D: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Most manifold methods are not inductive with precomputed D. Use fit_transform().")

    def _embed(self, D: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def _project_to_torus(self, X: np.ndarray) -> np.ndarray:
        if self.projection == "wrap":
            return _wrap_unit_square(X)
        elif self.projection == "robust_wrap":
            return _wrap_unit_square(_robust_affine_to_unit_square(X))
        else:
            raise ValueError(f"Unknown projection: {self.projection}")


@dataclass
class MDSTorusProjector(TorusProjector):
    projection: Literal["wrap", "robust_wrap"] = "wrap"

    # In the current schedule, larger learning_rate means smaller harmonic-tail
    # steps: step_tail = 1 / (learning_rate + epoch).
    learning_rate: float = 1

    def stochastic_gradient_descent(
            self,
            max_iters=10000,
            batch_size=4096,
            data=None,
            seed=0,
            alpha_init=1.0,
            r0_init=1.0,
            r1_init=1.0,
            learn_mode='alpha',   # 'fixed'|'alpha'|'square'|'rectangular'|'alpha_aspect'|'rhombic'|'parallelogram'
            geom_lr=0.01,
            sampled_unique=True,
            theta=90.0,           # torus angle in degrees; must be in [60, 120]
            init=None,            # optional (N, 2) initial positions in [0,1)^2 or spectral initialization dict
            learning_rate: float | Literal["auto"] | None = "auto",
            x_init=None,          # initial shear (parallelogram/rhombic); default derived from theta
            y_init=None,          # initial height (parallelogram)
            # --- position step-size schedule ---
            lr_warmup_init: float | Literal["auto"] | None = "auto",
            lr_warmup_decay: float = 25.0,  # e-folding scale (epochs) of the warmup
            # --- convergence criteria (sampled_unique path only) ---
            tol: float = 0.0,     # relative stress improvement threshold (0 = disabled)
            patience: int = 5,    # non-improving chunks before stopping
            finalize: Literal[None, "none", "lbfgs"] = None,
            finalize_iters: int = 50,
            finalize_options: Optional[dict] = None,
            # 'raw': minimize sum((alpha*r-d)^2). 'normalized': minimize
            # sum((alpha*r-d)^2 / d^2) -- each pair's term is divided by its own
            # d^2 before summing (not a single division after summing).
            stress_mode: Literal["raw", "normalized"] = "raw",
            coordinate_update: Literal["bounded", "gradient"] | None = None,
            pair_step_cap: float = 0.10,
            normalized_pair_sampling: Literal["auto", "importance", "uniform"] = "auto",
    ):
        if not (60.0 <= theta <= 120.0):
            raise ValueError(f"theta must be in [60, 120] degrees, got {theta}")
        if stress_mode not in ("raw", "normalized"):
            raise ValueError(f"stress_mode must be 'raw' or 'normalized', got {stress_mode!r}")
        if normalized_pair_sampling not in ("auto", "importance", "uniform"):
            raise ValueError(
                "normalized_pair_sampling must be 'auto', 'importance', or 'uniform', "
                f"got {normalized_pair_sampling!r}"
            )
        if coordinate_update is None:
            coordinate_update = "gradient" if stress_mode == "normalized" else "bounded"
        if coordinate_update not in ("bounded", "gradient"):
            raise ValueError(
                "coordinate_update must be 'bounded' or 'gradient', "
                f"got {coordinate_update!r}"
            )
        if normalized_pair_sampling == "importance" and (
            stress_mode != "normalized" or coordinate_update != "gradient" or not sampled_unique
        ):
            raise ValueError(
                "normalized_pair_sampling='importance' requires sampled_unique direct "
                "normalized-gradient optimization"
            )
        if pair_step_cap <= 0.0:
            raise ValueError(f"pair_step_cap must be positive, got {pair_step_cap}")
        normalize = stress_mode == "normalized"
        has_init = init is not None
        mode_int = {
            'fixed': 0, 'alpha': 1, 'square': 2, 'rectangular': 3,
            'alpha_aspect': 4, 'rhombic': 5, 'parallelogram': 6,
        }[learn_mode]

        # Modes 0-4 use the rect kernel, exact only for rectangular or equal-side
        # rhombic bases. At theta != 90, 'fixed' and 'alpha' are fine as a fixed-angle
        # rhombic torus (equal sides required); 'square'/'rectangular'/'alpha_aspect'
        # contradict a non-rectangular rhombus and are redirected to the dedicated
        # parallelogram modes (5, 6).
        if theta != 90.0:
            if mode_int in (2, 3, 4):
                raise ValueError(
                    f"learn_mode={learn_mode!r} does not support theta != 90 "
                    "(non-rectangular). use learn_mode='rhombic' to learn the angle, "
                    "'parallelogram' for a general shape, or 'alpha'/'fixed' with "
                    "r0_init == r1_init for a fixed-angle rhombic torus."
                )
            if mode_int in (0, 1) and not np.isclose(r0_init, r1_init):
                raise ValueError(
                    "theta != 90 with learn_mode='fixed'/'alpha' requires equal side "
                    "lengths (set r0_init == r1_init) for an exact rhombic torus."
                )
        theta_rad = float(np.radians(theta))

        if isinstance(init, dict) and "fundamental_directions" in init:
            dmax = float(np.asarray(data).max())
            if not np.isclose(dmax, 1.0):
                raise ValueError(
                    "spectral-dict init assumes distances normalized to max 1 "
                    f"(pass D / D.max()); got D.max()={dmax:.6g}."
                )
            from .initialization import rect_torus_init_from_spectral
            init, r0_init, r1_init = rect_torus_init_from_spectral(init)
            alpha_init = 1.0

        if learning_rate is None or learning_rate == "auto":
            # Random starts benefit from the exploratory main-branch schedule.
            # Supplied inits are already structured; preserve them with the old
            # spectral-init-sized first tail step: 1 / 10 == 0.1.
            learning_rate = 10.0 if has_init else self.learning_rate
        elif isinstance(learning_rate, str):
            raise ValueError(f"learning_rate must be 'auto' or positive, got {learning_rate!r}")
        if lr_warmup_init is None or lr_warmup_init == "auto":
            lr_warmup_init = 0.0 if has_init else 10.0
        elif isinstance(lr_warmup_init, str):
            raise ValueError(f"lr_warmup_init must be 'auto' or non-negative, got {lr_warmup_init!r}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if lr_warmup_init < 0:
            raise ValueError(f"lr_warmup_init must be non-negative, got {lr_warmup_init}")
        if lr_warmup_decay <= 0:
            raise ValueError(f"lr_warmup_decay must be positive, got {lr_warmup_decay}")

        # Legacy 'fixed'/'alpha' at theta != 90 is an equal-side rhombic torus; it now
        # runs on the parallelogram kernel (rect_grad is orthogonal only). The (x, y)
        # unit-circle basis carries no side length, so the fixed side length r0_init is
        # folded into the parallelogram scale here and unfolded out of alpha_ on report.
        non_orth_rect = mode_int in (0, 1) and theta != 90.0

        # Initial shear/height for the parallelogram modes (5, 6) and the legacy rhombic
        # path. lengths_angle_to_xy returns the unit-circle (cos t, sin t) for equal sides.
        if x_init is None or y_init is None:
            _, x0, y0 = lengths_angle_to_xy(r0_init, r1_init, theta_rad)
        else:
            x0, y0 = float(x_init), float(y_init)
        if mode_int == 5:
            # rhombic: start on the unit circle (equal sides) at the given angle
            x0, y0 = float(np.cos(theta_rad)), float(np.sin(theta_rad))
        if mode_int == 6 and x_init is None and abs(x0) < 1e-6:
            x0 = 0.05  # asymmetric init to escape the x -> -x (rectangular) saddle

        alpha_kernel = alpha_init * r0_init if non_orth_rect else alpha_init

        kwargs = dict(
            data=data,
            learning_rate=learning_rate,
            max_iters=max_iters,
            batch_pairs=batch_size,
            seed=seed,
            alpha_init=alpha_kernel,
            r0_init=r0_init,
            r1_init=r1_init,
            learn_mode=mode_int,
            geom_lr=geom_lr,
            theta=theta_rad,
            x_init=x0,
            y_init=y0,
            lr_warmup_init=lr_warmup_init,
            lr_warmup_decay=lr_warmup_decay,
            normalize=normalize,
        )
        if sampled_unique:
            sgd_fn = sgd_minibatch_njit  # uniform raw / importance-sampled normalized pairs
            kwargs["tol"] = tol
            kwargs["patience"] = patience
            kwargs["init"] = init
            kwargs["bounded"] = coordinate_update == "bounded"
            kwargs["pair_step_cap"] = pair_step_cap
            kwargs["normalized_pair_sampling"] = normalized_pair_sampling
        else:
            if init is not None:
                raise ValueError("init is only supported with sampled_unique=True")
            sgd_fn = _sgd_minibatch_njit_legacy  # sample pairs with replacement; no early stopping

        coords, alpha, r0, r1, x, y = sgd_fn(**kwargs)
        if not sampled_unique and mode_int in (1, 4, 5, 6):
            use_rect = mode_int <= 4 and abs(np.cos(theta_rad)) < 1e-12
            alpha = _profile_alpha_njit(data, coords, r0, r1, x, y, use_rect, normalize=normalize)

        final_info = {"mode": None}
        if finalize in (None, "none"):
            pass
        elif finalize == "lbfgs":
            from .exact_torus_stress import polish_torus_layout

            # Match the SGD objective: normalized stress is equivalent to a
            # per-pair weight of 1/d^2 in the (already weight-capable) polish
            # objective, so no separate normalized-stress kernel is needed there.
            polish_weights = 1.0 / (np.asarray(data, dtype=np.float64) ** 2 + 1e-12) if normalize else None
            result = polish_torus_layout(
                coords,
                data,
                weights=polish_weights,
                **_polish_shape_kwargs(mode_int, r0, r1, x, y, theta_rad),
                **_lbfgs_polish_options(finalize_iters, finalize_options),
            )
            coords = result.y
            alpha = result.alpha
            r0 = result.r0
            r1 = result.r1
            x = result.x
            y = result.y_shape
            theta_rad = result.theta
            final_info = {"mode": "lbfgs", **result.convergence_info, "n_iter": result.n_iter}
        else:
            raise ValueError("finalize must be None, 'none', or 'lbfgs'")

        if mode_int >= 5:
            # Report learned (alpha, x, y) as side lengths + angle; the b0 length
            # lives in alpha_, so r0_ = 1 and r1_ = |b1| / |b0|.
            self.alpha_ = float(alpha)
            self.x_ = float(x)
            self.y_ = float(y)
            self.r0_, self.r1_, self.theta_ = xy_to_lengths_angle(1.0, x, y)
        else:
            # Unfold the side length folded into the parallelogram scale for the legacy
            # rhombic path; a no-op (divide by 1) for the orthogonal rect modes.
            self.alpha_ = float(alpha) / r0_init if non_orth_rect and final_info["mode"] is None else float(alpha)
            self.r0_ = float(r0)
            self.r1_ = float(r1)
            self.theta_ = theta_rad
            _, self.x_, self.y_ = lengths_angle_to_xy(r0, r1, theta_rad)
        self.finalization_info_ = final_info
        return coords

    def _embed(self, D: np.ndarray, **kwargs) -> np.ndarray:
        return self.stochastic_gradient_descent(data=D, **kwargs)


@dataclass
class TSNETorusProjector(TorusProjector):
    perplexity: float = 30.0
    n_iter: int = 1000
    init: Literal["random", "pca"] = "random"
    learning_rate: Literal["auto"] | float = "auto"
    early_exaggeration: float = 12.0
    angle: float = 0.5
    method: Literal["barnes_hut", "exact"] = "barnes_hut"
    verbose: int = 0

    def _embed(self, D: np.ndarray, **kwargs) -> np.ndarray:
        from sklearn.manifold import TSNE

        tsne = TSNE(
            n_components=self.n_components,
            metric="precomputed",
            random_state=self.random_state,
            perplexity=self.perplexity,
            n_iter=self.n_iter,
            init=self.init,
            learning_rate=self.learning_rate,
            early_exaggeration=self.early_exaggeration,
            angle=self.angle,
            method=self.method,
            verbose=self.verbose,
        )
        return tsne.fit_transform(D)


@dataclass
class UMAPTorusProjector(TorusProjector):
    n_neighbors: int = 15
    min_dist: float = 0.1
    spread: float = 1.0
    n_epochs: Optional[int] = None
    learning_rate: float = 1.0
    repulsion_strength: float = 1.0
    negative_sample_rate: int = 5
    init: Literal["spectral", "random"] = "spectral"

    def _embed(self, D: np.ndarray, **kwargs) -> np.ndarray:
        import umap

        reducer = umap.UMAP(
            metric="precomputed",
            output_metric=torus_grad,
            learning_rate=1.0,
            repulsion_strength=0.1,
            negative_sample_rate=5,
            n_neighbors=100,
        )
        return reducer.fit_transform(D)
