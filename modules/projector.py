from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal, Tuple

import numpy as np
import numba
from numba.typed import List as NumbaList

from .geometry import grad_rect_torus, torus_grad

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



@numba.njit(cache=True, fastmath=True)
def _sgd_minibatch_njit_legacy(
    data,
    learning_rate,
    max_iters=500,
    batch_pairs=4096,
    seed=0,
    alpha_init=1.0,
    alpha_ema=0.05,
    eps=1e-12,
    alpha_min=1e-6,
    alpha_max=1e6,
    r0_init=1.0,
    r1_init=1.0,
    learn_mode=0,   # 0=fixed, 1=alpha, 2=square, 3=rectangular, 4=alpha_aspect
    geom_lr=0.01,
    geom_min=1e-6,
    theta=np.pi / 2,
):
    """
    Minibatch SGD for torus MDS on a flat rectangular/rhombic torus.

    learn_mode=0 ('fixed'):    positions only; alpha, r0, r1 fixed.
    learn_mode=1 ('alpha'):    positions + alpha via closed-form batch minimiser + EMA.
    learn_mode=2 ('square'):   positions + shared r = r0 = r1 via SGD; alpha fixed.
    learn_mode=3 ('rectangular'): positions + independent r0, r1 via SGD; alpha fixed.

    Returns: (params, alpha, r0, r1)
    """
    n = data.shape[0]
    np.random.seed(seed)

    params = np.random.rand(n, 2)
    alpha = alpha_init
    r0 = r0_init
    r1 = r1_init
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
        step_pos = max(1.0 / (learning_rate + it), 1e-4)
        if learn_mode == 4:
            r0 = np.exp(-0.5 * log_aspect)
            r1 = np.exp(0.5 * log_aspect)

        num = 0.0
        den = 0.0
        gr0_sum = 0.0
        gr1_sum = 0.0
        used = 0
        drawn = 0

        while drawn < batch_pairs:
            i = np.random.randint(0, n)
            j = np.random.randint(0, n)
            if i == j:
                continue

            d = data[i, j]
            g0, g1, dist, gr0_k, gr1_k = grad_rect_torus(
                params[i], params[j], d, alpha, r0, r1, eps, theta
            )

            params[i, 0] -= step_pos * g0
            params[i, 1] -= step_pos * g1
            params[j, 0] += step_pos * g0
            params[j, 1] += step_pos * g1

            num += d * dist
            den += dist * dist
            gr0_sum += gr0_k
            gr1_sum += gr1_k
            used += 1
            drawn += 1

        params %= 1.0

        if used > 0:
            if learn_mode == 1:
                alpha_hat = num / (den + eps)
                alpha_hat = max(alpha_min, min(alpha_max, alpha_hat))
                alpha = (1.0 - alpha_ema) * alpha + alpha_ema * alpha_hat
            elif learn_mode == 2:
                r = max(geom_min, r0 - geom_lr * (gr0_sum + gr1_sum) / used)
                r0 = r
                r1 = r
            elif learn_mode == 3:
                r0 = max(geom_min, r0 - geom_lr * gr0_sum / used)
                r1 = max(geom_min, r1 - geom_lr * gr1_sum / used)
            elif learn_mode == 4:
                alpha_hat = num / (den + eps)
                alpha_hat = max(alpha_min, min(alpha_max, alpha_hat))
                alpha = (1.0 - alpha_ema) * alpha + alpha_ema * alpha_hat
                grad_s = (-0.5 * gr0_sum * r0 + 0.5 * gr1_sum * r1) / used
                log_aspect -= geom_lr * grad_s
                if log_aspect < -6.0:
                    log_aspect = -6.0
                elif log_aspect > 6.0:
                    log_aspect = 6.0
                r0 = np.exp(-0.5 * log_aspect)
                r1 = np.exp(0.5 * log_aspect)

    return params, alpha, r0, r1


@numba.njit(cache=True, fastmath=True)
def _run_pair_sequence_online_njit(
    data,
    pair_sequence,
    learning_rate,
    init_params,
    start_iter=0,
    alpha_init=1.0,
    alpha_ema=0.05,
    eps=1e-12,
    alpha_min=1e-6,
    alpha_max=1e6,
    r0_init=1.0,
    r1_init=1.0,
    learn_mode=0,   # 0=fixed, 1=alpha, 2=square, 3=rectangular, 4=alpha_aspect
    geom_lr=0.01,
    geom_min=1e-6,
    theta=np.pi / 2,
):
    params = init_params.copy()
    alpha = alpha_init
    r0 = r0_init
    r1 = r1_init
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

    for it in range(epochs):
        step_pos = max(1.0 / (learning_rate + start_iter + it), 1e-4)
        if learn_mode == 4:
            r0 = np.exp(-0.5 * log_aspect)
            r1 = np.exp(0.5 * log_aspect)

        seq = pair_sequence[it]
        used = seq.shape[0]

        num = 0.0
        den = 0.0
        gr0_sum = 0.0
        gr1_sum = 0.0
        for k in range(used):
            i = seq[k, 0]
            j = seq[k, 1]

            d = data[i, j]
            g0, g1, dist, gr0_k, gr1_k = grad_rect_torus(
                params[i], params[j], d, alpha, r0, r1, eps, theta
            )

            params[i, 0] -= step_pos * g0
            params[i, 1] -= step_pos * g1
            params[j, 0] += step_pos * g0
            params[j, 1] += step_pos * g1

            num += d * dist
            den += dist * dist
            gr0_sum += gr0_k
            gr1_sum += gr1_k

        params %= 1.0

        if used > 0:
            if learn_mode == 1:
                alpha_hat = num / (den + eps)
                alpha_hat = max(alpha_min, min(alpha_max, alpha_hat))
                alpha = (1.0 - alpha_ema) * alpha + alpha_ema * alpha_hat
            elif learn_mode == 2:
                r = max(geom_min, r0 - geom_lr * (gr0_sum + gr1_sum) / used)
                r0 = r
                r1 = r
            elif learn_mode == 3:
                r0 = max(geom_min, r0 - geom_lr * gr0_sum / used)
                r1 = max(geom_min, r1 - geom_lr * gr1_sum / used)
            elif learn_mode == 4:
                alpha_hat = num / (den + eps)
                alpha_hat = max(alpha_min, min(alpha_max, alpha_hat))
                alpha = (1.0 - alpha_ema) * alpha + alpha_ema * alpha_hat
                grad_s = (-0.5 * gr0_sum * r0 + 0.5 * gr1_sum * r1) / used
                log_aspect -= geom_lr * grad_s
                if log_aspect < -6.0:
                    log_aspect = -6.0
                elif log_aspect > 6.0:
                    log_aspect = 6.0
                r0 = np.exp(-0.5 * log_aspect)
                r1 = np.exp(0.5 * log_aspect)

    return params, alpha, r0, r1


def sgd_minibatch_njit(
    data,
    learning_rate,
    max_iters=500,
    batch_pairs=4096,
    seed=0,
    alpha_init=1.0,
    alpha_ema=0.05,
    eps=1e-12,
    alpha_min=1e-6,
    alpha_max=1e6,
    r0_init=1.0,
    r1_init=1.0,
    learn_mode=0,   # 0=fixed, 1=alpha, 2=square, 3=rectangular, 4=alpha_aspect
    geom_lr=0.01,
    geom_min=1e-6,
    theta=np.pi / 2,
    chunk_epochs=DEFAULT_SEQUENCE_CHUNK_EPOCHS,
):
    """
    Minibatch SGD for torus MDS using online updates over unique sampled pairs.

    This replaces the older with-replacement pair sampling

    Returns: (params, alpha, r0, r1)
    """
    data = np.asarray(data, dtype=np.float64)
    n = data.shape[0]
    params = np.random.RandomState(seed).rand(n, 2).astype(np.float64, copy=False)
    alpha = float(alpha_init)
    r0 = float(r0_init)
    r1 = float(r1_init)
    if max_iters <= 0 or n < 2:
        return params, alpha, r0, r1

    rng = np.random.default_rng(seed)
    pairs = _all_unique_pairs(n)
    start_iter = 0
    while start_iter < max_iters:
        epochs = min(chunk_epochs, max_iters - start_iter)
        sequence = _build_sampled_unique_pair_sequence_from_pairs(pairs, rng, epochs, batch_pairs)
        params, alpha, r0, r1 = _run_pair_sequence_online_njit(
            data=data,
            pair_sequence=sequence,
            learning_rate=learning_rate,
            init_params=params,
            start_iter=start_iter,
            alpha_init=alpha,
            alpha_ema=alpha_ema,
            eps=eps,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            r0_init=r0,
            r1_init=r1,
            learn_mode=learn_mode,
            geom_lr=geom_lr,
            geom_min=geom_min,
            theta=theta,
        )
        start_iter += epochs
    return params, alpha, r0, r1


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

    learning_rate = 1

    def stochastic_gradient_descent(
            self,
            max_iters=10000,
            batch_size=4096,
            data=None,
            seed=0,
            alpha_init=1.0,
            r0_init=1.0,
            r1_init=1.0,
            learn_mode='alpha',   # 'fixed' | 'alpha' | 'square' | 'rectangular' | 'alpha_aspect'
            geom_lr=0.01,
            sampled_unique=True,
            theta=90.0,           # torus angle in degrees; must be in [60, 120]
    ):
        if not (60.0 <= theta <= 120.0):
            raise ValueError(f"theta must be in [60, 120] degrees, got {theta}")
        if theta != 90.0 and learn_mode in ('rectangular', 'alpha_aspect'):
            raise ValueError(
                "theta != 90 requires equal side lengths; use learn_mode='alpha' or 'square'"
            )
        theta_rad = float(np.radians(theta))
        mode_int = {'fixed': 0, 'alpha': 1, 'square': 2, 'rectangular': 3, 'alpha_aspect': 4}[learn_mode]

        if sampled_unique:
            sgd_fn = sgd_minibatch_njit # sample pairs without replacement
        else:
            sgd_fn = _sgd_minibatch_njit_legacy # sample pairs with replacement

        coords, alpha, r0, r1 = sgd_fn(
            data=data,
            learning_rate=self.learning_rate,
            max_iters=max_iters,
            batch_pairs=batch_size,
            seed=seed,
            alpha_init=alpha_init,
            r0_init=r0_init,
            r1_init=r1_init,
            learn_mode=mode_int,
            geom_lr=geom_lr,
            theta=theta_rad,
        )
        self.alpha_ = float(alpha)
        self.r0_ = float(r0)
        self.r1_ = float(r1)
        self.theta_ = theta_rad
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
