from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal, Tuple

import numpy as np 
import numba 

@numba.njit(fastmath=True)
def torus(x,y):
    result = 0.0
    for i in range(x.shape[0]):
        result += (((x[i] - y[i] + 0.5) % 1.0) - 0.5) ** 2

    return np.sqrt(result)

def torus_distance(p1,p2):
    delta = (p2 - p1 + 0.5) % 1.0 - 0.5
    return np.linalg.norm(delta)

@numba.njit(fastmath=True)
def torus_grad(x,y):
    delta = ((x - y + 0.5) % 1.0 - 0.5)
    result = 0.0 
    for i in range(x.shape[0]):
        result += delta[i] ** 2
    norm = np.sqrt(result)
    return norm, delta / (1e-6 + norm)

def torus_grad_old(p1,p2):
    delta = (p2 - p1 + 0.5) % 1.0 - 0.5 
    norm = np.linalg.norm(delta)
    if norm == 0.0: return np.zeros_like(p1)
    return -delta / norm

@numba.njit(fastmath=True)
def euc_grad(x,y):
    delta = x - y
    result = 0.0 
    for i in range(x.shape[0]):
        result += delta[i] ** 2
    norm = np.sqrt(result)
    return norm, delta / (1e-6 + norm)

@numba.njit(fastmath=True)
def euclidean_grad(x, y):
    """Standard euclidean distance and its gradient.

    ..math::
        D(x, y) = \sqrt{\sum_i (x_i - y_i)^2}
        \frac{dD(x, y)}{dx} = (x_i - y_i)/D(x,y)
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += (x[i] - y[i]) ** 2
    d = np.sqrt(result)
    grad = (x - y) / (1e-6 + d)
    return d, grad

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
    """
    Robustly affine-map each axis to [0,1] using quantiles, then clip.
    This makes modulo projection meaningful even if the embedding scale is arbitrary.
    """
    lo = np.quantile(X, q[0], axis=0)
    hi = np.quantile(X, q[1], axis=0)
    span = np.maximum(hi - lo, eps)
    Y = (X - lo) / span
    return np.clip(Y, 0.0, 1.0)


def _wrap_unit_square(X: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=np.float64) % 1.0


@dataclass
class TorusProjector:
    """
    Boilerplate base class:
      - accepts a precomputed distance matrix D (NxN)
      - runs an underlying embedding algorithm to produce Euclidean coords
      - projects/wraps the output to a square torus [0,1)^2

    Subclasses implement _embed(D) -> (N,2) or (N,k).

    Projection modes:
      - "wrap": just X % 1 (assumes embedding already on unit scale)
      - "robust_wrap": robust affine to [0,1] per axis, then wrap (recommended)
    """
    n_components: int = 2
    projection: Literal["wrap", "robust_wrap"] = "robust_wrap"
    random_state: Optional[int] = 42

    embedding_: Optional[np.ndarray] = field(default=None, init=False)
    torus_embedding_: Optional[np.ndarray] = field(default=None, init=False)

    def fit_transform(self, D: np.ndarray) -> np.ndarray:
        D = _check_distance_matrix(D)
        X = self._embed(D)
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] != self.n_components:
            raise ValueError(f"Expected embedding with {self.n_components} dims, got {X.shape}.")

        self.embedding_ = X
        self.torus_embedding_ = X % 1.0
        return self.torus_embedding_

    def fit(self, D: np.ndarray) -> "TorusProjector":
        self.fit_transform(D)
        return self

    def transform(self, D: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Most manifold methods aren't inductive with precomputed D. Use fit_transform().")

    # --- methods for subclasses to override ---
    def _embed(self, D: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # --- shared projection ---
    def _project_to_torus(self, X: np.ndarray) -> np.ndarray:
        if self.projection == "wrap":
            return _wrap_unit_square(X)
        elif self.projection == "robust_wrap":
            return _wrap_unit_square(_robust_affine_to_unit_square(X))
        else:
            raise ValueError(f"Unknown projection: {self.projection}")


# -------------------- Subclasses --------------------

import numpy as np
import numba


@numba.njit(fastmath=True, cache=True)
def stress_and_grad_unit_torus(p1, p2, d, alpha, eps=1e-12):
    """
    Consistent model:
      u in [0,1)^2
      du = min-image on unit torus
      delta = alpha * du
      norm = ||delta|| = alpha * r, where r = ||du||

    Loss: (norm - d)^2
    Gradient returned is dLoss/dp1 (and you apply antisymmetrically to p2).
    """
    du = p2 - p1
    du0 = (du[0] + 0.5) - np.floor(du[0] + 0.5) - 0.5  # wrap to [-0.5, 0.5)
    du1 = (du[1] + 0.5) - np.floor(du[1] + 0.5) - 0.5

    r2 = du0 * du0 + du1 * du1
    r = np.sqrt(r2) + eps

    norm = alpha * r
    diff = norm - d

    # dLoss/dp1 = 2*diff * d(norm)/dp1
    # norm = alpha * r, r = ||du||, du = p2 - p1 => dr/dp1 = -(du/r)
    # => d(norm)/dp1 = alpha * (-(du/r))
    scale = -2.0 * diff * alpha / r

    g0 = scale * du0
    g1 = scale * du1

    return diff * diff, np.array((g0, g1), dtype=p1.dtype), r


@numba.njit(cache=True, fastmath=True)
def sgd_minibatch_njit(
    data,
    learning_rate,
    max_iters=500,
    batch_pairs=4096,
    seed=0,
    alpha_init=1.0,
    alpha_ema=0.05,   # 0=no update, 1=replace with batch optimum
    eps=1e-12,
    alpha_min=1e-6,
    alpha_max=1e6
):
    """
    Stable, consistent alpha update for the SAME objective used in stress_and_grad_unit_torus:

      Loss per pair: (alpha*r - d)^2

    For a fixed batch {r_k, d_k}, the minimizer in alpha is:
      alpha_hat = sum(d_k * r_k) / sum(r_k^2)

    We compute alpha_hat once per minibatch and EMA it.

    IMPORTANT: data[i,j] must be in the same units as (alpha * r). Since r <= ~0.707,
    alpha roughly scales data by ~1/r.
    """
    n = data.shape[0]
    np.random.seed(seed)

    params = np.random.rand(n, 2)
    alpha = alpha_init

    for it in range(max_iters):
        step_pos = 1.0 / (learning_rate + it)
        if step_pos < 1e-4:
            step_pos = 1e-4

        num = 0.0  # sum d*r
        den = 0.0  # sum r^2
        used = 0

        drawn = 0
        while drawn < batch_pairs:
            i = np.random.randint(0, n)
            j = np.random.randint(0, n)
            if i == j:
                continue

            d = data[i, j]

            # compute grad and r CONSISTENTLY with alpha and unit-torus geometry
            _, grad, r = stress_and_grad_unit_torus(params[i], params[j], d, alpha, eps)

            # position update (antisymmetric)
            params[i, 0] -= step_pos * grad[0]
            params[i, 1] -= step_pos * grad[1]
            params[j, 0] += step_pos * grad[0]
            params[j, 1] += step_pos * grad[1]

            # alpha batch statistics for the same loss (alpha*r - d)^2
            num += d * r
            den += r * r
            used += 1

            drawn += 1

        # wrap once per outer iteration
        params %= 1.0

        # stable alpha update (closed-form + EMA)
        if used > 0:
            alpha_hat = num / (den + eps)

            # hard bounds to prevent runaway from bad batches / bad data
            if alpha_hat < alpha_min:
                alpha_hat = alpha_min
            elif alpha_hat > alpha_max:
                alpha_hat = alpha_max

            alpha = (1.0 - alpha_ema) * alpha + alpha_ema * alpha_hat

    return params, alpha

@dataclass
class MDSTorusProjector(TorusProjector):

    learning_rate: float = 100.0

    alpha_: Optional[float] = field(default=None, init=False)

    def stochastic_gradient_descent(self,
                max_iters=10000,
                batch_size=4096,
                data=None,
                seed=0,
                alpha_init=1.0
            ):

        coords, alpha = sgd_minibatch_njit(
            data=data,
            learning_rate=self.learning_rate,
            max_iters=max_iters,
            batch_pairs=batch_size,
            seed=seed,
            alpha_init=alpha_init,
        )
        return coords, float(alpha)

    def _embed(self, D: np.ndarray) -> np.ndarray:
        X, alpha = self.stochastic_gradient_descent(data=D)
        self.alpha_ = alpha
        return X


@dataclass
class TSNETorusProjector(TorusProjector):
    # sklearn TSNE knobs (metric="precomputed")
    perplexity: float = 30.0
    n_iter: int = 1000
    init: Literal["random", "pca"] = "random"  # pca not allowed with precomputed in some sklearn versions
    learning_rate: Literal["auto"] | float = "auto"
    early_exaggeration: float = 12.0
    angle: float = 0.5
    method: Literal["barnes_hut", "exact"] = "barnes_hut"
    verbose: int = 0

    def _embed(self, D: np.ndarray) -> np.ndarray:
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
    # umap-learn knobs (metric="precomputed")
    n_neighbors: int = 15
    min_dist: float = 0.1
    spread: float = 1.0
    n_epochs: Optional[int] = None
    learning_rate: float = 1.0
    repulsion_strength: float = 1.0
    negative_sample_rate: int = 5
    init: Literal["spectral", "random"] = "spectral"

    def _embed(self, D: np.ndarray) -> np.ndarray:
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


if __name__ == "__main__":
    pass
