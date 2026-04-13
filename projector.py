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
        self.torus_embedding_ = X % 1.0
        return self.torus_embedding_

    def fit(self, D: np.ndarray, **kwargs) -> "TorusProjector":
        self.fit_transform(D, **kwargs)
        return self

    def transform(self, D: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Most manifold methods aren't inductive with precomputed D. Use fit_transform().")

    # --- methods for subclasses to override ---
    def _embed(self, D: np.ndarray, **kwargs) -> np.ndarray:
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
def stress_and_grad_rect_torus(p1, p2, d, alpha, r0, r1, eps=1e-12, theta=np.pi/2):
    """
    Consistent model:
      u in [0,1)^2  (parameter space)
      du = min-image on unit torus, wrapped to [-0.5, 0.5)
      For a rectangular flat torus with side lengths r0, r1:
        geodesic offset: (r0*du0, r1*du1)
        r_rect = ||(r0*du0, r1*du1)||
        norm = alpha * r_rect

    theta (radians): torus angle. Default pi/2 = rectangular.
      For theta != pi/2 (rhombic, r0==r1 enforced by caller):
        min-image selected via 4-way check on q(u,v) = u^2 + 2*cos(theta)*u*v + v^2.
        Exact for |cos(theta)| <= 0.5, i.e. theta in [pi/3, 2*pi/3] (60-120 deg).

    Loss: (norm - d)^2

    Returns: (loss, grad_p1, r_rect, grad_r0, grad_r1)
      grad_p1  — dLoss/dp1 (apply antisymmetrically to p2)
      grad_r0  — dLoss/dr0  (used when learning side lengths)
      grad_r1  — dLoss/dr1
    """
    cos_theta = np.cos(theta)
    du = p2 - p1
    a = (du[0] + 0.5) - np.floor(du[0] + 0.5) - 0.5  # wrap to [-0.5, 0.5)
    b = (du[1] + 0.5) - np.floor(du[1] + 0.5) - 0.5
    # 4-way minimum-image check on skewed metric q(u,v) = u^2+2*cos_theta*u*v+v^2
    # (for theta=pi/2, cos_theta=0 so cross term vanishes and (a,b) always wins)
    a1 = a - 1.0 if a >= 0.0 else a + 1.0
    b1 = b - 1.0 if b >= 0.0 else b + 1.0
    q_best = a*a + 2.0*cos_theta*a*b + b*b
    du0 = a
    du1 = b
    q = a1*a1 + 2.0*cos_theta*a1*b + b*b
    if q < q_best:
        q_best = q
        du0 = a1
        du1 = b
    q = a*a + 2.0*cos_theta*a*b1 + b1*b1
    if q < q_best:
        q_best = q
        du0 = a
        du1 = b1
    q = a1*a1 + 2.0*cos_theta*a1*b1 + b1*b1
    if q < q_best:
        du0 = a1
        du1 = b1

    r2 = (r0 * du0) * (r0 * du0) + (r1 * du1) * (r1 * du1)
    r_rect = np.sqrt(r2) + eps

    norm = alpha * r_rect
    diff = norm - d

    # --- gradient w.r.t. positions ---
    # d(r_rect)/dp1_k carries a factor -rk^2 * duk / r_rect  (d(duk)/dp1_k = -1)
    scale = -2.0 * diff * alpha / r_rect
    g0 = scale * r0 * r0 * du0
    g1 = scale * r1 * r1 * du1

    # --- gradient w.r.t. side lengths ---
    # d(r_rect)/dr0 = r0 * du0^2 / r_rect
    # d(norm)/dr0   = alpha * r0 * du0^2 / r_rect
    # dLoss/dr0     = 2 * diff * alpha * r0 * du0^2 / r_rect
    gr0 = 2.0 * diff * alpha * r0 * du0 * du0 / r_rect
    gr1 = 2.0 * diff * alpha * r1 * du1 * du1 / r_rect

    return diff * diff, np.array((g0, g1), dtype=p1.dtype), r_rect, gr0, gr1


@numba.njit(cache=True, fastmath=True)
def sgd_minibatch_njit(
    data,
    learning_rate,
    max_iters=500,
    batch_pairs=4096,
    seed=0,
    alpha_init=1.0,
    alpha_ema=0.05,   # EMA weight for alpha update (learn_mode=0 only)
    eps=1e-12,
    alpha_min=1e-6,
    alpha_max=1e6,
    r0_init=1.0,
    r1_init=1.0,
    learn_mode=0,     # 0=fixed (geometry frozen), 1=alpha, 2=square (r0=r1), 3=rectangular (r0,r1 independent)
    geom_lr=0.01,     # SGD step size for side lengths (learn_mode 1 or 2)
    geom_min=1e-6,    # lower bound on r0, r1
    theta=np.pi/2,    # torus angle in radians; pi/2 = rectangular
):
    """
    Minibatch SGD for torus MDS on a rectangular flat torus.

    learn_mode=0 ('fixed'):
        Positions optimised only; alpha, r0, r1 all fixed.

    learn_mode=1 ('alpha'):
        Positions and alpha optimised; r0, r1 fixed.
        Alpha update: closed-form batch minimiser + EMA.

    learn_mode=2 ('square'):
        Positions and shared r = r0 = r1 optimised via SGD; alpha fixed.
        Gradient: dLoss/dr = dLoss/dr0 + dLoss/dr1  (chain rule, not double-counting).

    learn_mode=3 ('rectangular'):
        Positions, r0, r1 optimised independently via SGD; alpha fixed.

    Returns: (params, alpha, r0, r1)
    """
    n = data.shape[0]
    np.random.seed(seed)

    params = np.random.rand(n, 2)
    alpha = alpha_init
    r0 = r0_init
    r1 = r1_init

    for it in range(max_iters):
        step_pos = 1.0 / (learning_rate + it)
        if step_pos < 1e-4:
            step_pos = 1e-4

        num = 0.0      # sum d*dist  (for alpha closed-form)
        den = 0.0      # sum dist²   (for alpha closed-form)
        gr0_sum = 0.0  # accumulated dLoss/dr0
        gr1_sum = 0.0  # accumulated dLoss/dr1
        used = 0

        drawn = 0
        while drawn < batch_pairs:
            i = np.random.randint(0, n)
            j = np.random.randint(0, n)
            if i == j:
                continue

            d = data[i, j]
            _, grad, dist, gr0_k, gr1_k = stress_and_grad_rect_torus(
                params[i], params[j], d, alpha, r0, r1, eps, theta
            )

            # position update (antisymmetric)
            params[i, 0] -= step_pos * grad[0]
            params[i, 1] -= step_pos * grad[1]
            params[j, 0] += step_pos * grad[0]
            params[j, 1] += step_pos * grad[1]

            num += d * dist
            den += dist * dist
            gr0_sum += gr0_k
            gr1_sum += gr1_k
            used += 1
            drawn += 1

        # wrap once per outer iteration
        params %= 1.0

        if used > 0:
            if learn_mode == 1:
                # closed-form batch minimiser + EMA  ('alpha')
                alpha_hat = num / (den + eps)
                if alpha_hat < alpha_min:
                    alpha_hat = alpha_min
                elif alpha_hat > alpha_max:
                    alpha_hat = alpha_max
                alpha = (1.0 - alpha_ema) * alpha + alpha_ema * alpha_hat
            elif learn_mode == 2:
                # square: single shared r = r0 = r1
                # total dLoss/dr = dLoss/dr0 + dLoss/dr1 (chain rule)
                r = r0 - geom_lr * (gr0_sum + gr1_sum) / used
                if r < geom_min:
                    r = geom_min
                r0 = r
                r1 = r
            elif learn_mode == 3:
                # rectangular: independent r0, r1
                r0 -= geom_lr * gr0_sum / used
                r1 -= geom_lr * gr1_sum / used
                if r0 < geom_min:
                    r0 = geom_min
                if r1 < geom_min:
                    r1 = geom_min
            # learn_mode == 0 ('fixed'): geometry unchanged, nothing to do

    return params, alpha, r0, r1

@dataclass
class MDSTorusProjector(TorusProjector):

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
            learn_mode='alpha',  # 'fixed' | 'alpha' | 'square' | 'rectangular'
            geom_lr=0.01,
            theta=90.0,          # torus angle in degrees; must be in [60, 120]
        ):
        if not (60.0 <= theta <= 120.0):
            raise ValueError(f"theta must be in [60, 120] degrees, got {theta}")
        if theta != 90.0 and learn_mode == 'rectangular':
            raise ValueError(
                "theta != 90 requires equal side lengths; use learn_mode='alpha' or 'square'"
            )
        theta_rad = float(np.radians(theta))
        # numba (nopython mode) does not support string arguments, so convert here
        mode_int = {'fixed': 0, 'alpha': 1, 'square': 2, 'rectangular': 3}[learn_mode]
        coords, alpha, r0, r1 = sgd_minibatch_njit(
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
    # sklearn TSNE knobs (metric="precomputed")
    perplexity: float = 30.0
    n_iter: int = 1000
    init: Literal["random", "pca"] = "random"  # pca not allowed with precomputed in some sklearn versions
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
    # umap-learn knobs (metric="precomputed")
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


if __name__ == "__main__":
    pass
