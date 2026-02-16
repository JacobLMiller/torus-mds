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

@numba.njit(fastmath=True)
def stress_and_grad(p1,p2,d):
    delta = ((p2 - p1 + 0.5) % 1.0) - 0.5
    norm = np.sqrt(np.sum(np.square(delta)))

    if norm == 0.0: grad = np.zeros_like(p1)
    else: 
        grad = 2 * (norm - d) * (-delta / norm)
    
    return (norm - d)**2, grad

@dataclass
class MDSTorusProjector(TorusProjector):

    learning_rate = 1

    def stochastic_gradient_descent(self, 
                                max_iters=500, batch_size=1, data=None, 
                                tolerance=1e-6, verbose=False,colors=None):

        from itertools import combinations
        import tqdm

        lr = self.learning_rate

        n = data.shape[0]
        params = np.random.uniform(0,1,(n,2))

        pairs = np.array(list(combinations(range(n), 2)))


        for ind in tqdm.tqdm(range(max_iters)):
            np.random.shuffle(pairs)
            for i,j in pairs:
                stress, grad = stress_and_grad(params[i], params[j], data[i,j])
                update = (1/(lr+ind)) * grad
                params[i] -= update 
                params[j] += update
                params[i] %= 1.0
                params[j] %= 1.0
                
        return params
    
    def _embed(self, D: np.ndarray) -> np.ndarray:
        from sklearn.manifold import MDS

        X = self.stochastic_gradient_descent(data=D)
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
