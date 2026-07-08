import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from scipy.linalg import expm
from scipy.optimize import minimize
import types


def lowest_laplacian_eigenvalues(
    D,
    k=15,
    sigma=None,
    n_eigs=6,
    normalized=True,
):
    """
    Compute the lowest eigenvalues of a symmetrized graph Laplacian
    from a pairwise distance matrix.

    Parameters
    ----------
    D : (n, n) array
        Pairwise distance matrix.
    k : int
        Number of nearest neighbors per point.
    sigma : float or None
        Gaussian kernel bandwidth. If None, uses median kNN distance.
    n_eigs : int
        Number of lowest eigenvalues to compute.
    normalized : bool
        If True, use symmetric normalized Laplacian:
            L = I - D^{-1/2} W D^{-1/2}
        If False, use unnormalized Laplacian:
            L = D - W

    Returns
    -------
    evals : (n_eigs,) array
        Lowest eigenvalues.
    evecs : (n, n_eigs) array
        Corresponding eigenvectors.
    """

    D = np.asarray(D)
    n = D.shape[0]

    if D.shape != (n, n):
        raise ValueError("D must be a square distance matrix.")

    if sigma is None:
        k = min(k, n-1)
        knn_dists = np.partition(D, kth=k, axis=1)[:, 1:k + 1]
        sigma = np.median(knn_dists)

    # kNN indices, excluding self
    nn_idx = np.argpartition(D, kth=k, axis=1)[:, 1:k + 1]

    rows = np.repeat(np.arange(n), k)
    cols = nn_idx.ravel()
    vals = np.exp(-(D[rows, cols] ** 2) / (2 * sigma**2))

    W = csr_matrix((vals, (rows, cols)), shape=(n, n))

    # Symmetrize graph, but D should already be symmetric
    W = W.maximum(W.T)

    degrees = np.asarray(W.sum(axis=1)).ravel()

    if normalized:
        inv_sqrt_deg = np.zeros_like(degrees)
        mask = degrees > 0
        inv_sqrt_deg[mask] = 1.0 / np.sqrt(degrees[mask])

        D_inv_sqrt = diags(inv_sqrt_deg)
        L = diags(np.ones(n)) - D_inv_sqrt @ W @ D_inv_sqrt
    else:
        L = diags(degrees) - W

    evals, evecs = eigsh(L, k=n_eigs, which="SM")

    order = np.argsort(evals)
    return evals[order], evecs[:, order]


# ── helpers: phase extraction, harmonic scoring, initialisation ──────────────

def _phase_from_pair(evec_pair, eps=1e-12):
    """
    Convert a 2D eigenvector pair into circular phases.

    Parameters
    ----------
    evec_pair : (n, 2) array

    Returns
    -------
    z : (n,) complex array
        Unit-modulus complex phase representation.
    theta : (n,) array
        Angles in [-pi, pi].
    """
    u = evec_pair[:, 0]
    v = evec_pair[:, 1]

    u = u - np.mean(u)
    v = v - np.mean(v)

    z = u + 1j * v
    z = z / np.maximum(np.abs(z), eps)

    theta = np.angle(z)
    return z, theta


def torus_init_from_eigenpairs(pair1, pair2):
    _, theta1 = _phase_from_pair(pair1)
    _, theta2 = _phase_from_pair(pair2)

    x = np.mod(theta1, 2 * np.pi) / (2 * np.pi)
    y = np.mod(theta2, 2 * np.pi) / (2 * np.pi)

    return np.column_stack([x, y])


def harmonic_score(z_candidate, z_reference, m):
    """
    Score whether candidate phase is approximately an m-th harmonic
    of the reference phase.  Score is in [0, 1].
    """
    score_pos = np.abs(np.mean(z_candidate * np.conj(z_reference ** m)))
    score_neg = np.abs(np.mean(z_candidate * (z_reference ** m)))
    return max(score_pos, score_neg)


# ── helpers for degenerate eigenspaces ───────────────────────────────────────

def _group_eigenvectors(evals, evecs, degenerate_tol=0.2):
    """
    Partition consecutive eigenvectors into groups of size 2 or 4.

    Two consecutive size-2 pairs are merged into a 4-group when their mean
    eigenvalues differ by less than degenerate_tol (relative).  This detects
    the two cases where the standard pair-wise approach breaks:

      * aspect_ratio ≈ integer  — the k-th harmonic of direction 0 and the
        independent direction 1 land at the same eigenvalue.
      * aspect_ratio ≈ 1        — both fundamental directions share the first
        non-zero eigenspace.

    Returns a list of dicts with keys: size, eigenvalues, eigenvectors, lambda_mean.
    """
    groups = []
    j = 0
    n = len(evals)
    while j + 1 < n:
        lam_a = float(np.mean(evals[j:j + 2]))
        if j + 4 <= n:
            lam_b = float(np.mean(evals[j + 2:j + 4]))
            if abs(lam_b - lam_a) / max(lam_a, 1e-12) < degenerate_tol:
                groups.append({
                    "size": 4,
                    "eigenvalues": evals[j:j + 4],
                    "eigenvectors": evecs[:, j:j + 4],
                    "lambda_mean": float(np.mean(evals[j:j + 4])),
                })
                j += 4
                continue
        groups.append({
            "size": 2,
            "eigenvalues": evals[j:j + 2],
            "eigenvectors": evecs[:, j:j + 2],
            "lambda_mean": lam_a,
        })
        j += 2
    return groups


def _split_by_harmonic_projection(V, z_ref, m):
    """
    Split 4D subspace V into harmonic and independent components.

    Uses z_ref^m as a fingerprint for the m-th harmonic: projects its real
    and imaginary parts onto V, giving a (4, 2) coefficient matrix C.  The
    SVD of C identifies the 2D sub-direction in V that best represents the
    harmonic; the orthogonal complement is the independent direction.

    Parameters
    ----------
    V     : (N, 4) orthonormal basis of a degenerate eigenspace
    z_ref : (N,) complex, unit-modulus reference direction
    m     : int, harmonic order

    Returns
    -------
    harmonic_basis    : (N, 2)
    independent_basis : (N, 2)
    score             : float in [0, 1]
        Fraction of ||z_ref^m|| captured by V.  Near 1 means the harmonic
        truly lives in this 4D subspace.
    """
    h = z_ref ** m
    h_re = np.real(h)
    h_im = np.imag(h)

    C = np.column_stack([V.T @ h_re, V.T @ h_im])  # (4, 2)
    U, s, _ = np.linalg.svd(C, full_matrices=True)

    h_norm = np.sqrt(np.dot(h_re, h_re) + np.dot(h_im, h_im))
    score = float(np.sqrt(s[0] ** 2 + s[1] ** 2) / max(h_norm, 1e-12))

    return V @ U[:, :2], V @ U[:, 2:], score


def _split_by_flatness(V, n_starts=10, seed=0):
    """
    Find the 2+2 split of 4D subspace V minimising Var(|z1|^2) + Var(|z2|^2),
    where |z|^2 = u^2 + v^2 for a pair (u, v).

    For a correct torus pair {cos(ωx), sin(ωx)}, |z|^2 = 1 everywhere (flat).
    For a mixed pair the modulus varies, yielding a higher objective.  Used when
    no prior reference direction exists (aspect_ratio ≈ 1 case).

    Optimises over O(4) via the matrix exponential of a skew-symmetric matrix
    (6 free parameters).  Uses n_starts random initialisations.

    Returns
    -------
    pair1 : (N, 2)
    pair2 : (N, 2)
    """


    rng = np.random.default_rng(seed)

    def _skew(a):
        return np.array([
            [ 0,   -a[0], -a[1], -a[2]],
            [a[0],  0,    -a[3], -a[4]],
            [a[1],  a[3],  0,    -a[5]],
            [a[2],  a[4],  a[5],  0   ],
        ])

    def objective(angles):
        B = V @ expm(_skew(angles))
        return float(np.var(B[:, 0] ** 2 + B[:, 1] ** 2)
                     + np.var(B[:, 2] ** 2 + B[:, 3] ** 2))

    best_val, best_angles = np.inf, np.zeros(6)
    for _ in range(n_starts):
        x0 = rng.standard_normal(6) * 0.3
        res = minimize(objective, x0, method='L-BFGS-B')
        if res.fun < best_val:
            best_val, best_angles = res.fun, res.x

    B = V @ expm(_skew(best_angles))
    return B[:, :2], B[:, 2:]


# ── main function ─────────────────────────────────────────────────────────────

def find_fundamental_torus_directions(
    D,
    k=15,
    sigma=None,
    normalized=True,
    n_eigs=20,
    zero_tol=1e-8,
    harmonic_threshold=0.75,
    max_harmonic=8,
    degenerate_tol=0.2,
):
    """
    Find the first two independent torus directions from graph Laplacian
    eigenvectors, skipping harmonics and handling degenerate eigenspaces.

    Consecutive eigenvector pairs with nearly equal eigenvalues are merged into
    a 4-group (controlled by degenerate_tol).  Each 4-group is split into its
    harmonic and independent components via one of two strategies:

      * No prior reference (aspect_ratio ≈ 1): split by minimising Var(|z|^2)
        over all 2+2 partitions of the 4D space (_split_by_flatness).
      * Prior reference exists: project the m-th harmonic of the reference
        into the 4D space and use the SVD to separate harmonic from independent
        component (_split_by_harmonic_projection).

    Parameters
    ----------
    D                : (n, n) distance matrix
    degenerate_tol   : float
        Relative eigenvalue tolerance for merging two consecutive pairs into a
        4-group.  Default 0.2 (20 %).
    (all other parameters as in the previous version
    n_eigs          : int number of eigenvalues computed, (n_eig-1)// 2 is an upper bound on the detectable aspect ratio
    max
    Returns
    -------
    result : dict with keys:
        aspect_ratio, all_eigenvalues, nonzero_eigenvalues,
        fundamental_directions, skipped_harmonics.
        Each entry in fundamental_directions also contains side_length:
        the estimated circumference of that torus direction in units of D,
        derived from lambda = (2pi/L)^2  =>  L = 2pi / sqrt(lambda).
    """
    evals, evecs = lowest_laplacian_eigenvalues(
        D, k=k, sigma=sigma, n_eigs=n_eigs, normalized=normalized,
    )

    nz_idx = np.flatnonzero(evals > zero_tol)
    if len(nz_idx) < 4:
        raise ValueError("Need at least four nonzero eigenvectors.")

    evals_nz = evals[nz_idx]
    evecs_nz = evecs[:, nz_idx]

    groups = _group_eigenvectors(evals_nz, evecs_nz, degenerate_tol)

    accepted = []
    skipped  = []

    def _record(eigenvectors, lambda_mean, eigenvalues, z, theta):
        return {
            "eigenvalues":     eigenvalues,
            "lambda_mean":     lambda_mean,
            "lambda_splitting": float(abs(eigenvalues[-1] - eigenvalues[0])),
            "eigenvectors":    eigenvectors,
            "z":               z,
            "theta":           theta,
        }

    for group in groups:
        V       = group["eigenvectors"]
        evals_g = group["eigenvalues"]
        lmean   = group["lambda_mean"]

        if group["size"] == 2:
            z, theta = _phase_from_pair(V)
            is_harmonic = False
            for ref_id, ref in enumerate(accepted):
                scores    = [harmonic_score(z, ref["z"], m) for m in range(1, max_harmonic + 1)]
                best_m    = int(np.argmax(scores) + 1)
                best_scr  = float(np.max(scores))
                if best_scr >= harmonic_threshold:
                    is_harmonic = True
                    skipped.append({
                        "eigenvalues": evals_g, "lambda_mean": lmean,
                        "matched_to": {"reference_direction": ref_id,
                                       "harmonic": best_m, "score": best_scr},
                    })
                    break
            if not is_harmonic:
                accepted.append(_record(V, lmean, evals_g, z, theta))

        elif group["size"] == 4:
            if len(accepted) == 0:
                # No reference yet (aspect_ratio ≈ 1): split by modulus flatness.
                pair1, pair2 = _split_by_flatness(V)
                z1, theta1   = _phase_from_pair(pair1)
                z2, theta2   = _phase_from_pair(pair2)
                accepted.append(_record(pair1, lmean, evals_g, z1, theta1))
                accepted.append(_record(pair2, lmean, evals_g, z2, theta2))

            else:
                # Find the accepted direction and harmonic order whose fingerprint
                # best projects into this 4D subspace.
                best_scr_all  = -1.0
                best_h = best_ind = best_m_all = best_ref_id = None

                for ref_id, ref in enumerate(accepted):
                    for m in range(2, max_harmonic + 1):
                        h_basis, ind_basis, score = _split_by_harmonic_projection(
                            V, ref["z"], m)
                        if score > best_scr_all:
                            best_scr_all              = score
                            best_h, best_ind          = h_basis, ind_basis
                            best_m_all, best_ref_id   = m, ref_id

                if best_scr_all >= harmonic_threshold:
                    skipped.append({
                        "eigenvalues": evals_g, "lambda_mean": lmean,
                        "matched_to": {"reference_direction": best_ref_id,
                                       "harmonic": best_m_all, "score": best_scr_all},
                    })
                    # Check whether the independent component is itself a harmonic.
                    z_ind, theta_ind = _phase_from_pair(best_ind)
                    is_harmonic2 = False
                    for ref_id, ref in enumerate(accepted):
                        scores2   = [harmonic_score(z_ind, ref["z"], m)
                                     for m in range(1, max_harmonic + 1)]
                        best_m2   = int(np.argmax(scores2) + 1)
                        best_scr2 = float(np.max(scores2))
                        if best_scr2 >= harmonic_threshold:
                            is_harmonic2 = True
                            skipped.append({
                                "eigenvalues": evals_g, "lambda_mean": lmean,
                                "matched_to": {"reference_direction": ref_id,
                                               "harmonic": best_m2, "score": best_scr2},
                            })
                            break
                    if not is_harmonic2:
                        accepted.append(
                            _record(best_ind, lmean, evals_g, z_ind, theta_ind))

                else:
                    # No clear harmonic match; fall back to flatness split and
                    # take whichever of the two resulting pairs is flatter.
                    pair1, pair2 = _split_by_flatness(V)
                    z1, theta1   = _phase_from_pair(pair1)
                    z2, theta2   = _phase_from_pair(pair2)
                    var1 = float(np.var(pair1[:, 0] ** 2 + pair1[:, 1] ** 2))
                    var2 = float(np.var(pair2[:, 0] ** 2 + pair2[:, 1] ** 2))
                    if var1 <= var2:
                        accepted.append(_record(pair1, lmean, evals_g, z1, theta1))
                    else:
                        accepted.append(_record(pair2, lmean, evals_g, z2, theta2))

        if len(accepted) == 2:
            break

    if len(accepted) < 2:
        raise ValueError(
            "Could not identify two independent torus directions. "
            "Try increasing n_eigs or max_harmonic, or lowering harmonic_threshold."
        )

    lambda_1     = accepted[0]["lambda_mean"]
    lambda_2     = accepted[1]["lambda_mean"]
    aspect_ratio = np.sqrt(lambda_2 / lambda_1)

    # Spectral side-length estimate: from the continuous-Laplacian identity
    # lambda = (2pi/L)^2  =>  L = 2pi / sqrt(lambda).
    # Gives lengths in normalised units consistent with the distance matrix D.
    return {
        "aspect_ratio":        aspect_ratio,
        "all_eigenvalues":     evals,
        "nonzero_eigenvalues": evals_nz,
        "fundamental_directions": [
            {
                "lambda_mean":     a["lambda_mean"],
                "eigenvalues":     a["eigenvalues"],
                "lambda_splitting": a["lambda_splitting"],
                "theta":           a["theta"],
                "eigenvectors":    a["eigenvectors"],
                "side_length":     2 * np.pi / np.sqrt(max(a["lambda_mean"], 1e-12)),
            }
            for a in accepted
        ],
        "skipped_harmonics": skipped,
    }



def _torus_geom_from_result(result):
    """Build a minimal torus geometry object for plot_embedding_with_torus_edges."""
    d = result["fundamental_directions"]
    return types.SimpleNamespace(
        torus_embedding_=torus_init_from_eigenpairs(d[0]["eigenvectors"], d[1]["eigenvectors"]),
        alpha_=1.0,
        r0_=d[0]["side_length"],
        r1_=d[1]["side_length"],
        theta_=np.pi / 2,
    )