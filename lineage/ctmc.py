r"""Continuous-time Markov chain transitions on branch-length trees.

Reconstructed phylogenies (e.g. Cassiopeia trees from CRISPR recorders) have edges that span an
unknown number of divisions, so each edge e gets its own transition matrix
:math:`T(t_e) = \exp(Q t_e)` from a shared rate matrix :math:`Q`.

The M-step for :math:`Q` uses the endpoint-conditioned expected sufficient statistics of the
chain, following Hobolth & Jensen (2011, "Summary statistics for endpoint-conditioned
continuous-time Markov chains"). With the joint posterior :math:`\xi_e(a, b)` of the states at
both ends of each edge, the expected time spent in state i, :math:`R_i`, and the expected number
of jumps from i to j, :math:`N_{ij}`, are

.. math::

    E[R_i] = \sum_e \sum_{ab} \frac{\xi_e(a, b)}{P_{ab}(t_e)} I^{ii}_{ab}(t_e), \qquad
    E[N_{ij}] = q_{ij} \sum_e \sum_{ab} \frac{\xi_e(a, b)}{P_{ab}(t_e)} I^{ij}_{ab}(t_e),

with :math:`I^{ij}(t) = \int_0^t e^{Qs} e_i e_j^\top e^{Q(t-s)} ds`, and the update is
:math:`q_{ij} = E[N_{ij}] / E[R_i]`. With the eigendecomposition :math:`Q = U \Lambda V`,
:math:`V = U^{-1}`, both sums collapse to :math:`(V^\top A U^\top)_{ij}` for
:math:`A = \sum_e (U^\top W_e V^\top) \circ J(t_e)`, :math:`W_e = \xi_e / P(t_e)` and
:math:`J_{cd}(t) = \int_0^t e^{\lambda_c s} e^{\lambda_d (t - s)} ds`. This costs one
eigendecomposition and O(K^2) work per edge.
"""

import numpy as np


def rates_to_Q(rates: np.ndarray) -> np.ndarray:
    """Build a rate matrix from its off-diagonal rates (the diagonal of ``rates`` is ignored)."""
    Q = np.array(rates, dtype=float)
    np.fill_diagonal(Q, 0.0)
    np.fill_diagonal(Q, -Q.sum(axis=1))
    return Q


def random_Q(K: int, total_rate: float = 1.0, rng=None) -> np.ndarray:
    """Random rate matrix whose rows each leave their state at ``total_rate``."""
    rng = np.random.default_rng(rng)
    R = rng.dirichlet(np.ones(K - 1), size=K) * total_rate
    rates = np.zeros((K, K))
    for i in range(K):
        rates[i, np.arange(K) != i] = R[i]
    return rates_to_Q(rates)


def _eig(Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Eigendecomposition Q = U diag(lam) V, with a tiny perturbation if Q is (nearly) defective."""
    lam, U = np.linalg.eig(Q)
    if np.linalg.cond(U) > 1e8:
        # A defective or nearly-defective Q (e.g. repeated eigenvalues with a shared eigenvector)
        # has no stable eigenbasis. Splitting the eigenvalues slightly perturbs expm(Qt) by O(1e-9).
        jitter = np.diag(np.linspace(-1.0, 1.0, Q.shape[0])) * 1e-9 * max(1.0, np.abs(Q).max())
        lam, U = np.linalg.eig(Q + jitter)
    return lam, U, np.linalg.inv(U)


def transition_matrices(Q: np.ndarray, t: np.ndarray) -> np.ndarray:
    """:math:`\\exp(Q t)` for every entry of ``t``, shape (len(t), K, K). Rows are clipped and renormalized."""
    t = np.asarray(t, dtype=float)
    lam, U, V = _eig(Q)
    expo = np.exp(np.outer(t, lam))  # (E, K)
    P = np.real(np.einsum("ac,ec,cb->eab", U, expo, V))
    P = np.clip(P, 0.0, None)
    P /= P.sum(axis=2, keepdims=True)
    return P


def _J(lam: np.ndarray, t: np.ndarray) -> np.ndarray:
    """:math:`J_{cd}(t) = \\int_0^t e^{\\lambda_c s} e^{\\lambda_d (t-s)} ds` for each t, shape (E, K, K)."""
    lc = lam[np.newaxis, :, np.newaxis]
    ld = lam[np.newaxis, np.newaxis, :]
    tt = t[:, np.newaxis, np.newaxis]
    diff = lc - ld
    close = np.abs(diff * tt) < 1e-8
    safe = np.where(close, 1.0, diff)
    # e^{ld t} (e^{(lc - ld) t} - 1) / (lc - ld), falling back to t e^{ld t} when lc ~ ld
    general = np.exp(ld * tt) * np.expm1(diff * tt) / safe
    limit = tt * np.exp(0.5 * (lc + ld) * tt)
    return np.where(close, limit, general)


def expected_statistics(Q: np.ndarray, t: np.ndarray, xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Expected dwell times and jump counts given edge-endpoint posteriors.

    :param Q: current rate matrix (K by K)
    :param t: edge lengths (E)
    :param xi: joint posteriors of the parent and child states for each edge (E by K by K)
    :return: expected time in each state (K) and expected number of i -> j jumps (K by K, zero diagonal)
    """
    t = np.asarray(t, dtype=float)
    keep = t > 0  # zero-length edges carry no time and no jumps
    t, xi = t[keep], xi[keep]
    K = Q.shape[0]
    if t.size == 0:
        return np.zeros(K), np.zeros((K, K))

    lam, U, V = _eig(Q)
    P = transition_matrices(Q, t)
    W = xi / np.maximum(P, 1e-300)
    A = np.einsum("ac,eab,db->ecd", U, W, V) * _J(lam, t)
    S = np.real(V.T @ A.sum(axis=0) @ U.T)

    dwell = np.clip(np.diag(S).copy(), 0.0, None)
    jumps = np.clip(Q * S, 0.0, None)
    np.fill_diagonal(jumps, 0.0)
    return dwell, jumps


def M_step_Q(
    Q: np.ndarray, t: np.ndarray, xi: np.ndarray, pseudo_jumps: float = 1e-2, pseudo_time: float = 1e-2
) -> np.ndarray:
    """One EM update of the rate matrix. Small pseudocounts keep every rate strictly positive."""
    dwell, jumps = expected_statistics(Q, t, xi)
    K = Q.shape[0]
    rates = (jumps + pseudo_jumps / (K - 1)) / (dwell + pseudo_time)[:, np.newaxis]
    return rates_to_Q(rates)


def stationary(Q: np.ndarray) -> np.ndarray:
    """Stationary distribution of the chain, solving pi Q = 0."""
    K = Q.shape[0]
    A = np.vstack([Q.T, np.ones(K)])
    b = np.zeros(K + 1)
    b[-1] = 1.0
    pi = np.linalg.lstsq(A, b, rcond=None)[0]
    pi = np.clip(pi, 0.0, None)
    return pi / pi.sum()
