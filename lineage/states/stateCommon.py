"""Common utilities used between states regardless of distribution."""

import warnings
from ctypes import CFUNCTYPE, c_double
from typing import Literal

import numpy as np
import numpy.typing as npt
from numba import njit
from numba.extending import get_cython_function_address
from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.sparse import csr_array

arr_type = npt.NDArray[np.float64]


warnings.filterwarnings("ignore", message="Values in x were outside bounds")


def censor_lineage_gamma(
    tree: csr_array,
    obs: np.ndarray,
    states: np.ndarray,
    censor_condition: int,
    desired_experiment_time: float = 2e12,
) -> tuple[csr_array, np.ndarray, np.ndarray]:
    """Applies temporal and fate censorship to Gamma distribution lineages using arrays."""
    if censor_condition == 0:
        return tree, obs, states

    n = tree.shape[0]
    obs = obs.copy()
    states = states.copy()

    startT = np.zeros(n)
    endT = np.zeros(n)
    endT[0] = obs[0, 1]
    parents = np.repeat(np.arange(n), np.diff(tree.indptr))
    daughters = tree.indices

    for p, d in zip(parents, daughters, strict=False):
        startT[d] = endT[p]
        endT[d] = startT[d] + obs[d, 1]

    observed = np.ones(n, dtype=bool)

    # Fate censor (condition 1 or 3)
    if censor_condition in (1, 3):
        for i in range(n):
            if obs[i, 0] == 0:
                start, end = tree.indptr[i], tree.indptr[i + 1]
                for c in tree.indices[start:end]:
                    observed[c] = False

    # Time censor (condition 2 or 3)
    if censor_condition in (2, 3):
        for i in range(n):
            if endT[i] > desired_experiment_time:
                endT[i] = desired_experiment_time
                obs[i, 0] = np.nan
                obs[i, 1] = desired_experiment_time - startT[i]
                obs[i, 2] = 0  # censored
                start, end = tree.indptr[i], tree.indptr[i + 1]
                for c in tree.indices[start:end]:
                    observed[c] = False

    # Basic censor: downward propagation of unobserved
    for p, d in zip(parents, daughters, strict=False):
        if not observed[p]:
            observed[d] = False

    kept = np.nonzero(observed)[0]
    pruned_tree = tree[kept, :][:, kept]
    pruned_obs = obs[kept, :]
    pruned_states = states[kept]

    return pruned_tree, pruned_obs, pruned_states


def censor_lineage_gaphs(
    tree: csr_array,
    obs: np.ndarray,
    states: np.ndarray,
    censor_condition: int,
    desired_experiment_time: float = 2e12,
) -> tuple[csr_array, np.ndarray, np.ndarray]:
    """Applies temporal and fate censorship to 2-phase GaPhs lineages using arrays."""
    if censor_condition == 0:
        return tree, obs, states

    n = tree.shape[0]
    obs = obs.copy()
    states = states.copy()

    startT = np.zeros(n)
    transT = np.zeros(n)
    endT = np.zeros(n)

    transT[0] = obs[0, 2]
    endT[0] = obs[0, 2] + obs[0, 3]

    parents = np.repeat(np.arange(n), np.diff(tree.indptr))
    daughters = tree.indices

    for p, d in zip(parents, daughters, strict=False):
        startT[d] = endT[p]
        transT[d] = startT[d] + obs[d, 2]
        endT[d] = transT[d] + obs[d, 3]

    observed = np.ones(n, dtype=bool)

    # Fate censor (condition 1 or 3)
    if censor_condition in (1, 3):
        for i in range(n):
            if obs[i, 0] == 0 or obs[i, 1] == 0:
                start, end = tree.indptr[i], tree.indptr[i + 1]
                for c in tree.indices[start:end]:
                    observed[c] = False

                if obs[i, 0] == 0:  # dies in G1
                    obs[i, 1] = np.nan
                    obs[i, 3] = np.nan
                    obs[i, 5] = np.nan
                    endT[i] = startT[i] + obs[i, 2]
                    transT[i] = endT[i]
                elif obs[i, 1] == 0:  # dies in G2
                    endT[i] = startT[i] + obs[i, 2] + obs[i, 3]

    # Time censor (condition 2 or 3)
    if censor_condition in (2, 3):
        for i in range(n):
            if endT[i] > desired_experiment_time:
                endT[i] = desired_experiment_time
                obs[i, 1] = np.nan
                obs[i, 3] = desired_experiment_time - transT[i]
                obs[i, 5] = 0
                start, end = tree.indptr[i], tree.indptr[i + 1]
                for c in tree.indices[start:end]:
                    observed[c] = False

            if transT[i] > desired_experiment_time:
                endT[i] = desired_experiment_time
                transT[i] = desired_experiment_time
                obs[i, 0] = np.nan
                obs[i, 1] = np.nan
                obs[i, 2] = desired_experiment_time - startT[i]
                obs[i, 3] = np.nan
                obs[i, 4] = 0
                obs[i, 5] = np.nan
                start, end = tree.indptr[i], tree.indptr[i + 1]
                for c in tree.indices[start:end]:
                    observed[c] = False

    # Basic censor: downward propagation of unobserved
    for p, d in zip(parents, daughters, strict=False):
        if not observed[p]:
            observed[d] = False

    kept = np.nonzero(observed)[0]
    pruned_tree = tree[kept, :][:, kept]
    pruned_obs = obs[kept, :]
    pruned_states = states[kept]

    return pruned_tree, pruned_obs, pruned_states


def bern_estimator(bern_obs: np.ndarray, gammas: np.ndarray):
    """A weighted estimator for a Bernoulli distribution."""
    assert bern_obs.shape == gammas.shape
    assert bern_obs.dtype == float
    assert gammas.dtype == float

    # Add a pseudocount
    numerator = np.sum(gammas[bern_obs == 1.0]) + 1.0
    denominator = np.sum(gammas[np.isfinite(bern_obs)]) + 2.0
    return numerator / denominator


addr = get_cython_function_address("scipy.special.cython_special", "gammaincc")
gammaincc = CFUNCTYPE(c_double, c_double, c_double)(addr)

addr = get_cython_function_address("scipy.special.cython_special", "gammaln")
gammaln = CFUNCTYPE(c_double, c_double)(addr)

addr = get_cython_function_address("scipy.special.cython_special", "__pyx_fuse_0psi")
psi = CFUNCTYPE(c_double, c_double)(addr)


@njit
def gamma_LL(logX: arr_type, gamma_obs: arr_type, time_cen: arr_type, gammas: arr_type, param_idx):
    """Log-likelihood for the optionally censored Gamma distribution.
    The logX is the log transform of the parameters, in case of atonce estimation, it is [shape, scale1, scale2, scale3, scale4].
    """
    x = np.exp(logX)
    glnA = gammaln(x[0])

    gobs = gamma_obs / x[param_idx]
    outt = -1.0 * np.dot(
        gammas * time_cen,
        (x[0] - 1.0) * np.log(gobs) - gobs - glnA - logX[param_idx],
    )

    for jj, cen in enumerate(time_cen):
        if cen == 0:
            gamP = gammaincc(x[0], gobs[jj])
            gamP = np.maximum(gamP, 1e-35)  # Clip if the probability hits exactly 0
            outt -= gammas[jj] * np.log(gamP)

    assert np.isfinite(outt)
    return outt


@njit
def gamma_LL_grad(logX: arr_type, gamma_obs: arr_type, time_cen: arr_type, gammas: arr_type, param_idx) -> arr_type:
    """Analytical gradient of gamma_LL with respect to logX."""
    x = np.exp(logX)
    a = x[0]
    glnA = gammaln(a)
    psiA = psi(a)
    gobs = gamma_obs / x[param_idx]

    grad = np.zeros_like(logX)

    # Uncensored contribution for theta_0 (log shape):
    grad[0] = -a * np.dot(gammas * time_cen, np.log(gobs) - psiA)

    # Scale parameter contributions for theta_k (log scales):
    for i in range(len(gamma_obs)):
        k = param_idx[i]
        if time_cen[i] == 1:
            grad[k] += gammas[i] * (a - gobs[i])
        else:
            gamP = gammaincc(a, gobs[i])
            gamP = np.maximum(gamP, 1e-35)
            pdf_term = np.exp(a * np.log(gobs[i]) - gobs[i] - glnA)
            grad[k] -= gammas[i] * (pdf_term / gamP)

    # Censored contribution for theta_0:
    h = 1e-6
    for jj, cen in enumerate(time_cen):
        if cen == 0:
            gamP_plus = np.maximum(gammaincc(a * np.exp(h), gobs[jj]), 1e-35)
            gamP_minus = np.maximum(gammaincc(a * np.exp(-h), gobs[jj]), 1e-35)
            dlogQ_dtheta0 = (np.log(gamP_plus) - np.log(gamP_minus)) / (2 * h)
            grad[0] -= gammas[jj] * dlogQ_dtheta0

    return grad


@njit
def trigamma(x: float) -> float:
    """Asymptotic expansion for polygamma(1, x)."""
    ans = 0.0
    while x < 6.0:
        ans += 1.0 / (x * x)
        x += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    ans += inv + 0.5 * inv2 + inv2 * inv * (1.0 / 6.0 - inv2 * (1.0 / 30.0 - inv2 / 42.0))
    return ans


@njit
def pava_increasing(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Pool Adjacent Violators Algorithm for increasing monotonicity: y[0] <= y[1] <= ... <= y[K-1]."""
    K = len(y)
    val = y.copy()
    weight = w.copy()
    blocks = [[i] for i in range(K)]

    i = 0
    while i < len(blocks) - 1:
        if val[i] > val[i + 1]:
            new_w = weight[i] + weight[i + 1]
            new_v = (val[i] * weight[i] + val[i + 1] * weight[i + 1]) / new_w
            val[i] = new_v
            weight[i] = new_w
            blocks[i].extend(blocks[i + 1])
            val = np.delete(val, i + 1)
            weight = np.delete(weight, i + 1)
            blocks.pop(i + 1)
            if i > 0:
                i -= 1
        else:
            i += 1

    out = np.zeros(K)
    for b_idx, block in enumerate(blocks):
        for elem in block:
            out[elem] = val[b_idx]
    return out


@njit
def gamma_mle_closed_form(gamma_obs, gammas, param_idx, K, constrained=True):
    """1D profile likelihood solver using Minka initialization and Newton-Raphson."""
    W_k = np.zeros(K)
    sum_y_k = np.zeros(K)
    sum_logy_k = np.zeros(K)

    for i in range(len(gamma_obs)):
        k = param_idx[i] - 1
        w = gammas[i]
        y = gamma_obs[i]
        W_k[k] += w
        sum_y_k[k] += w * y
        sum_logy_k[k] += w * np.log(y)

    for k in range(K):
        if W_k[k] == 0:
            W_k[k] = 1e-12
            sum_y_k[k] = 1.0
            sum_logy_k[k] = 0.0

    y_bar_k = sum_y_k / W_k
    if constrained:
        y_bar_k = pava_increasing(y_bar_k, W_k)

    logy_bar_k = sum_logy_k / W_k
    W_total = np.sum(W_k)
    s = np.sum(W_k * (np.log(y_bar_k) - logy_bar_k)) / W_total
    s = max(s, 1e-12)

    a = (3.0 - s + np.sqrt((s - 3.0) ** 2 + 24.0 * s)) / (12.0 * s)
    a = max(a, 1e-6)

    for _ in range(5):
        g = np.log(a) - psi(a) - s
        g_prime = 1.0 / a - trigamma(a)
        if abs(g_prime) > 1e-12:
            step = g / g_prime
            a = max(a - step, 1e-6)

    b_k = y_bar_k / a
    return a, b_k


def gamma_estimator(
    gamma_obs: arr_type,
    time_cen: arr_type,
    gammas: arr_type,
    param_idx,
    x0: arr_type,
    phase: Literal["all", "G1", "G2"],
) -> arr_type:
    """
    This is a weighted estimator for the parameters of the Gamma distribution,
    estimating shared shape and separate scale parameters across drug concentrations.
    """
    has_censored = np.any(time_cen == 0)
    K = len(x0) - 1
    constrained = phase != "all"

    # For purely uncensored observations, 1D profile Newton + PAVA is 100% exact and instantaneous
    if not has_censored:
        a_est, b_est = gamma_mle_closed_form(gamma_obs, gammas, param_idx, K, constrained=constrained)
        return np.array([a_est] + list(b_est))

    # For censored observations, use the uncensored closed form to warm-start SLSQP
    uncen_mask = time_cen == 1
    if np.sum(uncen_mask) > 10:
        a_warm, b_warm = gamma_mle_closed_form(
            gamma_obs[uncen_mask],
            gammas[uncen_mask],
            param_idx[uncen_mask],
            K,
            constrained=constrained,
        )
        x0_used = np.array([a_warm] + list(b_warm))
    else:
        x0_used = x0

    arrgs = (
        gamma_obs,
        time_cen,
        gammas,
        param_idx,
    )

    if constrained:  # for constrained optimization
        A = np.zeros((3, 5))  # constraint Jacobian
        np.fill_diagonal(A[:, 1:], -1.0)
        np.fill_diagonal(A[:, 2:], 1.0)

        linc = LinearConstraint(A, lb=0.0, keep_feasible=False)
    else:
        linc = ()

    bnd = Bounds(-4.0, 7.0, keep_feasible=False)

    res = minimize(
        gamma_LL,
        jac=gamma_LL_grad,
        x0=np.log(x0_used),
        args=arrgs,
        bounds=bnd,
        method="SLSQP",
        constraints=linc,
    )

    assert res.success
    return np.exp(res.x)
