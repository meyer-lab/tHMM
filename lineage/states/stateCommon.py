""" Common utilities used between states regardless of distribution. """

import numpy as np
from ctypes import CFUNCTYPE, c_double
from numba.extending import get_cython_function_address
from numba import jit
from numba.typed import List
from scipy.optimize import minimize, LinearConstraint, Bounds


def basic_censor(cell):
    """
    Censors a cell, its daughters, its sister, and
    it's sister's daughters if the cell's parent is
    censored.
    """
    if not cell.isRootParent():
        if not cell.parent.observed:

            cell.observed = False
            if not cell.isLeafBecauseTerminal():
                cell.left.observed = False
                cell.right.observed = False

            cell.get_sister().observed = False
            if not cell.get_sister().isLeafBecauseTerminal():
                cell.get_sister().left.observed = False
                cell.get_sister().right.observed = False


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


@jit(nopython=True)
def gamma_LL(logX: np.ndarray, gamma_obs: List[np.ndarray], time_cen: List[np.ndarray], gammas: List[np.ndarray]):
    """ Log-likelihood for the optionally censored Gamma distribution. """
    logX = np.copy(logX)

    if len(logX) > 2:
        assert len(logX) == 5
        logX[2] += logX[1]
        logX[3] += logX[2]
        logX[4] += logX[3]

    x = np.exp(logX)
    glnA = gammaln(x[0])
    outt = 0.0
    for i in range(len(x) - 1):
        gobs = gamma_obs[i] / x[i + 1]
        outt -= np.dot(gammas[i] * time_cen[i], (x[0] - 1.0) * np.log(gobs) - gobs - glnA - logX[i + 1])

        for j in range(len(time_cen[i])):
            if time_cen[i][j] == 0.0:
                gamP = gammaincc(x[0], gobs[j])
                gamP = np.maximum(gamP, 1e-60)  # Clip if the probability hits exactly 0
                outt -= gammas[i][j] * np.log(gamP)

    assert np.isfinite(outt)
    return outt


@jit(nopython=True)
def gamma_LL_diff(x0: np.ndarray, gamma_obs: List[np.ndarray], time_cen: List[np.ndarray], gammas: List[np.ndarray]):
    """ Finite differencing of objective function. """
    f0 = gamma_LL(x0, gamma_obs, time_cen, gammas)
    grad = np.empty(x0.size)
    dx = 2e-8

    for i in range(x0.size):
        x = np.copy(x0)
        x[i] += dx
        fdx = gamma_LL(x, gamma_obs, time_cen, gammas)
        grad[i] = (fdx - f0) / dx

    return f0, grad


def gamma_estimator(gamma_obs: list[np.ndarray], time_cen: list[np.ndarray], gammas: list[np.ndarray], x0: np.ndarray):
    """
    This is a weighted, closed-form estimator for two parameters
    of the Gamma distribution for estimating shared shape and separate scale parameters of several drug concentrations at once.
    In the phase-specific case, we have 3 linear constraints: scale1 > scale2, scale2 > scale3, scale3 > scale 4.
    In the non-specific case, we have only 1 constraint: scale1 > scale2 ==> A = np.array([1, 3])
    """
    x0 = np.log(x0)

    # Handle no observations
    if np.sum([np.sum(g) for g in gammas]) < 0.1:
        gammas = [np.ones(g.size) for g in gammas]

    # Check shapes
    for i in range(len(gamma_obs)):
        assert gamma_obs[i].shape == time_cen[i].shape
        assert gamma_obs[i].shape == gammas[i].shape

    arrgs = (List(gamma_obs), List(time_cen), List(gammas))

    bnd = Bounds(np.full_like(x0, -3.5), np.full_like(x0, 6.0), keep_feasible=False)
    if x0.size > 2:
        bnd.lb[2:] = 0
        x0[2] -= x0[1]
        x0[3] -= x0[2]
        x0[4] -= x0[3]

    with np.errstate(all='raise'):
        opts = {"maxfun": 1e6, "maxiter": 1e6, "maxls": 100, "ftol": 0}
        res = minimize(gamma_LL_diff, jac=True, x0=x0, args=arrgs, bounds=bnd, method='L-BFGS-B', options=opts)

    if len(res.x) > 2:
        res.x[2] += res.x[1]
        res.x[3] += res.x[2]
        res.x[4] += res.x[3]

    assert res.success
    return np.exp(res.x)
