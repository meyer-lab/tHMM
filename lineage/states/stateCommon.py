""" Common utilities used between states regardless of distribution. """

import numpy as np
import ctypes
from numba.extending import get_cython_function_address
from numba import jit, prange
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


addr = get_cython_function_address("scipy.special.cython_special", "gammaincc")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)
gammaincc = functype(addr)

addr = get_cython_function_address("scipy.special.cython_special", "gammaln")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
gammaln = functype(addr)


@jit(nopython=True)
def xlogy(x, y):
    if x == 0 and not np.isnan(y):
        return 0
    else:
        return x * np.log(y)


@jit(nopython=True, parallel=True, fastmath=True)
def nLL_atonce(x: np.ndarray, gamma_obs: list[np.ndarray], time_cen: list[np.ndarray], gammas: list[np.ndarray]):
    """ uses the nLL_atonce and passes the vector of scales and the shared shape parameter. """
    x = np.exp(x)
    outt = 0.0
    for i in prange(len(x) - 1):
        gobs = gamma_obs[i] / x[i + 1]

        for j in range(len(time_cen[i])):
            if time_cen[i][j] == 1.0:
                outt -= gammas[i][j] * (xlogy(x[0] - 1.0, gobs[j]) - gobs[j] - gammaln(x[0]) - np.log(x[i + 1]))
            else:
                assert time_cen[i][j] == 0.0
                outt -= gammas[i][j] * np.log(gammaincc(x[0], gobs[j]))

    if np.isinf(outt):
        print(x)

    assert np.isfinite(outt)
    return outt


def gamma_estimator(gamma_obs: list[np.ndarray], time_cen: list[np.ndarray], gammas: list[np.ndarray], x0: np.ndarray):
    """
    This is a weighted, closed-form estimator for two parameters
    of the Gamma distribution for estimating shared shape and separate scale parameters of several drug concentrations at once.
    In the phase-specific case, we have 3 linear constraints: scale1 > scale2, scale2 > scale3, scale3 > scale 4.
    In the non-specific case, we have only 1 constraint: scale1 > scale2 ==> A = np.array([1, 3])
    """
    # Handle no observations
    if np.sum([np.sum(g) for g in gammas]) < 0.1:
        gammas = [np.ones(g.size) for g in gammas]

    # Check shapes
    for i in range(len(gamma_obs)):
        assert gamma_obs[i].shape == time_cen[i].shape
        assert gamma_obs[i].shape == gammas[i].shape

    arrgs = (List(gamma_obs), List(time_cen), List(gammas))

    if len(gamma_obs) == 4:  # for constrained optimization
        A = np.zeros((3, 5))  # is a matrix that contains the constraints. the number of rows shows the number of linear constraints.
        np.fill_diagonal(A[:, 1:], -1.0)
        np.fill_diagonal(A[:, 2:], 1.0)
        linc = [LinearConstraint(A, lb=np.zeros(3), ub=np.full(3, np.inf))]
        if np.allclose(np.dot(A, x0), 0.0):
            x0 = np.array([200.0, 0.2, 0.4, 0.6, 0.8])
    else:
        linc = list()

    bnd = Bounds(np.full_like(x0, -3.0), np.full_like(x0, 6.0), keep_feasible=True)
    opt = {"xtol": 1e-8}

    with np.errstate(all='raise'):
        res = minimize(nLL_atonce, x0=np.log(x0), args=arrgs, bounds=bnd, method="trust-constr", constraints=linc, options=opt)

    assert res.success or ("maximum number of function evaluations is exceeded" in res.message)
    return np.exp(res.x)
