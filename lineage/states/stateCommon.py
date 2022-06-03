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


addr = get_cython_function_address("scipy.special.cython_special", "gammaincc")
gammaincc = CFUNCTYPE(c_double, c_double, c_double)(addr)

addr = get_cython_function_address("scipy.special.cython_special", "gammaln")
gammaln = CFUNCTYPE(c_double, c_double)(addr)


@jit(nopython=True)
def nLL_atonce(logX: np.ndarray, gamma_obs: list[np.ndarray], time_cen: list[np.ndarray], gammas: list[np.ndarray]):
    """ uses the nLL_atonce and passes the vector of scales and the shared shape parameter. """
    x = np.exp(logX)
    glnA = gammaln(x[0])
    outt = 0.0
    for i in range(len(x) - 1):
        gobs = gamma_obs[i] / x[i + 1]

        for j in range(len(time_cen[i])):
            if time_cen[i][j] == 1.0:
                # Handle xlogy edge case
                if (x[0] - 1.0) == 0:
                    outt -= gammas[i][j] * (0.0 - gobs[j] - glnA - logX[i + 1])
                else:
                    outt -= gammas[i][j] * ((x[0] - 1.0) * np.log(gobs[j]) - gobs[j] - glnA - logX[i + 1])
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
    In the phase-specific case, we have 3 linear constraints: scale1 < scale2, scale2 < scale3, scale3 < scale 4, scale4 < shape.
    In the non-specific case, we have only 1 constraint: scale1 > scale2 ==> A = np.array([1, 3])
    """
    # Handle no observations
    if np.sum([np.sum(g) for g in gammas]) < 0.1:
        gammas = [np.ones(g.size) for g in gammas]

    # make sure all negative observations are removed
    gamma_obs_ = [gm[gamma_obs[i] >= 0] for i, gm in enumerate(gamma_obs)]
    time_cen_ = [tm[gamma_obs[i] >= 0] for i, tm in enumerate(time_cen)]
    gammas_ = [gs[gamma_obs[i] >= 0] for i, gs in enumerate(gammas)]

    # Check shapes
    for i in range(len(gamma_obs_)):
        assert gamma_obs_[i].shape == time_cen_[i].shape
        assert gamma_obs_[i].shape == gammas_[i].shape

    arrgs = (List(gamma_obs_), List(time_cen_), List(gammas_))

    if len(gamma_obs_) == 4:  # for constrained optimization
        A = np.zeros((4, 5))  # is a matrix that contains the constraints. the number of rows shows the number of linear constraints.
        np.fill_diagonal(A[1:, 1:], -1.0)
        np.fill_diagonal(A[1:, 2:], 1.0)
        A[0, 0] = 1.0
        A[0, 4] = -1.0
        linc = [LinearConstraint(A, lb=np.zeros(4), ub=np.full(4, np.inf))]
        if np.allclose(np.dot(A, x0), 0.0):
            x0 = np.array([50.0, 0.5, 1.0, 1.5, 2.0])

        if all(x==x0[0] for x in x0):
            print("all equal", x0)
            x0 = np.array([50.0, 0.5, 1.0, 1.5, 2.0])

    else:
        linc = list()

    bnd = Bounds(np.full_like(x0, -3.0), np.full_like(x0, 8.0), keep_feasible=False)

    with np.errstate(all='raise'):
        if len(linc) > 0:
            res = minimize(nLL_atonce, x0=np.log(x0), args=arrgs, bounds=bnd, method="trust-constr", constraints=linc)
        else:
            opts = {"maxfev": 1e6, "maxiter": 1e6, "xatol": 1e-6, "fatol": 1e-6}
            res = minimize(nLL_atonce, x0=np.log(x0), args=arrgs, bounds=bnd, method='Nelder-Mead', options=opts)

    assert res.success or ("maximum number of function evaluations is exceeded" in res.message)
    return np.exp(res.x)
