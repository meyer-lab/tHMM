""" Common utilities used between states regardless of distribution. """

import math
import numpy as np
import scipy.stats as sp
import scipy.special as sc
from scipy.optimize import toms748, LinearConstraint, minimize, Bounds, fmin_cobyla


def negative_LL(x, uncens_obs, uncens_gammas, cens_obs, cens_gammas):
    return negative_LL_sep(x[1], x[0], uncens_obs, uncens_gammas, cens_obs, cens_gammas)

def negative_LL_sep(scale, a, uncens_obs, uncens_gammas, cens_obs, cens_gammas):
    uncens = np.dot(uncens_gammas, sp.gamma.logpdf(uncens_obs, a=a, scale=scale))
    cens = np.dot(cens_gammas, sc.gammaincc(a, cens_obs / scale))
    out = -1 * (uncens + cens)
    if np.isnan(out):
        print("out is nan", out)
        print(scale, a)
        print(uncens_obs, cens_obs)
    return out

def gamma_uncensored(gamma_obs, gammas, constant_shape):
    """ An uncensored gamma estimator. """
    # Handle no observations
    if np.sum(gammas) == 0.0:
        gammas = np.ones_like(gammas)

    gammaCor = np.average(gamma_obs, weights=gammas)
    s = np.log(gammaCor) - np.average(np.log(gamma_obs), weights=gammas)

    def f(k):
        return np.log(k) - sc.polygamma(0, k) - s

    if constant_shape:
        a_hat0 = constant_shape
    else:
        flow = f(1.0)
        fhigh = f(100.0)
        if flow * fhigh > 0.0:
            if np.absolute(flow) < np.absolute(fhigh):
                a_hat0 = 1.0
            elif np.absolute(flow) > np.absolute(fhigh):
                a_hat0 = 100.0
            else:
                a_hat0 = 10.0
        else:
            a_hat0 = toms748(f, 1.0, 100.0)

    return [a_hat0, gammaCor / a_hat0]


def gamma_estimator(gamma_obs, time_cen, gammas, constant_shape, x0):
    """
    This is a weighted, closed-form estimator for two parameters
    of the Gamma distribution.
    """
    # Handle no observations
    if np.sum(gammas) == 0.0:
        gammas = np.ones_like(gammas)

    # If nothing is censored
    if np.all(time_cen == 1):
        return gamma_uncensored(gamma_obs, gammas, constant_shape)

    assert gammas.shape[0] == gamma_obs.shape[0]
    arrgs = (gamma_obs[time_cen == 1], gammas[time_cen == 1], gamma_obs[time_cen == 0], gammas[time_cen == 0])
    opt = {'gtol': 1e-12, 'ftol': 1e-12}
    bnd = (1.0, 800.0)

    if constant_shape is None:
        res = minimize(fun=negative_LL, jac="3-point", x0=x0, bounds=(bnd, bnd), args=arrgs, options=opt)
        xOut = res.x
    else:
        arrgs = (constant_shape, *arrgs)
        res = minimize(fun=negative_LL_sep, jac="3-point", x0=x0[1], bounds=(bnd, ), args=arrgs, options=opt)
        xOut = [constant_shape, res.x]

    return xOut

def get_experiment_time(lineageObj):
    """
    This function returns the longest experiment time
    experienced by cells in the lineage.
    We can simply find the leaf cell with the
    longest end time. This is effectively
    the same as the experiment time for synthetic lineages.
    """
    return max(cell.time.endT for cell in lineageObj.output_leaves)


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


def negative_LL_atonce(x, uncens_obs, uncens_gammas, cens_obs, cens_gammas):
    """ uses the negative_LL_atonce and passes the vector of scales and the shared shape parameter. """
    outt = 0.0
    for i in range(4):
        outt += negative_LL_sep(x[1 + i], x[0], uncens_obs[i], uncens_gammas[i], cens_obs[i], cens_gammas[i])
    return outt


def gamma_estimator_atonce(gamma_obs, time_cen, gamas):
    """
    This is a weighted, closed-form estimator for two parameters
    of the Gamma distribution for estimating shared shape and separate scale parameters of several drug concentrations at once.
    """
    gammas = []
    # Handle no observations
    for gamma in gamas:
        if np.sum(gamma) == 0.0:
            gammas.append(np.ones_like(gamma))
        else:
            gammas.append(gamma)

    for i, gamma in enumerate(gammas):
        assert gamma.shape[0] == gamma_obs[i].shape[0]
    arg1 = [np.squeeze(gamma_obs[i][time_cen[i] == 1]) for i in range(len(gamma_obs))]
    arg2 = [np.squeeze(gammas[i][time_cen[i] == 1]) for i in range(len(gamma_obs))]
    arg3 = [np.squeeze(gamma_obs[i][time_cen[i] == 0]) for i in range(len(gamma_obs))]
    arg4 = [np.squeeze(gammas[i][time_cen[i] == 0]) for i in range(len(gamma_obs))]

    arrgs = (arg1, arg2, arg3, arg4)

    # A is a matrix of coefficients of the constraints.
    # For example if we have x_1 - 2x_2 >= 0 then it forms a row in the A matrix as: [1, -2], and one indice in the b array [0].
    # the row array of independent variables are assumed to be [shape, scale1, scale2, scale3, scal4]
    x0 = np.array([10.0, 1.3, 1.5, 1.7, 1.9])
    def const1(x):
        return x[2] - x[1]
    def const2(x):
        return x[3] - x[2]
    def const3(x):
        return x[4] - x[3]
    res = fmin_cobyla(func=negative_LL_atonce, x0=x0, cons=[const1, const2, const3], args=arrgs, consargs=(), rhoend=1e-3, maxfun=50000, disp=True)
    return res
