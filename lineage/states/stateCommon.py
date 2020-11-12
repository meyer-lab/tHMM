""" Common utilities used between states regardless of distribution. """

import math
import numpy as np
import itertools as it
import scipy.stats as sp
import scipy.special as sc
from scipy.optimize import toms748, minimize


def negative_LL(x, uncens_obs, uncens_gammas, cens_obs, cens_gammas):
    return negative_LL_sep(x[1], x[0], uncens_obs, uncens_gammas, cens_obs, cens_gammas)


def negative_LL_sep(scale, a, uncens_obs, uncens_gammas, cens_obs, cens_gammas):
    uncens = np.dot(uncens_gammas, sp.gamma.logpdf(uncens_obs, a=a, scale=scale))
    cens = np.dot(cens_gammas, sc.gammaincc(a, cens_obs / scale))
    return -1 * (uncens + cens)


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


# ------ Functions for estimating shape and scale when shape is shared among all concentrations and scale only changes. ----- #

def negative_LL_atonce(x, uncens_obs, uncens_gammas, cens_obs, cens_gammas):
    return negative_LL_sep_atonce(x[1:7], x[0], uncens_obs, uncens_gammas, cens_obs, cens_gammas)


def negative_LL_sep_atonce(scales, a, uncens_obs, uncens_gammas, cens_obs, cens_gammas):
    uncens = np.sum([np.dot(uncens_gammas[i], sp.gamma.logpdf(uncens_obs[i], a=a, scale=scale)) for i, scale in enumerate(scales)])
    cens = np.sum([np.dot(cens_gammas[i], sc.gammaincc(a, cens_obs[i] / scale)) for i, scale in enumerate(scales)])
    return -1 * (uncens + cens)


def gamma_uncensored_atonce(gamma_obs_list, list_gammas, constant_shape):
    """ An uncensored gamma estimator. """
    # Handle no observations
    gammas_list = []
    for gammas in list_gammas:
        if np.sum(gammas) == 0.0:
            gammas = np.ones_like(gammas)
        gammas_list.append(gammas)

    gammaCor_list = [np.average(gamma_obs_list[i], weights=gammas_list[i]) for i in range(len(gamma_obs_list))]
    s_list = [np.log(gammaCor_list[i]) - np.average(np.log(gamma_obs_list[i]), weights=gammas_list[i]) for i in range(len(gamma_obs_list))]
    s = np.average(s_list) # s is only used to calculate the shape, doesn't have any effect on scale.

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

    res = [gammaCor / a_hat0 for gammaCor in gammaCor_list]
    return [a_hat0] + res


def gamma_estimator_atonce(gamma_obs_list, time_cen_list, list_gammas, constant_shape, x0):
    """
    This is a weighted, closed-form estimator for two parameters
    of the Gamma distribution.
    """
    # Handle no observation
    gammas_list = []
    for gammas in list_gammas:
        if np.sum(gammas) == 0.0:
            gammas = np.ones_like(gammas)
        gammas_list.append(gammas)

    # If nothing is censored
    if np.all(list(it.chain(*time_cen_list)) == 1):
        return gamma_uncensored_atonce(gamma_obs_list, gammas_list, constant_shape)

    for i, gammas in enumerate(gammas_list):
        assert gammas.shape[0] == gamma_obs_list[i].shape[0]

    # optimization settings
    opt = {'gtol': 1e-12, 'ftol': 1e-12}
    bnd = (1.0, 800.0)

    assert any(isinstance(el, list) for el in gamma_obs_list) # make sure you are getting a list of lists
    assert constant_shape is None
    arg1 = []
    arg2 = []
    arg3 = []
    arg4 = []
    for i in range(len(gamma_obs_list)):
        arg1.append(gamma_obs_list[i][time_cen_list[i] == 1])
        arg2.append(gammas_list[i][time_cen_list[i] == 1])
        arg3.append(gamma_obs_list[i][time_cen_list[i] == 0])
        arg4.append(gammas_list[i][time_cen_list[i] == 0])
    arrgs = (arg1, arg2, arg3, arg4)
    res = minimize(fun=negative_LL_atonce, jac="3-point", x0=x0, bounds=((bnd, ) * 4), args=arrgs, options=opt)
    return res.x
