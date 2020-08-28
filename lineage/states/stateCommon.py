""" Common utilities used between states regardless of distribution. """

import math
import numpy as np
from numba import njit
import scipy.stats as sp
import scipy.special as sc
from scipy.optimize import brentq, minimize


@njit
def bern_pdf(x, p):
    """
    This function takes in 1 observation and a Bernoulli rate parameter
    and returns the likelihood of the observation based on the Bernoulli
    probability distribution function.
    """
    # bern_ll = self.bern_p**(tuple_of_obs[0]) * (1.0-self.bern_p)**(1-tuple_of_obs[0])

    return (p ** x) * ((1.0 - p) ** (1 - x))


def bernoulli_estimator(bern_obs, gammas):
    """
    Add up all the 1s and divide by the total length (finding the average).
    """
    # Handle an empty state
    if np.sum(gammas) == 0.0:
        return np.average(bern_obs)

    return np.average(bern_obs, weights=gammas)


@njit
def gamma_pdf(x, a, scale):
    """
    This function takes in 1 observation and gamma shape and scale parameters
    and returns the likelihood of the observation based on the gamma
    probability distribution function.
    """
    return x ** (a - 1.0) * np.exp(-1.0 * x / scale) / math.gamma(a) / (scale ** a)


def gamma_estimator(gamma_obs, time_censor_obs, gammas, shape):
    """
    This is a weighted, closed-form estimator for two parameters
    of the Gamma distribution.
    """
    # Handle an empty state
    if np.sum(gammas) == 0.0:
        gammas = np.copy(gammas)
        gammas.fill(1.0)

    gammaCor = np.average(gamma_obs, weights=gammas)
    s = np.log(gammaCor) - np.average(np.log(gamma_obs), weights=gammas)

    def f(k):
        return np.log(k) - sc.polygamma(0, k) - s

    if shape is not None:
        a_hat0 = shape
    else:
        if f(0.001) * f(1000.0) > 0.0:
            a_hat0 = 10.0
        else:
            a_hat0 = brentq(f, 0.001, 1000.0)

    x0 = [a_hat0, gammaCor / a_hat0]

    uncens_gammas = gammas[time_censor_obs == 1]
    uncens_obs = gamma_obs[time_censor_obs == 1]
    assert uncens_gammas.shape[0] == uncens_obs.shape[0]
    cens_gammas = gammas[time_censor_obs == 0]
    cens_obs = gamma_obs[time_censor_obs == 0]
    assert cens_gammas.shape[0] == cens_obs.shape[0]

    def negative_LL(x):
        uncens = uncens_gammas * sp.gamma.logpdf(uncens_obs, a=x[0], scale=x[1])
        cens = cens_gammas * sp.gamma.logsf(cens_obs, a=x[0], scale=x[1])
        return -1 * (np.sum(uncens) + np.sum(cens))

    if np.all(time_censor_obs == 1):
        # if nothing is censored, then there is no need to use the numerical solver
        return x0[0], x0[1]
    else:
        res = minimize(fun=negative_LL, x0=x0, bounds=((1., 20.), (1., 20.),), options={'maxiter': 5})
        return res.x[0], res.x[1]


def get_experiment_time(lineageObj):
    """
    This function returns the longest experiment time
    experienced by cells in the lineage.
    We can simply find the leaf cell with the
    longest end time. This is effectively
    the same as the experiment time for synthetic lineages.
    """
    longest = 0.0
    for cell in lineageObj.output_leaves:
        if cell.time.endT > longest:
            longest = cell.time.endT
    return longest


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
