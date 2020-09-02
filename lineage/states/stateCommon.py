""" Common utilities used between states regardless of distribution. """

import math
import jax.numpy as jnp
from jax import value_and_grad, jit
import jax.scipy.stats as jsp
import jax.scipy.special as jsc
from jax.config import config
import numpy as np
import scipy.stats as sp
import scipy.special as sc
from scipy.optimize import brentq, minimize

config.update("jax_enable_x64", True)


def bernoulli_estimator(bern_obs, gammas):
    """
    Add up all the 1s and divide by the total length (finding the average).
    """
    # Handle an empty state
    if np.sum(gammas) == 0.0:
        return np.average(bern_obs)

    return np.average(bern_obs, weights=gammas)


def negative_LL(x, uncens_obs, uncens_gammas, cens_obs, cens_gammas):
    x = jnp.exp(x)
    uncens = jnp.dot(uncens_gammas, jsp.gamma.logpdf(uncens_obs, a=x[0], scale=x[1]))
    cens = jnp.dot(cens_gammas, jsc.gammaincc(x[0], cens_obs / x[1]))
    return -1 * (uncens + cens)


negative_LL_jit = jit(value_and_grad(negative_LL, 0))


def gamma_estimator(gamma_obs, time_censor_obs, gammas, shape):
    """
    This is a weighted, closed-form estimator for two parameters
    of the Gamma distribution.
    """
    # Handle an empty state
    if np.sum(gammas) == 0.0:
        gammas = np.copy(gammas)
        gammas.fill(1.0)

    # First assume nothing is censored
    gammaCor = np.average(gamma_obs, weights=gammas)
    s = np.log(gammaCor) - np.average(np.log(gamma_obs), weights=gammas)

    def f(k):
        return np.log(k) - sc.polygamma(0, k) - s

    if constant_shape:
        return [constant_shape, gammaCor /constant_shape]
    else:
        flow = f(0.1)
        fhigh = f(100.0)
        if flow * fhigh > 0.0:
            if np.absolute(flow) < np.absolute(fhigh):
                a_hat0 = 0.1
            elif np.absolute(flow) > np.absolute(fhigh):
                a_hat0 = 100.0
            else:
                a_hat0 = 10.0
        else:
            a_hat0 = brentq(f, 0.1, 100.0)

        x0 = [a_hat0, gammaCor / a_hat0]

    # If nothing is censored
    if np.all(time_censor_obs == 1):
        return x0

    uncens_gammas = gammas[time_censor_obs == 1]
    uncens_obs = gamma_obs[time_censor_obs == 1]
    assert uncens_gammas.shape[0] == uncens_obs.shape[0]
    cens_gammas = gammas[time_censor_obs == 0]
    cens_obs = gamma_obs[time_censor_obs == 0]
    assert cens_gammas.shape[0] == cens_obs.shape[0]

    arrgs = (uncens_obs, uncens_gammas, cens_obs, cens_gammas)
    res = minimize(fun=negative_LL_jit, jac=True, x0=np.log(x0), method="TNC", bounds=((None, 5.0), (None, 5.0)), args=arrgs)
    return np.exp(res.x)


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
