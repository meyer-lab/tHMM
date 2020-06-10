""" Common utilities used between states regardless of distribution. """

from math import gamma
import numpy as np
from numba import njit
import scipy.stats as sp
import scipy.special as sc
from scipy.optimize import brentq, minimize
import math


@njit
def bern_pdf(x, p):
    """
    This function takes in 1 observation and a Bernoulli rate parameter
    and returns the likelihood of the observation based on the Bernoulli
    probability distribution function.
    """
    # bern_ll = self.bern_p**(tuple_of_obs[0]) * (1.0-self.bern_p)**(1-tuple_of_obs[0])

    return (p ** x) * ((1.0 - p) ** (1 - x))


@njit
def gamma_pdf(x, a, scale):
    """
    This function takes in 1 observation and gamma shape and scale parameters
    and returns the likelihood of the observation based on the gamma
    probability distribution function.
    """
    return x ** (a - 1.0) * np.exp(-1.0 * x / scale) / gamma(a) / (scale ** a)


def gamma_estimator(gamma_obs, time_censor_obs, gammas):
    """
    This is a weighted, closed-form estimator for two parameters
    of the Gamma distribution.
    """
    gammaCor = sum(gammas * gamma_obs) / sum(gammas * time_censor_obs)
    s = np.log(gammaCor) - sum(gammas * np.log(gamma_obs)) / sum(gammas * time_censor_obs)

    def f(k): return np.log(k) - sc.polygamma(0, k) - s

    if f(0.01) * f(100.0) > 0.0:
        a_hat = 10.0
    else:
        a_hat = brentq(f, 0.01, 100.0)

    scale_hat = gammaCor / a_hat

    def LL(a_hat):
        scale_hat = gammaCor / a_hat
        uncens_gammas = np.array([gamma for gamma,idx in zip(gammas,time_censor_obs) if idx==1])
        uncens_obs = np.array([obs for obs,idx in zip(gamma_obs,time_censor_obs) if idx==1])
        assert uncens_gammas.shape[0] == uncens_obs.shape[0]
        uncens = uncens_gammas*sp.gamma.logpdf(uncens_obs, a=a_hat, scale=scale_hat)
        cens_gammas = np.array([gamma for gamma,idx in zip(gammas,time_censor_obs) if idx==0])
        cens_obs = np.array([obs for obs,idx in zip(gamma_obs,time_censor_obs) if idx==0])
        cens = cens_gammas*sp.gamma.logsf(cens_obs, a=a_hat, scale=scale_hat)

        return -1*np.sum(np.sum(uncens) + np.sum(cens))

    res = minimize(LL, a_hat, bounds=((1.,20.),), options={'maxiter': 5})

    return  res.x, gammaCor / res.x


def bernoulli_estimator(bern_obs, gammas):
    """
    Add up all the 1s and divide by the total length (finding the average).
    """
    return sum(gammas * bern_obs) / sum(gammas)


class Time:
    """
    Class that stores all the time related observations in a neater format.
    This will assist in pruning based on experimental time as well as
    obtaining attributes of the lineage as a whole, such as the
    average growth rate.
    """

    def __init__(self, startT, endT):
        self.startT = startT
        self.endT = endT


def assign_times(lineageObj, *kwargs):
    """
    Assigns the start and end time for each cell in the lineage.
    The time observation will be stored in the cell's observation parameter list
    in the second position (index 1). See the other time functions to understand.
    """
    # traversing the cells by generation
    for gen_minus_1, level in enumerate(lineageObj.full_list_of_gens[1:]):
        true_gen = gen_minus_1 + 1  # generations are 1-indexed
        if true_gen == 1:
            for cell in level:
                assert cell.isRootParent()
                if kwargs:
                    cell.time = Time(0, (cell.obs[1] + cell.ons[2]))
                else:
                    cell.time = Time(0, (cell.obs[1]))
        else:
            for cell in level:
                if kwargs:
                    cell.time = Time(cell.parent.time.endT, cell.parent.time.endT + cell.obs[1] + cell.obs[2])
                else:
                    cell.time = Time(cell.parent.time.endT, cell.parent.time.endT + cell.obs[1])


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
        if cell.parent.censored:

            cell.censored = True
            if not cell.isLeafBecauseTerminal():
                cell.left.censored = True
                cell.right.censored = True

            cell.get_sister().censored = True
            if not cell.get_sister().isLeafBecauseTerminal():
                cell.get_sister().left.censored = True
                cell.get_sister().right.censored = True


def fate_censor(cell):
    """
    User-defined function that checks whether a cell's subtree should be removed.
    Our example is based on the standard requirement that the first observation
    (index 0) is a measure of the cell's fate (1 being alive, 0 being dead).
    Clearly if a cell has died, its subtree must be removed.
    """
    if cell.obs[0] == 0:
        if not cell.isLeafBecauseTerminal():
            cell.left.censored = True
            cell.right.censored = True


def time_censor(cell, desired_experiment_time):
    """
    User-defined function that checks whether a cell's subtree should be removed.
    Our example is based on the standard requirement that the second observation
    (index 1) is a measure of the cell's lifetime.
    If a cell has lived beyond a certain experiment time, then its subtree
    must be removed.
    """
    if cell.time.endT > desired_experiment_time:
        cell.time.endT = desired_experiment_time
        cell.obs[1] = cell.time.endT - cell.time.startT
        cell.obs[2] = 0  # no longer observed
        if not cell.isLeafBecauseTerminal():
            cell.left.censored = True
            cell.right.censored = True
