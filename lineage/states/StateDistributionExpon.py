""" This file is completely user defined. We have provided a general starting point for the user to use as an example. """
import numpy as np
import scipy.stats as sp
from numba import njit

from .stateCommon import bern_pdf, bernoulli_estimator
from ..CellVar import Time


class StateDistribution:
    def __init__(self, bern_p=0.9, exp_beta=7.0):
        """ Initialization function should take in just in the parameters for the observations that comprise the multivariate random variable emission they expect their data to have. """
        self.params = [bern_p, exp_beta]

    def rvs(self, size):  # user has to identify what the multivariate (or univariate if he or she so chooses) random variable looks like
        """ User-defined way of calculating a random variable given the parameters of the state stored in that observation's object. """
        # {
        bern_obs = sp.bernoulli.rvs(p=self.params[0], size=size)  # bernoulli observations
        exp_obs = sp.expon.rvs(scale=self.params[1], size=size)  # gamma observations
        time_censor = [1] * len(exp_obs)  # 1 if observed
        # } is user-defined in that they have to define and maintain the order of the multivariate random variables.
        # These tuples of observations will go into the cells in the lineage tree.
        return bern_obs, exp_obs, time_censor

    def pdf(self, tuple_of_obs):  # user has to define how to calculate the likelihood
        """ User-defined way of calculating the likelihood of the observation stored in a cell. """
        # In the case of a univariate observation, the user still has to define how the likelihood is calculated,
        # but has the ability to just return the output of a known scipy.stats.<distribution>.<{pdf,pmf}> function.
        # In the case of a multivariate observation, the user has to decide how the likelihood is calculated.
        # In our example, we assume the observation's are uncorrelated across the dimensions (across the different
        # distribution observations), so the likelihood of observing the multivariate observation is just the product of
        # the individual observation likelihoods.

        bern_ll = bern_pdf(tuple_of_obs[0], self.params[0]) if tuple_of_obs[2] == 1 else 1.0

        if tuple_of_obs[2] == 1:
            exp_ll = exp_pdf(tuple_of_obs[1], self.params[1])
        else:
            exp_ll = exp_sf(tuple_of_obs[1], self.params[1])

        return bern_ll * exp_ll

    def estimator(self, list_of_tuples_of_obs, gammas):
        """ User-defined way of estimating the parameters given a list of the tuples of observations from a group of cells. """
        # unzipping the list of tuples
        unzipped_list_of_tuples_of_obs = list(zip(*list_of_tuples_of_obs))

        # getting the observations as individual lists
        # {
        bern_obs = list(unzipped_list_of_tuples_of_obs[0])
        exp_obs = list(unzipped_list_of_tuples_of_obs[1])
        time_censor_obs = np.array(unzipped_list_of_tuples_of_obs[2], dtype=bool)

        self.params[0] = bernoulli_estimator(bern_obs, gammas)
        self.params[1] = exp_estimator(exp_obs, time_censor_obs, gammas)
        # } requires the user's attention.
        # Note that we return an instance of the state distribution class, but now instantiated with the parameters
        # from estimation. This is then stored in the original state distribution object which then gets updated
        # if this function runs again.
        
    def assign_times(self, list_of_gens):
        """
        Assigns the start and end time for each cell in the lineage.
        The time observation will be stored in the cell's observation parameter list
        in the second position (index 1). See the other time functions to understand.
        This is used in the creation of LineageTrees
        """
        # traversing the cells by generation
        for gen_minus_1, level in enumerate(list_of_gens[1:]):
            true_gen = gen_minus_1 + 1  # generations are 1-indexed
            if true_gen == 1:
                for cell in level:
                    assert cell.isRootParent()
                        cell.time = Time(0, cell.obs[1])
            else:
                for cell in level:
                        cell.time = Time(cell.parent.time.endT, cell.parent.time.endT + cell.obs[1])

    def __repl__(self):
        return f"{self.params}"

    def __str__(self):
        return self.__repl__()


# Because parameter estimation requires that estimators be written or imported,
# the user should be able to provide
# estimators that can solve for the parameters that describe the distributions.
# We provide some estimators below as an example.
# Their use in the StateDistribution class is shown in the estimator class method.
# User must take care to define estimators that
# can handle the case where the list of observations is empty.


def exp_estimator(exp_obs, time_censor_obs, gammas):
    """
    This is a closed-form estimator for the lambda parameter of the
    exponential distribution, which is right-censored.
    """
    return sum(gammas * exp_obs) / sum(gammas * time_censor_obs)


@njit
def exp_pdf(x, beta):
    """
    This function takes in 1 observation and and an exponential parameter
    and returns the likelihood of the observation based on the exponential
    probability distribution function.
    """
    return (1.0 / beta) * np.exp(-1.0 * x / beta)


@njit
def exp_sf(x, beta):
    """
    This function takes in 1 observation and and an exponential parameter
    and returns the likelihood of the observation based on the exponential
    survival distribution function.
    """
    return np.exp(-1.0 * x / beta)
