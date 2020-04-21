""" This file is completely user defined. We have provided a general starting point for the user to use as an example. """
import numpy as np
import scipy.stats as sp
from numba import njit
import scipy.special as sc

from .stateCommon import bern_pdf


class StateDistribution:
    def __init__(self, bern_p, exp_lambda):
        """ Initialization function should take in just in the parameters for the observations that comprise the multivariate random variable emission they expect their data to have. """
        self.bern_p = bern_p
        self.exp_lambda = exp_lambda
        self.params = [self.bern_p, self.exp_lambda]

    def rvs(self, size):  # user has to identify what the multivariate (or univariate if he or she so chooses) random variable looks like
        """ User-defined way of calculating a random variable given the parameters of the state stored in that observation's object. """
        # {
        bern_obs = sp.bernoulli.rvs(p=self.bern_p, size=size)  # bernoulli observations
        exp_obs = sp.expon.rvs(scale=self.exp_lambda, size=size)  # gamma observations
        time_censor = [1] * len(exp_obs)
        # } is user-defined in that they have to define and maintain the order of the multivariate random variables.
        # These tuples of observations will go into the cells in the lineage tree.
        list_of_tuple_of_obs = list(map(list, zip(bern_obs, exp_obs, time_censor)))
        return list_of_tuple_of_obs

    def pdf(self, tuple_of_obs):  # user has to define how to calculate the likelihood
        """ User-defined way of calculating the likelihood of the observation stored in a cell. """
        # In the case of a univariate observation, the user still has to define how the likelihood is calculated,
        # but has the ability to just return the output of a known scipy.stats.<distribution>.<{pdf,pmf}> function.
        # In the case of a multivariate observation, the user has to decide how the likelihood is calculated.
        # In our example, we assume the observation's are uncorrelated across the dimensions (across the different
        # distribution observations), so the likelihood of observing the multivariate observation is just the product of
        # the individual observation likelihoods.

        bern_ll = bern_pdf(tuple_of_obs[0], self.bern_p)
        exp_ll = exp_pdf(tuple_of_obs[1], self.exp_lambda)

        return bern_ll * exp_ll

    def estimator(self, list_of_tuples_of_obs):
        """ User-defined way of estimating the parameters given a list of the tuples of observations from a group of cells. """
        # unzipping the list of tuples
        unzipped_list_of_tuples_of_obs = list(zip(*list_of_tuples_of_obs))

        # getting the observations as individual lists
        # {
        try:
            bern_obs = list(unzipped_list_of_tuples_of_obs[0])
            exp_obs = list(unzipped_list_of_tuples_of_obs[1])
        except BaseException:
            bern_obs = []
            gamma_obs = []

        bern_p_estimate = bernoulli_estimator(bern_obs)
        gamma_a_estimate, gamma_scale_estimate = gamma_estimator(gamma_obs)

        state_estimate_obj = StateDistribution(bern_p=bern_p_estimate, gamma_a=gamma_a_estimate, gamma_scale=gamma_scale_estimate)
        # } requires the user's attention.
        # Note that we return an instance of the state distribution class, but now instantiated with the parameters
        # from estimation. This is then stored in the original state distribution object which then gets updated
        # if this function runs again.
        return state_estimate_obj

    def __repr__(self):
        return "State object w/ parameters: {}, {}.".format(self.bern_p, self.exp_lambda)


def tHMM_E_init():
    """ Initialize a default state distribution. """
    return StateDistribution(0.9, 7*(np.random.uniform()))


# Because parameter estimation requires that estimators be written or imported,
# the user should be able to provide
# estimators that can solve for the parameters that describe the distributions.
# We provide some estimators below as an example.
# Their use in the StateDistribution class is shown in the estimator class method.
# User must take care to define estimators that
# can handle the case where the list of observations is empty.





def exp_estimator(exp_obs):
    """
    This is a closed-form estimator for the lambda parameter of the 
    exponential distribution, which is right-censored.
    """

    return lambda_hat


@njit
def exp_pdf(x, lambda_):
    """
    This function takes in 1 observation and and an exponential parameter
    and returns the likelihood of the observation based on the exponential
    probability distribution function.
    """
    gamma_ll = (1 / (gamma(a) * (scale ** a))) * x ** (a - 1) * np.exp(-x / scale)
    return exp_ll
