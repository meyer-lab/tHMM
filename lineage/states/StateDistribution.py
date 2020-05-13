""" This file is completely user defined. We have provided a general starting point for the user to use as an example. """
from math import gamma
import numpy as np
import scipy.stats as sp
from numba import njit
import scipy.special as sc
from scipy import optimize

from .stateCommon import bern_pdf, bernoulli_estimator, skew


class StateDistribution:
    def __init__(self, bern_p, gamma_a, gamma_scale):
        """ Initialization function should take in just in the parameters for the observations that comprise the multivariate random variable emission they expect their data to have. """
        self.bern_p = bern_p
        self.gamma_a = gamma_a
        self.gamma_scale = gamma_scale
        self.params = [self.bern_p, self.gamma_a, self.gamma_scale]

    def rvs(self, size):  # user has to identify what the multivariate (or univariate if he or she so chooses) random variable looks like
        """ User-defined way of calculating a random variable given the parameters of the state stored in that observation's object. """
        # {
        bern_obs = sp.bernoulli.rvs(p=self.bern_p, size=size)  # bernoulli observations
        gamma_obs = sp.gamma.rvs(a=self.gamma_a, scale=self.gamma_scale, size=size)  # gamma observations
        time_censor = [1] * len(gamma_obs)  # 1 if observed
        # } is user-defined in that they have to define and maintain the order of the multivariate random variables.
        # These tuples of observations will go into the cells in the lineage tree.
        list_of_tuple_of_obs = list(map(list, zip(bern_obs, gamma_obs, time_censor)))
        return list_of_tuple_of_obs

    def pdf(self, tuple_of_obs):  # user has to define how to calculate the likelihood
        """ User-defined way of calculating the likelihood of the observation stored in a cell. """
        # In the case of a univariate observation, the user still has to define how the likelihood is calculated,
        # but has the ability to just return the output of a known scipy.stats.<distribution>.<{pdf,pmf}> function.
        # In the case of a multivariate observation, the user has to decide how the likelihood is calculated.
        # In our example, we assume the observation's are uncorrelated across the dimensions (across the different
        # distribution observations), so the likelihood of observing the multivariate observation is just the product of
        # the individual observation likelihoods.

        try:
            bern_ll = bern_pdf(tuple_of_obs[0], self.bern_p)
            gamma_ll = gamma_pdf(tuple_of_obs[1], self.gamma_a, self.gamma_scale)
        except ZeroDivisionError:
            assert False

        return bern_ll * gamma_ll

    def estimator(self, list_of_tuples_of_obs, gammas):
        """ User-defined way of estimating the parameters given a list of the tuples of observations from a group of cells. """
        # unzipping the list of tuples
        unzipped_list_of_tuples_of_obs = list(zip(*list_of_tuples_of_obs))

        # getting the observations as individual lists
        # {
        try:
            bern_obs = list(unzipped_list_of_tuples_of_obs[0])
            gamma_obs = list(unzipped_list_of_tuples_of_obs[1])
            gamma_censor_obs = list(unzipped_list_of_tuples_of_obs[2])
        except BaseException:
            bern_obs = []
            gamma_obs = []
            gamma_censor_obs = []

        bern_p_estimate = bernoulli_estimator(bern_obs, (self.bern_p,), gammas)
        gamma_a_estimate, gamma_scale_estimate = gamma_estimator(gamma_obs, gamma_censor_obs, (self.gamma_a, self.gamma_scale,), gammas)
        
        state_estimate_obj = StateDistribution(bern_p=bern_p_estimate, gamma_a=gamma_a_estimate, gamma_scale=gamma_scale_estimate)
        # } requires the user's attention.
        # Note that we return an instance of the state distribution class, but now instantiated with the parameters
        # from estimation. This is then stored in the original state distribution object which then gets updated
        # if this function runs again.
        return state_estimate_obj

    def tHMM_E_init(self):
        """
        Initialize a default state distribution.
        """
        print("Initializing a new StateDistribution object!")
        return StateDistribution(0.9, 7, 3+(1*(np.random.uniform())))

    def __repr__(self):
        """
        Method to print out a state distribution object.
        """
        return "State object w/ parameters: {}, {}, {}.".format(self.bern_p, self.gamma_a, self.gamma_scale)


# Because parameter estimation requires that estimators be written or imported,
# the user should be able to provide
# estimators that can solve for the parameters that describe the distributions.
# We provide some estimators below as an example.
# Their use in the StateDistribution class is shown in the estimator class method.
# User must take care to define estimators that
# can handle the case where the list of observations is empty.


def gamma_estimator(gamma_obs, gamma_censor_obs, old_params, gammas):
    """
    This is a closed-form estimator for two parameters
    of the Gamma distribution, which is corrected for bias.
    """
    s = np.log(sum(gammas*gamma_obs)/sum(gammas)) - sum(gammas*np.log(gamma_obs))/sum(gammas)
    k0 = old_params[0]
    scale0 = old_params[1]
    
    f = lambda k,s : np.log(k) - sc.polygamma(0,k) - s
    fprime = lambda k,s : (1./k) - sc.polygamma(1,k)
    fprime2 = lambda k,s : (-1./k**2) - sc.polygamma(2,k)
    
    halley = lambda k,s: k - ((2*(f(k,s)*fprime(k,s)))/(2*(fprime(k,s)**2)-f(k,s)*fprime2(k,s)))
    a_hat = halley(k0, s)

    scale_hat = (sum(gammas*gamma_obs))/a_hat/sum(gammas)

    return a_hat, scale_hat


@njit
def gamma_pdf(x, a, scale):
    """
    This function takes in 1 observation and gamma shape and scale parameters
    and returns the likelihood of the observation based on the gamma
    probability distribution function.
    """
    return x**(a-1.) * np.exp(-1.*x/scale) / gamma(a) / (scale**a)
