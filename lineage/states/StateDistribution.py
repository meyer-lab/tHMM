""" This file is completely user defined. We have provided a general starting point for the user to use as an example. """
import numpy as np
import scipy.stats as sp
import scipy.special as sc
from scipy.optimize import brentq


from .stateCommon import bern_pdf, bernoulli_estimator


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

        bern_ll = bern_pdf(tuple_of_obs[0], self.bern_p) if tuple_of_obs[2] == 1 else 1.0

        try:
            if tuple_of_obs[2] == 1:
                gamma_ll = sp.gamma.pdf(tuple_of_obs[1], a=self.gamma_a, scale=self.gamma_scale)
            else:
                gamma_ll = sp.gamma.sf(tuple_of_obs[1], a=self.gamma_a, scale=self.gamma_scale)
        except ZeroDivisionError:
            print(f"{tuple_of_obs[1]}, {self.gamma_a}, {self.gamma_scale}")
            raise

        return bern_ll * gamma_ll

    def estimator(self, list_of_tuples_of_obs, gammas):
        """ User-defined way of estimating the parameters given a list of the tuples of observations from a group of cells. """
        # unzipping the list of tuples
        unzipped_list_of_tuples_of_obs = list(zip(*list_of_tuples_of_obs))

        # getting the observations as individual lists
        # {
        try:
            bern_obs = list(unzipped_list_of_tuples_of_obs[0])
            gamma_obs = np.array(unzipped_list_of_tuples_of_obs[1])
            gamma_censor_obs = np.array(unzipped_list_of_tuples_of_obs[2], dtype=bool)
        except BaseException:
            bern_obs = []
            gamma_obs = np.array([])
            gamma_censor_obs = np.array([], dtype=bool)

        bern_p_estimate = bernoulli_estimator(bern_obs, gammas)
        γ_a_hat, γ_scale_hat = gamma_estimator(gamma_obs, gamma_censor_obs, gammas)
        
        return StateDistribution(bern_p=bern_p_estimate, gamma_a=γ_a_hat, gamma_scale=γ_scale_hat)
        # } requires the user's attention.
        # Note that we return an instance of the state distribution class, but now instantiated with the parameters
        # from estimation. This is then stored in the original state distribution object which then gets updated
        # if this function runs again.

    def tHMM_E_init(self):
        """
        Initialize a default state distribution.
        """
        return StateDistribution(0.9, 7, 3 + (1 * (np.random.uniform())))


# Because parameter estimation requires that estimators be written or imported,
# the user should be able to provide
# estimators that can solve for the parameters that describe the distributions.
# We provide some estimators below as an example.
# Their use in the StateDistribution class is shown in the estimator class method.
# User must take care to define estimators that
# can handle the case where the list of observations is empty.


def gamma_estimator(gamma_obs, gamma_censor_obs, gammas):
    """
    This is a closed-form estimator for two parameters
    of the Gamma distribution, which is corrected for bias.
    """
    gammaCor = sum(gammas * gamma_obs) / sum(gammas)
    s = np.log(gammaCor) - sum(gammas * np.log(gamma_obs)) / sum(gammas)
    def f(k): return np.log(k) - sc.polygamma(0, k) - s

    if f(0.01) * f(100.0) > 0.0:
        a_hat = 10.0
    else:
        a_hat = brentq(f, 0.01, 100.0)

    scale_hat = gammaCor / a_hat

    
    def LL(x):
        uncens = sp.gamma.pdf(gamma_obs, a=x[0], scale=x[1])
        cens = sp.gamma.sf(gamma_obs, a=x[0], scale=x[1])

        # If the observation was censored, use the survival function
        uncens[np.logical_not(gamma_censor_obs)] = cens[np.logical_not(gamma_censor_obs)]

        # If gamma indicates the cell is very unlikely for this state, ignore it
        gamL = np.log(gammas)
        uncens[gamL < -9] = 1.0
        gamL[gamL < -9] = 0

        uncens = np.log(uncens)
        return -np.sum(uncens + gamL)

    #res = minimize(LL, [a_hat, scale_hat])

    return a_hat, scale_hat
