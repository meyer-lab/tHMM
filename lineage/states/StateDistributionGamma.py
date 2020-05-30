""" This file is completely user defined. We have provided a general starting point for the user to use as an example. """
import numpy as np
import scipy.stats as sp


from .stateCommon import bern_pdf, gamma_pdf, bernoulli_estimator, gamma_estimator


class StateDistribution:
    def __init__(self, bern_p=0.9, gamma_a=7, gamma_scale=3):
        """ Initialization function should take in just in the parameters for the observations that comprise the multivariate random variable emission they expect their data to have. """
        self.params = [bern_p, gamma_a, gamma_scale]

    def rvs(self, size):  # user has to identify what the multivariate (or univariate if he or she so chooses) random variable looks like
        """ User-defined way of calculating a random variable given the parameters of the state stored in that observation's object. """
        # {
        bern_obs = sp.bernoulli.rvs(p=self.params[0], size=size)  # bernoulli observations
        gamma_obs = sp.gamma.rvs(a=self.params[1], scale=self.params[2], size=size)  # gamma observations
        time_censor = [1] * len(gamma_obs)  # 1 if observed
        # } is user-defined in that they have to define and maintain the order of the multivariate random variables.
        # These tuples of observations will go into the cells in the lineage tree.
        return bern_obs, gamma_obs, time_censor

    def pdf(self, tuple_of_obs):  # user has to define how to calculate the likelihood
        """ User-defined way of calculating the likelihood of the observation stored in a cell. """
        # In the case of a univariate observation, the user still has to define how the likelihood is calculated,
        # but has the ability to just return the output of a known scipy.stats.<distribution>.<{pdf,pmf}> function.
        # In the case of a multivariate observation, the user has to decide how the likelihood is calculated.
        # In our example, we assume the observation's are uncorrelated across the dimensions (across the different
        # distribution observations), so the likelihood of observing the multivariate observation is just the product of
        # the individual observation likelihoods.

        bern_ll = bern_pdf(tuple_of_obs[0], self.params[0]) if tuple_of_obs[2] == 1 else 1.0

        try:
            if tuple_of_obs[2] == 1:
                gamma_ll = gamma_pdf(tuple_of_obs[1], self.params[1], self.params[2])
            else:
                gamma_ll = sp.gamma.sf(tuple_of_obs[1], a=self.params[1], scale=self.params[2])
        except ZeroDivisionError:
            print(f"{tuple_of_obs[1]}, {self.params[1]}, {self.params[2]}")
            raise

        return bern_ll * gamma_ll

    def estimator(self, list_of_tuples_of_obs, gammas):
        """ User-defined way of estimating the parameters given a list of the tuples of observations from a group of cells. """
        # unzipping the list of tuples
        unzipped_list_of_tuples_of_obs = list(zip(*list_of_tuples_of_obs))

        # getting the observations as individual lists
        # {
        bern_obs = list(unzipped_list_of_tuples_of_obs[0])
        γ_obs = np.array(unzipped_list_of_tuples_of_obs[1])
        γ_censor_obs = np.array(unzipped_list_of_tuples_of_obs[2], dtype=bool)

        self.params[0] = bernoulli_estimator(bern_obs, gammas)
        self.params[1], self.params[2] = gamma_estimator(γ_obs, γ_censor_obs, gammas)
        # } requires the user's attention.
        # Note that we return an instance of the state distribution class, but now instantiated with the parameters
        # from estimation. This is then stored in the original state distribution object which then gets updated
        # if this function runs again.
