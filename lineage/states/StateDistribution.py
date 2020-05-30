""" This file is completely user defined. We have provided a general starting point for the user to use as an example. """
import numpy as np
import scipy.stats as sp


from .stateCommon import bern_pdf, gamma_pdf, bernoulli_estimator, gamma_estimator


class StateDistribution:
    def __init__(self, bern_p=0.9, gamma_a=7, gamma_scale=3):
        """ Initialization function should take in just in the parameters for the observations that comprise the multivariate random variable emission they expect their data to have. """
        self.bern_p = bern_p
        self.gamma_a = gamma_a
        self.gamma_scale = gamma_scale

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
                gamma_ll = gamma_pdf(tuple_of_obs[1], self.gamma_a, self.gamma_scale)
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
            γ_obs = np.array(unzipped_list_of_tuples_of_obs[1])
            γ_censor_obs = np.array(unzipped_list_of_tuples_of_obs[2], dtype=bool)
        except BaseException:
            return StateDistribution()

        bern_p_estimate = bernoulli_estimator(bern_obs, gammas)
        γ_a_hat, γ_scale_hat = gamma_estimator(γ_obs, γ_censor_obs, gammas)

        return StateDistribution(bern_p=bern_p_estimate, gamma_a=γ_a_hat, gamma_scale=γ_scale_hat)
        # } requires the user's attention.
        # Note that we return an instance of the state distribution class, but now instantiated with the parameters
        # from estimation. This is then stored in the original state distribution object which then gets updated
        # if this function runs again.
