""" State distribution class for separated G1 and G2 phase durations as observation. """
import numpy as np
import scipy.stats as sp
from .StateDistribution import (gamma_estimator,
                                gamma_pdf,
                               )
from .stateCommon import (bern_pdf,
                          bernoulli_estimator,
                         )


class StateDistribution2:
    """ For G1 and G2 separated as observations. """
    def __init__(self, bern_p, gamma_a1, gamma_scale1, gamma_a2, gamma_scale2):  # user has to identify what parameters to use for each state
        """ Initialization function should take in just in the parameters for the observations that comprise the multivariate random variable emission they expect their data to have. """
        self.bern_p = bern_p
        self.gamma_a1 = gamma_a1
        self.gamma_scale1 = gamma_scale1
        self.gamma_a2 = gamma_a2
        self.gamma_scale2 = gamma_scale2
        self.params = [self.bern_p, self.gamma_a1, self.gamma_scale1, self.gamma_a2, self.gamma_scale2]

    def rvs(self, size):  # user has to identify what the multivariate (or univariate if he or she so chooses) random variable looks like
        """ User-defined way of calculating a random variable given the parameters of the state stored in that observation's object. """
        # {
        bern_obs = sp.bernoulli.rvs(p=self.bern_p, size=size)  # bernoulli observations
        gamma_obsG1 = sp.gamma.rvs(a=self.gamma_a1, scale=self.gamma_scale1, size=size)  # gamma observations
        gamma_obsG2 = sp.gamma.rvs(a=self.gamma_a2, scale=self.gamma_scale2, size=size)
        time_censor = [1] * (len(gamma_obsG1) + len(gamma_obsG2))
        # } is user-defined in that they have to define and maintain the order of the multivariate random variables.
        # These tuples of observations will go into the cells in the lineage tree.
        list_of_tuple_of_obs = list(map(list, zip(bern_obs, gamma_obsG1, gamma_obsG2, time_censor)))
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
        except ZeroDivisionError:
            assert False, f"{tuple_of_obs[0]}, {self.bern_p}"
        try:
            gamma_llG1 = gamma_pdf(tuple_of_obs[1], self.gamma_a1, self.gamma_scale1)
        except ZeroDivisionError:
            assert False, f"{tuple_of_obs[1]}, {self.gamma_a1}, {self.gamma_scale1}"
        try:
            gamma_llG2 = gamma_pdf(tuple_of_obs[2], self.gamma_a2, self.gamma_scale2)
        except ZeroDivisionError:
            assert False, f"{tuple_of_obs[2]}, {self.gamma_a2}, {self.gamma_scale2}"

        return bern_ll * gamma_llG1 * gamma_llG2

    def estimator(self, list_of_tuples_of_obs, gammas):
        """ User-defined way of estimating the parameters given a list of the tuples of observations from a group of cells. """
        # unzipping the list of tuples
        unzipped_list_of_tuples_of_obs = list(zip(*list_of_tuples_of_obs))

        # getting the observations as individual lists
        # {
        try:
            bern_obs = list(unzipped_list_of_tuples_of_obs[0])
            gamma_obsG1 = list(unzipped_list_of_tuples_of_obs[1])
            gamma_obsG2 = list(unzipped_list_of_tuples_of_obs[2])
            gamma_censor_obs = list(unzipped_list_of_tuples_of_obs[3])
        except BaseException:
            bern_obs = []
            gamma_obsG1 = []
            gamma_obsG2 = []
            gamma_censor_obs = []

        bern_p_estimate = bernoulli_estimator(bern_obs, (self.bern_p,), gammas)
        gamma_a1_estimate, gamma_scale1_estimate = gamma_estimator(gamma_obsG1, gamma_censor_obs, (self.gamma_a1, self.gamma_scale1,), gammas)
        gamma_a2_estimate, gamma_scale2_estimate = gamma_estimator(gamma_obsG2, gamma_censor_obs, (self.gamma_a2, self.gamma_scale2,), gammas)
        state_estimate_obj = StateDistribution2(bern_p=bern_p_estimate, gamma_a1=gamma_a1_estimate, gamma_scale1=gamma_scale1_estimate, gamma_a2=gamma_a2_estimate, gamma_scale2=gamma_scale2_estimate)

        # } requires the user's attention.
        # Note that we return an instance of the state distribution class, but now instantiated with the parameters
        # from estimation. This is then stored in the original state distribution object which then gets updated
        # if this function runs again.
        return state_estimate_obj
    def tHMM_E_init(self):
        """
        Initialize a default state distribution.
        """
        return StateDistribution2(0.9, 7, 3 + (1 * (np.random.uniform())), 14, 6 + (1 * (np.random.uniform())))

    def __repr__(self):
        """
        Method to print out a state distribution object.
        """
        return "State object w/ parameters: {}, {}, {}.".format(self.bern_p, self.gamma_a1, self.gamma_scale1, self.gamma_a2, self.gamma_scale2)
    