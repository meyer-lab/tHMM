""" State distribution class for separated G1 and G2 phase durations as observation. """
import scipy.stats as sp
from .stateCommon import bern_pdf, bernoulli_estimator, gamma_pdf, gamma_estimator


class StateDistribution2:
    """ For G1 and G2 separated as observations. """

    def __init__(self, bern_p=0.9, gamma_a1=7.0, gamma_scale1=3, gamma_a2=14.0, gamma_scale2=6):  # user has to identify what parameters to use for each state
        """ Initialization function should take in just in the parameters for the observations that comprise the multivariate random variable emission they expect their data to have. """
        self.params = [bern_p, gamma_a1, gamma_scale1, gamma_a2, gamma_scale2]

    def rvs(self, size):  # user has to identify what the multivariate (or univariate if he or she so chooses) random variable looks like
        """ User-defined way of calculating a random variable given the parameters of the state stored in that observation's object. """
        # {
        bern_obs = sp.bernoulli.rvs(p=self.bern_p, size=size)  # bernoulli observations
        gamma_obsG1 = sp.gamma.rvs(a=self.gamma_a1, scale=self.gamma_scale1, size=size)  # gamma observations
        gamma_obsG2 = sp.gamma.rvs(a=self.gamma_a2, scale=self.gamma_scale2, size=size)
        time_censor = [1] * (len(gamma_obsG1) + len(gamma_obsG2))
        # } is user-defined in that they have to define and maintain the order of the multivariate random variables.
        # These tuples of observations will go into the cells in the lineage tree.
        return list(map(list, zip(bern_obs, gamma_obsG1, gamma_obsG2, time_censor)))

    def pdf(self, tuple_of_obs):  # user has to define how to calculate the likelihood
        """ User-defined way of calculating the likelihood of the observation stored in a cell. """
        # In the case of a univariate observation, the user still has to define how the likelihood is calculated,
        # but has the ability to just return the output of a known scipy.stats.<distribution>.<{pdf,pmf}> function.
        # In the case of a multivariate observation, the user has to decide how the likelihood is calculated.
        # In our example, we assume the observation's are uncorrelated across the dimensions (across the different
        # distribution observations), so the likelihood of observing the multivariate observation is just the product of
        # the individual observation likelihoods.
        bern_ll = bern_pdf(tuple_of_obs[0], self.bern_p)

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
            self.params = [0.9, 7.0, 3, 14.0, 6]
            return

        self.params[0] = bernoulli_estimator(bern_obs, gammas)
        self.params[1], self.params[2] = gamma_estimator(gamma_obsG1, gamma_censor_obs, gammas)
        self.params[3], self.params[4] = gamma_estimator(gamma_obsG2, gamma_censor_obs, gammas)
        # } requires the user's attention.
        # Note that we return an instance of the state distribution class, but now instantiated with the parameters
        # from estimation. This is then stored in the original state distribution object which then gets updated
        # if this function runs again.
