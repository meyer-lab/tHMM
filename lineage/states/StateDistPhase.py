""" State distribution class for separated G1 and G2 phase durations as observation. """
import scipy.stats as sp

from .stateCommon import bern_pdf, bernoulli_estimator, gamma_pdf, gamma_estimator


class StateDistribution:
    """ For G1 and G2 separated as observations. """

    def __init__(self, bern_p1=0.9, bern_p2=0.75, gamma_a1=7.0, gamma_scale1=3, gamma_a2=14.0, gamma_scale2=6):  # user has to identify what parameters to use for each state
        """ Initialization function should take in just in the parameters for the observations that comprise the multivariate random variable emission they expect their data to have. """
        self.params = [bern_p1, bern_p2, gamma_a1, gamma_scale1, gamma_a2, gamma_scale2]

    def rvs(self, size):  # user has to identify what the multivariate (or univariate if he or she so chooses) random variable looks like
        """ User-defined way of calculating a random variable given the parameters of the state stored in that observation's object. """
        # {
        bern_obsG1 = sp.bernoulli.rvs(p=self.params[0], size=size)  # bernoulli observations
        bern_obsG2 = sp.bernoulli.rvs(p=self.params[1], size=size)
        gamma_obsG1 = sp.gamma.rvs(a=self.params[2], scale=self.params[3], size=size)  # gamma observations
        gamma_obsG2 = sp.gamma.rvs(a=self.params[4], scale=self.params[5], size=size)
        time_censor = [1] * (len(gamma_obsG1) + len(gamma_obsG2))
        # } is user-defined in that they have to define and maintain the order of the multivariate random variables.
        # These tuples of observations will go into the cells in the lineage tree.
        return bern_obsG1, bern_obsG2, gamma_obsG1, gamma_obsG2, time_censor

    def pdf(self, tuple_of_obs):  # user has to define how to calculate the likelihood
        """ User-defined way of calculating the likelihood of the observation stored in a cell. """
        # In the case of a univariate observation, the user still has to define how the likelihood is calculated,
        # but has the ability to just return the output of a known scipy.stats.<distribution>.<{pdf,pmf}> function.
        # In the case of a multivariate observation, the user has to decide how the likelihood is calculated.
        # In our example, we assume the observation's are uncorrelated across the dimensions (across the different
        # distribution observations), so the likelihood of observing the multivariate observation is just the product of
        # the individual observation likelihoods.
        bern_llG1 = bern_pdf(tuple_of_obs[0], self.params[0])
        bern_llG2 = bern_pdf(tuple_of_obs[1], self.params[1])
        gamma_llG1 = gamma_pdf(tuple_of_obs[2], self.params[2], self.params[3])
        gamma_llG2 = gamma_pdf(tuple_of_obs[3], self.params[4], self.params[5])

        if tuple_of_obs[0] == 0:
            return bern_llG2 * gamma_llG1 * gamma_llG2
        else:
            return bern_llG1 * bern_llG2 * gamma_llG1 * gamma_llG2

    def estimator(self, list_of_tuples_of_obs, gammas):
        """ User-defined way of estimating the parameters given a list of the tuples of observations from a group of cells. """
        # unzipping the list of tuples
        unzipped_list_of_tuples_of_obs = list(zip(*list_of_tuples_of_obs))

        # getting the observations as individual lists
        # {
        bern_obsG1 = list(unzipped_list_of_tuples_of_obs[0])
        bern_obsG2 = list(unzipped_list_of_tuples_of_obs[1])
        gamma_obsG1 = list(unzipped_list_of_tuples_of_obs[2])
        gamma_obsG2 = list(unzipped_list_of_tuples_of_obs[3])
        gamma_censor_obs = list(unzipped_list_of_tuples_of_obs[4])

        self.params[0] = bernoulli_estimator(bern_obsG1, gammas)
        self.params[1] = bernoulli_estimator(bern_obsG2, gammas)
        self.params[2], self.params[3] = gamma_estimator(gamma_obsG1, gamma_censor_obs, gammas)
        self.params[4], self.params[5] = gamma_estimator(gamma_obsG2, gamma_censor_obs, gammas)
        # } requires the user's attention.
        # Note that we return an instance of the state distribution class, but now instantiated with the parameters
        # from estimation. This is then stored in the original state distribution object which then gets updated
        # if this function runs again.

    def __repl__(self):
        return f"{self.params}"

    def __str__(self):
        return self.__repl__()
