""" This file is completely user defined. We have provided a general starting point for the user to use as an example. """
import numpy as np
import scipy.stats as sp


class StateDistribution:
    def __init__(self, norm_loc, norm_scale):
        """ Initialization function should take in just in the parameters for the observations that comprise the multivariate random variable emission they expect their data to have. """
        self.norm_loc = norm_loc
        assert norm_scale > 0, "A non-valid scale has been given. Please provide a scale > 0"
        self.norm_scale = norm_scale
        self.params = [self.norm_loc, self.norm_scale]

    def rvs(self, size):  # user has to identify what the multivariate (or univariate if he or she so chooses) random variable looks like
        """ User-defined way of calculating a random variable given the parameters of the state stored in that observation's object. """
        # {
        norm_obs = sp.norm.rvs(loc=self.norm_loc, scale=self.norm_scale, size=size)  # normal observations
        # time_censor = [1] * len(gamma_obs)  # 1 if observed
        # } is user-defined in that they have to define and maintain the order of the multivariate random variables.
        # These tuples of observations will go into the cells in the lineage tree.
        list_of_tuple_of_obs = list(map(list, zip(norm_obs)))
        return list_of_tuple_of_obs

    def pdf(self, tuple_of_obs):  # user has to define how to calculate the likelihood
        """ User-defined way of calculating the likelihood of the observation stored in a cell. """
        # In the case of a univariate observation, the user still has to define how the likelihood is calculated,
        # but has the ability to just return the output of a known scipy.stats.<distribution>.<{pdf,pmf}> function.
        # In the case of a multivariate observation, the user has to decide how the likelihood is calculated.
        # In our example, we assume the observation's are uncorrelated across the dimensions (across the different
        # distribution observations), so the likelihood of observing the multivariate observation is just the product of
        # the individual observation likelihoods.

        norm_ll = sp.norm.pdf(tuple_of_obs[0], self.norm_loc, self.norm_scale)

        return norm_ll

    def estimator(self, list_of_tuples_of_obs, gammas):
        """ User-defined way of estimating the parameters given a list of the tuples of observations from a group of cells. """
        # unzipping the list of tuples
        unzipped_list_of_tuples_of_obs = list(zip(*list_of_tuples_of_obs))

        # getting the observations as individual lists
        # {
        try:
            norm_obs = list(unzipped_list_of_tuples_of_obs[0])
        except BaseException:
            norm_obs = []

        norm_loc_estimate, norm_scale_estimate = norm_estimator(norm_obs, gammas)

        state_estimate_obj = StateDistribution(norm_loc=norm_loc_estimate, norm_scale=norm_scale_estimate)
        # } requires the user's attention.
        # Note that we return an instance of the state distribution class, but now instantiated with the parameters
        # from estimation. This is then stored in the original state distribution object which then gets updated
        # if this function runs again.
        return state_estimate_obj

    def tHMM_E_init(self):
        """
        Initialize a default state distribution.
        """
        return StateDistribution(10, 1 + 10 * (np.random.uniform()))

    def __repr__(self):
        """
        Method to print out a state distribution object.
        """
        return "State object w/ parameters: {}, {}.".format(self.norm_loc, self.norm_scale)


# Because parameter estimation requires that estimators be written or imported,
# the user should be able to provide
# estimators that can solve for the parameters that describe the distributions.
# We provide some estimators below as an example.
# Their use in the StateDistribution class is shown in the estimator class method.
# User must take care to define estimators that
# can handle the case where the list of observations is empty.


def norm_estimator(norm_obs, gammas):
    '''This function is an estimator for the mean and standard deviation of a normal distribution, including weighting for each state'''
    mu = (sum(gammas * norm_obs) + 1e-10) / (sum(gammas) + 1e-10)
    std = ((sum(gammas * (norm_obs - mu)**2) + 1e-10) / (sum(gammas) + 1e-10))**.5
    if mu == 0:
        print("mu == 0")
    if std == 0:
        print("std == 0")
    if sum(gammas) == 0:
        print("sum(gammas) == 0")
    return mu, std
