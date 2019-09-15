""" This file is completely user defined. We have provided a general starting point for the user to use as an example. """
import numpy as np
import scipy.stats as sp
import math


class StateDistribution:
    def __init__(self, state, bern_p, gamma_a, gamma_scale):  # user has to identify what parameters to use for each state
        """ Initialization function should take in just in the parameters for the observations that comprise the multivariate random variable emission they expect their data to have. """
        self.state = state
        self.bern_p = bern_p
        self.gamma_a = gamma_a
        self.gamma_scale = gamma_scale

    def rvs(self, size):  # user has to identify what the multivariate (or univariate if he or she so chooses) random variable looks like
        """ User-defined way of calculating a random variable given the parameters of the state stored in that observation's object. """
        # {
        bern_obs = sp.bernoulli.rvs(p=self.bern_p, size=size)  # bernoulli observations
        gamma_obs = sp.gamma.rvs(a=self.gamma_a, scale=self.gamma_scale, size=size)  # gamma observations
        # } is user-defined in that they have to define and maintain the order of the multivariate random variables.
        # These tuples of observations will go into the cells in the lineage tree.
        list_of_tuple_of_obs = list(zip(bern_obs, gamma_obs))
        return list_of_tuple_of_obs

    def pdf(self, tuple_of_obs):  # user has to define how to calculate the likelihood
        """ User-defined way of calculating the likelihood of the observation stored in a cell. """
        # In the case of a univariate observation, the user still has to define how the likelihood is calculated,
        # but has the ability to just return the output of a known scipy.stats.<distribution>.<{pdf,pmf}> function.
        # In the case of a multivariate observation, the user has to decide how the likelihood is calculated.
        # In our example, we assume the observation's are uncorrelated across the dimensions (across the different
        # distribution observations), so the likelihood of observing the multivariate observation is just the product of
        # the individual observation likelihoods.

        bern_ll = sp.bernoulli.pmf(k=tuple_of_obs[0], p=self.bern_p)  # bernoulli likelihood
        gamma_ll = sp.gamma.logpdf(x=tuple_of_obs[1], a=self.gamma_a, scale=self.gamma_scale)  # gamma likelihood

        assert not math.isnan(np.exp(gamma_ll)), "{} {} {} {}".format(gamma_ll, tuple_of_obs[1], self.gamma_a, self.gamma_scale)
        if bern_ll == 0 or np.exp(gamma_ll) == 0:
            print(tuple_of_obs[1], ',', self.gamma_a, ',', self.gamma_scale, ',', np.exp(gamma_ll))

        return np.exp(gamma_ll)

    def estimator(self, list_of_tuples_of_obs):
        """ User-defined way of estimating the parameters given a list of the tuples of observations from a group of cells. """
        # unzipping the list of tuples
        unzipped_list_of_tuples_of_obs = list(zip(*list_of_tuples_of_obs))

        # getting the observations as individual lists
        # {
        try:
            bern_obs = list(unzipped_list_of_tuples_of_obs[0])
            gamma_obs = list(unzipped_list_of_tuples_of_obs[1])
        except BaseException:
            bern_obs = []
            gamma_obs = []

        bern_p_estimate = bernoulli_estimator(bern_obs)
        gamma_a_estimate, gamma_scale_estimate = gamma_estimator(gamma_obs)

        state_estimate_obj = StateDistribution(state=self.state,
                                               bern_p=bern_p_estimate,
                                               gamma_a=gamma_a_estimate,
                                               gamma_scale=gamma_scale_estimate)
        # } requires the user's attention.
        # Note that we return an instance of the state distribution class, but now instantiated with the parameters
        # from estimation. This is then stored in the original state distribution object which then gets updated
        # if this function runs again.
        return state_estimate_obj

    def __repr__(self):
        return "State object w/ parameters: {}, {}, {}.".format(self.bern_p, self.gamma_a, self.gamma_scale)


def prune_rule(cell):
    """ User-defined function that checks whether a cell's subtree should be removed. """
    truther = False
    if cell.obs[0] == 0:
        truther = True  # cell has died; subtree must be removed
    return truther


def tHMM_E_init(state):
    return StateDistribution(state,
                             0.9,
                             10 * (np.random.uniform()),
                             1)

# Because parameter estimation requires that estimators be written or imported, the user should be able to provide
# estimators that can solve for the parameters that describe the distributions. We provide some estimators below as an example.
# Their use in the StateDistribution class is shown in the estimator class method. User must take care to define estimators that
# can handle the case where the list of observations is empty.


def report_time(cell):
    """ Given any cell in the lineage, this function walks through the cell's ancestors and return how long it has taken so far. """
    list_parents = [cell]
    taus = 0.0 + cell.obs[1]

    for cell in list_parents:
        if cell._isRootParent():
            break
        elif cell.parent not in list_parents:
            list_parents.append(cell.parent)
            taus += cell.parent.obs[1]
    return taus


def get_experiment_time(lineage):
    """ This function is to find the amount of time it took for the cells to be generated and reach to the desired number of cells. """
    leaf_times = []
    for cell in lineage.output_leaves:
        temp = report_time(cell)
        leaf_times.append(temp)
    longest = max(leaf_times)
    return longest


def bernoulli_estimator(bern_obs):
    """ Add up all the 1s and divide by the total length (finding the average). """
    return (sum(bern_obs) + 1e-10) / (len(bern_obs) + 2e-10)


def exponential_estimator(exp_obs):
    """ Trivial exponential """
    return (sum(exp_obs) + 50e-10) / (len(exp_obs) + 1e-10)


def gamma_estimator(gamma_obs):
    """ This is a closed-form estimator for two parameters of the Gamma distribution, which is corrected for bias. """
    N = len(gamma_obs)
    if N == 0:
        return 10, 1
    #assert N != 0, "The number of gamma observations is zero!"
    print("the number of gamma observations", N)
    x_lnx = [x * np.log(x) for x in gamma_obs]
    lnx = [np.log(x) for x in gamma_obs]
    # gamma_a
    a_hat = (N * (sum(gamma_obs)) + 1e-10) / (N * sum(x_lnx) - (sum(lnx)) * (sum(gamma_obs)) + 1e-10)
    # gamma_scale
    b_hat = ((1 + 1e-10) / (N**2 + 1e-10)) * (N * (sum(x_lnx)) - (sum(lnx)) * (sum(gamma_obs)))
    # bias correction
#     if N>1:
#         a_hat = (N /(N - 1)) * a_hat
#         b_hat = b_hat - (1/N) * (3*b_hat - (2/3) * (b_hat/(b_hat + 1)) - (4/5)* (b_hat)/((1 + b_hat)**2

    if b_hat < 1.0 or 50. < a_hat < 5.:
        return 10, 1

    return a_hat, b_hat