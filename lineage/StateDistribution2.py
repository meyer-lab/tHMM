""" This file is completely user defined. We have provided a general starting point for the user to use as an example. """
import numpy as np
import scipy.stats as sp
from .StateDistribution import gamma_estimator, bernoulli_estimator


class StateDistribution2:
    def __init__(self, bern_p, gamma_a1, gamma_scale1, gamma_a2, gamma_scale2):
        """ Initialization function should take in just in the parameters for the observations that comprise the multivariate random variable emission they expect their data to have. """
        self.bern_p = bern_p
        self.gamma_a1 = gamma_a1
        self.gamma_scale1 = gamma_scale1
        self.gamma_a2 = gamma_a2
        self.gamma_scale2 = gamma_scale2

    def rvs(self, size):  # user has to identify what the multivariate (or univariate if he or she so chooses) random variable looks like
        """ User-defined way of calculating a random variable given the parameters of the state stored in that observation's object. """
        # {
        bern_obs = sp.bernoulli.rvs(p=self.bern_p, size=size)  # bernoulli observations
        gamma_obsG1 = sp.gamma.rvs(a=self.gamma_a1, scale=self.gamma_scale1, size=size)  # gamma observations
        gamma_obsG2 = sp.gamma.rvs(a=self.gamma_a2, scale=self.gamma_scale2, size=size)
        # } is user-defined in that they have to define and maintain the order of the multivariate random variables.
        # These tuples of observations will go into the cells in the lineage tree.
        list_of_tuple_of_obs = list(zip(bern_obs, gamma_obsG1, gamma_obsG2))
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
        gamma_llG1 = sp.gamma.pdf(x=tuple_of_obs[1], a=self.gamma_a1, scale=self.gamma_scale1)  # gamma likelihood for G1
        gamma_llG2 = sp.gamma.pdf(x=tuple_of_obs[2], a=self.gamma_a2, scale=self.gamma_scale2)  # gamma likelihood for G2

        return bern_ll * gamma_llG1 * gamma_llG2

    def estimator(self, list_of_tuples_of_obs):
        """ User-defined way of estimating the parameters given a list of the tuples of observations from a group of cells. """
        # unzipping the list of tuples
        unzipped_list_of_tuples_of_obs = list(zip(*list_of_tuples_of_obs))

        # getting the observations as individual lists
        # {
        try:
            bern_obs = list(unzipped_list_of_tuples_of_obs[0])
            gamma_obsG1 = list(unzipped_list_of_tuples_of_obs[1])
            gamma_obsG2 = list(unzipped_list_of_tuples_of_obs[2])
        except BaseException:
            bern_obs = []
            gamma_obsG1 = []
            gamma_obsG2 = []

        bern_p_estimate = bernoulli_estimator(bern_obs)
        gamma_a1_estimate, gamma_scale1_estimate = gamma_estimator(gamma_obsG1)
        gamma_a2_estimate, gamma_scale2_estimate = gamma_estimator(gamma_obsG2)

        state_estimate_obj = StateDistribution2(bern_p=bern_p_estimate,
                                                gamma_a1=gamma_a1_estimate,
                                                gamma_scale1=gamma_scale1_estimate,
                                                gamma_a2=gamma_a2_estimate,
                                                gamma_scale2=gamma_scale2_estimate)
        # } requires the user's attention.
        # Note that we return an instance of the state distribution class, but now instantiated with the parameters
        # from estimation. This is then stored in the original state distribution object which then gets updated
        # if this function runs again.
        return state_estimate_obj

    def __repr__(self):
        return "State object w/ parameters: {}, {}, {}, {}, {}, {}.".format(self.bern_p,
                                                                            self.gamma_a1,
                                                                            self.gamma_scale1,
                                                                            self.gamma_a2,
                                                                            self.gamma_scale2)


def prune_rule(cell):
    """ User-defined function that checks whether a cell's subtree should be removed. """
    truther = False
    if cell.obs[0] == 0:
        truther = True  # cell has died; subtree must be removed
    return truther


def tHMM_E_init2():
    return StateDistribution2(0.9,
                              10 * (np.random.uniform()),
                              1.5,
                              10 * (np.random.uniform()),
                              1.5)

# Because parameter estimation requires that estimators be written or imported, the user should be able to provide
# estimators that can solve for the parameters that describe the distributions. We provide some estimators below as an example.
# Their use in the StateDistribution class is shown in the estimator class method. User must take care to define estimators that
# can handle the case where the list of observations is empty.


def report_time2(cell):
    """ Given any cell in the lineage, this function walks through the cell's ancestors and return how long it has taken so far. """
    list_parents = [cell]
    taus = 0.0 + cell.obs[1] + cell.obs[2]

    for cell in list_parents:
        if cell._isRootParent():
            break
        elif cell.parent not in list_parents:
            list_parents.append(cell.parent)
            taus = taus + cell.parent.obs[1] + cell.parent.obs[2]
    return taus


def get_experiment_time(lineage):
    """ This function is to find the amount of time it took for the cells to be generated and reach to the desired number of cells. """
    leaf_times = []
    for cell in lineage.output_leaves:
        temp = report_time2(cell)
        leaf_times.append(temp)
    longest = max(leaf_times)
    return longest
