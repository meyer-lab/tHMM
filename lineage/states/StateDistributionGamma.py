""" This file is completely user defined. We have provided a general starting point for the user to use as an example. """
import numpy as np
import scipy.stats as sp

from .stateCommon import gamma_estimator, basic_censor
from ..CellVar import Time


class StateDistribution:
    """
    StateDistribution for cells with gamma distributed times.
    """

    def __init__(self, bern_p=0.9, gamma_a=7, gamma_scale=4.5, shape=None):
        """ Initialization function should take in just in the parameters for the observations that comprise the multivariate random variable emission they expect their data to have. """
        self.params = np.array([bern_p, gamma_a, gamma_scale])
        self.const_shape = shape

    def rvs(self, size):  # user has to identify what the multivariate (or univariate if he or she so chooses) random variable looks like
        """ User-defined way of calculating a random variable given the parameters of the state stored in that observation's object. """
        # {
        bern_obs = sp.bernoulli.rvs(p=self.params[0], size=size)  # bernoulli observations
        gamma_obs = sp.gamma.rvs(a=self.params[1], scale=self.params[2], size=size)  # gamma observations
        gamma_obs_censor = [1] * size  # 1 if observed
        # } is user-defined in that they have to define and maintain the order of the multivariate random variables.
        # These tuples of observations will go into the cells in the lineage tree.
        return bern_obs, gamma_obs, gamma_obs_censor

    def dist(self, other):
        """ Calculate the Wasserstein distance between this state emissions and the given.
        Note that this does not take the Bernoulli into account. """
        assert isinstance(self, type(other))
        dist = np.absolute(self.params[1] * self.params[2] - other.params[1] * other.params[2])
        return dist

    def dof(self):
        """ Return the degrees of freedom. """
        if self.const_shape is None:
            return 3

        return 2

    def pdf(self, x):  # user has to define how to calculate the likelihood
        """ User-defined way of calculating the likelihood of the observation stored in a cell. """
        # In the case of a univariate observation, the user still has to define how the likelihood is calculated,
        # but has the ability to just return the output of a known scipy.stats.<distribution>.<{pdf,pmf}> function.
        # In the case of a multivariate observation, the user has to decide how the likelihood is calculated.
        # In our example, we assume the observation's are uncorrelated across the dimensions (across the different
        # distribution observations), so the likelihood of observing the multivariate observation is just the product of
        # the individual observation likelihoods.
        ll = np.zeros(x.shape[0])

        # Update uncensored Gamma
        ll[x[:, 2] == 1] = sp.gamma.logpdf(x[x[:, 2] == 1, 1], a=self.params[1], scale=self.params[2])

        # Update censored Gamma
        ll[x[:, 2] == 0] = sp.gamma.logsf(x[x[:, 2] == 0, 1], a=self.params[1], scale=self.params[2])

        # Remove dead cells
        ll[x[:, 0] == 0] = 0.001

        # Update for observed Bernoulli
        ll[np.isfinite(x[:, 0])] += sp.bernoulli.logpmf(x[np.isfinite(x[:, 0]), 0], self.params[0])

        return np.exp(ll)

    def estimator(self, x, gammas):
        """ User-defined way of estimating the parameters given a list of the tuples of observations from a group of cells. """
        # unzipping the list of tuples
        x = np.array(x)

        # getting the observations as individual lists
        # {
        bern_obs = x[:, 0]
        γ_obs = x[:, 1]
        gamma_obs_censor = x[:, 2]

        b_mask = np.isfinite(bern_obs)
        # Both unoberved and dead cells should be removed from gamma
        g_mask = np.logical_and(np.isfinite(γ_obs), bern_obs == 1)

        # Handle an empty state
        if np.sum(gammas[b_mask]) == 0.0:
            self.params[0] = np.average(bern_obs[b_mask])
        else:
            self.params[0] = np.average(bern_obs[b_mask], weights=gammas[b_mask])

        self.params[1], self.params[2] = gamma_estimator(γ_obs[g_mask], gamma_obs_censor[g_mask], gammas[g_mask], self.const_shape, self.params[1:3])

        # } requires the user's attention.
        # Note that we return an instance of the state distribution class, but now instantiated with the parameters
        # from estimation. This is then stored in the original state distribution object which then gets updated
        # if this function runs again.

    def assign_times(self, list_of_gens):
        """
        Assigns the start and end time for each cell in the lineage.
        The time observation will be stored in the cell's observation parameter list
        in the second position (index 1). See the other time functions to understand.
        This is used in the creation of LineageTrees
        """
        # traversing the cells by generation
        for gen_minus_1, level in enumerate(list_of_gens[1:]):
            true_gen = gen_minus_1 + 1  # generations are 1-indexed
            if true_gen == 1:
                for cell in level:
                    assert cell.isRootParent()
                    cell.time = Time(0, cell.obs[1])
            else:
                for cell in level:
                    cell.time = Time(cell.parent.time.endT, cell.parent.time.endT + cell.obs[1])

    def censor_lineage(self, censor_condition, full_list_of_gens, full_lineage, **kwargs):
        """
        This function removes those cells that are intended to be remove
        from the output binary tree based on emissions.
        It takes in LineageTree object, walks through all the cells in the output binary tree,
        applies the pruning to each cell that is supposed to be removed,
        and returns the censored list of cells.
        """
        if kwargs:
            desired_experiment_time = kwargs.get("desired_experiment_time", 2e12)

        if censor_condition == 0:
            output_lineage = full_lineage
            return output_lineage

        output_lineage = []
        for gen_minus_1, level in enumerate(full_list_of_gens[1:]):
            true_gen = gen_minus_1 + 1  # generations are 1-indexed
            if true_gen == 1:
                for cell in level:
                    assert cell.isRootParent()
                    basic_censor(cell)
                    if censor_condition == 1:
                        fate_censor(cell)
                    elif censor_condition == 2:
                        time_censor(cell, desired_experiment_time)
                    elif censor_condition == 3:
                        fate_censor(cell)
                        time_censor(cell, desired_experiment_time)
                    if cell.observed:
                        output_lineage.append(cell)
            else:
                for cell in level:
                    basic_censor(cell)
                    if censor_condition == 1:
                        fate_censor(cell)
                    elif censor_condition == 2:
                        time_censor(cell, desired_experiment_time)
                    elif censor_condition == 3:
                        fate_censor(cell)
                        time_censor(cell, desired_experiment_time)
                    if cell.observed:
                        output_lineage.append(cell)
        return output_lineage


def fate_censor(cell):
    """
    User-defined function that checks whether a cell's subtree should be removed.
    Our example is based on the standard requirement that the first observation
    (index 0) is a measure of the cell's fate (1 being alive, 0 being dead).
    Clearly if a cell has died, its subtree must be removed.
    """
    if cell.obs[0] == 0:
        if not cell.isLeafBecauseTerminal():
            cell.left.observed = False
            cell.right.observed = False


def time_censor(cell, desired_experiment_time):
    """
    User-defined function that checks whether a cell's subtree should be removed.
    Our example is based on the standard requirement that the second observation
    (index 1) is a measure of the cell's lifetime.
    If a cell has lived beyond a certain experiment time, then its subtree
    must be removed.
    """
    if cell.time.endT > desired_experiment_time:
        cell.time.endT = desired_experiment_time
        cell.obs[0] = float('nan')
        cell.obs[1] = desired_experiment_time - cell.time.startT
        cell.obs[2] = 0  # censored
        if not cell.isLeafBecauseTerminal():
            # the daughters are no longer observed
            cell.left.observed = False
            cell.right.observed = False
