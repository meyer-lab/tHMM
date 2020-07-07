""" This file is completely user defined. We have provided a general starting point for the user to use as an example. """
import numpy as np
import scipy.stats as sp

from .stateCommon import basic_censor
from ..CellVar import Time


class StateDistribution:
    """
    StateDistribution for cells with Gaussian observations.
    """

    def __init__(self, norm_loc=10.0, norm_scale=1.0):
        """ Initialization function should take in just in the parameters for the observations that comprise the multivariate random variable emission they expect their data to have. """
        assert norm_scale > 0
        self.params = [norm_loc, norm_scale]

    def rvs(self, size):  # user has to identify what the multivariate (or univariate if he or she so chooses) random variable looks like
        """ User-defined way of calculating a random variable given the parameters of the state stored in that observation's object. """
        # {
        norm_obs = sp.norm.rvs(loc=self.params[0], scale=self.params[1], size=size)  # normal observations
        # } is user-defined in that they have to define and maintain the order of the multivariate random variables.
        # These tuples of observations will go into the cells in the lineage tree.
        return (norm_obs, )

    def pdf(self, tuple_of_obs):  # user has to define how to calculate the likelihood
        """ User-defined way of calculating the likelihood of the observation stored in a cell. """
        # In the case of a univariate observation, the user still has to define how the likelihood is calculated,
        # but has the ability to just return the output of a known scipy.stats.<distribution>.<{pdf,pmf}> function.
        # In the case of a multivariate observation, the user has to decide how the likelihood is calculated.
        # In our example, we assume the observation's are uncorrelated across the dimensions (across the different
        # distribution observations), so the likelihood of observing the multivariate observation is just the product of
        # the individual observation likelihoods.
        return sp.norm.pdf(tuple_of_obs[0], self.params[0], self.params[1])

    def estimator(self, list_of_tuples_of_obs, gammas):
        """ User-defined way of estimating the parameters given a list of the tuples of observations from a group of cells. """
        # unzipping the list of tuples
        unzipped_list_of_tuples_of_obs = list(zip(*list_of_tuples_of_obs))

        # getting the observations as individual lists
        # {
        norm_obs = list(unzipped_list_of_tuples_of_obs[0])

        eps = np.finfo(float).eps
        self.params[0] = (sum(gammas * norm_obs) + eps) / (sum(gammas) + eps)
        self.params[1] = ((sum(gammas * (norm_obs - self.params[0]) ** 2) + eps) / (sum(gammas) + eps)) ** 0.5
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
                    cell.time = Time(0, cell.obs[0])
            else:
                for cell in level:
                    cell.time = Time(cell.parent.time.endT, cell.parent.time.endT + cell.obs[0])

    def censor_lineage(self, censor_condition, full_list_of_gens, full_lineage, **kwargs):
        """
        This function removes those cells that are intended to be remove
        from the output binary tree based on emissions.
        It takes in LineageTree object, walks through all the cells in the output binary tree,
        applies the pruning to each cell that is supposed to be removed,
        and returns the censored list of cells.
        """
        if kwargs:
            _ = kwargs.get("desired_experiment_time", 2e12)

        if censor_condition == 0:
            output_lineage = full_lineage
            return output_lineage

        output_lineage = []
        for gen_minus_1, level in enumerate(full_list_of_gens[1:]):
            for cell in level:
                basic_censor(cell)
                if cell.observed:
                    output_lineage.append(cell)
        return output_lineage

    def __repl__(self):
        return f"{self.params}"

    def __str__(self):
        return self.__repl__()
