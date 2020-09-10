""" State distribution class for separated G1 and G2 phase durations as observation. """
import math
import numpy as np
import scipy.stats as sp

from .stateCommon import basic_censor
from .StateDistributionGamma import StateDistribution as GammaSD
from ..CellVar import Time


class StateDistribution:
    """ For G1 and G2 separated as observations. """

    def __init__(self, bern_p1=0.9, bern_p2=0.75, gamma_a1=7.0, gamma_scale1=3, gamma_a2=14.0, gamma_scale2=6, shape1=None, shape2=None):  # user has to identify what parameters to use for each state
        """ Initialization function should take in just in the parameters for the observations that comprise the multivariate random variable emission they expect their data to have. """
        self.params = np.array([bern_p1, bern_p2, gamma_a1, gamma_scale1, gamma_a2, gamma_scale2])
        self.G1 = GammaSD(bern_p=bern_p1, gamma_a=gamma_a1, gamma_scale=gamma_scale1, shape=shape1)
        self.G2 = GammaSD(bern_p=bern_p2, gamma_a=gamma_a2, gamma_scale=gamma_scale2, shape=shape2)

    def rvs(self, size):  # user has to identify what the multivariate (or univariate if he or she so chooses) random variable looks like
        """ User-defined way of calculating a random variable given the parameters of the state stored in that observation's object. """
        # {
        bern_obsG1, gamma_obsG1, gamma_censor_obsG1 = self.G1.rvs(size)
        bern_obsG2, gamma_obsG2, gamma_censor_obsG2 = self.G2.rvs(size)
        # } is user-defined in that they have to define and maintain the order of the multivariate random variables.
        # These tuples of observations will go into the cells in the lineage tree.
        return bern_obsG1, bern_obsG2, gamma_obsG1, gamma_obsG2, gamma_censor_obsG1, gamma_censor_obsG2

    def dist(self, other):
        """ Calculate the Wasserstein distance between this state emissions and the given. """
        assert isinstance(self, type(other))
        return self.G1.dist(other.G1) + self.G2.dist(other.G2)

    def dof(self):
        """ Return the degrees of freedom. """
        return self.G1.dof() + self.G2.dof()

    def pdf(self, x):  # user has to define how to calculate the likelihood
        """ User-defined way of calculating the likelihood of the observation stored in a cell. """
        # In the case of a univariate observation, the user still has to define how the likelihood is calculated,
        # but has the ability to just return the output of a known scipy.stats.<distribution>.<{pdf,pmf}> function.
        # In the case of a multivariate observation, the user has to decide how the likelihood is calculated.
        # In our example, we assume the observation's are uncorrelated across the dimensions (across the different
        # distribution observations), so the likelihood of observing the multivariate observation is just the product of
        # the individual observation likelihoods.
        G1_LL = self.G1.pdf(x[:, np.array([0, 2, 4])])
        G2_LL = self.G2.pdf(x[:, np.array([1, 3, 5])])

        return G1_LL * G2_LL

    def estimator(self, x, gammas):
        """ User-defined way of estimating the parameters given a list of the tuples of observations from a group of cells. """
        x = np.array(x)

        self.G1.estimator(x[:, np.array([0, 2, 4])], gammas)
        self.G2.estimator(x[:, np.array([1, 3, 5])], gammas)

        self.params[0] = self.G1.params[0]
        self.params[1] = self.G2.params[0]
        self.params[2:4] = self.G1.params[1:3]
        self.params[4:6] = self.G2.params[1:3]

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
                    cell.time = Time(0, cell.obs[2] + cell.obs[3])
                    cell.time.transition_time = 0 + cell.obs[2]
            else:
                for cell in level:
                    cell.time = Time(cell.parent.time.endT, cell.parent.time.endT + cell.obs[2] + cell.obs[3])
                    cell.time.transition_time = cell.parent.time.endT + cell.obs[2]

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
    if cell.obs[0] == 0 or cell.obs[1] == 0:
        if not cell.isLeafBecauseTerminal():
            cell.left.observed = False
            cell.right.observed = False
        if cell.obs[0] == 0:  # dies in G1
            cell.obs[1] = float('nan')  # unobserved
            cell.obs[3] = float('nan')  # unobserved
            cell.obs[5] = float('nan')  # unobserved
            cell.time.endT = cell.time.startT + cell.obs[2]
            cell.time.transition_time = cell.time.endT
        if cell.obs[1] == 0:  # dies in G2
            cell.time.endT = cell.time.startT + cell.obs[2] + cell.obs[3]


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
        cell.obs[1] = float('nan')  # unobserved
        cell.obs[3] = desired_experiment_time - cell.time.transition_time
        cell.obs[5] = 0  # censored
        if not cell.isLeafBecauseTerminal():
            cell.left.observed = False
            cell.right.observed = False

    if cell.time.transition_time > desired_experiment_time:
        cell.time.endT = desired_experiment_time
        cell.time.transition_time = desired_experiment_time
        cell.obs[0] = float('nan')  # unobserved
        cell.obs[1] = float('nan')  # unobserved
        cell.obs[2] = desired_experiment_time - cell.time.startT
        cell.obs[3] = float('nan')  # unobserved
        cell.obs[4] = 0  # censored
        cell.obs[5] = float('nan')  # unobserved
        if not cell.isLeafBecauseTerminal():
            cell.left.observed = False
            cell.right.observed = False
