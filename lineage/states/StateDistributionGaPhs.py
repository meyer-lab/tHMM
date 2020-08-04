""" State distribution class for separated G1 and G2 phase durations as observation. """
import math
import numpy as np
import scipy.stats as sp

from .stateCommon import bern_pdf, bernoulli_estimator, gamma_pdf, gamma_estimator, basic_censor, gamma_sf
from ..CellVar import Time


class StateDistribution:
    """ For G1 and G2 separated as observations. """

    # user has to identify what parameters to use for each state
    def __init__(self, bern_p1=0.9, bern_p2=0.75, gamma_a1=7.0, gamma_scale1=3, gamma_a2=14.0, gamma_scale2=6):
        """ Initialization function should take in just in the parameters for the observations that comprise the multivariate random variable emission they expect their data to have. """
        self.params = [bern_p1, bern_p2, gamma_a1,
                       gamma_scale1, gamma_a2, gamma_scale2]

    def rvs(self, size):  # user has to identify what the multivariate (or univariate if he or she so chooses) random variable looks like
        """ User-defined way of calculating a random variable given the parameters of the state stored in that observation's object. """
        # {
        bern_obsG1 = sp.bernoulli.rvs(
            p=self.params[0], size=size)  # bernoulli observations
        bern_obsG2 = sp.bernoulli.rvs(p=self.params[1], size=size)
        gamma_obsG1 = sp.gamma.rvs(
            a=self.params[2], scale=self.params[3], size=size)  # gamma observations
        gamma_obsG2 = sp.gamma.rvs(
            a=self.params[4], scale=self.params[5], size=size)
        gamma_censor_obsG1 = [1] * size
        gamma_censor_obsG2 = [1] * size
        # } is user-defined in that they have to define and maintain the order of the multivariate random variables.
        # These tuples of observations will go into the cells in the lineage tree.
        return bern_obsG1, bern_obsG2, gamma_obsG1, gamma_obsG2, gamma_censor_obsG1, gamma_censor_obsG2

    def pdf(self, tuple_of_obs):  # user has to define how to calculate the likelihood
        """ User-defined way of calculating the likelihood of the observation stored in a cell. """
        # In the case of a univariate observation, the user still has to define how the likelihood is calculated,
        # but has the ability to just return the output of a known scipy.stats.<distribution>.<{pdf,pmf}> function.
        # In the case of a multivariate observation, the user has to decide how the likelihood is calculated.
        # In our example, we assume the observation's are uncorrelated across the dimensions (across the different
        # distribution observations), so the likelihood of observing the multivariate observation is just the product of
        # the individual observation likelihoods.

        bern_llG1 = 1
        if not math.isnan(tuple_of_obs[0]):
            # observed
            assert tuple_of_obs[0] == 0 or tuple_of_obs[0] == 1
            bern_llG1 = bern_pdf(tuple_of_obs[0], self.params[0])

        bern_llG2 = 1
        if not math.isnan(tuple_of_obs[1]):
            # observed
            assert tuple_of_obs[1] == 0 or tuple_of_obs[1] == 1
            bern_llG2 = bern_pdf(tuple_of_obs[1], self.params[1])

        gamma_llG1 = 1
        if tuple_of_obs[4] == 1:
            # uncensored
            gamma_llG1 = gamma_pdf(
                tuple_of_obs[2], self.params[2], self.params[3])
        elif tuple_of_obs[4] == 0:
            # censored
            gamma_llG1 = gamma_sf(
                tuple_of_obs[2], a=self.params[2], scale=self.params[3])
        else:
            assert math.isnan(tuple_of_obs[4])
            # G1 lifetime not observed
            assert math.isnan(tuple_of_obs[2])
            gamma_llG1 = 1

        gamma_llG2 = 1
        if tuple_of_obs[5] == 1:
            # uncensored
            gamma_llG2 = gamma_pdf(
                tuple_of_obs[3], self.params[4], self.params[5])
        elif tuple_of_obs[5] == 0:
            # censored
            gamma_llG2 = gamma_sf(
                tuple_of_obs[3], a=self.params[4], scale=self.params[5])
        elif math.isnan(tuple_of_obs[5]):
            # unobserved
            assert math.isnan(tuple_of_obs[3]) and math.isnan(
                tuple_of_obs[5]) and math.isnan(tuple_of_obs[1])
            gamma_llG2 = 1

        return bern_llG1 * bern_llG2 * gamma_llG1 * gamma_llG2

    def estimator(self, list_of_tuples_of_obs, gammas):
        """ User-defined way of estimating the parameters given a list of the tuples of observations from a group of cells. """
        # unzipping the list of tuples
        unzipped_list_of_tuples_of_obs = list(zip(*list_of_tuples_of_obs))

        # getting the observations as individual lists
        # {
        bern_obsG1 = np.array(unzipped_list_of_tuples_of_obs[0])
        bern_obsG2 = np.array(unzipped_list_of_tuples_of_obs[1])
        gamma_obsG1 = np.array(unzipped_list_of_tuples_of_obs[2])
        gamma_obsG2 = np.array(unzipped_list_of_tuples_of_obs[3])
        gamma_censor_obsG1 = np.array(unzipped_list_of_tuples_of_obs[4])
        gamma_censor_obsG2 = np.array(unzipped_list_of_tuples_of_obs[5])

        b1_mask = np.logical_not(np.isnan(bern_obsG1))
        self.params[0] = bernoulli_estimator(
            bern_obsG1[b1_mask], gammas[b1_mask])
        b2_mask = np.logical_not(np.isnan(bern_obsG2))
        self.params[1] = bernoulli_estimator(
            bern_obsG2[b2_mask], gammas[b2_mask])
        ga1_mask = np.logical_not(np.isnan(gamma_obsG1))
        self.params[2], self.params[3] = gamma_estimator(
            gamma_obsG1[ga1_mask], gamma_censor_obsG1[ga1_mask], gammas[ga1_mask])
        ga2_mask = np.logical_not(np.isnan(gamma_obsG2))
        self.params[4], self.params[5] = gamma_estimator(
            gamma_obsG2[ga2_mask], gamma_censor_obsG2[ga2_mask], gammas[ga2_mask])
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
                    cell.time = Time(
                        cell.parent.time.endT, cell.parent.time.endT + cell.obs[2] + cell.obs[3])
                    cell.time.transition_time = cell.parent.time.endT + \
                        cell.obs[2]

    def censor_lineage(self, censor_condition, full_list_of_gens, full_lineage, **kwargs):
        """
        This function removes those cells that are intended to be remove
        from the output binary tree based on emissions.
        It takes in LineageTree object, walks through all the cells in the output binary tree,
        applies the pruning to each cell that is supposed to be removed,
        and returns the censored list of cells.
        """
        if kwargs:
            desired_experiment_time = kwargs.get(
                "desired_experiment_time", 2e12)

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

    def __repl__(self):
        return f"{self.params}"

    def __str__(self):
        return self.__repl__()


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
        if cell.obs[0] == 0:
            cell.obs[1] = float('nan')  # unobserved
            cell.obs[3] = float('nan')  # unobserved
            cell.obs[5] = float('nan')  # unobserved
            cell.time.endT = cell.time.startT + cell.obs[2]
            cell.time.transition_time = cell.time.endT


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
