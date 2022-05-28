""" To plot a summary of cross validation. """
import numpy as np
from copy import deepcopy
import scipy.stats as sp
import itertools as it
from ..LineageTree import LineageTree
from .common import getSetup
from ..Analyze import Analyze_list, Results
from ..tHMM import tHMM, fit_list
from ..BaumWelch import calculate_stationary
from ..states.StateDistributionGamma import StateDistribution

desired_num_states = np.arange(1, 8)

T = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=float)

# pi: the initial probability vector
pi = calculate_stationary(T)

# bern, gamma_a, gamma_scale
state0 = StateDistribution(0.99, 100, 0.1)
state1 = StateDistribution(0.75, 80, 0.5)
E = [state0, state1]

def makeFigure():
    """
    Makes figure 19.
    """
    ax, f = getSetup((4, 4), (1, 1))

    likelihoods = np.zeros((7, 10))
    for i in range(10):
        likelihoods[:, i] = all_cv()

    ax[0].scatter(desired_num_states, likelihoods)

    return f

def all_cv():
    """ find out the likelihoods for various masking percentages."""
    likelihood = []
    for i in desired_num_states:
        likelihood.append(cv(i))

        print("likelihood for states ", likelihood)
    return likelihood

def cv(num_states):
    """Simplest case of cross validation."""

    # create a lineage
    complete_lineages = [LineageTree.init_from_parameters(pi, T, E, 31) for _ in range(10)]
    true_states_by_lineage = [[cell.state for cell in complete_lineage.output_lineage] for complete_lineage in complete_lineages]

    # hide some percentage of observations
    lineages, indexes = [], []
    for complete_lin in complete_lineages:
        lineage, hide_index = hide_observation(complete_lin, 0.1)
        lineages.append(lineage)
        indexes.append(hide_index)

    # fit the not-hidden data from the lineages to the model
    tHMMobj_list, LL = Analyze_list([lineages], num_states)

    # assign the predicted states to each cell
    states_list = [tHMMobj.predict() for tHMMobj in tHMMobj_list]

    for idx, tHMMobj in enumerate(tHMMobj_list):
        for lin_indx, lin in enumerate(tHMMobj.X):
            for cell_indx, cell in enumerate(lin.output_lineage):
                cell.state = states_list[idx][lin_indx][cell_indx]

    # print the total likelihood of the observations to the assigned compared to the other state
    lls = calc_likelihood(tHMMobj_list[0], complete_lineages, indexes)[0]
    return lls


def hide_observation(lineage, percentage):
    """This assumes we have cell lifetime and bernoulli as observations.
    We mark a random number of cells' lifetime as negative, to be removed from fitting."""

    new_lineage = deepcopy(lineage)
    num_cells = len(lineage.output_lineage)
    hide_index = sp.multinomial.rvs(n=int(percentage * num_cells), p=[1 / num_cells]*num_cells, size=1)[0]

    for ix, cell in enumerate(new_lineage.output_lineage):
        if hide_index[ix] == 1: # means we hide the cell lifetime
            cell.obs = -1 * np.ones(len(cell.obs))

    return new_lineage, hide_index


def calc_likelihood(tHMMobj, complete_lineages, hidden_index):
    """ Calculates the likelihood of cell's observation to the state it is assigned to, and sums over all for each lineage. """

    # find the list of hidden observations with the same order find their corresponding estimated state
    hidden_obs = []
    hidden_states = []
    for i, complete_lin in enumerate(complete_lineages):
        tmp1, tmp2 = [], []
        for ix, cell in enumerate(complete_lin.output_lineage):
            if hidden_index[i][ix] == 1:
                tmp1.append(cell.obs)
                tmp2.append(tHMMobj.X[i].output_lineage[ix].state)
            hidden_obs.append(tmp1)
            hidden_states.append(tmp2)

    # calculate the likelihood of the observation, to the state it is assigned to
    Ls = 0
    for i, hid in enumerate(hidden_obs):
        for ix, obs in enumerate(hid):
            Ls -= np.log(tHMMobj.estimate.E[hidden_states[i][ix]].pdf(np.array(obs)[np.newaxis, :]))

    return Ls
