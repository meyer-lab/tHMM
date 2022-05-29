""" To plot a summary of cross validation. """
import numpy as np
from copy import deepcopy
import itertools as it
from ..LineageTree import LineageTree
from .common import getSetup
from ..Analyze import Analyze_list
from ..BaumWelch import calculate_stationary
from ..states.StateDistributionGamma import StateDistribution

desired_num_states = np.arange(1, 8)

T = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=float)

# pi: the initial probability vector
pi = calculate_stationary(T)

# bern, gamma_a, gamma_scale
state0 = StateDistribution(0.99, 40, 1)
state1 = StateDistribution(0.99, 40, 2)
E = [state0, state1]

def makeFigure():
    """
    Makes figure 19.
    """
    ax, f = getSetup((4, 4), (1, 1))

    # create a population
    complete_lineages = [LineageTree.init_from_parameters(pi, T, E, 63) for _ in range(50)]

    # create training data by hiding 20% of cells in each lineage
    train_lineages, hidden_indexes, hidden_obs = [], [], []
    for complete_lin in complete_lineages:
        lineage, hide_index, hide_obs = hide_observation(complete_lin, 0.2)
        train_lineages.append(lineage)
        hidden_indexes.append(hide_index)
        hidden_obs.append(hide_obs)

    ll = []
    for i in desired_num_states:
        ll.append(crossval(train_lineages, hidden_indexes, hidden_obs, i))
    print(ll)

    ax[0].plot(desired_num_states, ll)
    return f


def hide_observation(lineage, percentage):
    """This assumes we have cell lifetime and bernoulli as observations.
    We mark a random number of cells' lifetime as negative, to be removed from fitting."""

    new_lineage = deepcopy(lineage)
    num_cells = len(lineage.output_lineage)
    # create the indexes for hidden observations
    hide_index = np.zeros(num_cells)
    hide_index[:int(num_cells*percentage)] = 1
    np.random.shuffle(hide_index)

    obss = []
    for ix, cell in enumerate(new_lineage.output_lineage):
        if hide_index[ix] == 1: # means we hide the cell lifetime
            obss.append(cell.obs)
            cell.obs = -1 * np.ones(len(cell.obs))

    return new_lineage, hide_index, obss


def crossval(train_lineages, hidden_indexes, hidden_obs, num_states):

    # fit training data
    tHMMobj_list, LL = Analyze_list([train_lineages], num_states)

    # predict states of hidden cells
    states_list = tHMMobj_list[0].predict()

    # hidden states
    hidden_states = []
    for i, st in enumerate(states_list):
        tmp1 = []
        for ii, c_st in enumerate(st):
            if hidden_indexes[i][ii] == 1:
                tmp1.append(c_st)
            hidden_states.append(tmp1)

    Ls = 0
    for i, hid in enumerate(hidden_obs):
        for ix, obs in enumerate(hid):
            Ls += tHMMobj_list[0].estimate.E[hidden_states[i][ix]].pdf(np.array(hidden_obs[i][ix])[np.newaxis, :])
    return Ls