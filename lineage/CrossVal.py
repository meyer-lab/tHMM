""" Cross validation. """
import numpy as np
from copy import deepcopy
import itertools as it
from .Analyze import Analyze_list

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
        if hid == []:
            continue
        else:
            for ix, obs in enumerate(hid):
                Ls += tHMMobj_list[0].estimate.E[hidden_states[i][ix]].pdf(np.array(hidden_obs[i][ix])[np.newaxis, :])
    return Ls
