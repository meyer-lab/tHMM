""" Cross validation. """
import numpy as np
import itertools
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
    
    assert np.sum(hide_index) == len(obss)

    return new_lineage, hide_index, obss


def crossval(train_lineages: list, hidden_indexes: list, hidden_obs: list, num_states: int):
    """ Perform cross validation for a population of lineages.
    train_lineages: the lineages after applying hide_observation.
    hidden_indexes: is a list of np.arrays for each lineage, 
    filled with zeros and ones. ones refer to the index of those cells that have been hidden.
    hidden_obs: list of tuples of observations that have been masked in the train_lineage.
    """

    # fit training data
    tHMMobj_list, LL = Analyze_list([train_lineages], num_states)

    # predict states of hidden cells
    states_list = tHMMobj_list[0].predict()

    # hidden states
    hidden_states = []
    for i, lineage_st in enumerate(states_list):
        hidden_states.append(lineage_st[hidden_indexes[i] == 1])


    Ls = 0
    for i, obs_lin in enumerate(hidden_obs):
        if obs_lin:
            for i2, obs_cell in enumerate(obs_lin): 
                Ls += tHMMobj_list[0].estimate.E[hidden_states[i][i2]].pdf(np.array(obs_cell)[np.newaxis, :])
    return Ls
