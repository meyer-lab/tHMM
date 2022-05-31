""" Cross validation. """
import numpy as np
from sklearn.utils import shuffle
import itertools
from copy import deepcopy
import itertools as it
from typing import Tuple
from .Analyze import run_Analyze_over

def hide_observation(lineages: list, percentage: float) -> Tuple[list, list, list]:
    """ Taking a list of lineages and the percentage of cells want to be masked, it marks those x% as -1.
    We mark a random number of cells' observations as negative, to be removed from fitting."""

    new_lineages = deepcopy(lineages)
    num_cells = 0
    len_lineage = [] # remember the length of each lineage
    for lin in lineages:
        num_cells += len(lin.output_lineage)
        len_lineage.append(len(lin.output_lineage))

    # create the indexes for hidden observations
    hide_index = np.zeros(num_cells)
    hide_index[:int(num_cells*percentage)] = 1
    hide_index = shuffle(hide_index)

    # to partition the hide_index (which is an array that has all cells in all lineages together) 
    # for each lineage as a list of arrays, each array for each lineage
    prev = 0
    new_hide_index = []
    for i in len_lineage:
        new_hide_index.append(hide_index[prev:prev+i])
        prev += i

    obss = [] # save those observations that will be masked
    for i, new_lineage in enumerate(new_lineages):
        tmp1 = []
        for ix, cell in enumerate(new_lineage.output_lineage):
            if new_hide_index[i][ix] == 1: # means we hide the cell lifetime
                tmp1.append(cell.obs)
                cell.obs = -1 * np.ones(len(cell.obs))
        obss.append(tmp1)
    
    for i, ob in enumerate(obss):
        assert np.sum(new_hide_index[i]) == len(ob)

    return new_lineages, new_hide_index, obss

def hide_for_population(complete_lineages, perc):
    """ Use the hide_obsrvation function for a population of cells. """

    train_population, hidden_indexes, hidden_obs = [], [], []
    for complete_lin in complete_lineages:
        lineage, hide_index, hide_obs = hide_observation(complete_lin, perc)
        train_population.append(lineage)
        hidden_indexes.append(hide_index)
        hidden_obs.append(hide_obs)
    return train_population, hidden_indexes, hidden_obs


def crossval(train_populations: list, hidden_indexes: list, hidden_obs: list, num_states: int):
    """ Perform cross validation for a drug treated population.
    train_populations: the populations after applying hide_observation.
    hidden_indexes: is a list of list of np.arrays for each lineage, 
    filled with zeros and ones. ones refer to the index of those cells that have been hidden.
    hidden_obs: list of list of tuples of observations that have been masked in the train_lineage.
    """

    # fit training data
    output = run_Analyze_over(train_populations, num_states)
    tHMMobj_list = []
    for out in output:
        tHMMobj_list.append(out[0])

    # predict states of hidden cells
    states_list = [tHMMobj.predict() for tHMMobj in tHMMobj_list]

    # hidden states
    hidden_states = []
    for i, lineage_st in enumerate(states_list):
        tmp = []
        for j, lin_st in enumerate(lineage_st):
            tmp.append(lin_st[hidden_indexes[i][j] == 1])
        hidden_states.append(tmp)

    Ls = 0
    for i, obs_lins in enumerate(hidden_obs):
        for j, obs_lin in enumerate(obs_lins):
            if obs_lin:
                for i2, obs_cell in enumerate(obs_lin): 
                    Ls += tHMMobj_list[i].estimate.E[hidden_states[i][j][i2]].pdf(np.array(obs_cell)[np.newaxis, :])
    return Ls
