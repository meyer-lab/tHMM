""" Cross validation. """
import numpy as np
import itertools
from copy import deepcopy
import itertools as it
from .Analyze import run_Analyze_over

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


def hide_for_population(complete_population):
    """ Use the hide_obsrvation function for a population of cells. """

    train_population, hidden_indexes, hidden_obs = [], [], []
    for complete_lineages in complete_population:
        t_population, h_indexes, h_obs = [], [], []
        for complete_lin in complete_lineages:
            lineage, hide_index, hide_obs = hide_observation(complete_lin, 0.25)
            t_population.append(lineage)
            h_indexes.append(hide_index)
            h_obs.append(hide_obs)
        train_population.append(t_population)
        hidden_indexes.append(h_indexes)
        hidden_obs.append(h_obs)
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
