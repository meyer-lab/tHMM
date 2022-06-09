""" Cross validation. """
import numpy as np
from sklearn.utils import shuffle
import itertools
from copy import deepcopy
import itertools as it
from typing import Tuple
from .Analyze import run_Analyze_over

def hide_for_population(complete_lineages: list, perc: float) -> Tuple[list, list, list]:
    """ Use the hide_obsrvation function for a population of cells. 
    This is used for running the cross validation for parallel fitting that we create list of list of lineages."""

    train_population, hidden_indexes, hidden_obs = [], [], []
    for complete_lin in complete_lineages:
        lineage, hide_index, hide_obs = hide_observation(complete_lin, perc)
        train_population.append(lineage)
        hidden_indexes.append(hide_index)
        hidden_obs.append(hide_obs)
    return train_population, hidden_indexes, hidden_obs


def crossval(train_populations: list, hidden_indexes: list, hidden_obs: list, num_states: np.array):
    """ Perform cross validation for the experimental data which runs in parallel for all states.
    :param train_populations: the populations after applying hide_observation. This includes the list of list of lineages.
    :param hidden_indexes: is a list of list of np.arrays for each lineage, 
    filled with zeros and ones. ones refer to the index of those cells that have been hidden.
    :param hidden_obs: list of list of tuples of observations that have been masked in the train_lineage.
    :param num_states: is a range of states we want to run the cross validation for.
    """

    # fit training data
    output = run_Analyze_over(train_populations, num_states, atonce=True)
    tHMMobj_list_states = []
    for out in output:
        tHMMobj_list_states.append(out[0])

    # predict states of hidden cells
    # states_list: len(states_list) = num_states
    states_list = [[tHMMobj.predict() for tHMMobj in tHMMobj_list] for tHMMobj_list in tHMMobj_list_states]

    # calculate the likelihood of observations of masked cells to their assigned state
    LLs = []
    for k in range(len(num_states)):
        Ls = 0
        tHMMobj_list = output[k][0]

        # find the states of masked cells
        hidden_states = []
        for i, lineage_st in enumerate(states_list[k]):
            tmp = []
            for j, lin_st in enumerate(lineage_st):
                tmp.append(lin_st[hidden_indexes[i][j] == 1])
            hidden_states.append(tmp)


        for i, obs_lins in enumerate(hidden_obs):
            for j, obs_lin in enumerate(obs_lins):
                if obs_lin:
                    for i2, obs_cell in enumerate(obs_lin): 
                        Ls += tHMMobj_list[i].estimate.E[hidden_states[i][j][i2]].pdf(np.array(obs_cell)[np.newaxis, :])
        LLs.append(Ls)
    return LLs
