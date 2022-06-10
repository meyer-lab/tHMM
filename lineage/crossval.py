""" Cross validation. """
import numpy as np
from scipy.stats import bernoulli
from copy import deepcopy
from .Analyze import run_Analyze_over


def hide_observation(lineages: list, percentage: float) -> list:
    """Taking a list of lineages and the percentage of cells want to be masked, it marks those x% negative."""
    new_lineages = deepcopy(lineages)
    for new_lineage in new_lineages:
        for cell in new_lineage.output_lineage:
            if bernoulli.rvs(p=percentage, size=1):
                # negate the cell observations to mask them
                cell.obs = [-1 * o for o in cell.obs]

    return new_lineages


def crossval(train_populations: list, num_states: np.array):
    """Perform cross validation for the experimental data which runs in parallel for all states.
    :param train_populations: the populations after applying hide_observation. This includes the list of list of lineages.
    :param hidden_indexes: is a list of list of np.arrays for each lineage,
    filled with zeros and ones. ones refer to the index of those cells that have been hidden.
    :param hidden_obs: list of list of tuples of observations that have been masked in the train_lineage.
    :param num_states: is a range of states we want to run the cross validation for.
    """
    # fit training data by parallel.
    output = run_Analyze_over(train_populations, num_states, atonce=True)
    # save the tHMMobj for each number of states that is being run
    tHMMobj_list_states = []
    for out in output:
        tHMMobj_list_states.append(out[0])

    # predict states of hidden cells
    # states_list: len(states_list) = num_states. states_list[0] for the 1-state model, etc.
    states_list = [[tHMMobj.predict() for tHMMobj in tHMMobj_list] for tHMMobj_list in tHMMobj_list_states]

    # calculate the log likelihood of observations of masked cells to their assigned state
    LLs = []
    for k in range(len(num_states)):
        tHMMobj_list = output[k][0]

        # assign the predicted states to each cell
        for idx, tHMMobj in enumerate(tHMMobj_list):
            for lin_indx, lin in enumerate(tHMMobj.X):
                for cell_indx, cell in enumerate(lin.output_lineage):
                    cell.state = states_list[k][idx][lin_indx][cell_indx]

        Logls = 0
        # calculate the log likelihood of hidden observations
        for idx, tHMMobj in enumerate(tHMMobj_list):
            for lin_indx, lin in enumerate(tHMMobj.X):
                for cell_indx, cell in enumerate(lin.output_lineage):
                    if cell.obs[2] < 0:
                        positive_obs = [-1 * o for o in cell.obs]
                        Logls += tHMMobj.estimate.E[cell.state].logpdf(np.array(positive_obs)[np.newaxis, :])

        LLs.append(Logls)
    return LLs
