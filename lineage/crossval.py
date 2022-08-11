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
    gamma_lists = []
    for out in output:
        tHMMobj_list_states.append(out[0])
        gamma_lists.append(out[2])

    # calculate the log likelihood of observations of masked cells to each state based on the soft assignement
    LLs = []
    for k in range(len(num_states)):
        tHMMobj_list = output[k][0]
        gamma_list = output[k][2]

        Logls = 0
        # calculate the log likelihood of hidden observations
        for idx, tHMMobj in enumerate(tHMMobj_list):
            for lin_indx, lin in enumerate(tHMMobj.X):
                for cell_indx, cell in enumerate(lin.output_lineage):
                    if cell.obs[2] < 0:
                        positive_obs = [-1 * o for o in cell.obs]
                        tmp = 0
                        for i in range(k + 1):
                            tmp += np.exp(tHMMobj.estimate.E[i].logpdf(np.array(positive_obs)[np.newaxis, :])) * gamma_list[idx][lin_indx][cell_indx][i]

                        Logls += np.log(tmp)
        LLs.append(Logls)
    return LLs

def output_LL(complete_population):
    """ Given the complete population, it masks 25% of cells and prepares the data for parallel fitting using crossval function."""
    # create training data by hiding 25% of cells in each lineage
    train_population = [hide_observation(complete_pop, 0.25) for complete_pop in complete_population]
    # Copy out data to full set
    dataFull = []
    for _ in desired_num_states:
        dataFull.append(train_population)

    return crossval(dataFull, desired_num_states)
