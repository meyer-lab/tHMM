""" Cross validation. """
from scipy.stats import bernoulli
from copy import deepcopy

def hide_observation(lineages: list, percentage: float) -> list:
    """ Taking a list of lineages and the percentage of cells want to be masked, it marks those x% as -1.
    This x% is selected from all cells and all lineages together such that we create a list of all cells in a population, hide x% and then regroup to their lineages.
    We mark a random number of cells' observations as negative, to be removed from fitting.
    :return new_lineages: list of lineages in which x% of cells were randomly masked. This forms our train lineages.
    :return new_hide_index: list of arrays, each array corresponding to indexes of masked cells in each lineage.
    :return obss: Those observations that have been masked, which are technically the test data.
    """

    new_lineages = deepcopy(lineages)

    for i, new_lineage in enumerate(new_lineages):
        for ix, cell in enumerate(new_lineage.output_lineage):

            bern = bernoulli.rvs(p=0.25, size=1)
            if bern == 1:
                # negate the cell observations at those specific indexes
                cell.obs = [-1 * o for o in cell.obs]

    return new_lineages
