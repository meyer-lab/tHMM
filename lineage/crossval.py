""" Cross validation. """
from scipy.stats import bernoulli
from copy import deepcopy

def hide_observation(lineages: list, percentage: float) -> list:
    """ Taking a list of lineages and the percentage of cells want to be masked, it marks those x% negative."""

    new_lineages = deepcopy(lineages)
    for i, new_lineage in enumerate(new_lineages):
        for ix, cell in enumerate(new_lineage.output_lineage):

            bern = bernoulli.rvs(p=percentage, size=1)
            if bern == 1:
                # negate the cell observations at those specific indexes
                cell.obs = [-1 * o for o in cell.obs]

    return new_lineages
