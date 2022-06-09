""" Cross validation. """
from scipy.stats import bernoulli
from copy import deepcopy


def hide_observation(lineages: list, percentage: float) -> list:
    """ Taking a list of lineages and the percentage of cells want to be masked, it marks those x% negative."""
    new_lineages = deepcopy(lineages)
    for new_lineage in new_lineages:
        for cell in new_lineage.output_lineage:
            if bernoulli.rvs(p=percentage, size=1):
                # negate the cell observations to mask them
                cell.obs = [-1 * o for o in cell.obs]

    return new_lineages
