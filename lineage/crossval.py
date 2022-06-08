""" Cross validation. """
import numpy as np
from sklearn.utils import shuffle
from copy import deepcopy
import itertools as it
from typing import Tuple

def hide_observation(lineages: list, percentage: float) -> Tuple[list, list, list]:
    """ Taking a list of lineages and the percentage of cells want to be masked, it marks those x% as -1.
    This x% is selected from all cells and all lineages together such that we create a list of all cells in a population, hide x% and then regroup to their lineages.
    We mark a random number of cells' observations as negative, to be removed from fitting.
    :return new_lineages: list of lineages in which x% of cells were randomly masked. This forms our train lineages.
    :return new_hide_index: list of arrays, each array corresponding to indexes of masked cells in each lineage.
    :return obss: Those observations that have been masked, which are technically the test data.
    """

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
    # each array for each lineage
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
