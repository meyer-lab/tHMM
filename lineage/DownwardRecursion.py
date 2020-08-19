""" File holds the code for the downward recursion. """

import math
import numpy as np
import logging
from .UpwardRecursion import beta_parent_child_func


def get_root_gammas(tHMMobj, betas):
    """Need the first gamma terms in the baum welch, which are just the beta values of the root nodes.
    """
    logging.info('started get_root_gammas')
    gammas = []

    for num, lineageObj in enumerate(tHMMobj.X):  # for each lineage in our Population
        gamma_array = np.zeros((len(lineageObj.output_lineage), tHMMobj.num_states))
        gamma_array[0, :] = betas[num][0, :]
        gammas.append(gamma_array)

    for _, gamma in enumerate(gammas):  # for each lineage in our Population
        assert np.isclose(np.sum(gamma[0]), 1.0)

    return gammas


def get_nonroot_gammas(tHMMobj, MSD, gammas, betas):
    """Get the gammas for all other nodes using recursion from the root nodes.
    """
    logging.info('started get_nonroot_gammas')
    T = tHMMobj.estimate.T

    for num, lineageObj in enumerate(tHMMobj.X):  # for each lineage in our Population
        lineage = lineageObj.output_lineage

        assert not math.isnan(np.all(MSD[num])), f"MSD for some lineage/lineages is nan!"
        with np.errstate(divide="ignore", invalid="ignore"):
            coeffs = betas[num] / MSD[num]
            assert not math.isnan(np.all(coeffs)), f"beta over MSD which is coeffs is nan"

        for level in lineageObj.output_list_of_gens[1:]:
            for cell in level:
                parent_idx = lineage.index(cell)

                for daughter in cell.get_daughters():
                    child_idx = lineage.index(daughter)

                    beta_parent = beta_parent_child_func(beta_array=betas[num], T=T, MSD_array=MSD[num], node_child_n_idx=child_idx)

                    with np.errstate(divide="ignore", invalid="ignore"):
                        sum_holder = np.matmul(gammas[num][parent_idx, :] / beta_parent, T)
                        assert not math.isnan(np.all(sum_holder))

                    gammas[num][child_idx, :] = coeffs[child_idx, :] * sum_holder

        assert np.all(gammas[num][0, :] == betas[num][0, :])

    for _, gg in enumerate(gammas):
        assert np.allclose(np.sum(gg, axis=1), 1.0, atol=0.01), f"sum of gammas are not right {gg}"


def sum_nonleaf_gammas(lineageObj, gamma_arr):
    """Sum of the gammas of the cells that are able to divide, that is,
    sum the of the gammas of all the nonleaf cells.
    """
    holder_wo_leaves = np.zeros(gamma_arr.shape[1])
    for level in lineageObj.output_list_of_gens[1:]:  # sum the gammas for cells that are transitioning
        for cell in level:
            if not cell.isLeaf():
                cell_idx = lineageObj.output_lineage.index(cell)
                holder_wo_leaves += gamma_arr[cell_idx, :]

    return holder_wo_leaves
