""" File holds the code for the downward recursion. """

import numpy as np


def get_gammas(tHMMobj, MSD, betas):
    """ Get the gammas for all other nodes using recursion from the root nodes. """
    T = tHMMobj.estimate.T
    gammas = []

    for num, lineageObj in enumerate(tHMMobj.X):  # for each lineage in our Population
        gamma_array = np.zeros((len(lineageObj.output_lineage), tHMMobj.num_states))
        gamma_array[0, :] = betas[num][0, :]
        gammas.append(gamma_array)

    for num, lineageObj in enumerate(tHMMobj.X):  # for each lineage in our Population
        lineage = lineageObj.output_lineage
        coeffs = betas[num] / (MSD[num] + np.finfo(np.float).eps)

        for level in lineageObj.output_list_of_gens[1:]:
            for cell in level:
                parent_idx = lineage.index(cell)
                gam = gammas[num][parent_idx, :]

                for daughter in cell.get_daughters():
                    child_idx = lineage.index(daughter)

                    beta_parent = np.clip(T @ coeffs[child_idx, :], np.finfo(np.float).eps, np.inf)
                    gammas[num][child_idx, :] = coeffs[child_idx, :] * np.matmul(gam / beta_parent, T)

    for _, gamma in enumerate(gammas):  # for each lineage in our Population
        assert np.all(np.isfinite(gamma))
        np.testing.assert_allclose(np.sum(gamma[0]), 1.0)

    return gammas


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
