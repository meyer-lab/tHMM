""" File holds the code for the downward recursion. """

import numpy as np


def get_gammas(tHMMobj, MSD, betas):
    """ Get the gammas for all other nodes using recursion from the root nodes. """
    T = tHMMobj.estimate.T
    gammas = []

    for num, lO in enumerate(tHMMobj.X):  # for each lineage in our Population
        gamma_array = np.zeros((len(lO.output_lineage), tHMMobj.num_states))
        gamma_array[0, :] = betas[num][0, :]
        gammas.append(gamma_array)

    for num, lO in enumerate(tHMMobj.X):  # for each lineage in our Population
        lineage = lO.output_lineage
        MSDn = np.clip(MSD[num], np.finfo(np.float).eps, np.inf)

        for level in lO.output_list_of_gens[1:]:
            for cell in level:
                parent_idx = lineage.index(cell)
                gam = gammas[num][parent_idx, :]

                for d in cell.get_daughters():
                    ci = lineage.index(d)

                    coeffs = betas[num] / MSDn
                    beta_parent = np.clip(T @ coeffs[ci, :], np.finfo(np.float).eps, np.inf)
                    gammas[num][ci, :] = coeffs[ci, :] * np.matmul(gam / beta_parent, T)

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
