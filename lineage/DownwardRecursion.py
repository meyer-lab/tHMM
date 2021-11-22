""" File holds the code for the downward recursion. """

import numpy as np


def get_gammas(tHMMobj, MSD: list, betas: list) -> list:
    """
    Get the gammas for all other nodes using recursion from the root nodes.
    The conditional probability of states, given the observation of the whole tree P(z_n = k | X_bar = x_bar)
    x_bar is the observations for the whole tree.
    gamma_1 (k) = P(z_1 = k | X_bar = x_bar)
    gamma_n (k) = P(z_n = k | X_bar = x_bar)

    :param tHMMobj: A class object with properties of the lineages of cells
    :param MSD: The marginal state distribution P(z_n = k)
    :param betas: beta values. The conditional probability of states, given observations of the sub-tree rooted in cell_n
    :type betas: list of ndarray
    """
    T = tHMMobj.estimate.T
    gammas = []

    for num, lO in enumerate(tHMMobj.X):  # for each lineage in our Population
        gamma_array = np.zeros((len(lO.output_lineage), tHMMobj.num_states))
        gamma_array[0, :] = betas[num][0, :]
        gammas.append(gamma_array)

    for num, lO in enumerate(tHMMobj.X):  # for each lineage in our Population
        lineage = lO.output_lineage
        coeffs = betas[num] / np.clip(MSD[num], np.finfo(float).eps, np.inf)
        coeffs = np.clip(coeffs, np.finfo(float).eps, np.inf)
        beta_parents = np.einsum("ij,kj->ik", T, coeffs)

        for level in lO.output_list_of_gens[1:]:
            for cell in level:
                parent_idx = lineage.index(cell)
                gam = gammas[num][parent_idx, :]

                for d in cell.get_daughters():
                    ci = lineage.index(d)
                    gammas[num][ci, :] = coeffs[ci, :] * np.matmul(gam / beta_parents[:, ci], T)

    for gamm in gammas:
        assert np.all(np.isfinite(gamm))

    return gammas


def sum_nonleaf_gammas(lineageObj, gamma_arr: np.ndarray) -> np.ndarray:
    """
    Sum of the gammas of the cells that are able to divide, that is,
    sum the of the gammas of all the nonleaf cells. It is used in estimating the transition probability matrix.
    This is an inner component in calculating the overall transition probability matrix.

    :param lineageObj: the object of lineage tree
    :param gamma_arr: the gamma values for each lineage
    :return: the sum of gamma values for each state for non-leaf cells.
    """
    holder_wo_leaves = np.zeros(gamma_arr.shape[1])
    for level in lineageObj.output_list_of_gens[1:]:  # sum the gammas for cells that are transitioning
        for cell in level:
            if not cell.isLeaf():
                cell_idx = lineageObj.output_lineage.index(cell)
                holder_wo_leaves += gamma_arr[cell_idx, :]
    assert np.all(np.isfinite(holder_wo_leaves))

    return holder_wo_leaves
