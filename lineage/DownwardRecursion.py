'''File holds the code for the downward recursion.'''

import numpy as np
from .UpwardRecursion import beta_parent_child_func


def get_root_gammas(tHMMobj, betas):
    '''need the first gamma terms in the baum welch, which are just the beta values of the root nodes.'''
    gammas = []

    for num, lineageObj in enumerate(tHMMobj.X):  # for each lineage in our Population
        gamma_array = np.zeros((len(lineageObj.output_lineage), tHMMobj.numStates))

        gamma_array[0, :] = betas[num][0, :]
        assert np.isclose(np.sum(gamma_array[0]), 1.)
        gammas.append(gamma_array)

    for num, lineageObj in enumerate(tHMMobj.X):  # for each lineage in our Population
        gammas_0_row_sum = np.sum(gammas[num][0])
        assert np.isclose(gammas_0_row_sum, 1.)

    return gammas


def get_nonroot_gammas(tHMMobj, gammas, betas):
    '''get the gammas for all other nodes using recursion from the root nodes'''
    for num, lineageObj in enumerate(tHMMobj.X):  # for each lineage in our Population
        lineage = lineageObj.output_lineage
        MSD_array = tHMMobj.MSD[num]  # getting the MSD of the respective lineage
        T = tHMMobj.estimate.T
        beta_array = betas[num]  # instantiating N by K array

        for level in lineageObj.output_list_of_gens[1:]:
            for cell in level:
                parent_idx = lineage.index(cell)

                for daughter_idx in cell._get_daughters():
                    child_idx = lineage.index(daughter_idx)
                    coeffs = beta_array[child_idx, :] / MSD_array[child_idx, :]

                    beta_parent = beta_parent_child_func(beta_array=beta_array,
                                                         T=T,
                                                         MSD_array=MSD_array,
                                                         node_child_n_idx=child_idx)

                    for child_state_k in range(tHMMobj.numStates):
                        sum_holder = np.sum(T[:, child_state_k] * gammas[num][parent_idx, :]) / beta_parent

                        gammas[num][child_idx, child_state_k] = coeffs[child_state_k] * sum_holder
                        assert np.all(gammas[num][0, :] == betas[num][0, :])

    for _, gg in enumerate(gammas):
        assert np.allclose(np.sum(gg, axis=1), 1.)
