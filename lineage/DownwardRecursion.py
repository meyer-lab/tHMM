'''File holds the code for the downward recursion.'''

import numpy as np
from .UpwardRecursion import beta_parent_child_func


def get_root_gammas(tHMMobj, betas):
    '''need the first gamma terms in the baum welch, which are just the beta values of the root nodes.'''
    gammas = []

    for num, lineageObj in enumerate(tHMMobj.X):  # for each lineage in our Population
        gamma_array = np.zeros((len(lineageObj.output_lineage), tHMMobj.numStates))
        gamma_array[0, :] = betas[num][0, :]
        gammas.append(gamma_array)

    for _, gamma in enumerate(gammas):  # for each lineage in our Population
        assert np.isclose(np.sum(gamma[0]), 1.)

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

                    for parent_state_j in range(tHMMobj.numStates):
                        beta_parent = beta_parent_child_func(beta_array=beta_array,
                                                             T=T,
                                                             MSD_array=MSD_array,
                                                             state_j=parent_state_j,
                                                             node_child_n_idx=child_idx)

                    sum_holder = np.sum(np.matmul(gammas[num][parent_idx, :], T)) / beta_parent

                    gammas[num][child_idx, :] = coeffs * sum_holder

        assert np.all(gammas[num][0, :] == betas[num][0, :])

    for _, gg in enumerate(gammas):
        assert np.allclose(np.sum(gg, axis=1), 1.)
