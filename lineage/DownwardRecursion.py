'''File holds the code for the downward recursion.'''

import numpy as np
from .tHMM_utils import max_gen, get_gen, get_daughters
from .UpwardRecursion import beta_parent_child_func


def get_root_gammas(tHMMobj, betas):
    '''need the first gamma terms in the baum welch, which are just the beta values of the root nodes.'''
    numStates = tHMMobj.numStates

    gammas = []

    for num, lineageObj in enumerate(tHMMobj.X):  # for each lineage in our Population
        lineage = lineageObj.output_lineage
        gamma_array = np.zeros((len(lineage), numStates))
        gamma_array[0, :] = betas[num][0, :]
        gammas.append(gamma_array)

    for num, lineageObj in enumerate(tHMMobj.X):  # for each lineage in our Population
        gammas_0_row_sum = np.sum(gammas[num][0])
        assert np.isclose(gammas_0_row_sum, 1.)

    return gammas


def get_nonroot_gammas(tHMMobj, gammas, betas):
    '''get the gammas for all other nodes using recursion from the root nodes'''
    numStates = tHMMobj.numStates

    MSD = tHMMobj.MSD

    for num, lineageObj in enumerate(tHMMobj.X):  # for each lineage in our Population
        lineage = lineageObj.output_lineage  # getting the lineage in the Population by index
        MSD_array = MSD[num]  # getting the MSD of the respective lineage
        beta_array = betas[num]  # instantiating N by K array
        T = tHMMobj.estimate.T

        curr_level = 1
        max_level = max_gen(lineage)

        while curr_level < max_level:
            level = get_gen(curr_level, lineage)  # get lineage for the gen
            for cell in level:
                parent_idx = lineage.index(cell)
                daughter_idxs_list = get_daughters(cell)

                for daughter_idx in daughter_idxs_list:
                    child_idx = lineage.index(daughter_idx)

                    for child_state_k in range(numStates):
                        beta_child = beta_array[child_idx, child_state_k]
                        MSD_child = MSD_array[child_idx, child_state_k]
                        coeff = beta_child / MSD_child
                        sum_holder = []

                        for parent_state_j in range(numStates):
                            T_fac = T[parent_state_j, child_state_k]
                            gamma_parent = gammas[num][parent_idx, parent_state_j]
                            beta_parent = beta_parent_child_func(numStates=numStates,
                                                                 lineage=lineage,
                                                                 beta_array=beta_array,
                                                                 T=T,
                                                                 MSD_array=MSD_array,
                                                                 state_j=parent_state_j,
                                                                 node_parent_m_idx=parent_idx,
                                                                 node_child_n_idx=child_idx)
                            sum_holder.append(T_fac * gamma_parent / beta_parent)
                        gamma_child_state_k = coeff * sum(sum_holder)
                        gammas[num][child_idx, child_state_k] = gamma_child_state_k
                        for state_k in range(numStates):
                            assert gammas[num][0, state_k] == betas[num][0, state_k]
            curr_level += 1
    for num, lineageObj in enumerate(tHMMobj.X):  # for each lineage in our Population
        gammas_row_sum = np.sum(gammas[num], axis=1)
        # print(gammas_row_sum)
        #assert np.allclose(gammas_row_sum, 1.)
