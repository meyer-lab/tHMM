'''File holds the code for the downward recursion.'''

import numpy as np
from .tHMM_utils import max_gen, get_gen, get_daughters
from .UpwardRecursion import beta_parent_child_func


def get_root_gammas(tHMMobj, betas):
    ''' Need the first gamma terms in the baum welch, which are just the beta values of the root nodes. '''
    gammas = []

    for num, lineage in enumerate(tHMMobj.population):  # for each lineage in our Population
        gamma_array = np.zeros((len(lineage), tHMMobj.numStates))
        gamma_array[0, :] = betas[num][0, :]
        assert np.isclose(np.sum(gamma_array[0]), 1.)
        gammas.append(gamma_array)

    return gammas


def get_nonroot_gammas(tHMMobj, gammas, betas):
    '''get the gammas for all other nodes using recursion from the root nodes'''
    numStates = tHMMobj.numStates
    numLineages = tHMMobj.numLineages
    population = tHMMobj.population
    paramlist = tHMMobj.paramlist
    MSD = tHMMobj.MSD

    for num, lineage in enumerate(population):  # for each lineage in our Population
        MSD_array = MSD[num]  # getting the MSD of the respective lineage
        T = paramlist[num]['T']
        beta_array = betas[num]  # instantiating N by K array

        for curr_level in range(1, max_gen(lineage)):
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

    for num in range(numLineages):
        assert np.allclose(np.sum(gammas[num], axis=1), 1.)
