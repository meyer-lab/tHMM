'''File holds the code for the downward recursion.'''

import numpy as np
from .tHMM_utils import max_gen, get_gen, get_daughters
from .UpwardRecursion import beta_parent_child_func


def get_root_gammas(tHMMobj, betas):
<<<<<<< HEAD
    '''need the first gamma terms in the baum welch, which are just the beta values of the root nodes.'''
    numStates = tHMMobj.numStates

    gammas = []

    for num, lineageObj in enumerate(tHMMobj.X):  # for each lineage in our Population
        lineage = lineageObj.output_lineage
        gamma_array = np.zeros((len(lineage), numStates))
=======
    ''' Need the first gamma terms in the baum welch, which are just the beta values of the root nodes. '''
    gammas = []

    for num, lineage in enumerate(tHMMobj.population):  # for each lineage in our Population
        gamma_array = np.zeros((len(lineage), tHMMobj.numStates))
>>>>>>> master
        gamma_array[0, :] = betas[num][0, :]
        assert np.isclose(np.sum(gamma_array[0]), 1.)
        gammas.append(gamma_array)

<<<<<<< HEAD
    for num, lineageObj in enumerate(tHMMobj.X):  # for each lineage in our Population
        gammas_0_row_sum = np.sum(gammas[num][0])
        assert np.isclose(gammas_0_row_sum, 1.)

=======
>>>>>>> master
    return gammas


def get_nonroot_gammas(tHMMobj, gammas, betas):
    '''get the gammas for all other nodes using recursion from the root nodes'''
<<<<<<< HEAD
    numStates = tHMMobj.numStates

    MSD = tHMMobj.MSD

    for num, lineageObj in enumerate(tHMMobj.X):  # for each lineage in our Population
        lineage = lineageObj.output_lineage  # getting the lineage in the Population by index
        MSD_array = MSD[num]  # getting the MSD of the respective lineage
        beta_array = betas[num]  # instantiating N by K array
        T = tHMMobj.estimate.T

        curr_level = 1
        max_level = max_gen(lineage)
=======
    for num, lineage in enumerate(tHMMobj.population):  # for each lineage in our Population
        MSD_array = tHMMobj.MSD[num]  # getting the MSD of the respective lineage
        T = tHMMobj.paramlist[num]['T']
        beta_array = betas[num]  # instantiating N by K array
>>>>>>> master

        for curr_level in range(1, max_gen(lineage)):
            level = get_gen(curr_level, lineage)  # get lineage for the gen
            for cell in level:
                parent_idx = lineage.index(cell)

                for daughter_idx in get_daughters(cell):
                    child_idx = lineage.index(daughter_idx)
                    coeffs = beta_array[child_idx, :] / MSD_array[child_idx, :]

                    for child_state_k in range(tHMMobj.numStates):
                        sum_holder = 0.0

                        for parent_state_j in range(tHMMobj.numStates):
                            beta_parent = beta_parent_child_func(beta_array=beta_array,
                                                                 T=T,
                                                                 MSD_array=MSD_array,
                                                                 state_j=parent_state_j,
                                                                 node_child_n_idx=child_idx)
<<<<<<< HEAD
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
=======
                            sum_holder += T[parent_state_j, child_state_k] * gammas[num][parent_idx, parent_state_j] / beta_parent

                        gammas[num][child_idx, child_state_k] = coeffs[child_state_k] * sum_holder

                        assert np.all(gammas[num][0, :] == betas[num][0, :])

    for _, gg in enumerate(gammas):
        assert np.allclose(np.sum(gg, axis=1), 1.)
>>>>>>> master
