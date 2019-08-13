''' Re-calculates the tHMM parameters of pi, T, and emissions using Baum Welch. '''
import logging
import numpy as np

from .DownwardRecursion import get_root_gammas, get_nonroot_gammas
from .UpwardRecursion import get_leaf_Normalizing_Factors, get_leaf_betas, get_nonleaf_NF_and_betas, calculate_log_likelihood, beta_parent_child_func


def zeta_parent_child_func(node_parent_m_idx, node_child_n_idx, parent_state_j,
                           child_state_k, lineage, beta_array, MSD_array, gamma_array, T):
    '''calculates the zeta value that will be used to fill the transition matrix in baum welch'''

    # check the child-parent relationship
    err_str = "Something wrong with your parent-daughter linkage when trying to use the zeta-related functions... Check again that your lineage is constructed clearly."
    assert lineage[node_child_n_idx].parent is lineage[node_parent_m_idx], err_str
    # if the child-parent relationship is correct, then the child must
    assert lineage[node_child_n_idx]._isChild(
    ), "Something wrong with your parent-daughter linkage when trying to use the zeta-related functions... Check again that your lineage is constructed clearly."
    # either be the left daughter or the right daughter

    beta_child_state_k = beta_array[node_child_n_idx, child_state_k]
    gamma_parent_state_j = gamma_array[node_parent_m_idx, parent_state_j]
    MSD_child_state_k = MSD_array[node_child_n_idx, child_state_k]
    beta_parent_child_state_j = beta_parent_child_func(beta_array=beta_array,
                                                       T=T,
                                                       MSD_array=MSD_array,
                                                       state_j=parent_state_j,
                                                       node_child_n_idx=node_child_n_idx)

    zeta = beta_child_state_k * T[parent_state_j, child_state_k] * \
        gamma_parent_state_j / \
        (MSD_child_state_k * beta_parent_child_state_j)
    return zeta


def get_all_gammas(lineageObj, gamma_array_at_state_j):
    '''sum of the list of all the gamma parent child for all the parent child relationships'''
    lineage = lineageObj.output_lineage
    holder = 0.0
    # get all the gammas but not the ones at the last level
    for level in lineageObj.output_list_of_gens[1:]:
        for cell in level:
            if not cell._isLeaf():
                cell_idx = lineage.index(cell)
                holder += gamma_array_at_state_j[cell_idx]

    return holder


def get_all_zetas(parent_state_j, child_state_k, lineageObj,
                  beta_array, MSD_array, gamma_array, T):
    '''sum of the list of all the zeta parent child for all the parent cells for a given state transition pair'''
    assert MSD_array.shape[1] == gamma_array.shape[1] == beta_array.shape[1], "Number of states in tHMM object mismatched!"
    lineage = lineageObj.output_lineage
    holder = 0.0
    for level in lineageObj.output_list_of_gens[1:]:
        for cell in level:  # get lineage for the gen
            node_parent_m_idx = lineage.index(cell)

            for daughter_idx in cell._get_daughters():
                node_child_n_idx = lineage.index(daughter_idx)
                holder += zeta_parent_child_func(node_parent_m_idx=node_parent_m_idx,
                                                 node_child_n_idx=node_child_n_idx,
                                                 parent_state_j=parent_state_j,
                                                 child_state_k=child_state_k,
                                                 lineage=lineage,
                                                 beta_array=beta_array,
                                                 MSD_array=MSD_array,
                                                 gamma_array=gamma_array,
                                                 T=T)
    return holder


def fit(tHMMobj, tolerance=1e-10, max_iter=100):
    '''Runs the tHMM function through Baum Welch fitting'''
    numStates = tHMMobj.numStates

    # first E step

    NF = get_leaf_Normalizing_Factors(tHMMobj)
    betas = get_leaf_betas(tHMMobj, NF)
    get_nonleaf_NF_and_betas(tHMMobj, NF, betas)
    gammas = get_root_gammas(tHMMobj, betas)
    get_nonroot_gammas(tHMMobj, gammas, betas)

    # first stopping condition check
    new_LL_list = calculate_log_likelihood(tHMMobj, NF)

    for _ in range(max_iter):
        old_LL_list = new_LL_list

        # code for grouping all states in cell lineages
        cell_groups = {}
        for state in range(numStates):
            cell_groups[str(state)] = []

        for num, lineageObj in enumerate(tHMMobj.X):
            lineage = lineageObj.output_lineage
            gamma_array = gammas[num]
            tHMMobj.estimate.pi = gamma_array[0, :]
            T_holder = np.zeros((numStates, numStates), dtype=float)
            for state_j in range(numStates):
                gamma_array_at_state_j = gamma_array[:, state_j]
                denom = get_all_gammas(lineageObj, gamma_array_at_state_j)
                for state_k in range(numStates):
                    numer = get_all_zetas(parent_state_j=state_j,
                                          child_state_k=state_k,
                                          lineageObj=lineageObj,
                                          beta_array=betas[num],
                                          MSD_array=tHMMobj.MSD[num],
                                          gamma_array=gamma_array,
                                          T=tHMMobj.estimate.T)
                    T_holder[state_j, state_k] = numer / denom

            tHMMobj.estimate.T = T_holder / \
                T_holder.sum(axis=1)[:, np.newaxis]

            # a list the size of lineage, that contains max state for each
            # cell
            max_state_holder = []
            for ii, cell in enumerate(lineage):
                assert lineage[ii] is cell
                # says which state is maximal
                max_state_holder.append(np.argmax(gammas[num][ii, :]))

            # this bins the cells by lineage to the population cell lists
            for ii, state in enumerate(max_state_holder):
                cell_groups[str(state)].append(lineage[ii])

        # after iterating through each lineage, do the population wide E
        # calculation
        for state_j in range(numStates):
            # this array has the correct cells classified per group
            cells = cell_groups[str(state_j)]
            tHMMobj.estimate.E[state_j] = tHMMobj.estimate.E[state_j].estimator(
                [cell.obs for cell in cells])

        tHMMobj.MSD = tHMMobj.get_Marginal_State_Distributions()
        tHMMobj.EL = tHMMobj.get_Emission_Likelihoods()

        NF = get_leaf_Normalizing_Factors(tHMMobj)
        betas = get_leaf_betas(tHMMobj, NF)
        get_nonleaf_NF_and_betas(tHMMobj, NF, betas)
        gammas = get_root_gammas(tHMMobj, betas)
        get_nonroot_gammas(tHMMobj, gammas, betas)

        # tolerance checking
        new_LL_list = calculate_log_likelihood(tHMMobj, NF)

        logging.info(
            "Average Log-Likelihood across all lineages: {}".format(np.mean(new_LL_list)))

        if np.allclose(np.array(old_LL_list), np.array(
                new_LL_list), atol=tolerance):
            return(tHMMobj, NF, betas, gammas, new_LL_list)

    logging.info(
        "Max iteration of {} steps achieved. Exiting Baum-Welch EM while loop.".format(max_iter))
    return(tHMMobj, NF, betas, gammas, new_LL_list)
