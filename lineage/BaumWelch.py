""" Re-calculates the tHMM parameters of pi, T, and emissions using Baum Welch. """
import numpy as np

from .DownwardRecursion import get_root_gammas, get_nonroot_gammas
from .UpwardRecursion import get_leaf_Normalizing_Factors, get_leaf_betas, get_nonleaf_NF_and_betas, calculate_log_likelihood, beta_parent_child_func


def zeta_parent_child_func(node_parent_m_idx, node_child_n_idx, lineage, beta_array, MSD_array, gamma_array, T):
    """calculates the zeta value that will be used to fill the transition matrix in baum welch"""

    # check the child-parent relationship
    assert lineage[node_child_n_idx].parent is lineage[node_parent_m_idx]
    # if the child-parent relationship is correct, then the child must
    assert lineage[node_child_n_idx].isChild()
    # either be the left daughter or the right daughter

    beta_child_state_k = beta_array[node_child_n_idx, :]  # x by k
    gamma_parent = gamma_array[node_parent_m_idx, :]  # x by j
    MSD_child_state_k = MSD_array[node_child_n_idx, :]  # x by k
    beta_parent_child = beta_parent_child_func(beta_array=beta_array, T=T, MSD_array=MSD_array, node_child_n_idx=node_child_n_idx)

    js = gamma_parent / beta_parent_child
    ks = beta_child_state_k / MSD_child_state_k

    return np.outer(js, ks) * T


def get_all_gammas(lineageObj, gamma_arr):
    """sum of the list of all the gamma parent child for all the parent child relationships"""
    holder_wo_leaves = np.zeros(gamma_arr.shape[1])
    for level in lineageObj.output_list_of_gens[1:]:  # get all the gammas but not the ones at the last level
        for cell in level:
            if not cell.isLeaf():
                cell_idx = lineageObj.output_lineage.index(cell)
                holder_wo_leaves += gamma_arr[cell_idx, :]

    return holder_wo_leaves


def get_all_zetas(lineageObj, beta_array, MSD_array, gamma_array, T):
    """sum of the list of all the zeta parent child for all the parent cells for a given state transition pair"""
    assert MSD_array.shape[1] == gamma_array.shape[1] == beta_array.shape[1], "Number of states in tHMM object mismatched!"
    lineage = lineageObj.output_lineage
    holder = np.zeros(T.shape)
    for level in lineageObj.output_list_of_gens[1:]:
        for cell in level:  # get lineage for the gen
            node_parent_m_idx = lineage.index(cell)
            if not cell.isLeaf():
                for daughter_idx in cell.get_daughters():
                    holder += zeta_parent_child_func(
                        node_parent_m_idx=node_parent_m_idx,
                        node_child_n_idx=lineage.index(daughter_idx),
                        lineage=lineage,
                        beta_array=beta_array,
                        MSD_array=MSD_array,
                        gamma_array=gamma_array,
                        T=T,
                    )
    return holder


def calculateQuantities(tHMMobj):
    """ Calculate NF, gamma, beta, LL from tHMM model. """
    NF = get_leaf_Normalizing_Factors(tHMMobj)
    betas = get_leaf_betas(tHMMobj, NF)
    get_nonleaf_NF_and_betas(tHMMobj, NF, betas)
    gammas = get_root_gammas(tHMMobj, betas)
    get_nonroot_gammas(tHMMobj, gammas, betas)
    LL = calculate_log_likelihood(NF)

    return NF, betas, gammas, LL


def fit(tHMMobj, tolerance=np.spacing(1), max_iter=200):
    """Runs the tHMM function through Baum Welch fitting"""
    num_states = tHMMobj.num_states

    # first E step
    NF, betas, gammas, new_LL = calculateQuantities(tHMMobj)

    # first stopping condition check
    new_LL = calculate_log_likelihood(NF)
    for iter_number in range(max_iter):
        old_LL = new_LL

        # code for grouping all states in cell lineages
        pi_estimate = np.zeros((num_states), dtype=float)
        numer_estimate = np.zeros((num_states, num_states), dtype=float)
        denom_estimate = np.zeros((num_states,), dtype=float)
        for num, lineageObj in enumerate(tHMMobj.X):
            gamma_array = gammas[num]

            # local pi estimate
            pi_estimate += gamma_array[0, :]

            # local T estimate
            numer_estimate += get_all_zetas(lineageObj, betas[num], tHMMobj.MSD[num], gamma_array, tHMMobj.estimate.T)
            denom_estimate += get_all_gammas(lineageObj, gamma_array)

        if tHMMobj.estimate.fpi is None:
            # population wide pi calculation
            tHMMobj.estimate.pi = pi_estimate / sum(pi_estimate)
        if tHMMobj.estimate.fT is None:
            # population wide T calculation
            T_estimate = numer_estimate / denom_estimate[:, np.newaxis]
            tHMMobj.estimate.T = T_estimate / T_estimate.sum(axis=1)[:, np.newaxis]
        if tHMMobj.estimate.fE is None:
            # population wide E calculation
            all_cells = [cell.obs for lineage in tHMMobj.X for cell in lineage.output_lineage]
            all_gammas = np.vstack(gammas)
            for state_j in range(tHMMobj.num_states):
                tHMMobj.estimate.E[state_j] = tHMMobj.estimate.E[state_j].estimator(all_cells, all_gammas[:, state_j])

        tHMMobj.MSD = tHMMobj.get_Marginal_State_Distributions()
        tHMMobj.EL = tHMMobj.get_Emission_Likelihoods()

        NF, betas, gammas, new_LL = calculateQuantities(tHMMobj)

        if np.allclose(old_LL, new_LL, atol=tolerance) and iter_number > 2:
            break

    return (tHMMobj, NF, betas, gammas, new_LL)