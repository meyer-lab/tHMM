''' Re-calculates the tHMM parameters of pi, T, and emissions using Baum Welch. '''
import numpy as np

from .DownwardRecursion import get_root_gammas, get_nonroot_gammas
from .UpwardRecursion import get_leaf_Normalizing_Factors, get_leaf_betas, get_nonleaf_NF_and_betas, calculate_log_likelihood, beta_parent_child_func


def zeta_parent_child_func(node_parent_m_idx, node_child_n_idx, lineage, beta_array, MSD_array, gamma_array, T):
    '''calculates the zeta value that will be used to fill the transition matrix in baum welch'''

    # check the child-parent relationship
    assert lineage[node_child_n_idx].parent is lineage[node_parent_m_idx]
    # if the child-parent relationship is correct, then the child must
    assert lineage[node_child_n_idx]._isChild()
    # either be the left daughter or the right daughter

    beta_child_state_k = beta_array[node_child_n_idx, :]  # x by k
    gamma_parent = gamma_array[node_parent_m_idx, :]  # x by j
    MSD_child_state_k = MSD_array[node_child_n_idx, :]  # x by k
    beta_parent_child = beta_parent_child_func(beta_array=beta_array,
                                               T=T,
                                               MSD_array=MSD_array,
                                               node_child_n_idx=node_child_n_idx)

    js = gamma_parent / beta_parent_child
    ks = beta_child_state_k / MSD_child_state_k

    return np.outer(js, ks) * T


def get_all_gammas(lineageObj, gamma_arr):
    '''sum of the list of all the gamma parent child for all the parent child relationships'''
    holder = np.zeros(gamma_arr.shape[1])
    for level in lineageObj.output_list_of_gens[1:]:  # get all the gammas but not the ones at the last level
        for cell in level:
            if not cell._isLeaf():
                cell_idx = lineageObj.output_lineage.index(cell)
                holder += gamma_arr[cell_idx, :]

    return holder


@jit(forceobj=True)
def get_all_zetas(lineageObj, beta_array, MSD_array, gamma_array, T):
    '''sum of the list of all the zeta parent child for all the parent cells for a given state transition pair'''
    assert MSD_array.shape[1] == gamma_array.shape[1] == beta_array.shape[1], "Number of states in tHMM object mismatched!"
    lineage = lineageObj.output_lineage
    holder = np.zeros(T.shape)
    for level in lineageObj.output_list_of_gens[1:]:
        for cell in level:  # get lineage for the gen
            node_parent_m_idx = lineage.index(cell)

            for daughter_idx in cell._get_daughters():
                holder += zeta_parent_child_func(node_parent_m_idx=node_parent_m_idx,
                                                 node_child_n_idx=lineage.index(daughter_idx),
                                                 lineage=lineage,
                                                 beta_array=beta_array,
                                                 MSD_array=MSD_array,
                                                 gamma_array=gamma_array,
                                                 T=T)
    return holder


def fit(tHMMobj, tolerance=np.spacing(1), max_iter=200):
    '''Runs the tHMM function through Baum Welch fitting'''
    numStates = tHMMobj.numStates

    # first E step

    NF = get_leaf_Normalizing_Factors(tHMMobj)
    betas = get_leaf_betas(tHMMobj, NF)
    get_nonleaf_NF_and_betas(tHMMobj, NF, betas)
    gammas = get_root_gammas(tHMMobj, betas)
    get_nonroot_gammas(tHMMobj, gammas, betas)

    # first stopping condition check
    new_LL = calculate_log_likelihood(NF)
    for _ in range(max_iter):
        old_LL = new_LL

        # code for grouping all states in cell lineages
        cell_groups = [[] for state in range(numStates)]
        pi_estimate = np.zeros((numStates), dtype=float)
        T_estimate = np.zeros((numStates, numStates), dtype=float)
        for num, lineageObj in enumerate(tHMMobj.X):
            lineage = lineageObj.output_lineage
            gamma_array = gammas[num]
            pi_estimate += gamma_array[0, :]

            denom = get_all_gammas(lineageObj, gamma_array)
            numer = get_all_zetas(lineageObj=lineageObj,
                                  beta_array=betas[num],
                                  MSD_array=tHMMobj.MSD[num],
                                  gamma_array=gamma_array,
                                  T=tHMMobj.estimate.T)

            T_holder = (numer + np.spacing(1)) / (denom[:, np.newaxis] + np.spacing(1))
            T_estimate += T_holder

            max_state_holder = []  # a list the size of lineage, that contains max state for each cell
            for ii, cell in enumerate(lineage):
                assert lineage[ii] is cell
                max_state_holder.append(np.argmax(gamma_array[ii, :]))  # says which state is maximal

            # this bins the cells by lineage to the population cell lists
            for ii, state in enumerate(max_state_holder):
                cell_groups[state].append(lineage[ii])
        tHMMobj.estimate.pi = pi_estimate / sum(pi_estimate)
        tHMMobj.estimate.T = T_estimate / T_estimate.sum(axis=1)[:, np.newaxis]
        # after iterating through each lineage, do the population wide E calculation
        for state_j in range(numStates):
            tHMMobj.estimate.E[state_j] = tHMMobj.estimate.E[state_j].estimator([cell.obs for cell in cell_groups[state_j]])

        tHMMobj.MSD = tHMMobj.get_Marginal_State_Distributions()
        tHMMobj.EL = tHMMobj.get_Emission_Likelihoods()

        NF = get_leaf_Normalizing_Factors(tHMMobj)
        betas = get_leaf_betas(tHMMobj, NF)
        get_nonleaf_NF_and_betas(tHMMobj, NF, betas)
        gammas = get_root_gammas(tHMMobj, betas)
        get_nonroot_gammas(tHMMobj, gammas, betas)

        # tolerance checking
        new_LL = calculate_log_likelihood(NF)

        if np.allclose([old_LL], [new_LL], atol=tolerance):
            return(tHMMobj, NF, betas, gammas, new_LL)

    return(tHMMobj, NF, betas, gammas, new_LL)
