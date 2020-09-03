""" Re-calculates the tHMM parameters of pi, T, and emissions using Baum Welch. """
import numpy as np

from .UpwardRecursion import (
    get_Marginal_State_Distributions,
    get_Emission_Likelihoods,
    get_leaf_Normalizing_Factors,
    get_leaf_betas,
    get_nonleaf_NF_and_betas,
    beta_parent_child_func,
)

from .DownwardRecursion import (
    get_root_gammas,
    get_nonroot_gammas,
    sum_nonleaf_gammas,
)


def do_E_step(tHMMobj):
    """
    Calculate MSD, EL, NF, gamma, beta, LL from tHMM model.
    """
    MSD = get_Marginal_State_Distributions(tHMMobj)
    EL = get_Emission_Likelihoods(tHMMobj)
    tHMMobj.EL = EL
    NF = get_leaf_Normalizing_Factors(tHMMobj, MSD, EL)
    betas = get_leaf_betas(tHMMobj, MSD, EL, NF)
    get_nonleaf_NF_and_betas(tHMMobj, MSD, EL, NF, betas)
    gammas = get_root_gammas(tHMMobj, betas)
    get_nonroot_gammas(tHMMobj, MSD, gammas, betas)

    return MSD, NF, betas, gammas


def calculate_log_likelihood(NF):
    """
    Calculates log likelihood of NF for each lineage.
    """
    # NF is a list of arrays, an array for each lineage in the population
    return np.array([sum(np.log(arr)) for arr in NF])


def do_M_step(tHMMobj, MSD, betas, gammas, constant_params):
    """
    Calculates the M-step of the Baum Welch algorithm
    given output of the E step.
    The individual parameter estimations are performed in
    separate functions.
    """
    if tHMMobj.estimate.fpi is None:
        assert tHMMobj.fpi is None
        tHMMobj.estimate.pi = do_M_pi_step(tHMMobj, gammas)

    if tHMMobj.estimate.fT is None:
        assert tHMMobj.fT is None
        tHMMobj.estimate.T = do_M_T_step(tHMMobj, MSD, betas, gammas)

    if tHMMobj.estimate.fE is None:
        assert tHMMobj.fE is None
        do_M_E_step(tHMMobj, gammas, constant_params)


def do_M_pi_step(tHMMobj, gammas):
    """
    Calculates the M-step of the Baum Welch algorithm
    given output of the E step.
    Does the parameter estimation for the pi
    initial probability vector.
    """
    num_states = tHMMobj.num_states

    pi_estimate = np.zeros((num_states), dtype=float)
    for num, _ in enumerate(tHMMobj.X):
        gamma_array = gammas[num]

        # local pi estimate
        pi_estimate += gamma_array[0, :]

    pi_estimate = pi_estimate / sum(pi_estimate)

    return pi_estimate


def do_M_T_step(tHMMobj, MSD, betas, gammas):
    """
    Calculates the M-step of the Baum Welch algorithm
    given output of the E step.
    Does the parameter estimation for the T
    Markov stochastic transition matrix.
    """
    num_states = tHMMobj.num_states

    numer_estimate = np.zeros((num_states, num_states))
    denom_estimate = np.zeros((num_states,)) + np.finfo(np.float).eps
    for num, lineageObj in enumerate(tHMMobj.X):
        # local T estimate
        numer_estimate += get_all_zetas(lineageObj, betas[num], MSD[num], gammas[num], tHMMobj.estimate.T)
        denom_estimate += sum_nonleaf_gammas(lineageObj, gammas[num])

    T_estimate = numer_estimate / denom_estimate[:, np.newaxis]

    # Add a small amount of identity in case a state is completely unobserved
    T_estimate += np.identity(num_states) * np.finfo(np.float).eps
    T_estimate /= T_estimate.sum(axis=1)[:, np.newaxis]
    assert np.all(np.isfinite(T_estimate))

    return T_estimate


def do_M_E_step(tHMMobj, gammas, constant_params):
    """
    Calculates the M-step of the Baum Welch algorithm
    given output of the E step.
    Does the parameter estimation for the E
    Emissions matrix (state probabilistic distributions).
    """
    all_cells = [cell.obs for lineage in tHMMobj.X for cell in lineage.output_lineage]
    all_gammas = np.vstack(gammas)
    for state_j in range(tHMMobj.num_states):
        tHMMobj.estimate.E[state_j].estimator(all_cells, all_gammas[:, state_j], constant_params)


def get_all_zetas(lineageObj, beta_array, MSD_array, gamma_array, T):
    """
    Sum of the list of all the zeta parent child for all the parent cells for a given state transition pair.
    """
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


def zeta_parent_child_func(node_parent_m_idx, node_child_n_idx, lineage, beta_array, MSD_array, gamma_array, T):
    """
    Calculates the zeta value that will be used to fill the transition matrix in baum welch.
    """

    # check the child-parent relationship
    assert lineage[node_child_n_idx].parent is lineage[node_parent_m_idx]
    # if the child-parent relationship is correct, then the child must
    assert lineage[node_child_n_idx].isChild()
    # either be the left daughter or the right daughter

    beta_child_state_k = beta_array[node_child_n_idx, :]  # x by k
    gamma_parent = gamma_array[node_parent_m_idx, :]  # x by j
    MSD_child_state_k = MSD_array[node_child_n_idx, :]  # x by k
    beta_parent_child = beta_parent_child_func(beta_array=beta_array, T=T, MSD_array=MSD_array, node_child_n_idx=node_child_n_idx)

    js = gamma_parent / (beta_parent_child + np.finfo(np.float).eps)
    ks = beta_child_state_k / (MSD_child_state_k + np.finfo(np.float).eps)

    return np.outer(js, ks) * T
