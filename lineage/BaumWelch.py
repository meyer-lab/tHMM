""" Re-calculates the tHMM parameters of pi, T, and emissions using Baum Welch. """
import numpy as np

from .UpwardRecursion import (
    get_Marginal_State_Distributions,
    get_Emission_Likelihoods,
    get_leaf_Normalizing_Factors,
    get_leaf_betas,
    get_nonleaf_NF_and_betas,
)

from .DownwardRecursion import (
    get_gammas,
    sum_nonleaf_gammas,
)

from .states.StateDistributionGamma import atonce_estimator


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
    gammas = get_gammas(tHMMobj, MSD, betas)

    return MSD, NF, betas, gammas


def calculate_log_likelihood(NF):
    """
    Calculates log likelihood of NF for each lineage.
    """
    # NF is a list of arrays, an array for each lineage in the population
    return np.array([sum(np.log(arr)) for arr in NF])


def calculate_stationary(T):
    """ Calculate the stationary distribution of states from T.
    Note that this does not take into account potential influences of the emissions. """
    eigenvalues, eigenvectors = np.linalg.eig(T.T)
    idx = np.argmin(np.abs(eigenvalues - 1))
    w = np.real(eigenvectors[:, idx]).T
    return w / np.sum(w)


def do_M_step(tHMMobj, MSD, betas, gammas):
    """
    Calculates the M-step of the Baum Welch algorithm
    given output of the E step.
    The individual parameter estimations are performed in
    separate functions.
    """
    if not isinstance(tHMMobj, list):
        tHMMobj = [tHMMobj]
        MSD = [MSD]
        betas = [betas]
        gammas = [gammas]

    # the first object is representative of the whole population.
    # If thmmObj[0] satisfies this "if", then all the objects in this population do.
    if tHMMobj[0].estimate.fT is None:
        assert tHMMobj[0].fT is None
        T = do_M_T_step(tHMMobj, MSD, betas, gammas)

        # all the objects in the population have the same T
        for t in tHMMobj:
            t.estimate.T = T

    if tHMMobj[0].estimate.fpi is None:
        assert tHMMobj[0].fpi is None
        pi = do_M_pi_step(tHMMobj, gammas)
    elif tHMMobj[0].estimate.fpi is True:
        # True indicates that pi should be set based on the stationary distribution of T
        assert tHMMobj[0].fpi is True
        pi = calculate_stationary(tHMMobj[0].estimate.T)
    else:
        pi = tHMMobj[0].fpi

    # all the objects in the population have the same pi
    for t in tHMMobj:
        t.estimate.pi = pi

    if tHMMobj[0].estimate.fE is None:
        assert tHMMobj[0].fE is None
        if len(tHMMobj) == 1:  # means it only performs calculation on one condition at a time.
            do_M_E_step(tHMMobj[0], gammas[0])
        else:  # means it performs the calculations on several concentrations at once.
            do_M_E_step_atonce(tHMMobj, gammas)


def do_M_pi_step(tHMMobj, gammas):
    """
    Calculates the M-step of the Baum Welch algorithm
    given output of the E step.
    Does the parameter estimation for the pi
    initial probability vector.
    """
    pi_e = np.zeros(tHMMobj[0].num_states, dtype=float)
    for i, tt in enumerate(tHMMobj):
        for num in range(len(tt.X)):
            # local pi estimate
            pi_e += gammas[i][num][0, :]

    return pi_e / np.sum(pi_e)


def do_M_T_step(tHMMobj, MSD, betas, gammas):
    """
    Calculates the M-step of the Baum Welch algorithm
    given output of the E step.
    Does the parameter estimation for the T
    Markov stochastic transition matrix.
    """
    n = tHMMobj[0].num_states

    # One pseudocount spread across states
    numer_e = np.full((n, n), 0.1 / n)
    denom_e = np.ones(n) + 0.1

    for i, tt in enumerate(tHMMobj):
        for num, lO in enumerate(tt.X):
            # local T estimate
            numer_e += get_all_zetas(lO, betas[i][num], MSD[i][num], gammas[i][num], tt.estimate.T)
            denom_e += sum_nonleaf_gammas(lO, gammas[i][num])

    T_estimate = numer_e / denom_e[:, np.newaxis]
    T_estimate /= T_estimate.sum(axis=1)[:, np.newaxis]

    assert np.all(np.isfinite(T_estimate))

    return T_estimate


def do_M_E_step(tHMMobj, gammas):
    """
    Calculates the M-step of the Baum Welch algorithm
    given output of the E step.
    Does the parameter estimation for the E
    Emissions matrix (state probabilistic distributions).
    """
    all_cells = [cell.obs for lineage in tHMMobj.X for cell in lineage.output_lineage]
    all_gammas = np.vstack(gammas)
    for state_j in range(tHMMobj.num_states):
        tHMMobj.estimate.E[state_j].estimator(all_cells, all_gammas[:, state_j])


def do_M_E_step_atonce(all_tHMMobj, all_gammas):
    """ perform the M_E step when all the concentrations are given at once for all the states.
    gms is a list of arrays with size = N x K.
    After reshaping, we will have a list of lists for each state.
    """
    gms = []
    for gm in all_gammas:
        gms.append(np.vstack(gm))
    G1cells = []
    G2cells = []
    for tHMMobj in all_tHMMobj:
        all_cells = np.array([cell.obs for lineage in tHMMobj.X for cell in lineage.output_lineage])
        G1cells.append(all_cells[:, np.array([0, 2, 4])])
        G2cells.append(all_cells[:, np.array([1, 3, 5])])

    # reshape the gammas so that each list in this list of lists is for each state.
    for j in range(all_tHMMobj[0].num_states):
        gammas_1st = [array[:, j] for array in gms]
        atonce_estimator(all_tHMMobj, G1cells, gammas_1st, "G1", j)  # [shape, sc1, sc2, sc3, sc4] for G1
        atonce_estimator(all_tHMMobj, G2cells, gammas_1st, "G2", j)  # [shape, sc1, sc2, sc3, sc4] for G2


def get_all_zetas(lineageObj, beta_array, MSD_array, gamma_array, T):
    """
    Sum of the list of all the zeta parent child for all the parent cells for a given state transition pair.
    """
    assert MSD_array.shape[1] == gamma_array.shape[1] == beta_array.shape[1], "Number of states in tHMM object mismatched!"
    betaMSD = beta_array / np.clip(MSD_array, np.finfo(np.float).eps, np.inf)
    TbetaMSD = np.clip(betaMSD @ T.T, np.finfo(np.float).eps, np.inf)
    lineage = lineageObj.output_lineage
    holder = np.zeros(T.shape)

    for level in lineageObj.output_list_of_gens[1:]:
        for cell in level:  # get lineage for the gen
            gamma_parent = gamma_array[lineage.index(cell), :]  # x by j

            if not cell.isLeaf():
                for daughter_idx in cell.get_daughters():
                    d_idx = lineage.index(daughter_idx)
                    js = gamma_parent / TbetaMSD[d_idx, :]
                    holder += np.outer(js, betaMSD[d_idx, :])
    return holder * T
