""" Re-calculates the tHMM parameters of pi, T, and emissions using Baum Welch. """
import numpy as np
from typing import Tuple, Any

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


def do_E_step(tHMMobj) -> Tuple[list, list, list, list]:
    """
    Calculate MSD, EL, NF, gamma, beta, LL from tHMM model.

    :param tHMMobj: A tHMM object with properties of the lineages of cells, such as
    :return MSD: Marginal state distribution
    :return NF: normalizing factor
    :return betas: beta values (conditional probability of cell states given cell observations)
    :return gammas: gamma values (used to calculate the downward reursion)
    """
    MSD = get_Marginal_State_Distributions(tHMMobj)
    EL = get_Emission_Likelihoods(tHMMobj)
    NF = get_leaf_Normalizing_Factors(tHMMobj, MSD, EL)
    betas = get_leaf_betas(tHMMobj, MSD, EL, NF)
    get_nonleaf_NF_and_betas(tHMMobj, MSD, EL, NF, betas)
    gammas = get_gammas(tHMMobj, MSD, betas)

    return MSD, NF, betas, gammas


def calculate_log_likelihood(NF: Any) -> np.ndarray:
    """
    Calculates log likelihood of NF for each lineage.

    :param NF: normalizing factor
    return: the sum of log likelihoods for each lineage
    """
    # NF is a list of arrays, an array for each lineage in the population
    return np.array([sum(np.log(arr)) for arr in NF])


def calculate_stationary(T: np.ndarray) -> np.ndarray:
    """
    Calculate the stationary distribution of states from T.
    Note that this does not take into account potential influences of the emissions.

    :param T: transition matrix, a square matrix with probabilities of transitioning from one state to the other
    :return: The stationary distribution of states which can be obtained by solving w = w * T
    """
    eigenvalues, eigenvectors = np.linalg.eig(T.T)
    idx = np.argmin(np.abs(eigenvalues - 1))
    w = np.real(eigenvectors[:, idx]).T
    return w / np.sum(w)


def do_M_step(tHMMobj: list, MSD: list, betas: list, gammas: list):
    """
    Calculates the maximization step of the Baum Welch algorithm
    given output of the expectation step.
    The individual parameter estimations are performed in
    separate functions.

    :param tHMMobj: A class object with properties of the lineages of cells
    :type tHMMobj: list
    :param MSD: The marginal state distribution P(z_n = k)
    :param betas: beta values. The conditional probability of states, given observations of the sub-tree rooted in cell_n
    :param gammas: gamma values. The conditional probability of states, given the observation of the whole tree
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

        # each tHMMobj has its own T
        for jj, t in enumerate(tHMMobj):
            t.estimate.T = T[jj]

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


def do_M_pi_step(tHMMobj, gammas: list) -> np.ndarray:
    """
    Calculates the M-step of the Baum Welch algorithm
    given output of the E step.
    Does the parameter estimation for the pi
    initial probability vector.

    :param tHMMobj: A class object with properties of the lineages of cells
    :type tHMMobj: object
    :param gammas: gamma values. The conditional probability of states, given the observation of the whole tree
    """
    pi_e = np.zeros(tHMMobj[0].num_states, dtype=float)
    for i, tt in enumerate(tHMMobj):
        for num in range(len(tt.X)):
            # local pi estimate
            pi_e += gammas[i][num][0, :]

    return pi_e / np.sum(pi_e)


def do_M_T_step(tHMMobj, MSD: list, betas: list, gammas: list) -> list:
    """
    Calculates the M-step of the Baum Welch algorithm
    given output of the E step.
    Does the parameter estimation for the T
    Markov stochastic transition matrix.

    :param tHMMobj: A class object with properties of the lineages of cells
    :type tHMMobj: list of tHMMobj s
    :param MSD: The marginal state distribution P(z_n = k)
    :param betas: beta values. The conditional probability of states, given observations of the sub-tree rooted in cell_n
    :param gammas: gamma values. The conditional probability of states, given the observation of the whole tree
    """
    n = tHMMobj[0].num_states

    T_estimate = []
    for i, tt in enumerate(tHMMobj):
        numer_e = np.full((n, n), 0.1 / n)
        denom_e = np.ones(n) + 0.1

        for num, lO in enumerate(tt.X):
            # local T estimate
            numer_e += get_all_zetas(lO, betas[i][num], MSD[i][num], gammas[i][num], tt.estimate.T)
            denom_e += sum_nonleaf_gammas(lO, gammas[i][num])

        T_temp = numer_e / denom_e[:, np.newaxis]
        T_temp /= T_temp.sum(axis=1)[:, np.newaxis]

        assert np.all(np.isfinite(T_temp))
        T_estimate.append(T_temp)

    return T_estimate


def do_M_E_step(tHMMobj, gammas: list):
    """
    Calculates the M-step of the Baum Welch algorithm
    given output of the E step.
    Does the parameter estimation for the E
    Emissions matrix (state probabilistic distributions).

    :param tHMMobj: A class object with properties of the lineages of cells
    :type tHMMobj: object
    :param gammas: gamma values. The conditional probability of states, given the observation of the whole tree
    """
    all_cells = [cell.obs for lineage in tHMMobj.X for cell in lineage.output_lineage]
    all_gammas = np.vstack(gammas)
    for state_j in range(tHMMobj.num_states):
        tHMMobj.estimate.E[state_j].estimator(all_cells, all_gammas[:, state_j])


def do_M_E_step_atonce(all_tHMMobj: list, all_gammas: list):
    """
    Performs the maximization step for emission estimation when data for all the concentrations are given at once for all the states.
    After reshaping, we will have a list of lists for each state.
    This function is specifically written for the experimental data of G1 and S-G2 cell cycle fates and durations.
    """
    gms = []
    for gm in all_gammas:
        gms.append(np.vstack(gm))

    all_cells = np.array([cell.obs for lineage in all_tHMMobj[0].X for cell in lineage.output_lineage])
    if len(all_cells[1, :]) == 6:
        phase = True
    else:
        phase = False

    G1cells = []
    G2cells = []
    cells = []
    for tHMMobj in all_tHMMobj:
        all_cells = np.array([cell.obs for lineage in tHMMobj.X for cell in lineage.output_lineage])
        if phase:
            G1cells.append(all_cells[:, np.array([0, 2, 4])])
            G2cells.append(all_cells[:, np.array([1, 3, 5])])
        else:
            cells.append(all_cells)

    # reshape the gammas so that each list in this list of lists is for each state.
    for j in range(all_tHMMobj[0].num_states):
        gammas_1st = [array[:, j] for array in gms]
        if phase:
            atonce_estimator(all_tHMMobj, G1cells, gammas_1st, "G1", j)  # [shape, scale1, scale2, scale3, scale4] for G1
            atonce_estimator(all_tHMMobj, G2cells, gammas_1st, "G2", j)  # [shape, scale1, scale2, scale3, scale4] for G2
        else:
            atonce_estimator(all_tHMMobj, cells, gammas_1st, "all", j)  # [shape, scale1, scale2]


def get_all_zetas(lineageObj, beta_array: np.ndarray, MSD_array: np.ndarray, gamma_array: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Sum of the list of all the zeta parent child for all the parent cells for a given state transition pair.
    This is an inner component in calculating the overall transition probability matrix.

    :param lineageObj: the lineage tree of cells
    :param beta_array: beta values. The conditional probability of states, given observations of the sub-tree rooted in cell_n
    :param MSD_array: marginal state distribution
    :param gamma_array: gamma values. The conditional probability of states, given the observation of the whole tree
    :param T: transition probability matrix
    :return: numerator for calculating the transition probabilities
    """
    assert MSD_array.shape[1] == gamma_array.shape[1] == beta_array.shape[1], "Number of states in tHMM object mismatched!"
    betaMSD = beta_array / np.clip(MSD_array, np.finfo(float).eps, np.inf)
    TbetaMSD = np.clip(betaMSD @ T.T, np.finfo(float).eps, np.inf)
    lineage = lineageObj.output_lineage
    holder = np.zeros(T.shape)

    for level in lineageObj.output_list_of_gens[1:]:
        for cell in level:  # get lineage for the generation
            gamma_parent = gamma_array[lineage.index(cell), :]  # x by j

            if not cell.isLeaf():
                for daughter_idx in cell.get_daughters():
                    d_idx = lineage.index(daughter_idx)
                    js = gamma_parent / TbetaMSD[d_idx, :]
                    holder += np.outer(js, betaMSD[d_idx, :])
    return holder * T
