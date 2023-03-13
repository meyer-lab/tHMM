""" Re-calculates the tHMM parameters of pi, T, and emissions using Baum Welch. """
import numpy as np
from typing import Tuple
from .tHMM import tHMM
from .LineageTree import get_Emission_Likelihoods
from .states.StateDistributionGamma import atonce_estimator
from .HMM.M_step import get_all_zetas, sum_nonleaf_gammas
from .HMM.E_step import get_leaf_Normalizing_Factors, get_MSD, get_beta, get_gamma


def do_E_step(tHMMobj: tHMM) -> Tuple[list, list, list, list]:
    """
    Calculate MSD, EL, NF, gamma, beta, LL from tHMM model.

    :param tHMMobj: A tHMM object with properties of the lineages of cells, such as
    :return MSD: Marginal state distribution
    :return NF: normalizing factor
    :return betas: beta values (conditional probability of cell states given cell observations)
    :return gammas: gamma values (used to calculate the downward reursion)
    """
    MSD = list()
    NF = list()
    betas = list()
    gammas = list()
    EL = get_Emission_Likelihoods(tHMMobj.X, tHMMobj.estimate.E)

    for ii, lO in enumerate(tHMMobj.X):
        MSD.append(
            get_MSD(lO.cell_to_parent, tHMMobj.estimate.pi, tHMMobj.estimate.T)
        )
        NF.append(get_leaf_Normalizing_Factors(lO.leaves_idx, MSD[ii], EL[ii]))
        betas.append(get_beta(lO.leaves_idx, lO.cell_to_daughters, tHMMobj.estimate.T, MSD[ii], EL[ii], NF[ii]))
        gammas.append(get_gamma(lO.cell_to_daughters, tHMMobj.estimate.T, MSD[ii], betas[ii]))

    return MSD, NF, betas, gammas


def calculate_log_likelihood(NF: list) -> float:
    """
    Calculates log likelihood of NF for each lineage.

    :param NF: list of normalizing factors
    return: the sum of log likelihoods for each lineage
    """
    summ = 0.0
    for N in NF:
        if isinstance(N, np.ndarray):
            summ += np.sum(np.log(N))
        else:
            summ += np.sum([np.sum(np.log(a)) for a in N])

    return summ


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


def do_M_step(tHMMobj: list[tHMM], MSD: list, betas: list, gammas: list):
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
    # the first object is representative of the whole population.
    # If thmmObj[0] satisfies this "if", then all the objects in this population do.
    if tHMMobj[0].estimate.fT is None:
        assert tHMMobj[0].fT is None
        T = do_M_T_step(tHMMobj, MSD, betas, gammas)
        # the following line will replace line 89 in case we want to have equal transitions between all states.
        # T = np.ones((gammas[0][0].shape[1], gammas[0][0].shape[1])) / gammas[0][0].shape[1]

        # all the objects in the population have the same T
        for t in tHMMobj:
            t.estimate.T = T

    if tHMMobj[0].estimate.fpi is None:
        pi = calculate_stationary(tHMMobj[0].estimate.T)
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
        if (
            len(tHMMobj) == 1
        ):  # means it only performs calculation on one condition at a time.
            do_M_E_step(tHMMobj[0], gammas[0])
        else:  # means it performs the calculations on several concentrations at once.
            do_M_E_step_atonce(tHMMobj, gammas)


def do_M_pi_step(tHMMobj: list[tHMM], gammas: list[np.ndarray]) -> np.ndarray:
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


def do_M_T_step(
    tHMMobj: list[tHMM],
    MSD: list[list[np.ndarray]],
    betas: list[list[np.ndarray]],
    gammas: list[list[np.ndarray]],
) -> np.ndarray:
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

    # One pseudocount spread across states
    numer_e = np.full((n, n), 0.1 / n)
    denom_e = np.ones(n) + 0.1

    for i, tt in enumerate(tHMMobj):
        for num, lO in enumerate(tt.X):
            # local T estimate
            numer_e += get_all_zetas(lO.leaves_idx, lO.cell_to_daughters,
                betas[i][num], MSD[i][num], gammas[i][num], tt.estimate.T
            )
            denom_e += sum_nonleaf_gammas(lO.leaves_idx, gammas[i][num])

    T_estimate = numer_e / denom_e[:, np.newaxis]
    T_estimate /= T_estimate.sum(axis=1)[:, np.newaxis]

    assert np.all(np.isfinite(T_estimate))

    return T_estimate


def do_M_E_step(tHMMobj: tHMM, gammas: list[np.ndarray]):
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


def do_M_E_step_atonce(all_tHMMobj: list[tHMM], all_gammas: list[list[np.ndarray]]):
    """
    Performs the maximization step for emission estimation when data for all the concentrations are given at once for all the states.
    After reshaping, we will have a list of lists for each state.
    This function is specifically written for the experimental data of G1 and S-G2 cell cycle fates and durations.
    """
    gms = []
    for gm in all_gammas:
        gms.append(np.vstack(gm))

    all_cells = np.array(
        [cell.obs for lineage in all_tHMMobj[0].X for cell in lineage.output_lineage]
    )
    if len(all_cells[1, :]) == 6:
        phase = True
    else:
        phase = False

    G1cells = []
    G2cells = []
    cells = []
    for tHMMobj in all_tHMMobj:
        all_cells = np.array(
            [cell.obs for lineage in tHMMobj.X for cell in lineage.output_lineage]
        )
        if phase:
            G1cells.append(all_cells[:, np.array([0, 2, 4])])
            G2cells.append(all_cells[:, np.array([1, 3, 5])])
        else:
            cells.append(all_cells)

    # reshape the gammas so that each list in this list of lists is for each state.
    for j in range(all_tHMMobj[0].num_states):
        gammas_1st = [array[:, j] for array in gms]
        if phase:
            atonce_estimator(
                all_tHMMobj, G1cells, gammas_1st, "G1", j
            )  # [shape, scale1, scale2, scale3, scale4] for G1
            atonce_estimator(
                all_tHMMobj, G2cells, gammas_1st, "G2", j
            )  # [shape, scale1, scale2, scale3, scale4] for G2
        else:
            atonce_estimator(
                all_tHMMobj, cells, gammas_1st, "all", j
            )  # [shape, scale1, scale2]
