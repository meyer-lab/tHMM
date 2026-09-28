"""Re-calculates the tHMM parameters of pi, T, and emissions using Baum Welch."""

from typing import Any

import numpy as np

from .HMM.E_step import get_beta_and_NF, get_gamma, get_MSD
from .HMM.M_step import get_all_zetas, sum_nonleaf_gammas
from .LineageTree import get_Emission_Likelihoods
from .states.StateDistributionGamma import atonce_estimator as gamma_atonce_estimator
from .tHMM import tHMM


def do_E_step(tHMMobj: tHMM) -> tuple[list, list, list, list]:
    """
    Calculate MSD, EL, NF, gamma, beta, LL from tHMM model.

    :param tHMMobj: A tHMM object with properties of the lineages of cells.
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
        MSD.append(get_MSD(lO.tree, tHMMobj.estimate.pi, tHMMobj.estimate.T))

        NF_one, beta = get_beta_and_NF(lO.leaves_idx, lO.tree, tHMMobj.estimate.T, MSD[ii], EL[ii])
        NF.append(NF_one)
        betas.append(beta)
        gammas.append(get_gamma(lO.tree, tHMMobj.estimate.T, MSD[ii], betas[ii]))

    return MSD, NF, betas, gammas


def calculate_log_likelihood(
    NF: list[np.ndarray] | list[list[np.ndarray]] | list[Any],
) -> float:
    """
    Calculates log likelihood of NF for each lineage.

    :param NF: list of normalizing factors
    :return: the sum of log likelihoods for each lineage
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


def do_M_step(
    tHMMobj: list[tHMM],
    MSD: list,
    betas: list,
    gammas: list,
    shared_T: bool = True,
    independent_T: bool = False,
    estimate_pi: bool = False,
):
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
    :param shared_T: when fitting several conditions at once, estimate one transition matrix
        for all of them (the default) or a separate one for each. Emissions are fit jointly either way.
    :param independent_T: constrain every row of T to be identical, so that a daughter's
        state does not depend on its mother's; the null model for heritability.
    """
    # the first object is representative of the whole population.
    # If thmmObj[0] satisfies this "if", then all the objects in this population do.
    if tHMMobj[0].estimate.fT is None:
        assert tHMMobj[0].fT is None
        if shared_T:
            T = do_M_T_step(tHMMobj, MSD, betas, gammas, independent_T)

            # all the objects in the population have the same T
            for t in tHMMobj:
                t.estimate.T = T
        else:
            for i, t in enumerate(tHMMobj):
                t.estimate.T = do_M_T_step([t], [MSD[i]], [betas[i]], [gammas[i]], independent_T)

    for i, t in enumerate(tHMMobj):
        if estimate_pi and t.estimate.fpi is None:
            t.estimate.pi = do_M_pi_step([t], [gammas[i]])
        elif t.estimate.fpi is None or t.estimate.fpi is True:
            # True indicates that pi should be set based on the stationary distribution of T
            t.estimate.pi = calculate_stationary(t.estimate.T)
        else:
            t.estimate.pi = t.fpi

    if tHMMobj[0].estimate.fE is None:
        assert tHMMobj[0].fE is None
        if len(tHMMobj) == 1:  # means it only performs calculation on one condition at a time.
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

    # A small pseudocount keeps an unused state from getting exactly zero prior mass.
    pi_e += 1e-3
    return pi_e / np.sum(pi_e)


def do_M_T_step(
    tHMMobj: list[tHMM],
    MSD: list[list[np.ndarray]],
    betas: list[list[np.ndarray]],
    gammas: list[list[np.ndarray]],
    independent_T: bool = False,
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
    :param independent_T: constrain all rows of T to be equal. The expected transition counts
        are then pooled over mother states, which is the constrained MLE.
    """
    n = tHMMobj[0].num_states

    # One pseudocount spread across states
    numer_e = np.full((n, n), 0.1 / n)
    denom_e = np.ones(n) + 0.1

    for i, tt in enumerate(tHMMobj):
        for num, lO in enumerate(tt.X):
            # local T estimate
            numer_e += get_all_zetas(
                lO.tree,
                betas[i][num],
                MSD[i][num],
                gammas[i][num],
                tt.estimate.T,
            )
            denom_e += sum_nonleaf_gammas(lO.leaves_idx, gammas[i][num])

    if independent_T:
        T_estimate = np.tile(numer_e.sum(axis=0), (n, 1))
    else:
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
    cell_arr = np.vstack([lineage.obs for lineage in tHMMobj.X])
    all_gammas = np.vstack(gammas)
    for state_j in range(tHMMobj.num_states):
        tHMMobj.estimate.E[state_j].estimator(cell_arr, all_gammas[:, state_j])


def do_M_E_step_atonce(all_tHMMobj: list[tHMM], all_gammas: list[list[np.ndarray]]):
    """
    Performs the maximization step for emission estimation when data for all the concentrations are given at once for all the states.
    After reshaping, we will have a list of lists for each state.
    This function is specifically written for the experimental data of G1 and S-G2 cell cycle fates and durations.
    """
    gms = [np.vstack(gm) for gm in all_gammas]

    E0 = all_tHMMobj[0].estimate.E[0]
    # Six-column observations are G1 and S-G2 phases, unless the emission says otherwise.
    phase = getattr(E0, "split_phases", all_tHMMobj[0].X[0].obs.shape[1] == 6)

    G1cells = []
    G2cells = []
    cells = []
    for tHMMobj in all_tHMMobj:
        all_cells = np.vstack([lineage.obs for lineage in tHMMobj.X])
        if phase:
            G1cells.append(all_cells[:, [0, 2, 4]])
            G2cells.append(all_cells[:, [1, 3, 5]])
        else:
            cells.append(all_cells)

    # Emission classes may supply their own at-once estimator; fall back to the Gamma one.
    atonce_estimator = getattr(E0, "atonce_estimator", gamma_atonce_estimator)

    # reshape the gammas so that each list in this list of lists is for each state.
    if phase:
        atonce_estimator(all_tHMMobj, G1cells, gms, "G1")  # [shape, scale1, scale2, scale3, scale4] for G1
        atonce_estimator(all_tHMMobj, G2cells, gms, "G2")  # [shape, scale1, scale2, scale3, scale4] for G2
    else:
        atonce_estimator(all_tHMMobj, cells, gms, "all")  # [shape, scale1, scale2]
