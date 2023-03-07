import numpy as np
import numpy.typing as npt


def sum_nonleaf_gammas(
    leaves_idx, gammas: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Sum of the gammas of the cells that are able to divide, that is,
    sum the of the gammas of all the nonleaf cells. It is used in estimating the transition probability matrix.
    This is an inner component in calculating the overall transition probability matrix.

    This is downward recursion.

    :param lO: the object of lineage tree
    :param gamma_arr: the gamma values for each lineage
    :return: the sum of gamma values for each state for non-leaf cells.
    """
    # Remove leaves
    gs = np.delete(gammas, leaves_idx, axis=0)

    # sum the gammas for cells that are transitioning (all but gen 0)
    return np.sum(gs[1:, :], axis=0)


def get_all_zetas(
    leaves_idx: np.ndarray,
    cell_to_daughters: np.ndarray,
    beta_array: npt.NDArray[np.float64],
    MSD_array: npt.NDArray[np.float64],
    gammas: npt.NDArray[np.float64],
    T: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Sum of the list of all the zeta parent child for all the parent cells for a given state transition pair.
    This is an inner component in calculating the overall transition probability matrix.

    :param lineageObj: the lineage tree of cells
    :param beta_array: beta values. The conditional probability of states, given observations of the sub-tree rooted in cell_n
    :param MSD_array: marginal state distribution
    :param gammas: gamma values. The conditional probability of states, given the observation of the whole tree
    :param T: transition probability matrix
    :return: numerator for calculating the transition probabilities
    """
    betaMSD = beta_array / np.clip(MSD_array, np.finfo(float).eps, np.inf)
    TbetaMSD = np.clip(betaMSD @ T.T, np.finfo(float).eps, np.inf)

    cIDXs = np.arange(gammas.shape[0])
    cIDXs = np.delete(cIDXs, leaves_idx, axis=0)

    dIDXs = cell_to_daughters[cIDXs, :]

    # Getting lineage by generation, but it is sorted this way
    js = gammas[cIDXs, np.newaxis, :] / TbetaMSD[dIDXs, :]
    holder = np.einsum("ijk,ijl->kl", js, betaMSD[dIDXs, :])
    return holder * T
