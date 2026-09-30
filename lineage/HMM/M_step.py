import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_array


def sum_nonleaf_gammas(leaves_idx, gammas: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Sum of the gammas of the cells that are able to divide, that is,
    sum the of the gammas of all the nonleaf cells. It is used in estimating the transition probability matrix.
    This is an inner component in calculating the overall transition probability matrix.

    This is downward recursion.

    :param leaves_idx: leaf cell indices of the lineage tree
    :param gammas: the gamma values for each lineage
    :return: the sum of gamma values for each state for non-leaf cells.
    """
    # Remove leaves
    gs = np.delete(gammas, leaves_idx, axis=0)

    # sum the gammas for cells that are transitioning (all but gen 0)
    return np.sum(gs[1:, :], axis=0)


def get_all_zetas(
    tree: csr_array,
    beta_array: npt.NDArray[np.float64],
    MSD_array: npt.NDArray[np.float64],
    gammas: npt.NDArray[np.float64],
    T: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Sum of the list of all the zeta parent child for all the parent cells for a given state transition pair.
    This is an inner component in calculating the overall transition probability matrix.

    :param tree: CSR array representing the lineage tree
    :param beta_array: beta values. The conditional probability of states, given observations of the sub-tree rooted in cell_n
    :param MSD_array: marginal state distribution
    :param gammas: gamma values. The conditional probability of states, given the observation of the whole tree
    :param T: transition probability matrix
    :return: numerator for calculating the transition probabilities
    """
    if tree.nnz == 0:
        return np.zeros_like(T)

    betaMSD = beta_array / np.clip(MSD_array, np.finfo(float).eps, np.inf)
    TbetaMSD = np.clip(betaMSD @ T.T, np.finfo(float).eps, np.inf)

    parents = np.repeat(np.arange(tree.shape[0]), np.diff(tree.indptr))
    daughters = tree.indices

    # Getting lineage by generation, but it is sorted this way
    js = gammas[parents, :] / TbetaMSD[daughters, :]
    holder = np.einsum("ik,il->kl", js, betaMSD[daughters, :])
    return holder * T


def get_edge_posteriors(
    tree: csr_array,
    beta_array: npt.NDArray[np.float64],
    MSD_array: npt.NDArray[np.float64],
    gammas: npt.NDArray[np.float64],
    T: npt.NDArray[np.float64],
) -> tuple[np.ndarray, np.ndarray, npt.NDArray[np.float64]]:
    """
    Edge-wise joint posterior :math:`P(z_p = k, z_c = l | X)` for every parent-child edge.

    :param T: transition matrix, shared (K by K) or per edge (N by K by K, indexed by the child)
    :return: parents, children, and an (edges by K by K) array of joint posteriors; each slice sums to 1.
    """
    parents = np.repeat(np.arange(tree.shape[0]), np.diff(tree.indptr))
    daughters = tree.indices
    K = beta_array.shape[1]
    if tree.nnz == 0:
        return parents, daughters, np.zeros((0, K, K))

    Te = T[daughters] if T.ndim == 3 else np.broadcast_to(T, (len(daughters), K, K))
    betaMSD = beta_array[daughters] / np.clip(MSD_array[daughters], np.finfo(float).eps, np.inf)
    TbetaMSD = np.clip(np.einsum("ekl,el->ek", Te, betaMSD), np.finfo(float).eps, np.inf)
    js = gammas[parents, :] / TbetaMSD
    xi = js[:, :, np.newaxis] * Te * betaMSD[:, np.newaxis, :]
    return parents, daughters, xi
