import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_array


def get_MSD(
    tree: csr_array,
    pi: npt.NDArray[np.float64],
    T: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    r"""Marginal State Distribution (MSD) matrix by upward recursion.
    This is the probability that a hidden state variable :math:`z_n` is of
    state k, that is, each value in the N by K MSD array for each lineage is
    the probability

    :math:`P(z_n = k)`,

    for all :math:`z_n` in the hidden state tree
    and for all k in the total number of discrete states. Each MSD array is
    an N by K array (an entry for each cell and an entry for each state),
    and each lineage has its own MSD array.

    Every element in MSD matrix is essentially sum over all transitions from any state to
    state j (from parent to daughter):

    :math:`P(z_u = k) = \sum_j(Transition(j -> k) * P(parent_{cell_u}) = j)`

    :param tree: CSR array representing the lineage tree adjacency
    :param pi: Initial probabilities vector
    :param T: State transitions matrix
    :return: The marginal state distribution
    """
    m = np.zeros((tree.shape[0], pi.size))
    m[0, :] = pi

    # recursion based on parent cell
    for pIDX in range(tree.shape[0]):
        start, end = tree.indptr[pIDX], tree.indptr[pIDX + 1]
        if start < end:
            p_trans = m[pIDX, :] @ T
            for cIDX in tree.indices[start:end]:
                m[cIDX, :] = p_trans

    # Assert all ~= 1.0
    assert np.linalg.norm(np.sum(m, axis=1) - 1.0) < 1e-9
    return m


def get_beta_and_NF(
    leaves_idx: np.ndarray,
    tree: csr_array,
    T: np.ndarray,
    MSD: np.ndarray,
    EL: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Normalizing factor (NF) matrix and base case at the leaves.

    Each element in this N by 1 matrix is the normalizing
    factor for each beta value calculation for each node.
    This normalizing factor is essentially the marginal
    observation distribution for a node.

    :param leaves_idx: array of indices corresponding to leaf cells
    :param tree: CSR array representing the lineage tree adjacency
    :param T: Transition probability matrix
    :param MSD: The marginal state distribution P(z_n = k)
    :param EL: The emissions likelihood
    :return: normalizing factor. The marginal observation distribution P(x_n = x)
    :return: beta values. The conditional probability of states, given observations of the sub-tree rooted in cell_n
    """
    # MSD of the respective lineage
    MSD_array = np.maximum(MSD, np.finfo(MSD.dtype).eps)
    ELMSD = EL * MSD

    ### NF leaf calculation
    NF = np.zeros(MSD.shape[0], dtype=float)  # instantiating N by 1 array
    NF[leaves_idx] = np.sum(ELMSD[leaves_idx, :], axis=1)
    assert np.all(np.isfinite(NF))

    ### beta calculation
    beta = np.zeros_like(MSD)
    beta[leaves_idx, :] = ELMSD[leaves_idx, :] / NF[leaves_idx, np.newaxis]

    # Assert all ~= 1.0
    assert np.abs(np.sum(beta[-1]) - 1.0) < 1e-9

    cIDXs = np.nonzero(np.diff(tree.indptr) > 0)[0][::-1]

    for pii in cIDXs:
        ch_ii = tree.indices[tree.indptr[pii] : tree.indptr[pii + 1]]
        ratt = (beta[ch_ii, :] / MSD_array[ch_ii, :]) @ T.T
        fac1 = np.prod(ratt, axis=0) * ELMSD[pii, :]

        NF[pii] = np.sum(fac1)
        beta[pii, :] = fac1 / NF[pii]

    return NF, beta


def get_gamma(
    tree: csr_array,
    T: npt.NDArray[np.float64],
    MSD: npt.NDArray[np.float64],
    beta: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Get the gammas using downward recursion from the root nodes.
    The conditional probability of states, given observation of the whole tree P(z_n = k | X_bar = x_bar)
    x_bar is the observations for the whole tree.

    :param tree: CSR array representing the lineage tree adjacency
    :param T: State transitions matrix
    :param MSD: The marginal state distribution P(z_n = k)
    :param beta: beta values. The conditional probability of states, given observations of the sub-tree rooted in cell_n
    """
    gamma = np.zeros_like(beta)
    gamma[0, :] = beta[0, :]

    epss = np.finfo(np.float32).eps
    coeffs = beta / np.maximum(MSD, epss)
    coeffs = np.maximum(coeffs, epss)
    beta_parents = T @ coeffs.T

    non_leaves = np.nonzero(np.diff(tree.indptr) > 0)[0]
    for pidx in non_leaves:
        A_base = gamma[pidx, :].T
        for ci in tree.indices[tree.indptr[pidx] : tree.indptr[pidx + 1]]:
            A = A_base / beta_parents[:, ci]
            gamma[ci, :] = coeffs[ci, :] * (A @ T)

    assert np.all(np.isfinite(gamma))
    return gamma
