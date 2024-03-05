import numpy as np
import numpy.typing as npt


def get_leaf_Normalizing_Factors(
    MSD: npt.NDArray[np.float64],
    EL: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Normalizing factor (NF) matrix and base case at the leaves.

    Each element in this N by 1 matrix is the normalizing
    factor for each beta value calculation for each node.
    This normalizing factor is essentially the marginal
    observation distribution for a node.

    This function gets the normalizing factor for
    the upward recursion only for the leaves.
    We first calculate the joint probability
    using the definition of conditional probability:

    :math:`P(x_n = x | z_n = k) * P(z_n = k) = P(x_n = x , z_n = k)`,
    where n are the leaf nodes.

    We can then sum this joint probability over k,
    which are the possible states z_n can be,
    and through the law of total probability,
    obtain the marginal observation distribution
    :math:`P(x_n = x) = sum_k ( P(x_n = x , z_n = k) ) = P(x_n = x)`.

    :param EL: The emissions likelihood
    :param MSD: The marginal state distribution P(z_n = k)
    :return: normalizing factor. The marginal observation distribution P(x_n = x)
    """
    NF_array = np.zeros(MSD.shape[0], dtype=float)  # instantiating N by 1 array
    first_leaf = int(np.floor(MSD.shape[0] / 2))

    # P(x_n = x , z_n = k) = P(x_n = x | z_n = k) * P(z_n = k)
    # this product is the joint probability
    # P(x_n = x) = sum_k ( P(x_n = x , z_n = k) )
    # the sum of the joint probabilities is the marginal probability
    NF_array[first_leaf:] = np.sum(MSD[first_leaf:, :] * EL[first_leaf:, :], axis=1)
    assert np.all(np.isfinite(NF_array))
    return NF_array


def get_MSD(
    n_cells: int, pi: npt.NDArray[np.float64], T: npt.NDArray[np.float64]
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

    :param pi: Initial probabilities vector
    :param T: State transitions matrix
    :return: The marginal state distribution
    """
    m = np.zeros((n_cells, pi.size))
    m[0, :] = pi

    # recursion based on parent cell
    for cIDX in range(1, n_cells):
        pIDX = int(np.floor(cIDX / 2))
        m[cIDX, :] = m[pIDX, :] @ T

    # Assert all ~= 1.0
    assert np.linalg.norm(np.sum(m, axis=1) - 1.0) < 1e-9
    return m


def get_beta(
    T: npt.NDArray[np.float64],
    MSD: npt.NDArray[np.float64],
    EL: npt.NDArray[np.float64],
    NF: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    r"""Beta matrix and base case at the leaves.

    Each element in this N by K matrix is the beta value
    for each cell and at each state. In particular, this
    value is derived from the Marginal State Distributions
    (MSD), the Emission Likelihoods (EL), and the
    Normalizing Factors (NF). Each beta value
    for the leaves is exactly the probability

    :math:`beta[n,k] = P(z_n = k | x_n = x)`.

    Using Bayes Theorem, we see that the above equals

    numerator = :math:`P(x_n = x | z_n = k) * P(z_n = k)`
    denominator = :math:`P(x_n = x)`
    :math:`beta[n,k] = numerator / denominator`

    For non-leaf cells, the first value in the numerator is the Emission
    Likelihoods. The second value in the numerator is
    the Marginal State Distributions. The value in the
    denominator is the Normalizing Factor.

    Traverses upward through each tree and calculates the
    beta value for each non-leaf cell. The normalizing factors (NFs)
    are also calculated as an intermediate for determining each
    beta term. Helper functions are called to determine one of
    the terms in the NF equation. This term is also used in the calculation
    of the betas.

    :param tHMMobj: A class object with properties of the lineages of cells
    :param MSD: The marginal state distribution P(z_n = k)
    :param EL: The emissions likelihood
    :return: normalizing factor. The marginal observation distribution P(x_n = x)
    :return: beta values. The conditional probability of states, given observations of the sub-tree rooted in cell_n
    """
    # MSD of the respective lineage
    MSD_array = np.maximum(MSD, np.finfo(MSD.dtype).eps)
    ELMSD = EL * MSD

    ### NF leaf calculation
    # P(x_n = x , z_n = k) = P(x_n = x | z_n = k) * P(z_n = k)
    # this product is the joint probability
    # P(x_n = x) = sum_k ( P(x_n = x , z_n = k) )
    # the sum of the joint probabilities is the marginal probability
    NF = np.zeros(MSD.shape[0], dtype=float)  # instantiating N by 1 array
    NF[leaves_idx] = np.sum(ELMSD[leaves_idx, :], axis=1)
    assert np.all(np.isfinite(NF))

    ### beta calculation
    beta = np.zeros_like(MSD)
    first_leaf = int(np.floor(MSD.shape[0] / 2))

    # Emission Likelihood, Marginal State Distribution, Normalizing Factor (same regardless of state)
    # P(x_n = x | z_n = k), P(z_n = k), P(x_n = x)
    ZZ = EL[first_leaf:, :] * MSD[first_leaf:, :] / NF[first_leaf:, np.newaxis]
    beta[first_leaf:, :] = ZZ

    # Assert all ~= 1.0
    assert np.abs(np.sum(beta[-1]) - 1.0) < 1e-9

    MSD_array = np.maximum(
        MSD, np.finfo(MSD.dtype).eps
    )  # MSD of the respective lineage
    ELMSD = EL * MSD

    for pii in range(first_leaf - 1, -1, -1):
        ch_ii = np.array([pii * 2 + 1, pii * 2 + 2])

        ratt = (beta[ch_ii, :] / MSD_array[ch_ii, :]) @ T.T
        fac1 = ratt[0, :] * ratt[1, :] * ELMSD[pii, :]

        NF[pii] = np.sum(fac1)
        beta[pii, :] = fac1 / NF[pii]

    return NF, beta


def get_gamma(
    T: npt.NDArray[np.float64],
    MSD: npt.NDArray[np.float64],
    beta: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Get the gammas using downward recursion from the root nodes.
    The conditional probability of states, given observation of the whole tree P(z_n = k | X_bar = x_bar)
    x_bar is the observations for the whole tree.
    gamma_1 (k) = P(z_1 = k | X_bar = x_bar)
    gamma_n (k) = P(z_n = k | X_bar = x_bar)

    :param MSD: The marginal state distribution P(z_n = k)
    :param betas: beta values. The conditional probability of states, given observations of the sub-tree rooted in cell_n
    """
    gamma = np.zeros_like(beta)
    gamma[0, :] = beta[0, :]

    epss = np.finfo(np.float32).eps
    coeffs = beta / np.maximum(MSD, epss)
    coeffs = np.maximum(coeffs, epss)
    beta_parents = T @ coeffs.T

    first_leaf = int(np.floor(MSD.shape[0] / 2))

    # Getting lineage by generation, but it is sorted this way
    for pidx in range(first_leaf):
        for ci in [pidx * 2 + 1, pidx * 2 + 2]:
            A = gamma[pidx, :].T / beta_parents[:, ci]

            gamma[ci, :] = coeffs[ci, :] * (A @ T)

    assert np.all(np.isfinite(gamma))
    return gamma
