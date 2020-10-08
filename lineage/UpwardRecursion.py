"""This file contains the methods that completes the downward recursion and evaulates the beta values."""

import math
import numpy as np


def get_Marginal_State_Distributions(tHMMobj):
    r"""Marginal State Distribution (MSD) matrix and recursion.
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
    """
    MSD = [np.zeros((len(lO.output_lineage), tHMMobj.num_states)) for lO in tHMMobj.X]
    np.testing.assert_almost_equal(np.sum(tHMMobj.estimate.pi), 1.0)
    assert np.all(np.isfinite(tHMMobj.estimate.T))

    for m in MSD:  # for each lineage in our Population
        m[0, :] = tHMMobj.estimate.pi

    for num, lO in enumerate(tHMMobj.X):  # for each lineage in our Population
        lineage = lO.output_lineage  # getting the lineage in the Population by lineage index

        for level in lO.output_list_of_gens[2:]:
            for cell in level:
                pCellIDX = lineage.index(cell.parent)  # get the index of the parent cell
                cCellIDX = lineage.index(cell)

                # recursion based on parent cell
                MSD[num][cCellIDX, :] = MSD[num][pCellIDX, :] @ tHMMobj.estimate.T

    for m in MSD:  # for each lineage in our Population
        np.testing.assert_allclose(np.sum(m, axis=1), 1.0)

    return MSD


def get_Emission_Likelihoods(tHMMobj, E=None):
    """Emission Likelihood (EL) matrix.

    Each element in this N by K matrix represents the probability

    :math:`P(x_n = x | z_n = k)`,

    for all :math:`x_n` and :math:`z_n` in our observed and hidden state tree
    and for all possible discrete states k.
    """
    if E is None:
        E = tHMMobj.estimate.E

    all_cells = np.array([cell.obs for lineage in tHMMobj.X for cell in lineage.output_lineage])
    ELstack = np.zeros((len(all_cells), tHMMobj.num_states))

    for k in range(tHMMobj.num_states):  # for each state
        ELstack[:, k] = E[k].pdf(all_cells)
    assert np.all(np.isfinite(ELstack))

    EL = []
    ii = 0
    for lineageObj in tHMMobj.X:  # for each lineage in our Population
        nl = len(lineageObj.output_lineage)  # getting the lineage length
        EL.append(ELstack[ii:(ii + nl), :])  # append the EL_array for each lineage
        ii += nl

    return EL


def get_leaf_Normalizing_Factors(tHMMobj, MSD, EL):
    """Normalizing factor (NF) matrix and base case at the leaves.

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
    """

    NF = []  # full Normalizing Factors holder

    for num, lineageObj in enumerate(tHMMobj.X):  # for each lineage in our Population
        lineage = lineageObj.output_lineage  # getting the lineage in the Population by index
        MSD_array = MSD[num]  # getting the MSD of the respective lineage
        EL_array = EL[num]  # geting the EL of the respective lineage
        NF_array = np.zeros(len(lineage), dtype=float)  # instantiating N by 1 array

        for ii, cell in enumerate(lineageObj.output_leaves):  # for each cell in the lineage's leaves
            assert cell.isLeaf()
            leaf_idx = lineageObj.output_leaves_idx[ii]

            # P(x_n = x , z_n = k) = P(x_n = x | z_n = k) * P(z_n = k)
            # this product is the joint probability
            # P(x_n = x) = sum_k ( P(x_n = x , z_n = k) )
            # the sum of the joint probabilities is the marginal probability
            NF_array[leaf_idx] = np.dot(MSD_array[leaf_idx, :], EL_array[leaf_idx, :])  # def of conditional prob
            assert NF_array[leaf_idx] > 0.0, f"MSD: {MSD_array[leaf_idx, :]}, EL: {EL_array[leaf_idx, :]}"

        NF.append(NF_array)
    return NF


def get_leaf_betas(tHMMobj, MSD, EL, NF):
    """Beta matrix and base case at the leaves.

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

    The first value in the numerator is the Emission
    Likelihoods. The second value in the numerator is
    the Marginal State Distributions. The value in the
    denominator is the Normalizing Factor.
    """
    betas = []  # full betas holder

    for num, lineageObj in enumerate(tHMMobj.X):  # for each lineage in our Population
        lineage = lineageObj.output_lineage  # getting the lineage in the Population by index
        MSD_arr = MSD[num]  # getting the MSD of the respective lineage
        EL_arr = EL[num]  # geting the EL of the respective lineage
        NF_arr = NF[num]  # getting the NF of the respective lineage

        for _, cell in enumerate(lineageObj.output_leaves):  # for each cell in the lineage's leaves
            assert cell.isLeaf()

        # Emission Likelihood, Marginal State Distribution, Normalizing Factor (same regardless of state)
        # P(x_n = x | z_n = k), P(z_n = k), P(x_n = x)
        beta_array = np.zeros((len(lineage), tHMMobj.num_states))  # instantiating N by K array
        ii = lineageObj.output_leaves_idx

        with np.errstate(divide="ignore", invalid="ignore"):
            beta_array[ii, :] = EL_arr[ii, :] * MSD_arr[ii, :] / NF_arr[ii, np.newaxis]

        betas.append(beta_array)

    for num, _ in enumerate(tHMMobj.X):
        assert np.isclose(np.sum(betas[num][-1]), 1.0)

    return betas


def get_nonleaf_NF_and_betas(tHMMobj, MSD, EL, NF, betas):
    """Traverses through each tree and calculates the
    beta value for each non-leaf cell. The normalizing factors (NFs)
    are also calculated as an intermediate for determining each
    beta term. Helper functions are called to determine one of
    the terms in the NF equation. This term is also used in the calculation
    of the betas. The recursion is upwards from the leaves to
    the roots.
    """
    for num, lineageObj in enumerate(tHMMobj.X):  # for each lineage in our Population
        lineage = lineageObj.output_lineage  # getting the lineage in the Population by index
        MSD_array = MSD[num]  # getting the MSD of the respective lineage
        EL_array = EL[num]  # geting the EL of the respective lineage
        T = tHMMobj.estimate.T  # getting the transition matrix of the respective lineage

        for level in lineageObj.output_list_of_gens[2:][::-1]:  # a reversed list of generations
            for node_parent_m_idx in lineageObj.get_parents_for_level(level):
                fac1 = get_beta_parent_child_prod(
                    lineage=lineage, MSD_array=MSD_array, T=T, beta_array=betas[num], node_parent_m_idx=node_parent_m_idx
                )
                fac1 *= EL_array[node_parent_m_idx, :] * MSD_array[node_parent_m_idx, :]

                NF[num][node_parent_m_idx] = sum(fac1)
                assert NF[num][node_parent_m_idx] > 0.0

                betas[num][node_parent_m_idx, :] = fac1 / NF[num][node_parent_m_idx]

    for num, lineageObj in enumerate(tHMMobj.X):  # for each lineage in our Population
        betas_row_sum = np.sum(betas[num], axis=1)
        assert np.allclose(betas_row_sum, 1.0)


def get_beta_parent_child_prod(lineage, beta_array, T, MSD_array, node_parent_m_idx):
    """Calculates the product of beta-links for every parent-child
    relationship of a given parent cell in a given state.
    """
    node_parent_m = lineage[node_parent_m_idx]  # get the index of the parent
    children_list = node_parent_m.get_daughters()
    children_idx_list = [lineage.index(daughter) for daughter in children_list]

    beta = beta_array[children_idx_list, :]
    MSD = np.clip(MSD_array[children_idx_list, :], np.finfo(np.float).eps, np.inf)
    return T @ np.prod(beta / MSD, axis=0)
