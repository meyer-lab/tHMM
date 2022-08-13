"""This file contains the methods that completes the downward recursion and evaulates the beta values."""

import numpy as np
from .tHMM import tHMM


def get_leaf_betas(
    tHMMobj: tHMM, MSD: list[np.ndarray], EL: list[np.ndarray], NF: list
):
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

    The first value in the numerator is the Emission
    Likelihoods. The second value in the numerator is
    the Marginal State Distributions. The value in the
    denominator is the Normalizing Factor.
    :param tHMMobj: A class object with properties of the lineages of cells
    :param MSD: The marginal state distribution P(z_n = k)
    :param EL: The emissions likelihood
    :param NF: normalizing factor. The marginal observation distribution P(x_n = x)
    :return: beta values. The conditional probability of states, given observations of the sub-tree rooted in cell_n
    """
    betas = [
        np.zeros((len(lO.output_lineage), tHMMobj.num_states)) for lO in tHMMobj.X
    ]  # full betas holder

    for num, b_arr in enumerate(betas):  # for each lineage in our Population
        # Emission Likelihood, Marginal State Distribution, Normalizing Factor (same regardless of state)
        # P(x_n = x | z_n = k), P(z_n = k), P(x_n = x)
        ii = tHMMobj.X[num].leaves_idx
        b_arr[ii, :] = EL[num][ii, :] * MSD[num][ii, :] / NF[num][ii, np.newaxis]
        assert np.isclose(np.sum(b_arr[-1]), 1.0)

    return betas


def get_nonleaf_NF_and_betas(
    tHMMobj: tHMM, MSD: list[np.ndarray], EL: list[np.ndarray], NF: list, betas: list
):
    """
    Traverses through each tree and calculates the
    beta value for each non-leaf cell. The normalizing factors (NFs)
    are also calculated as an intermediate for determining each
    beta term. Helper functions are called to determine one of
    the terms in the NF equation. This term is also used in the calculation
    of the betas. The recursion is upwards from the leaves to
    the roots.

    :param tHMMobj: A class object with properties of the lineages of cells
    :param MSD: The marginal state distribution P(z_n = k)
    :param EL: The emissions likelihood
    :param NF: normalizing factor. The marginal observation distribution P(x_n = x)
    :param betas: beta values. The conditional probability of states, given observations of the sub-tree rooted in cell_n
    """
    for num, lO in enumerate(tHMMobj.X):  # for each lineage in our Population
        lineage = lO.output_lineage  # lineage in the population
        MSD_array = np.clip(
            MSD[num], np.finfo(float).eps, np.inf
        )  # MSD of the respective lineage
        T = tHMMobj.estimate.T  # get the lineage transition matrix
        ELMSD = EL[num] * MSD[num]

        for level in lO.output_list_of_gens[2:][::-1]:  # a reversed list of generations
            for pii in lO.get_parents_for_level(level):
                ch_ii = [lineage.index(d) for d in lineage[pii].get_daughters()]
                ratt = betas[num][ch_ii, :] / MSD_array[ch_ii, :]
                fac1 = np.prod(ratt @ T.T, axis=0) * ELMSD[pii, :]

                NF[num][pii] = sum(fac1)
                betas[num][pii, :] = fac1 / NF[num][pii]
