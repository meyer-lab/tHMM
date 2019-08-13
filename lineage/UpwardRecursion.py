'''This file contains the methods that completes the downward recursion and evaulates the beta values.'''

import numpy as np


def get_leaf_Normalizing_Factors(tHMMobj):
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

    P(x_n = x | z_n = k) * P(z_n = k) = P(x_n = x , z_n = k),
    where n are the leaf nodes.

    We can then sum this joint probability over k,
    which are the possible states z_n can be,
    and through the law of total probability,
    obtain the marginal observation distribution
    P(x_n = x):

    sum_k ( P(x_n = x , z_n = k) ) = P(x_n = x).
    """
    numStates = tHMMobj.numStates

    MSD = tHMMobj.MSD
    EL = tHMMobj.EL

    NF = []  # full Normalizing Factors holder

    for num, lineageObj in enumerate(
            tHMMobj.X):  # for each lineage in our Population
        # getting the lineage in the Population by index
        lineage = lineageObj.output_lineage
        MSD_array = MSD[num]  # getting the MSD of the respective lineage
        EL_array = EL[num]  # geting the EL of the respective lineage
        # instantiating N by 1 array
        NF_array = np.zeros((len(lineage)), dtype=float)

        for ii, cell in enumerate(
                lineageObj.output_leaves):  # for each cell in the lineage's leaves
            assert cell._isLeaf()
            leaf_cell_idx = lineageObj.output_leaves_idx[ii]
            temp_sum_holder = []  # create a temporary list
            for state_k in range(numStates):  # for each state
                joint_prob = MSD_array[leaf_cell_idx,
                                       state_k] * EL_array[leaf_cell_idx,
                                                           state_k]  # def of conditional prob
                # P(x_n = x , z_n = k) = P(x_n = x | z_n = k) * P(z_n = k)
                # this product is the joint probability
                # append the joint probability to be summed
                temp_sum_holder.append(joint_prob)

            marg_prob = sum(temp_sum_holder)  # law of total probability
            # P(x_n = x) = sum_k ( P(x_n = x , z_n = k) )
            # the sum of the joint probabilities is the marginal
            # probability
            # each leaf is now intialized
            NF_array[leaf_cell_idx] = marg_prob
        NF.append(NF_array)
    return NF


def get_leaf_betas(tHMMobj, NF):
    '''
    beta matrix and base case at the leaves.

    Each element in this N by K matrix is the beta value
    for each cell and at each state. In particular, this
    value is derived from the Marginal State Distributions
    (MSD), the Emission Likelihoods (EL), and the
    Normalizing Factors (NF). Each beta value
    for the leaves is exactly the probability

    beta[n,k] = P(z_n = k | x_n = x).

    Using Bayes Theorem, we see that the above equals

    numerator = P(x_n = x | z_n = k) * P(z_n = k)
    denominator = P(x_n = x)
    beta[n,k] = numerator / denominator

    The first value in the numerator is the Emission
    Likelihoods. The second value in the numerator is
    the Marginal State Distributions. The value in the
    denominator is the Normalizing Factor.
    '''
    numStates = tHMMobj.numStates

    MSD = tHMMobj.MSD
    EL = tHMMobj.EL
    # NF is an input argument

    betas = []  # full betas holder

    for num, lineageObj in enumerate(
            tHMMobj.X):  # for each lineage in our Population
        # getting the lineage in the Population by index
        lineage = lineageObj.output_lineage
        MSD_array = MSD[num]  # getting the MSD of the respective lineage
        EL_array = EL[num]  # geting the EL of the respective lineage
        NF_array = NF[num]  # getting the NF of the respective lineage

        # instantiating N by K array
        beta_array = np.zeros((len(lineage), numStates))
        for ii, cell in enumerate(
                lineageObj.output_leaves):  # for each cell in the lineage's leaves
            assert cell._isLeaf()
            leaf_cell_idx = lineageObj.output_leaves_idx[ii]
            for state_k in range(numStates):  # for each state
                # see expression in docstring
                # Emission Likelihood
                numer1 = EL_array[leaf_cell_idx, state_k]
                # P(x_n = x | z_n = k)
                # Marginal State Distribution
                numer2 = MSD_array[leaf_cell_idx, state_k]
                # P(z_n = k)
                # Normalizing Factor (same regardless of state)
                denom = NF_array[leaf_cell_idx]
                # P(x_n = x)
                beta_array[leaf_cell_idx,
                           state_k] = numer1 * numer2 / denom
        betas.append(beta_array)
    for num, lineageObj in enumerate(tHMMobj.X):
        betas_last_row_sum = np.sum(betas[num][-1])
        assert np.isclose(betas_last_row_sum, 1.)
    return betas


def get_nonleaf_NF_and_betas(tHMMobj, NF, betas):
    '''
    Traverses through each tree and calculates the
    beta value for each non-leaf cell. The normalizing factors (NFs)
    are also calculated as an intermediate for determining each
    beta term. Helper functions are called to determine one of
    the terms in the NF equation. This term is also used in the calculation
    of the betas. The recursion is upwards from the leaves to
    the roots.
    '''
    numStates = tHMMobj.numStates

    MSD = tHMMobj.MSD
    EL = tHMMobj.EL
    # NF is an input argument
    # betas is an input argument

    for num, lineageObj in enumerate(
            tHMMobj.X):  # for each lineage in our Population
        # getting the lineage in the Population by index
        lineage = lineageObj.output_lineage
        MSD_array = MSD[num]  # getting the MSD of the respective lineage
        EL_array = EL[num]  # geting the EL of the respective lineage
        T = tHMMobj.estimate.T  # getting the transition matrix of the respective lineage

        for level in lineageObj.output_list_of_gens[2:][::-1]:
            parent_holder = lineageObj._get_parents_for_level(level)
            for node_parent_m_idx in parent_holder:
                numer_holder = []
                for state_j in range(numStates):
                    fac1 = get_beta_parent_child_prod(numStates=numStates,
                                                      lineage=lineage,
                                                      MSD_array=MSD_array,
                                                      T=T,
                                                      beta_array=betas[num],
                                                      state_j=state_j,
                                                      node_parent_m_idx=node_parent_m_idx)
                    fac2 = EL_array[node_parent_m_idx, state_j]
                    fac3 = MSD_array[node_parent_m_idx, state_j]
                    numer_holder.append(fac1 * fac2 * fac3)
                NF[num][node_parent_m_idx] = sum(numer_holder)
                for state_j in range(numStates):
                    betas[num][node_parent_m_idx,
                               state_j] = numer_holder[state_j] / NF[num][node_parent_m_idx]
    for num, lineageObj in enumerate(
            tHMMobj.X):  # for each lineage in our Population
        betas_row_sum = np.sum(betas[num], axis=1)
        assert np.allclose(betas_row_sum, 1.)


def get_beta_parent_child_prod(
        numStates, lineage, beta_array, T, MSD_array, state_j, node_parent_m_idx):
    '''
    Calculates the product of beta-links for every parent-child
    relationship of a given parent cell in a given state.
    '''
    beta_m_n_holder = 1.0  # list to hold the factors in the product
    # get the index of the parent
    node_parent_m = lineage[node_parent_m_idx]
    children_list = node_parent_m._get_daughters()
    children_idx_list = [lineage.index(daughter)
                         for daughter in children_list]
    for node_child_n_idx in children_idx_list:
        # check the child-parent relationship
        assert lineage[node_child_n_idx].parent is lineage[node_parent_m_idx]
        # if the child-parent relationship is correct, then the child must
        assert lineage[node_child_n_idx]._isChild()
        beta_m_n = beta_parent_child_func(beta_array=beta_array,
                                          T=T,
                                          MSD_array=MSD_array,
                                          state_j=state_j,
                                          node_child_n_idx=node_child_n_idx)
        beta_m_n_holder *= beta_m_n

    return beta_m_n_holder


def beta_parent_child_func(
        beta_array, T, MSD_array, state_j, node_child_n_idx):
    '''
    This "helper" function calculates the probability
    described as a 'beta-link' between parent and child
    nodes in our tree for some state j. This beta-link
    value is what lets you calculate the values of
    higher (in the direction from the leave
    to the root node) node beta and Normalizing Factor
    values.
    '''
    # beta at node n for state k; transition rate for going from state j to state k; MSD for node n at state k
    # P( z_n = k | z_m = j); P(z_n = k)
    return np.sum(beta_array[node_child_n_idx, :] *
                  T[state_j, :] / MSD_array[node_child_n_idx, :])


def calculate_log_likelihood(tHMMobj, NF):
    '''
    Calculates log likelihood of NF for each lineage.
    '''
    LL = []

    for num, _ in enumerate(
            tHMMobj.X):  # for each lineage in our Population

        NF_array = NF[num]  # getting the NF of the respective lineage
        log_NF_array = np.log(NF_array)
        ll_per_num = sum(log_NF_array)
        LL.append(ll_per_num)

    return LL
