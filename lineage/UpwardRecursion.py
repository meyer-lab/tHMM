'''This file contains the methods that completes the downward recursion and evaulates the beta values.'''

import numpy as np
from .tHMM_utils import max_gen, get_gen, get_parents_for_level

def get_leaf_Normalizing_Factors(tHMMobj):
    '''
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
    '''
    numStates = tHMMobj.numStates
    numLineages = tHMMobj.numLineages
    population = tHMMobj.population
    assert numLineages == len(population)
    MSD = tHMMobj.MSD
    EL = tHMMobj.EL

    NF = [] # full Normalizing Factors holder

    for num in range(numLineages): # for each lineage in our Population
        lineage = population[num] # getting the lineage in the Population by index
        MSD_array = MSD[num] # getting the MSD of the respective lineage
        EL_array = EL[num] # geting the EL of the respective lineage
        NF_array = np.zeros((len(lineage)), dtype=float) # instantiating N by 1 array

        for cell in lineage: # for each cell in the lineage
            if cell.isLeaf(): # if it is a leaf
                leaf_cell_idx = lineage.index(cell) # get the index of the leaf
                temp_sum_holder = [] # create a temporary list

                for state_k in range(numStates): # for each state
                    joint_prob = MSD_array[leaf_cell_idx, state_k] * EL_array[leaf_cell_idx, state_k] # def of conditional prob
                    # P(x_n = x , z_n = k) = P(x_n = x | z_n = k) * P(z_n = k)
                    # this product is the joint probability
                    temp_sum_holder.append(joint_prob) # append the joint probability to be summed

                marg_prob = sum(temp_sum_holder) # law of total probability
                # P(x_n = x) = sum_k ( P(x_n = x , z_n = k) )
                # the sum of the joint probabilities is the marginal probability
                NF_array[leaf_cell_idx] = marg_prob # each leaf is now intialized
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
    numLineages = tHMMobj.numLineages
    population = tHMMobj.population
    MSD = tHMMobj.MSD
    EL = tHMMobj.EL
    # NF is an input argument

    betas = [] # full betas holder

    for num in range(numLineages): # for each lineage in our Population
        lineage = population[num] # getting the lineage in the Population by index
        MSD_array = MSD[num] # getting the MSD of the respective lineage
        EL_array = EL[num] # geting the EL of the respective lineage
        NF_array = NF[num] # getting the NF of the respective lineage

        beta_array = np.zeros((len(lineage), numStates)) # instantiating N by K array

        for cell in lineage: # for each cell in the lineage
            if cell.isLeaf(): # if it is a leaf
                leaf_cell_idx = lineage.index(cell) # get the index of the leaf

                for state_k in range(numStates): # for each state
                    # see expression in docstring
                    numer1 = EL_array[leaf_cell_idx, state_k] # Emission Likelihood
                    # P(x_n = x | z_n = k)
                    numer2 = MSD_array[leaf_cell_idx, state_k] # Marginal State Distribution
                    # P(z_n = k)
                    denom = NF_array[leaf_cell_idx] # Normalizing Factor (same regardless of state)
                    # P(x_n = x)
                    beta_array[leaf_cell_idx, state_k] = numer1 * numer2 / denom

        betas.append(beta_array)
    for num in range(numLineages):
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
    numLineages = tHMMobj.numLineages
    population = tHMMobj.population
    paramlist = tHMMobj.paramlist
    MSD = tHMMobj.MSD
    EL = tHMMobj.EL
    # NF is an input argument
    # betas is an input argument

    for num in range(numLineages): # for each lineage in our Population
        lineage = population[num] # getting the lineage in the Population by index
        MSD_array = MSD[num] # getting the MSD of the respective lineage
        EL_array = EL[num] # geting the EL of the respective lineage
        params = paramlist[num] # getting the respective params by lineage index
        T = params["T"] # getting the transition matrix of the respective lineage

        curr_gen = max_gen(lineage) # start at the lowest generation of the lineage (at the leaves)
        while curr_gen > 1:
            level = get_gen(curr_gen, lineage)
            parent_holder = get_parents_for_level(level, lineage)
            for node_parent_m_idx in parent_holder:
                numer_holder = []
                for state_j in range(tHMMobj.numStates):
                    fac1 = get_beta_parent_child_prod(numStates=numStates,
                                                      lineage=lineage,
                                                      MSD_array=MSD_array,
                                                      T=T,
                                                      beta_array=betas[num],
                                                      state_j=state_j,
                                                      node_parent_m_idx=node_parent_m_idx)
                    fac2 = EL_array[node_parent_m_idx, state_j]
                    fac3 = MSD_array[node_parent_m_idx, state_j]
                    numer_holder.append(fac1*fac2*fac3)
                NF[num][node_parent_m_idx] = sum(numer_holder)
                for state_j in range(numStates):
                    betas[num][node_parent_m_idx, state_j] = numer_holder[state_j] / NF[num][node_parent_m_idx]
            curr_gen -= 1
    for num in range(numLineages):
        betas_row_sum = np.sum(betas[num], axis=1)
        assert np.allclose(betas_row_sum, 1.)



def get_beta_parent_child_prod(numStates, lineage, beta_array, T, MSD_array, state_j, node_parent_m_idx):
    '''
    Calculates the product of beta-links for every parent-child
    relationship of a given parent cell in a given state.
    '''
    beta_m_n_holder = [] # list to hold the factors in the product
    node_parent_m = lineage[node_parent_m_idx] # get the index of the parent
    children_idx_list = [] # list to hold the children
    if node_parent_m.left is not None:
        node_child_n_left_idx = lineage.index(node_parent_m.left)
        children_idx_list.append(node_child_n_left_idx)
    if node_parent_m.right is not None:
        node_child_n_right_idx = lineage.index(node_parent_m.right)
        children_idx_list.append(node_child_n_right_idx)
    for node_child_n_idx in children_idx_list:
        beta_m_n = beta_parent_child_func(numStates=numStates,
                                          lineage=lineage,
                                          beta_array=beta_array,
                                          T=T,
                                          MSD_array=MSD_array,
                                          state_j=state_j,
                                          node_parent_m_idx=node_parent_m_idx,
                                          node_child_n_idx=node_child_n_idx)
        beta_m_n_holder.append(beta_m_n)
    result = np.prod(beta_m_n_holder) # calculates the product of items in a list
    return result

def beta_parent_child_func(numStates, lineage, beta_array, T, MSD_array, state_j, node_parent_m_idx, node_child_n_idx):
    '''
    This "helper" function calculates the probability
    described as a 'beta-link' between parent and child
    nodes in our tree for some state j. This beta-link
    value is what lets you calculate the values of
    higher (in the direction from the leave
    to the root node) node beta and Normalizing Factor
    values.
    '''
    assert lineage[node_child_n_idx].parent is lineage[node_parent_m_idx] # check the child-parent relationship
    assert lineage[node_child_n_idx].isChild() # if the child-parent relationship is correct, then the child must
    # either be the left daughter or the right daughter
    summand_holder=[] # summing over the states

    for state_k in range(numStates): # for each state k
        numer1 = beta_array[node_child_n_idx, state_k] # get the already calculated beta at node n for state k
        numer2 = T[state_j, state_k] # get the transition rate for going from state j to state k
        # P( z_n = k | z_m = j)
        denom = MSD_array[node_child_n_idx, state_k] # get the MSD for node n at state k
        # P(z_n = k)
        summand_holder.append(numer1*numer2/denom)

    return sum(summand_holder)

def calculate_log_likelihood(tHMMobj, NF):
    '''
    Calculates log likelihood of NF for each lineage.
    '''
    numLineages = tHMMobj.numLineages

    LL = []

    for num in range(numLineages): # for each lineage in our Population
        NF_array = NF[num] # getting the NF of the respective lineage
        log_NF_array = np.log(NF_array)
        ll_per_num = sum(log_NF_array)
        LL.append(ll_per_num)

    return LL
