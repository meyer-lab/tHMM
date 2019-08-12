'''This file contains the methods for the Viterbi algorithm implemented in an a upward recursion.'''
import numpy as np


def get_leaf_deltas(tHMMobj):
    '''delta matrix and base case at the leaves. Each element in this N by K matrix is the probability for the leaves P(x_n = x | z_n = k).'''
    numStates = tHMMobj.numStates

    EL = tHMMobj.EL

    deltas = []
    state_ptrs = []

    for num, lineageObj in enumerate(
            tHMMobj.X):  # for each lineage in our Population
        # getting the lineage in the Population by index
        lineage = lineageObj.output_lineage
        EL_array = EL[num]  # geting the EL of the respective lineage
        # instantiating N by K array
        delta_array = np.zeros((len(lineage), numStates))
        state_ptrs_array = np.empty(
            (len(lineage), numStates), dtype=object)  # instantiating N by K array

        for cell in lineage:  # for each cell in the lineage
            if cell._isLeaf():  # if it is a leaf
                # get the index of the leaf
                leaf_cell_idx = lineage.index(cell)
                delta_array[leaf_cell_idx, :] = EL_array[leaf_cell_idx, :]

        deltas.append(delta_array)
        state_ptrs.append(state_ptrs_array)
    return deltas, state_ptrs


def get_nonleaf_deltas(tHMMobj, deltas, state_ptrs):
    '''Calculates the delta values for all non-leaf cells.'''
    numStates = tHMMobj.numStates

    EL = tHMMobj.EL

    for num, lineageObj in enumerate(
            tHMMobj.X):  # for each lineage in our Population
        # getting the lineage in the Population by index
        lineage = lineageObj.output_lineage
        T = tHMMobj.estimate.T  # getting the transition matrix of the respective lineage
        EL_array = EL[num]  # geting the EL of the respective lineage

        # move up one generation until the 2nd generation is the children
        # and the root nodes are the parents
        for level in lineageObj.output_list_of_gens[2:][::-1]:
            parent_holder = lineageObj._get_parents_for_level(level)

            for node_parent_m_idx in parent_holder:
                for state_k in range(numStates):
                    fac1, max_state_ptr = get_delta_parent_child_prod(numStates=numStates,
                                                                      lineage=lineage,
                                                                      delta_array=deltas[num],
                                                                      T=T,
                                                                      state_k=state_k,
                                                                      node_parent_m_idx=node_parent_m_idx)
                    fac2 = EL_array[node_parent_m_idx, state_k]
                    deltas[num][node_parent_m_idx, state_k] = fac1 * fac2
                    state_ptrs[num][node_parent_m_idx,
                                    state_k] = max_state_ptr


def get_delta_parent_child_prod(
        numStates, lineage, delta_array, T, state_k, node_parent_m_idx):
    '''Calculates the delta coefficient for every parent-child relationship of a given parent cell in a given state.'''
    delta_m_n_holder = []  # list to hold the factors in the product
    max_state_ptr = []
    # get the index of the parent
    node_parent_m = lineage[node_parent_m_idx]
    children_idx_list = []  # list to hold the children

    if node_parent_m.left:
        node_child_n_left_idx = lineage.index(node_parent_m.left)
        children_idx_list.append(node_child_n_left_idx)

    if node_parent_m.right:
        node_child_n_right_idx = lineage.index(node_parent_m.right)
        children_idx_list.append(node_child_n_right_idx)

    for node_child_n_idx in children_idx_list:
        delta_m_n, state_ptr = delta_parent_child_func(numStates=numStates,
                                                       lineage=lineage,
                                                       delta_array=delta_array,
                                                       T=T,
                                                       state_j=state_k,
                                                       node_parent_m_idx=node_parent_m_idx,
                                                       node_child_n_idx=node_child_n_idx)
        delta_m_n_holder.append(delta_m_n)
        max_state_ptr.append((node_child_n_idx, state_ptr))

    # calculates the product of items in a list
    result = np.prod(delta_m_n_holder)
    return result, max_state_ptr


def delta_parent_child_func(
        numStates, lineage, delta_array, T, state_j, node_parent_m_idx, node_child_n_idx):
    '''Calculates the delta value for a single parent-child relationship where the parent is in a given state.'''
    assert lineage[node_child_n_idx].parent is lineage[node_parent_m_idx]  # check the child-parent relationship
    # if the child-parent relationship is correct, then the child must be
    # either the left daughter or the right daughter
    assert lineage[node_child_n_idx]._isChild()
    max_holder = []  # maxing over the states
    for state_k in range(numStates):  # for each state k
        # get the already calculated delta at node n for state k
        num1 = delta_array[node_child_n_idx, state_k]
        # get the transition rate for going from state j to state k
        num2 = T[state_j, state_k]
        # P( z_n = k | z_m = j)

        max_holder.append(num1 * num2)
        result = max(max_holder)
        # gets the state of the maximum value
        state_ptr = np.argmax(max_holder)
    return result, state_ptr


def Viterbi(tHMMobj, deltas, state_ptrs):
    '''Runs the viterbi algorithm and returns a list of arrays containing the optimal state of each cell.'''
    all_states = []

    for num, lineageObj in enumerate(tHMMobj.X):
        lineage = lineageObj.output_lineage
        pi = tHMMobj.estimate.pi
        delta_array = deltas[num]
        state_ptrs_array = state_ptrs[num]

        opt_state_tree = np.zeros((len(lineage)), dtype=int)
        possible_first_states = np.multiply(delta_array[0, :], pi)
        opt_state_tree[0] = np.argmax(possible_first_states)
        for level in lineageObj.output_list_of_gens[1:]:
            for cell in level:
                parent_idx = lineage.index(cell)
                temp = cell._get_daughters()
                for n in temp:
                    child_idx = lineage.index(n)
                    parent_state = opt_state_tree[parent_idx]
                    temp = state_ptrs_array[parent_idx, parent_state]
                    for child_state_tuple in temp:
                        if child_state_tuple[0] == child_idx:
                            opt_state_tree[child_idx] = child_state_tuple[1]

        all_states.append(opt_state_tree)

    return all_states
