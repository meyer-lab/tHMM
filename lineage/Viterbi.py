""" This file contains the methods for the Viterbi algorithm implemented in an a upward recursion. """
import numpy as np
from typing import Tuple


# pylint: disable=too-many-nested-blocks


def get_leaf_deltas(tHMMobj) -> Tuple[list, list]:
    """
    Delta matrix and base case at the leaves. 
    Each element in this N by K matrix is the probability for the leaves :math:`P(x_n = x | z_n = k)`.

    :param tHMMobj: the tHMM object
    :return: a list of N by K matrices for each lineage, initialized from the leaf cells by EL(n,k).
    """
    num_states = tHMMobj.num_states

    deltas = []
    state_ptrs = []

    for num, lineageObj in enumerate(tHMMobj.X): # getting the lineage in the Population by index
        lineage = lineageObj.output_lineage
        delta_array = np.zeros((len(lineage), num_states)) # instantiating N by K array
        state_ptrs_array = np.empty((len(lineage), 2), dtype=object)

        for cell in lineage:
            if cell.isLeaf():
                # get the index of the leaf
                leaf_cell_idx = lineage.index(cell)
                delta_array[leaf_cell_idx, :] = tHMMobj.EL[num][leaf_cell_idx, :]

        deltas.append(delta_array)
        state_ptrs.append(state_ptrs_array)
    return deltas, state_ptrs


def get_nonleaf_deltas(tHMMobj, deltas: list, state_ptrs: list):
    """
    Calculates the delta values for all non-leaf cells by filling out the delta matrix passed to it from the :func:`get_leaf_deltas`.

    :param tHMMobj: the tHMM object
    :param deltas: a list of N by K matrices for each lineage, initialized from the leaf cells by EL(n,k).
    :param state_ptrs: a list of N by K matrices that are state pointers, to obtain nonleaf deltas.
    """

    for num, lineageObj in enumerate(tHMMobj.X):
        lineage = lineageObj.output_lineage # getting the lineage in the Population by index
        T = tHMMobj.estimate.T  # getting the transition matrix of the respective lineage

        # move up one generation until the 2nd generation is the children
        # and the root nodes are the parents
        for level in lineageObj.output_list_of_gens[2:][::-1]:
            parent_holder = lineageObj.get_parents_for_level(level)

            for node_parent_m_idx in parent_holder:
                fac1, max_state_ptr = get_delta_parent_child_prod(lineage=lineage, delta_array=deltas[num], T=T, node_parent_m_idx=node_parent_m_idx)

                deltas[num][node_parent_m_idx, :] = fac1 * tHMMobj.EL[num][node_parent_m_idx, :]
                state_ptrs[num][node_parent_m_idx, :] = max_state_ptr


def get_delta_parent_child_prod(lineage: list, delta_array: np.ndarray, T: np.ndarray, node_parent_m_idx: int) -> Tuple[np.array, list]:
    """
    Calculates the delta coefficient for every parent-child relationship of a given parent cell in a given state.

    :param lineage: A list containing cells (which are objects with their own properties).
    :param delta_array: A N by K matrix containing the delta values that will be used in Viterbi.
    :param T: The K by K transition matrix.
    :param node_parent_m_index: The index of the parent to the currently-intended-cell.
    :return delta_m_n_holder: A list to hold the factors in the product.
    :return max_state_ptr: A list of tuples of daughter cell indexes and their state pointers.
    """
    delta_m_n_holder = np.ones(T.shape[0])  # list to hold the factors in the product
    max_state_ptr = []
    # get the index of the parent
    node_parent_m = lineage[node_parent_m_idx]
    children_idx_list = []  # list to hold the children

    if node_parent_m.left:
        children_idx_list.append(lineage.index(node_parent_m.left))

    if node_parent_m.right:
        children_idx_list.append(lineage.index(node_parent_m.right))

    for node_child_n_idx in children_idx_list:
        # get the already calculated delta at node n for state k
        # get the transition rate for going from state j to state k
        # P( z_n = k | z_m = j)
        max_holder = T * delta_array[node_child_n_idx, :]

        state_ptr = np.argmax(max_holder, axis=1)

        delta_m_n_holder *= np.max(max_holder, axis=1)
        max_state_ptr.append((node_child_n_idx, state_ptr))

    return delta_m_n_holder, max_state_ptr


def Viterbi(tHMMobj, deltas: list, state_ptrs: list) -> list:
    """
    Runs the viterbi algorithm and returns a list of arrays containing the optimal state of each cell.
    This function returns the most likely sequence of states for each lineage.

    :param tHMMobj: a class object with properties of the lineages of cells
    :param deltas: a list of N by K matrices containing the delta values for each lineage
    :param state_ptrs: a list of tuples of daughter cell indexes and their state pointers
    :return: assigned states to each cell in all lineages
    """

    all_states = []

    for num, lineageObj in enumerate(tHMMobj.X):
        lineage = lineageObj.output_lineage

        opt_state_tree = np.zeros(len(lineage), dtype=int)
        possible_first_states = np.multiply(deltas[num][0, :], tHMMobj.estimate.pi)
        opt_state_tree[0] = np.argmax(possible_first_states)
        for level in lineageObj.output_list_of_gens[1:]:
            for cell in level:
                parent_idx = lineage.index(cell)
                for n in cell.get_daughters():
                    child_idx = lineage.index(n)
                    parent_state = opt_state_tree[parent_idx]

                    for ii in range(state_ptrs[num].shape[1]):
                        child_state_tuple = state_ptrs[num][parent_idx, ii]

                        if child_state_tuple[0] == child_idx:
                            opt_state_tree[child_idx] = child_state_tuple[1][parent_state]
                            break

        all_states.append(opt_state_tree)

    return all_states
