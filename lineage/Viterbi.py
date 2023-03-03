""" This file contains the methods for the Viterbi algorithm implemented in an a upward recursion. """
import numpy as np
from typing import Tuple

from .LineageTree import get_Emission_Likelihoods


def get_deltas(X: list, E: list, T: np.ndarray) -> Tuple[list[np.ndarray], list]:
    """
    Delta matrix and base case at the leaves.
    Each element in this N by K matrix is the probability for the leaves :math:`P(x_n = x | z_n = k)`.

    Then calculates delta values for non-leaf cells by filling out the delta matrix.

    :param tHMMobj: the tHMM object
    :return: a list of N by K matrices for each lineage, initialized from the leaf cells by EL(n,k).
    """
    EL = get_Emission_Likelihoods(X, E)

    deltas = []
    state_ptrs = []

    for num, linObj in enumerate(X):  # getting the lineage in the Population by index
        delta_array = np.zeros((len(linObj), len(E)))  # instantiating N by K array
        state_ptrs_array = np.empty((len(linObj), 2), dtype=object)
        delta_array[linObj.leaves_idx, :] = EL[num][linObj.leaves_idx, :]

        # move up one generation until the 2nd generation is the children
        # and the root nodes are the parents
        for level in linObj.idx_by_gen[1:][::-1]:
            for pIDX in np.unique(linObj.cell_to_parent[level]):
                fac1, max_state_ptr = get_delta_parent_child_prod(
                    linObj, delta_array, T, pIDX
                )

                delta_array[pIDX, :] = fac1 * EL[num][pIDX, :]
                state_ptrs_array[pIDX, :] = max_state_ptr

        deltas.append(delta_array)
        state_ptrs.append(state_ptrs_array)

    return deltas, state_ptrs


def get_delta_parent_child_prod(
    linObj, delta_array: np.ndarray, T: np.ndarray, node_parent_m_idx: int
) -> Tuple[np.ndarray, list]:
    """
    Calculates the delta coefficient for every parent-child relationship of a given parent cell in a given state.
    In these set of functions
    state pointer is an array of size K, that holds the state number with the highest probability in each row of the max_holder.
    max_holder is a matrix of size K x K that initially starts from T, and gets updated.
    delta_m_n_holder is a vector of size K that has the highest of those probabilities.

    :param lineage: A list containing cells (which are objects with their own properties).
    :param delta_array: A N by K matrix containing the delta values that will be used in Viterbi.
    :param T: The K by K transition matrix.
    :param node_parent_m_index: The index of the parent to the currently-intended-cell.
    :return delta_m_n_holder: A list to hold the factors in the product.
    :return max_state_ptr: A list of tuples of daughter cell indexes and their state pointers.
    """
    delta_m_n_holder = np.ones(T.shape[0])  # list to hold the factors in the product
    max_state_ptr = []
    children_idx_list = linObj.cell_to_daughters[
        node_parent_m_idx, :
    ]  # list to hold the children
    assert np.all(children_idx_list != -1)  # Make sure we found cells

    for cIDX in children_idx_list:
        # get the already calculated delta at node n for state k
        # get the transition rate for going from state j to state k
        # P( z_n = k | z_m = j)
        max_holder = T * delta_array[cIDX, :]

        state_ptr = np.argmax(max_holder, axis=1)
        delta_m_n_holder *= np.max(max_holder, axis=1)
        max_state_ptr.append((cIDX, state_ptr))

    return delta_m_n_holder, max_state_ptr


def Viterbi(tHMMobj) -> list[np.ndarray]:
    """
    Runs the viterbi algorithm and returns a list of arrays containing the optimal state of each cell.
    This function returns the most likely sequence of states for each lineage.

    :param tHMMobj: a class object with properties of the lineages of cells
    :param deltas: a list of N by K matrices containing the delta values for each lineage
    :param state_ptrs: a list of tuples of daughter cell indexes and their state pointers
    :return: assigned states to each cell in all lineages
    """
    deltas, state_ptrs = get_deltas(tHMMobj.X, tHMMobj.E, tHMMobj.T)
    all_states = []

    for num, lineageObj in enumerate(tHMMobj.X):
        opt_state_tree = np.zeros(len(lineageObj), dtype=int)
        possible_first_states = np.multiply(deltas[num][0, :], tHMMobj.estimate.pi)
        opt_state_tree[0] = np.argmax(possible_first_states)

        for pIDX, cell in enumerate(lineageObj.output_lineage):
            if cell.gen == 0:
                continue

            parent_state = opt_state_tree[pIDX]

            for cIDX in lineageObj.cell_to_daughters[pIDX, :]:
                if cIDX == -1:  # If a daughter does not exist
                    continue

                for ii in range(state_ptrs[num].shape[1]):
                    child_state_tuple = state_ptrs[num][pIDX, ii]

                    if child_state_tuple[0] == cIDX:
                        opt_state_tree[cIDX] = child_state_tuple[1][parent_state]
                        break

        all_states.append(opt_state_tree)

    return all_states
