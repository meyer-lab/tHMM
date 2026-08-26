"""This file contains the methods for the Viterbi algorithm implemented in an upward recursion."""

import numpy as np

from .LineageTree import get_Emission_Likelihoods


def get_deltas(X: list, E: list, T: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Delta matrix and base case at the leaves.
    Each element in this N by K matrix is the probability for the leaves :math:`P(x_n = x | z_n = k)`.

    Then calculates delta values for non-leaf cells by filling out the delta matrix.

    :param X: list of lineage trees
    :param E: list of emission distributions
    :param T: transition probability matrix
    :return: deltas and state pointers for each lineage
    """
    EL = get_Emission_Likelihoods(X, E)

    deltas = []
    state_ptrs = []

    for num, linObj in enumerate(X):
        tree = linObj.tree
        delta_array = np.zeros((len(linObj), len(E)))
        state_ptrs_array = np.zeros((len(linObj), T.shape[0]), dtype=int)
        delta_array[linObj.leaves_idx, :] = EL[num][linObj.leaves_idx, :]

        # Get non-leaves in reverse topological order
        pIDXs = np.nonzero(np.diff(tree.indptr) > 0)[0][::-1]

        for pIDX in pIDXs:
            fac1 = np.ones(T.shape[0])
            children = tree.indices[tree.indptr[pIDX] : tree.indptr[pIDX + 1]]

            for cIDX in children:
                max_holder = T * delta_array[cIDX, :]
                state_ptrs_array[cIDX, :] = np.argmax(max_holder, axis=1)
                fac1 *= np.max(max_holder, axis=1)

            delta_array[pIDX, :] = fac1 * EL[num][pIDX, :]

        deltas.append(delta_array)
        state_ptrs.append(state_ptrs_array)

    return deltas, state_ptrs


def Viterbi(tHMMobj) -> list[np.ndarray]:
    """
    Runs the Viterbi algorithm and returns a list of arrays containing the optimal state of each cell.
    This function returns the most likely sequence of states for each lineage.

    :param tHMMobj: a class object with properties of the lineages of cells
    :return: assigned states to each cell in all lineages
    """
    deltas, state_ptrs = get_deltas(tHMMobj.X, tHMMobj.estimate.E, tHMMobj.estimate.T)
    all_states = []

    for num, lineageObj in enumerate(tHMMobj.X):
        tree = lineageObj.tree
        opt_state_tree = np.zeros(len(lineageObj), dtype=int)
        possible_first_states = np.multiply(deltas[num][0, :], tHMMobj.estimate.pi)
        opt_state_tree[0] = np.argmax(possible_first_states)

        for pIDX in range(len(lineageObj)):
            parent_state = opt_state_tree[pIDX]
            children = tree.indices[tree.indptr[pIDX] : tree.indptr[pIDX + 1]]

            for cIDX in children:
                opt_state_tree[cIDX] = state_ptrs[num][cIDX, parent_state]

        all_states.append(opt_state_tree)

    return all_states
