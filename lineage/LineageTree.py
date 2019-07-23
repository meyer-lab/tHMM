""" This file contains the LineageTree class. """

from .CellVar import CellVar
from .StateDistribution import prune_rule

import scipy.stats as sp
import numpy as np

# temporary style guide:
# Boolean functions are in camelCase.
# Functions that return cells or lists of cells will be spaced with underscores.
# Functions that are not to be used by a general user are prefixed with an underscore.
# States of cells are 0-indexed and discrete (states start at 0 and are whole numbers).
# Variables with the prefix num (i.e., num_states, num_cells) have an underscore following the 'num'.
# Docstrings use """ and not '''.
# Class names use camelCase.
# The transition matrix must be a square matrix, not a list of lists.


class LineageTree:
    def __init__(self, pi, T, E, desired_num_cells, prune_boolean):
        """
        A class for the structure of the lineage tree. Every lineage from this class is a binary tree built based on initial probabilities and transition probabilities given by the user that builds up the states based off of these until it reaches the desired number of cells in the tree, and then stops. Given the desired distributions for emission, the object will have the "E" a list of state distribution objects assigned to them.

        Args:
        -----
        pi {numpy array}: The initial probability matrix; its shape must be the same as the number of states and all of them must sum up to 1.

        T {numpy square matrix}: The transition probability matrix; every row must sum up to 1.

        E {list}: A list containing state distribution objects, the length of it is the same as the number of states. 

        desired_num_cells {Int}: The desired number of cells we want the lineage to end up with.

        prune_boolean {bool}: If it is True, it means the user want this lineage to be pruned, if False it means the user want this lineage as a full binary tree -- in which none of the cells die.        
        """
        self.pi = pi
        pi_num_states = len(pi)
        self.T = T
        T_shape = self.T.shape
        assert T_shape[0] == T_shape[1], "Transition numpy array is not square. Ensure that your transition numpy array has the same number of rows and columns."
        T_num_states = self.T.shape[0]
        self.E = E
        E_num_states = len(E)
        assert pi_num_states == T_num_states == E_num_states, "The number of states in your input Markov probability parameters are mistmatched. Please check that the dimensions and states match. "
        self.num_states = pi_num_states
        self.desired_num_cells = desired_num_cells
        
        self.fullLineage_list = self._generate_lineage_list()
        
        for state in range(self.num_states):
            self.E[state].num_full_lin_cells, self.E[state].full_lin_cells, self.E[state].full_lin_cells_idx = self._full_assign_obs(state)
    
        self.prune_boolean = prune_boolean # this is given by the user, true of they want the lineage to be pruned, false if they want the full binary tree
        self.pruned_list = self._prune_lineage()

        for state in range(self.num_states):
            self.E[state].num_pruned_lin_cells, self.E[state].pruned_lin_cells, self.E[state].pruned_lin_cells_idx = self._get_state_count(state, prune=True)

        # Based on the user's decision, if they want the lineage to be pruned (prune_boolean == True), 
        # the lineage tree that is given to the tHMM, will be the pruned one.
        # If the user decides that they want the full binary tree (prune_boolean == False), 
        # then the full_lin_list will be passed to the output_lineage.
        if prune_boolean:
            self.output_lineage = self.pruned_lin_list
        else:
            self.output_lineage = self.full_lin_list


    def _generate_lineage_list(self):
        """ Generates a single lineage tree given Markov variables. This only generates the hidden variables (i.e., the states) in a full binary tree manner. It generates the tree until it reaches the desired number of cells in the lineage.
        Args:
        -----
        It takes in the LineageTree object

        Returns:
        --------
        full_lin_list {list}: A list containing cells with assigned hidden states based on initial and transition probabilities.
        """
        first_state_results = sp.multinomial.rvs(1, self.pi)  # roll the dice and yield the state for the first cell
        first_cell_state = first_state_results.tolist().index(1)
        first_cell = CellVar(state=first_cell_state, left=None, right=None, parent=None, gen=1)  # create first cell
        self.full_lin_list = [first_cell]

        for cell in self.full_lin_list:  # letting the first cell proliferate
            if not cell.left:  # if the cell has no daughters...
                left_cell, right_cell = cell._divide(self.T)  # make daughters by dividing and assigning states
                self.full_lin_list.append(left_cell)  # add daughters to the list of cells

            if len(self.full_lin_list) == self.desired_num_cells:
                break

        return self.full_lin_list

    def _prune_lineage(self):
        """ This function removes those cells that are intended to be remove from the full binary tree based on emissions.
        It takes in LineageTree object, walks through all the cells in the full binary tree, applies the pruning to each cell that is supposed to be removed, and returns the pruned list of cells.
        """
        self.pruned_lin_list = self.full_lin_list.copy()
        for cell in self.pruned_lin_list:
            if prune_rule(cell):
                _, _, self.pruned_lin_list = find_two_subtrees(cell, self.pruned_lin_list)
                cell.left = None
                cell.right = None
                assert cell._isLeaf()
        return self.pruned_lin_list

    def _get_state_count(self, state, prune):
        """ Counts the number of cells in a specific state and makes a list out of those numbers. Used for generating emissions for that specific state.
        Args:
        -----
        state {Int}: The number assigned to a state.

        Returns:
        --------
        num_cells_in_state {Int}: The number of cells in the given state.

        cells_in_state {list}: A list of cells being in the given state.

        indices_of_cells_in_state {list}: Holding the indexes of the cells being in the given state 
        """
        cells_in_state = []  # a list holding cells in the same state
        indices_of_cells_in_state = []
        list_to_use = []
        if prune:
            list_to_use = self.pruned_lin_list
        else:
            list_to_use = self.full_lin_list
        for cell in list_to_use:
            if cell.state == state:  # if the cell is in the given state...
                cells_in_state.append(cell)  # append them to a list
                indices_of_cells_in_state.append(list_to_use.index(cell))
        num_cells_in_state = len(cells_in_state)  # gets the number of cells in the list

        return num_cells_in_state, cells_in_state, indices_of_cells_in_state

    def _full_assign_obs(self, state):
        """ Observation assignment give a state. """
        num_cells_in_state, cells_in_state, indices_of_cells_in_state = self._get_state_count(state, prune=False)
        list_of_tuples_of_obs = self.E[state].rvs(size=num_cells_in_state)

        assert len(cells_in_state) == len(list_of_tuples_of_obs) == num_cells_in_state

        for i, cell in enumerate(cells_in_state):
            cell.obs = list_of_tuples_of_obs[i]

        return num_cells_in_state, cells_in_state, indices_of_cells_in_state
    
    def __repr__(self):
        if self.prune_boolean:
            s1 = "This tree is pruned. It is made of {} states.\n For each state in this tree: ".format(self.num_states)
            s_list = []
            for state in range(self.num_states):
                s_list.append("\n \t There are {} cells of state {}".format(self.E[state].num_pruned_lin_cells, state))
            seperator = ', '
            s2 = seperator.join(s_list)
            s3 = ".\n This pruned tree has {} many cells in total".format(len(self.pruned_lin_list))
            return s1 + s2 + s3
        else:
            s1 = "This tree is NOT pruned. It is made of {} states.\n For each state in this tree: ".format(self.num_states)
            s_list = []
            for state in range(self.num_states):
                s_list.append("\n \t There are {} cells of state {}".format(self.E[state].num_full_lin_cells, state))
            seperator = ', '
            s2 = seperator.join(s_list)
            s3 = ".\n This UNpruned tree has {} many cells in total".format(len(self.full_lin_list))
            return s1 + s2 + s3
            
    def __str__(self):
        if self.prune_boolean:
            s1 = "This tree is pruned. It is made of {} states.\n For each state in this tree: ".format(self.num_states)
            s_list = []
            for state in range(self.num_states):
                s_list.append("\n \t There are {} cells of state {}".format(self.E[state].num_pruned_lin_cells, state))
            seperator = ', '
            s2 = seperator.join(s_list)
            s3 = ".\n This pruned tree has {} cells in total".format(len(self.pruned_lin_list))
            return s1 + s2 + s3
        else:
            s1 = "This tree is NOT pruned. It is made of {} states.\n For each state in this tree: ".format(self.num_states)
            s_list = []
            for state in range(self.num_states):
                s_list.append("\n \t There are {} cells of state {}".format(self.E[state].num_full_lin_cells, state))
            seperator = ', '
            s2 = seperator.join(s_list)
            s3 = ".\n This UNpruned tree has {} cells in total".format(len(self.full_lin_list))
            return s1 + s2 + s3

# tools for traversing trees

def tree_recursion(cell, subtree):
    """ a recurssive function that traverses upwards from the leaf to the root. """
    if cell._isLeaf():
        return
    subtree.append(cell.left)
    subtree.append(cell.right)
    tree_recursion(cell.left, subtree)
    tree_recursion(cell.right, subtree)
    return

def get_subtrees(node, lineage):
    """ Given one cell, return the subtree of that cell, and return all the tree other than that subtree. """
    subtree = [node]
    tree_recursion(node, subtree)
    not_subtree = []
    for cell in lineage:
        if cell not in subtree:
            not_subtree.append(cell)
    return subtree, not_subtree

def find_two_subtrees(cell, lineage):
    """ Gets the left and right subtrees from a cell. """
    left_sub, _ = get_subtrees(cell.left, lineage)
    right_sub, _ = get_subtrees(cell.right, lineage)
    neither_subtree = []
    for node in lineage:
        if node not in left_sub and node not in right_sub:
            neither_subtree.append(node)
    return left_sub, right_sub, neither_subtree

def get_mixed_subtrees(node_m, node_n, lineage):
    m_sub, _ = get_subtrees(node_m, lineage)
    n_sub, _ = get_subtrees(node_n, lineage)
    mixed_sub = []
    for cell in m_sub:
        if cell not in n_sub:
            mixed_sub.append(cell)
    not_mixed = []
    for cell in lineage:
        if cell not in mixed_sub:
            not_mixed.append(cell)
    return mixed_sub, not_mixed
