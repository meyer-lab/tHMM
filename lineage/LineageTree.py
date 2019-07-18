""" This file contains the LineageTree class. """

from .Lineage import CellVar
from .Lineage import ObservationEmission

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


class LineageTree:
    def __init__(self, pi, T, E, desired_num_cells):
        self.pi = pi
        pi_num_states = len(pi)
        self.T = T
        T_shape = self.T.shape()
        assert T_shape[0] == T_shape[1], "Transition numpy array is not square. Ensure that your transition numpy array has the same number of rows and columns."
        T_num_states = self.T.shape[0]
        self.E = E
        E_num_states = len[state for state in self.E.keys()]
        assert pi_num_states == T_num_states == E_num_states, "The number of states in your input Markov probability parameters are mistmatched. Please check that the dimensions and states match. "
        self.num_states = pi_num_states
        self.desired_num_cells = desired_num_cells
        self.lineage_list = self._generate_lineage_list()
        for state in self.num_states:
            self.E["{}".format(state)].cells = self._assign_obs(state)

    def _generate_lineage_list(self):
        """ Generates a single lineage tree given Markov variables. This only generates the hidden variables (i.e., the states). """
        first_state_results = sp.multinomial.rvs(1, self.pi)  # roll the dice and yield the state for the first cell
        first_cell_state = first_state_results.index(1)
        first_cell = CellVar(state=first_cell_state, left=None, right=None, parent=None, gen=1)  # create first cell
        lineage_list = [first_cell]

        for cell in lineage_list:  # letting the first cell proliferate
            if not cell.left:  # if the cell has no daughters...
                left_cell, right_cell = cell._divide(self.T)  # make daughters by dividing and assigning states
                lineage_list.append(left_cell)  # add daughters to the list of cells
                lineage_list.append(right_cell)

            if len(lineage_list) >= desired_num_cells:
                break

        return lineage_list

    def _get_state_count(self, state):
        """ Counts the number of cells in a specific state and makes a list out of those numbers. Used for generating emissions for that specific state. """
        cells_in_state = []  # a list holding cells in the same state
        indices_of_cells_in_state = []
        for cell in self.lineage_list:
            if cell.state == state:  # if the cell is in the given state...
                cells_in_state.append(cell)  # append them to a list
                indices_of_cells_in_state.append(self.lineage_list)

        num_cells_in_state = len(cells_in_state)  # gets the number of cells in the list

        return num_cells_in_state, cells_in_state, indices_of_cells_in_state

    def _assign_obs(self, state):
        """ Observation assignment give a state. """
        num_cells_in_state, cells_in_state, _ = self._get_state_count(state)
        list_of_tuples_of_obs = self.E["{}".format{state}].rvs(size=num_cells_in_state)

        assert len(cells_in_state) == len(list_of_tuples_of_obs) == num_cells_in_state

        for i, cell in enumerate(cells_in_state):
            cell.obs = list_of_tuples_of_obs[i]

        return cells_in_state
