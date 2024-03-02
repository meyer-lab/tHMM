""" Unit test file. """

import unittest
import numpy as np
from ..CellVar import CellVar as c


# pylint: disable=protected-access


class TestModel(unittest.TestCase):
    """
    Unit test class for the cell class.
    """

    def test_cellVar(self):
        """
        Make sure cell state assignment is correct.
        """
        left_state = 0
        right_state = 1

        cell_left = c(state=left_state, parent=None)
        cell_right = c(state=right_state, parent=None)

        self.assertTrue(cell_left.state == 0)
        self.assertTrue(cell_right.state == 1)

    def test_cell_divide(self):
        """
        Tests the division of the cells.
        """
        T = np.array([[1.0, 0.0], [0.0, 1.0]])

        parent_state = 1
        cell = c(state=parent_state, parent=None)
        left_cell, right_cell = cell.divide(T)
        # the probability of switching states is 0
        self.assertTrue(left_cell.state == 1)
        self.assertTrue(right_cell.state == 1)
        self.assertTrue(right_cell.parent is cell and left_cell.parent is cell)
        self.assertTrue(cell.left is left_cell and cell.right is right_cell)
        self.assertTrue(not cell.parent)
        self.assertTrue(cell.gen == 1)
        self.assertTrue(left_cell.gen == 2 and right_cell.gen == 2)

        parent_state = 0
        cell = c(state=parent_state, parent=None)
        left_cell, right_cell = cell.divide(T)
        # the probability of switching states is 0
        self.assertTrue(left_cell.state == 0)
        self.assertTrue(right_cell.state == 0)
        self.assertTrue(right_cell.parent is cell and left_cell.parent is cell)
        self.assertTrue(cell.left is left_cell and cell.right is right_cell)
        self.assertTrue(not cell.parent)
        self.assertTrue(cell.gen == 1)
        self.assertTrue(left_cell.gen == 2 and right_cell.gen == 2)

    def test_isLeafBecauseTerminal(self):
        """
        Tests whether the leaf cells are correctly checked.
        """
        T = np.array([[1.0, 0.0], [0.0, 1.0]])

        parent_state = 1
        cell = c(state=parent_state, parent=None)
        self.assertTrue(cell.isLeafBecauseTerminal())
        left_cell, right_cell = cell.divide(T)
        self.assertFalse(cell.isLeafBecauseTerminal())
        self.assertTrue(
            left_cell.isLeafBecauseTerminal() and right_cell.isLeafBecauseTerminal()
        )
