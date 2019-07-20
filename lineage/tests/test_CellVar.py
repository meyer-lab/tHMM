""" Unit test file. """
import unittest
import numpy as np
from ..CellVar import CellVar as c, _double


class TestModel(unittest.TestCase):

    def test_cellVar(self):
        """ Make sure cell state assignment is correct. """
        left_state = 0
        right_state = 1
        
        cell_left = c(state=left_state, left=None, right=None, parent=None, gen=1)
        cell_right = c(state=right_state, left=None, right=None, parent=None, gen=1)

        self.assertTrue(cell_left.state == 0)
        self.assertTrue(cell_right.state == 1)
    
    def test_cell_divide(self):
        T = np.array([[1.0, 0.0],
                      [0.0, 1.0]])

        parent_state = 1
        cell = c(state=parent_state, left=None, right=None, parent=None, gen=1)
        left_cell, right_cell = cell._divide(T)
        self.assertTrue(left_cell.state == 1) # the probability of switching states is 0
        self.assertTrue(right_cell.state == 1)
        self.assertTrue(right_cell.parent is cell and left_cell.parent is cell)
        self.assertTrue(cell.left is left_cell and cell.right is right_cell)
        self.assertTrue(not cell.parent)
        self.assertTrue(cell.gen == 1)
        self.assertTrue(left_cell.gen == 2 and right_cell.gen == 2)

        parent_state = 0
        cell = c(state=parent_state, left=None, right=None, parent=None, gen=1)
        left_cell, right_cell = cell._divide(T)
        self.assertTrue(left_cell.state == 0) # the probability of switching states is 0
        self.assertTrue(right_cell.state == 0)
        self.assertTrue(right_cell.parent is cell and left_cell.parent is cell)
        self.assertTrue(cell.left is left_cell and cell.right is right_cell)
        self.assertTrue(not cell.parent)
        self.assertTrue(cell.gen == 1)
        self.assertTrue(left_cell.gen == 2 and right_cell.gen == 2)
        
    def test_isParent(self):
        parent_state = 1
        cell = c(state=parent_state, left=None, right=None, parent=None, gen=1)
        self.assertFalse(cell._isParent())
        left_cell, right_cell = cell._divide(T)
        self.assertTrue(cell._isParent())
        
    def test_isChild(self):
        parent_state = 1
        cell = c(state=parent_state, left=None, right=None, parent=None, gen=1)
        self.assertFalse(cell._isChild())
        left_cell, right_cell = cell._divide(T)
        self.assertTrue(left_cell._isChild() and right_cell._isChild())

    def test_isRootParent(self):
        T = np.array([[1.0, 0.0],
                      [0.0, 1.0]])

        parent_state = 1
        cell = c(state=parent_state, left=None, right=None, parent=None, gen=1)
        self.assertFalse(cell._isRootParent())
        left_cell, right_cell = cell._divide(T)
        self.assertFalse(cell._isRootParent())

    def test_isLeaf(self):
        T = np.array([[1.0, 0.0],
                      [0.0, 1.0]])

        parent_state = 1
        cell = c(state=parent_state, left=None, right=None, parent=None, gen=1)
        self.assertTrue(cell._isLeaf())
        left_cell, right_cell = cell._divide(T)
        self.assertFalse(cell._isLeaf())
        self.assertTrue(left_cell._isLeaf() and right_cell._isLeaf())

    def test_get_sister(self):
        T = np.array([[1.0, 0.0],
                      [0.0, 1.0]])

        parent_state = 1
        cell = c(state=parent_state, left=None, right=None, parent=None, gen=1)
        left_cell, right_cell = cell._divide(T)
        self.assertTrue(left_cell._get_sister() is right_cell and right_cell._get_sister() is left_cell)

    def test_get_root_cell(self):
        T = np.array([[1.0, 0.0],
                      [0.0, 1.0]])

        parent_state = 1
        cell = c(state=parent_state, left=None, right=None, parent=None, gen=1)
        self.assertTrue(cell._get_root_cell() is cell)
        left_cell, right_cell = cell._divide(T)
        self.assertTrue(left_cell._get_root_cell() is cell and right_cell._get_root_cell() is cell)

    def test_cell_double(self):
        """ Make sure double function creates the right and left states properly. """
        # transition matrix
        T = np.array([[1.0, 0.0],
                      [0.0, 1.0]])

        parent_state = 0
        left_state, right_state = _double(parent_state, T)
        self.assertTrue(left_state == 0)
        self.assertTrue(right_state == 0)

        parent_state = 1
        left_state2, right_state2 = _double(parent_state2, T)
        self.assertTrue(left_state2 == 1)
        self.assertTrue(right_state2 == 1)
        
