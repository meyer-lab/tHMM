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
        cell = c(state = parent_state, left = None, right = None, parent = None, gen = 1)
        left_cell, right_cell = cell._divide(T)

        self.assertTrue(left_cell.state == 1)
        self.assertTrue(right_cell.state == 1)


    def test_cell_double(self):
        """ Make sure double function creates the right and left states properly. """

        # transition matrix
        T = np.array([[1.0, 0.0],
             [0.0, 1.0]])

        # arbitrary parent state, based on T given above, we get the two daughter cell states.
        parent_state = 0

        left_state, right_state = _double(parent_state, T)
        print(left_state)
        self.assertTrue(left_state == 0)
        self.assertTrue(right_state == 0), " double function is not working properly based on transition matrix. "

        # second arbitrary parent state
        parent_state2 = 1

        left_state2, right_state2 = _double(parent_state2, T)
        self.assertTrue(left_state2 == 1)
        self.assertTrue(right_state2 == 1), " double function is not working properly based on transition matrix. "
        
        
        
            
    #        cell1 = c(startT=20)
    #        cell2, cell3 = cell1.divide(40)

    #        # cell divides at correct time & parent dies
    #        self.assertFalse(cell1.isUnfinished())
    #        self.assertTrue(cell1.tau == 20)
    #        self.assertTrue(cell2.startT == 40)
    #        self.assertTrue(cell2.isUnfinished())

        # left and right children exist for cell1 with proper linking
    #        self.assertTrue(cell1.left is cell2)
    #        self.assertTrue(cell1.right is cell3)
    #        self.assertTrue(cell2.parent is cell1)
    #        self.assertTrue(cell3.parent is cell1)
