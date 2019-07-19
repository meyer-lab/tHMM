""" Unit test file. """
import unittest
import numpy as np
from ..LineageTree import LineageTree


class TestModel(unittest.TestCase):

    def test_cellVar(self):
        """ Make sure cell state assignment is correct. """
        left_state = 0
        right_state = 1
        
        cell_left = c(state=left_state, left=None, right=None, parent=None, gen=1)
        cell_right = c(state=right_state, left=None, right=None, parent=None, gen=1)

        self.assertTrue(cell_left.state == 0)
        self.assertTrue(cell_right.state == 1)
