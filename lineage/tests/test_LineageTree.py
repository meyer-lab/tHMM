""" Unit test file. """
import unittest
import numpy as np
from ..LineageTree import LineageTree


class TestModel(unittest.TestCase):

    def test_generate_lineage_list(self):
        
        # state0: resistant
        # state1: susciptible
        self.pi = np.array([0.85, 0.15]) # initial probability matrix
        self.T = np.array([[0.95, 0.05],
                      [0.3, 0.7]])
        self.desired_num_cells = 2**8 - 1

        full_lin_list = _generate_lineage_list(self)
        self.assertTrue(len(full_lin_list) == len(self.desired_num_cells))
        self.assertTrue(self.full_lin_list == full_lin_list)

        for cell in full_lin_list:
            self.assertTrue(cell is not None)

            
    def test_prune_lineage(self):
        pruned_list = _prune_lineage(self)
        self.assertTrue(len(pruned_list) != 0)
        

    def test_get_state_count(self):
        pass

    def test_full_assign_obs(self):
        pass

    def test_tree_recursion():
        pass
        
