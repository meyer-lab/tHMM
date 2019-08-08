""" Unit test file. """
import unittest
import numpy as np
from ..LineageTree import LineageTree
from ..StateDistribution import StateDistribution


class TestModel(unittest.TestCase):

    def setUp(self):
        # pi: the initial probability vector
        pi = np.array([0.75, 0.25])

        # T: transition probability matrix
        T = np.array([[0.85, 0.15],
                      [0.2, 0.8]])

        # E: states are defined as StateDistribution objects
        # State 0 parameters "Resistant"
        state0 = 0
        bern_p0 = 0.99
        expon_scale_beta0 = 20
        gamma_a0 = 5.0
        gamma_scale0 = 1.0

        # State 1 parameters "Susciptible"
        state1 = 1
        bern_p1 = 0.8
        expon_scale_beta1 = 80
        gamma_a1 = 10.0
        gamma_scale1 = 2.0

        state_obj0 = StateDistribution(state0, bern_p0, expon_scale_beta0, gamma_a0, gamma_scale0)
        state_obj1 = StateDistribution(state1, bern_p1, expon_scale_beta1, gamma_a1, gamma_scale1)

        E = [state_obj0, state_obj1]

        self.lineage1_big = LineageTree(pi, T, E, desired_num_cells=2**9 - 1, prune_boolean=True)
        self.lineage2_small = LineageTree(pi, T, E, desired_num_cells=2**5 - 1, prune_boolean=True)
        self.lineage3_big_pruned = LineageTree(pi, T, E, desired_num_cells=2**9 - 1, prune_boolean=False)
        self.lineage4_small_pruned = LineageTree(pi, T, E, desired_num_cells=2**3 - 1, prune_boolean=False)

    def test_generate_lineage_list(self):
        """ A unittest for generate_lineage_list. """
        # checking the number of cells generated is equal to the desired number of cells given by the user.
        self.assertTrue(len(self.lineage1_big.full_lin_list) == 2**9 -1)
        self.assertTrue(len(self.lineage2_small.full_lin_list) == 2**5 -1)
        self.assertTrue(self.lineage1_big.full_lin_list == self.lineage1_big.output_lineage), "The output lineage is wrong according to prune boolean"
        self.assertTrue(self.lineage3_big_pruned.pruned_lin_list == self.lineage3_big_pruned.output_lineage), "The output lineage is wrong according to prune boolean"

    def test_prune_lineage(self):
        """ A unittest for prune_lineage. """
        # checking the number of cells in the pruned version is smaller than the full version.
        self.assertTrue(len(self.lineage1_big.full_lin_list) >= len(self.lineage1_big.pruned_lin_list))
        self.assertTrue(len(self.lineage2_small.full_lin_list) >= len(self.lineage2_small.pruned_lin_list))

        # checking all the cells in the pruned version should have all the bernoulli observations == 1 (dead cells have been removed.)
        for cell in self.lineage1_big.pruned_lin_list:
            self.assertFalse(cell.obs[0] == 0), "There are still cells that have died but not removed!"
            if cell._isLeaf():
                self.assertTrue(cell.left == None)
                self.assertTrue(cell.right == None)

        for cell in self.lineage2_small.pruned_lin_list:
            self.assertFalse(cell.obs[0] == 0), "There are still cells that have died but not removed!"
            if cell._isLeaf():
                self.assertTrue(cell.left == None)
                self.assertTrue(cell.right == None)
            
        
        
        