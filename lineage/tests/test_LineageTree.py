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
                
        self.lineage1_big_full = LineageTree(pi, T, E, desired_num_cells=2**9 - 1, prune_boolean=False)
        self.lineage2_small_full = LineageTree(pi, T, E, desired_num_cells=2**3 - 1, prune_boolean=False)
        self.lineage3_big_pruned = LineageTree(pi, T, E, desired_num_cells=2**9 - 1, prune_boolean=True)
        self.lineage4_small_pruned = LineageTree(pi, T, E, desired_num_cells=2**3 - 1, prune_boolean=True)
            
    def test_prune_lineage(self):
        pass
        
    def test_get_state_count(self):
        pass

    def test_full_assign_obs(self):
        pass

    def test_tree_recursion():
        pass
        
