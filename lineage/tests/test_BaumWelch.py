""" Unit test file. """
import unittest
import numpy as np
from ..StateDistribution import StateDistribution
from ..UpwardRecursion import get_leaf_Normalizing_Factors, calculate_log_likelihood
from ..BaumWelch import fit
from ..LineageTree import LineageTree
from ..tHMM import tHMM


class TestModel(unittest.TestCase):
    """ Unit tests for Baum-Welch methods. """
        
    def test_step(self):
        """ This tests that one step of Baum-Welch increases the likelihood of the fit. """
        
        # pi: the initial probability vector
        pi = np.array([0.6, 0.4], dtype="float")

        # T: transition probability matrix
        T = np.array([[0.85, 0.15],
                      [0.15, 0.85]], dtype="float")

        # State 0 parameters "Resistant"
        state0 = 0
        bern_p0 = 0.95
        gamma_a0 = 5.0
        gamma_scale0 = 1.0

        # State 1 parameters "Susciptible"
        state1 = 1
        bern_p1 = 0.85
        gamma_a1 = 10.0
        gamma_scale1 = 2.0

        state_obj0 = StateDistribution(state0, bern_p0, gamma_a0, gamma_scale0)
        state_obj1 = StateDistribution(state1, bern_p1, gamma_a1, gamma_scale1)

        E = [state_obj0, state_obj1]
        
        num = 10000

        # Using an unpruned lineage to avoid unforseen issues
        X = LineageTree(pi, T, E, num, prune_boolean=False)
        
        tHMMobj = tHMM([X], numStates=2)  # build the tHMM class with X
        
        LLbefore = calculate_log_likelihood(tHMMobj, get_leaf_Normalizing_Factors(tHMMobj))
        
        fit(tHMMobj, max_iter=4)

        LL = calculate_log_likelihood(tHMMobj, get_leaf_Normalizing_Factors(tHMMobj))
        
        self.assertGreater(LL, LLbefore)
