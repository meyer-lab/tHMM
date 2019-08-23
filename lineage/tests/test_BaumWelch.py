""" Unit test file. """
import unittest
import numpy as np
from ..StateDistribution import StateDistribution
from ..UpwardRecursion import get_leaf_Normalizing_Factors, get_leaf_betas, get_nonleaf_NF_and_betas, calculate_log_likelihood
from ..BaumWelch import fit
from ..LineageTree import LineageTree
from ..tHMM import tHMM


class TestBW(unittest.TestCase):
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
        
        num = 2**7-1

        # Using an unpruned lineage to avoid unforseen issues
        X = LineageTree(pi, T, E, num, prune_boolean=False)
        tHMMobj = tHMM([X], numStates=2)  # build the tHMM class with X
        
        # Get the likelihoods before fitting
        NF_before = get_leaf_Normalizing_Factors(tHMMobj)
        betas_before = get_leaf_betas(tHMMobj, NF_before)
        get_nonleaf_NF_and_betas(tHMMobj, NF_before, betas_before)
        LL_before = calculate_log_likelihood(tHMMobj, NF_before)
        self.assertTrue(np.isfinite(LL_before[0]))

        # Get the likelihoods after fitting
        tHMMobj_after, NF_after, betas_after, gammas_after, new_LL_list_after = fit(tHMMobj, max_iter=4)
        LL_after = calculate_log_likelihood(tHMMobj, NF_after)
        self.assertTrue(np.isfinite(LL_after[0]))
        self.assertTrue(np.isfinite(new_LL_list_after[0]))

        self.assertTrue(np.isfinite(LL_after[0]))
        self.assertGreater(LL_after[0], LL_before[0])
