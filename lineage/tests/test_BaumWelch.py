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
        bern_p0 = 0.99
        gamma_a0 = 20
        gamma_loc = 0
        gamma_scale0 = 5

        # State 1 parameters "Susceptible"
        state1 = 1
        bern_p1 = 0.8
        gamma_a1 = 10
        gamma_scale1 = 1

        state_obj0 = StateDistribution(state0, bern_p0, gamma_a0, gamma_loc, gamma_scale0)
        state_obj1 = StateDistribution(state1, bern_p1, gamma_a1, gamma_loc, gamma_scale1)

        E = [state_obj0, state_obj1]

        # Using an unpruned lineage to avoid unforseen issues
        X = LineageTree(pi, T, E, desired_num_cells=(2**11) - 1, desired_experiment_time=500, prune_condition='die', prune_boolean=False)
        tHMMobj = tHMM([X], numStates=2)  # build the tHMM class with X

        # Test cases below
        # Get the likelihoods before fitting
        NF_before = get_leaf_Normalizing_Factors(tHMMobj)
        betas_before = get_leaf_betas(tHMMobj, NF_before)
        get_nonleaf_NF_and_betas(tHMMobj, NF_before, betas_before)
        LL_before = calculate_log_likelihood(NF_before)
        self.assertTrue(np.isfinite(LL_before))

        # Get the likelihoods after fitting
        tHMMobj_after, NF_after, _, _, new_LL_list_after = fit(tHMMobj, max_iter=4)
        LL_after = calculate_log_likelihood(NF_after)
        self.assertTrue(np.isfinite(LL_after))
        self.assertTrue(np.isfinite(new_LL_list_after))

        self.assertGreater(LL_after, LL_before)
