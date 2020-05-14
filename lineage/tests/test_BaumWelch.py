""" Unit test file. """
import unittest
import numpy as np
from ..states.StateDistribution import StateDistribution
from ..LineageTree import LineageTree
from ..tHMM import tHMM
from ..Analyze import Analyze


class TestBW(unittest.TestCase):
    """ Unit tests for Baum-Welch methods. """

    def test_step(self):
        """ This tests that one step of Baum-Welch increases the likelihood of the fit. """

        # pi: the initial probability vector
        pi = np.array([0.6, 0.4], dtype="float")

        # T: transition probability matrix
        T = np.array([[0.85, 0.15], [0.15, 0.85]], dtype="float")
        # State 0 parameters "Resistant"
        bern_p0 = 0.99
        gamma_a0 = 20
        gamma_scale0 = 5

        # State 1 parameters "Susceptible"
        bern_p1 = 0.8
        gamma_a1 = 10
        gamma_scale1 = 1

        state_obj0 = StateDistribution(bern_p0, gamma_a0, gamma_scale0)
        state_obj1 = StateDistribution(bern_p1, gamma_a1, gamma_scale1)

        E = [state_obj0, state_obj1]

        # Using an unpruned lineage to avoid unforseen issues
        X = LineageTree(pi, T, E, desired_num_cells=(2 ** 11) - 1)
        tHMMobj, pred_states_by_lineage, LL_before = Analyze([X], num_states, max_iter=1)

        # Get the likelihoods after fitting
        tHMMobj, pred_states_by_lineage, LL_after = Analyze([X], num_states, max_iter=5)
        self.assertTrue(np.isfinite(LL_after))
        self.assertTrue(np.isfinite(new_LL_list_after))

        self.assertGreater(LL_after, LL_before)

    def test_step1(self):
        """ This tests that one step of Baum-Welch increases the likelihood of the fit. """

        # pi: the initial probability vector
        pi = np.array([0.6, 0.4], dtype="float")

        # T: transition probability matrix
        T = np.array([[0.85, 0.15], [0.15, 0.85]], dtype="float")
        # State 0 parameters "Resistant"
        bern_p0 = 0.99
        gamma_a0 = 20
        gamma_scale0 = 5

        # State 1 parameters "Susceptible"
        bern_p1 = 0.8
        gamma_a1 = 10
        gamma_scale1 = 1

        state_obj0 = StateDistribution(bern_p0, gamma_a0, gamma_scale0)
        state_obj1 = StateDistribution(bern_p1, gamma_a1, gamma_scale1)

        E = [state_obj0, state_obj1]

        X = LineageTree(pi, T, E, desired_num_cells=(2 ** 11) - 1)
        tHMMobj, pred_states_by_lineage, LL_before = Analyze([X], num_states, max_iter=1)

        # Get the likelihoods after fitting
        tHMMobj, pred_states_by_lineage, LL_after = Analyze([X], num_states, max_iter=5)
        self.assertTrue(np.isfinite(LL_after))
        self.assertTrue(np.isfinite(new_LL_list_after))

        self.assertGreater(LL_after, LL_before)

    def test_step2(self):
        """ This tests that one step of Baum-Welch increases the likelihood of the fit. """

        # pi: the initial probability vector
        pi = np.array([0.6, 0.4], dtype="float")

        # T: transition probability matrix
        T = np.array([[0.85, 0.15], [0.15, 0.85]], dtype="float")
        # State 0 parameters "Resistant"
        bern_p0 = 0.99
        gamma_a0 = 20
        gamma_scale0 = 5

        # State 1 parameters "Susceptible"
        bern_p1 = 0.8
        gamma_a1 = 10
        gamma_scale1 = 1

        state_obj0 = StateDistribution(bern_p0, gamma_a0, gamma_scale0)
        state_obj1 = StateDistribution(bern_p1, gamma_a1, gamma_scale1)

        E = [state_obj0, state_obj1]

        X = LineageTree(pi, T, E, desired_num_cells=(2 ** 11) - 1)
        tHMMobj, pred_states_by_lineage, LL_before = Analyze([X], num_states, max_iter=1)

        # Get the likelihoods after fitting
        tHMMobj, pred_states_by_lineage, LL_after = Analyze([X], num_states, max_iter=5)
        self.assertTrue(np.isfinite(LL_after))
        self.assertTrue(np.isfinite(new_LL_list_after))

        self.assertGreater(LL_after, LL_before)

    def test_step3(self):
        """ This tests that one step of Baum-Welch increases the likelihood of the fit. """

        # pi: the initial probability vector
        pi = np.array([0.6, 0.4], dtype="float")

        # T: transition probability matrix
        T = np.array([[0.85, 0.15], [0.15, 0.85]], dtype="float")
        # State 0 parameters "Resistant"
        bern_p0 = 0.99
        gamma_a0 = 20
        gamma_scale0 = 5

        # State 1 parameters "Susceptible"
        bern_p1 = 0.8
        gamma_a1 = 10
        gamma_scale1 = 1

        state_obj0 = StateDistribution(bern_p0, gamma_a0, gamma_scale0)
        state_obj1 = StateDistribution(bern_p1, gamma_a1, gamma_scale1)

        E = [state_obj0, state_obj1]

        X = LineageTree(pi, T, E, desired_num_cells=(2 ** 11) - 1)
        tHMMobj, pred_states_by_lineage, LL_before = Analyze([X], num_states, max_iter=1)

        # Get the likelihoods after fitting
        tHMMobj, pred_states_by_lineage, LL_after = Analyze([X], num_states, max_iter=5)
        self.assertTrue(np.isfinite(LL_after))
        self.assertTrue(np.isfinite(new_LL_list_after))

        self.assertGreater(LL_after, LL_before)
