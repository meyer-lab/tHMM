""" Unit test file. """
import unittest
import numpy as np
from ..UpwardRecursion import calculate_log_likelihood
from ..BaumWelch import fit, calculateQuantities
from ..LineageTree import LineageTree
from ..tHMM import tHMM
from ..states.StateDistPhase import StateDistribution2 as StateDistPhase
from ..figures.figureCommon import pi, T, E


class TestBW(unittest.TestCase):
    """ Unit tests for Baum-Welch methods. """

    def setUp(self):
        """ This setup will be used to test the model for the cases with 3 number of states. """
        # ingredients for LineageTree with 3 states
        self.pi = np.array([0.55, 0.35, 0.10])
        self.T = np.array([[0.75, 0.20, 0.05], [0.1, 0.85, 0.05], [0.1, 0.1, 0.8]])

        # Emissions
        self.E = [StateDistPhase(0.99, 0.9, 20, 5, 10, 3), StateDistPhase(0.88, 0.75, 10, 2, 15, 4), StateDistPhase(0.77, 0.85, 15, 7, 20, 5)]

    def commonTest(self, **kwargs):
        """ This tests that one step of Baum-Welch increases the likelihood of the fit. """
        # Using an unpruned lineage to avoid unforseen issues
        X = LineageTree(pi, T, E, desired_num_cells=(2 ** 11) - 1, **kwargs)
        tHMMobj = tHMM([X], num_states=2)  # build the tHMM class with X
        X3s = LineageTree(self.pi, self.T, self.E, desired_num_cells=(2 ** 11) - 1, **kwargs)
        tHMMobj3s = tHMM([X3s], num_states=3)

        # Test cases below
        # Get the likelihoods before fitting
        _, _, _, LL_before = calculateQuantities(tHMMobj)
        self.assertTrue(np.isfinite(LL_before))
        # For 3 states
        _, _, _, LL_before3 = calculateQuantities(tHMMobj3s)
        self.assertTrue(np.isfinite(LL_before3))

        # Get the likelihoods after fitting
        _, NF_after, _, _, new_LL_list_after = fit(tHMMobj, max_iter=4)
        LL_after = calculate_log_likelihood(NF_after)
        self.assertTrue(np.isfinite(LL_after))
        self.assertTrue(np.isfinite(new_LL_list_after))

        # for 3 states
        _, NF_after3, _, _, new_LL_list_after3 = fit(tHMMobj3s, max_iter=4)
        LL_after3 = calculate_log_likelihood(NF_after3)
        self.assertGreater(LL_after3, LL_before3)

    def test_step(self):
        """ This tests that one step of Baum-Welch increases the likelihood of the fit. """
        self.commonTest()

    def test_step1(self):
        """ This tests that one step of Baum-Welch increases the likelihood of the fit. """
        self.commonTest(censor_condition=1)

    def test_step2(self):
        """ This tests that one step of Baum-Welch increases the likelihood of the fit. """
        self.commonTest(censor_condition=2, desired_experimental_time=500)

    def test_step3(self):
        """ This tests that one step of Baum-Welch increases the likelihood of the fit. """
        self.commonTest(censor_condition=3, desired_experimental_time=1000)
