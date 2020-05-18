""" Unit test file. """
import unittest
import numpy as np
from ..UpwardRecursion import calculate_log_likelihood
from ..BaumWelch import fit, calculateQuantities
from ..LineageTree import LineageTree
from ..tHMM import tHMM
from lineage.figures.figureCommon import pi, T, E


class TestBW(unittest.TestCase):
    """ Unit tests for Baum-Welch methods. """

    def commonTest(self, **kwargs):
        """ This tests that one step of Baum-Welch increases the likelihood of the fit. """
        # Using an unpruned lineage to avoid unforseen issues
        X = LineageTree(pi, T, E, desired_num_cells=(2 ** 11) - 1, **kwargs)
        tHMMobj = tHMM([X], num_states=2)  # build the tHMM class with X

        # Test cases below
        # Get the likelihoods before fitting
        _, _, _, LL_before = calculateQuantities(tHMMobj)
        self.assertTrue(np.isfinite(LL_before))

        # Get the likelihoods after fitting
        _, NF_after, _, _, new_LL_list_after = fit(tHMMobj, max_iter=4)
        LL_after = calculate_log_likelihood(NF_after)
        self.assertTrue(np.isfinite(LL_after))
        self.assertTrue(np.isfinite(new_LL_list_after))

        self.assertGreater(LL_after, LL_before)

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
