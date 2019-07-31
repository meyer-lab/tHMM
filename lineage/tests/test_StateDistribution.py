""" Unit test file. """
import unittest
import numpy as np
import scipy.stats as sp
from ..StateDistribution import StateDistribution, bernoulli_estimator, exponential_estimator, gamma_estimator, report_time
from ..LineageTree import LineageTree
from ..CellVar import CellVar as c


class TestModel(unittest.TestCase):
    """Here are the unit tests."""

    def test_bernoulli_estimator(self):
        """ blah """
        bern_obs = sp.bernoulli.rvs(p=0.90, size=1000)  # bernoulli observations
        self.assertTrue(0.87 <= bernoulli_estimator(bern_obs) <= 0.93)

    def test_exponential_estimator(self):
        """ blah """
        exp_obs = sp.expon.rvs(scale=50, size=1000)  # exponential observations
        self.assertTrue(45 <= exponential_estimator(exp_obs) <= 55)  # +/- 5 of beta

    def test_gamma_estimator(self):
        """ blah """
        gamma_obs = sp.gamma.rvs(a=12.5, scale=3, size=1000)  # gamma observations
        shape, scale = gamma_estimator(gamma_obs)

        self.assertTrue(10 <= shape <= 15)
        self.assertTrue(2 <= scale <= 4)
