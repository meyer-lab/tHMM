""" Unit test file. """
import unittest
import numpy as np
from ..StateDistribution import bernoulli_estimator, exponential_estimator, gamma_estimator

class TestModel(unittest.TestCase):
    """Here are the unit tests."""

    def test_bernoulli_estimator(self):
        """ blah """
        bern_obs = sp.bernoulli.rvs(p=0.9, size=100)  # bernoulli observations
        self.assertTrue(0.89 <= bernoulli_estimator(bern_obs) <= 0.91)

    def test_exponential_estimator(self):
        """ blah """
        exp_obs = sp.expon.rvs(scale=50, size=100)  # exponential observations
        self.assertTrue(45 <= exponential_estimator(exp_obs) <= 55)  # +/- 5 of beta

    def test_gamma_estimator(self):
        """ blah """
        gamma_obs = sp.gamma.rvs(a=12.5, scale=3, size=100) # gamma observations
        shape, scale = gamma_estimator(gamma_obs)

        self.assertTrue(10 <= shape <= 15)
        self.assertTrue(2 <= scale <= 4)
