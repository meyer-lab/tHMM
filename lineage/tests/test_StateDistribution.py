""" Unit test file. """
import unittest
import numpy as np
from ..StateDistribution import bernoulli_estimator, exponential_estimator, gamma_estimator

class TestModel(unittest.TestCase):
    def test_bernoulli_estimator(self):
        """ blah """
        self.assertTrue(0.899 <= bernoulli_estimator() <= 1.0)

    def test_exponential_estimator(self):
        """ blah """
        self.assertTrue(45 <= exponential_estimator() <= 55)  # +/- 5 of beta

    def test_gamma_estimator(self):
        """ blah """
        shape, scale = gamma_estimator()

        self.assertTrue(11 <= shape <= 15)
        self.assertTrue(2 <= scale <= 4)
