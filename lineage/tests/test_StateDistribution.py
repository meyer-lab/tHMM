""" Unit test file. """
import unittest
import numpy as np
import scipy.stats as sp
from ..StateDistribution import StateDistribution, bernoulli_estimator, exponential_estimator, gamma_estimator, report_time, get_experiment_time
from ..LineageTree import LineageTree
from ..CellVar import CellVar as c


class TestModel(unittest.TestCase):
    """Here are the unit tests."""
    def setUp(self):
        # observation parameters for state0
        self.state0 = 0
        self.bern0 = 0.95
        self.expon0 = 40.0
        self.gamma_a0 = 10.0
        self.gamma_b0 = 2.0
        self.stateDist0 = StateDistribution(self.state0, self.bern0, self.expon0, self.gamma_a0, self.gamma_b0)

        # observation parameters for state1
        self.state1 = 1
        self.bern1 = 0.8
        self.expon1 = 20
        self.gamma_a1 = 2.0
        self.gamma_b1 = 10.0
        self.stateDist1 = StateDistribution(self.state1, self.bern1, self.expon1, self.gamma_a1, self.gamma_b1)


    def test_rvs(self):
        """ A unittest for random generator function, given the number of random variables we want from each distribution, that each corresponds to one of the observation types. """
        tuple_of_obs = self.stateDist0.rvs(size=30)
        bern_obs, exp_obs, gamma_obs = list(zip(*tuple_of_obs))
        self.assertTrue(len(bern_obs) == len(exp_obs) == len(gamma_obs) == 30)

        tuple_of_obs1 = self.stateDist1.rvs(size=40)
        bern_obs1, exp_obs1, gamma_obs1 = list(zip(*tuple_of_obs1))
        self.assertTrue(len(bern_obs1) == len(exp_obs1) == len(gamma_obs1) == 40)
        
    def test_pdf(self):
        """ A unittest for the likelihood function. Here we generate one set of observation (the size == 1 which mean we just have one bernoulli, one exponential, and one gamma) although we don't need gamma AND exponential  together, for now we will leave it this way. """
        # for stateDist0
        tuple_of_obs = self.stateDist0.rvs(size=1)
        likelihood = self.stateDist0.pdf(tuple_of_obs)
        self.assertTrue(0.0 <= likelihood <= 1.0), " The likelihood function calculation is not working properly."

        # for stateDist1
        tuple_of_obs1 = self.stateDist1.rvs(size=1)
        likelihood1 = self.stateDist1.pdf(tuple_of_obs1)
        self.assertTrue(0.0 <= likelihood1 <= 1.0), " The likelihood function calculation is not working properly."
        
    def test_estimator(self):
        """ A unittest for the estimator function, by generating 150 observatopns for each of the distribution functions, we use the estimator and compare. """
        tuples_of_obs = self.stateDist0.rvs(size=150)
        estimator_obj = self.stateDist0.estimator(tuples_of_obs)

        # here we check the estimated parameters to be close
        self.assertEqual(estimator_obj.state, self.stateDist0.state)
        self.assertTrue(0.0 <= abs(estimator_obj.bern_p - self.stateDist0.bern_p) <= 0.5)
        self.assertTrue(0.0 <= abs(estimator_obj.expon_scale_beta - self.stateDist0.expon_scale_beta) <= 7.0)
        self.assertTrue(0.0 <= abs(estimator_obj.gamma_a - self.stateDist0.gamma_a) <= 1.0)
        self.assertTrue(0.0 <= abs(estimator_obj.gamma_scale - self.stateDist0.gamma_scale) <= 1.0)

    def test_report_time(self):
        pass
    def test_get_experiment_time(self):
        pass

    def test_bernoulli_estimator(self):
        """ Testing the bernoulli estimator, by comparing the result of the estimator to the result of scipy random variable generator. """
        bern_obs = sp.bernoulli.rvs(p=0.90, size=1000)  # bernoulli observations
        self.assertTrue(0.87 <= bernoulli_estimator(bern_obs) <= 0.93)

    def test_exponential_estimator(self):
        """ Testing the exponential estimator, by comparing the result of the estimator to the result of scipy random variable generator. """
        exp_obs = sp.expon.rvs(scale=50, size=1000)  # exponential observations
        self.assertTrue(45 <= exponential_estimator(exp_obs) <= 55)  # +/- 5 of beta

    def test_gamma_estimator(self):
        """ Testing the gamma estimator, by comparing the result of the estimator to the result of scipy random variable generator. """
        gamma_obs = sp.gamma.rvs(a=12.5, scale=3, size=1000)  # gamma observations
        shape, scale = gamma_estimator(gamma_obs)

        self.assertTrue(10 <= shape <= 15)
        self.assertTrue(2 <= scale <= 4)
