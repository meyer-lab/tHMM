""" Unit test file. """
import unittest
import numpy as np
import scipy.stats as sp
from ..states.StateDistribution import (
    StateDistribution,
    gamma_estimator,
    gamma_pdf,
)
from ..states.stateCommon import (
    bern_pdf,
    bernoulli_estimator,
    get_experiment_time,
)
from ..LineageTree import LineageTree


class TestModel(unittest.TestCase):
    """
    Unit test class for state distributions.
    """

    def setUp(self):
        # ingredients for LineageTree!
        self.pi = np.array([0.75, 0.25])
        self.T = np.array([[0.85, 0.15], [0.20, 0.80]])

        # bern, gamma_a, gamma_scale
        self.E = [StateDistribution(0.99, 20, 5), StateDistribution(0.80, 10, 1)]

        # creating two lineages, one with False for pruning, one with True.
        self.lineage = LineageTree(self.pi, self.T, self.E, desired_num_cells=(2 ** 11) - 1)
        self.lineage2 = LineageTree(self.pi, self.T, self.E, desired_num_cells=(2 ** 5.5) - 1, censor_condition=2, desired_experiment_time=50)
        self.lineage3 = LineageTree(self.pi, self.T, self.E, desired_num_cells=(2 ** 11) - 1, censor_condition=3, desired_experiment_time=800)
        self.population = [LineageTree(self.pi, self.T, self.E, desired_num_cells=(2 ** 11) - 1, censor_condition=3, desired_experiment_time=800) for i in range(50)]

    def test_rvs(self):
        """
        A unittest for random generator function,
        given the number of random variables we want from each distribution,
        that each corresponds to one of the observation types
        """
        tuple_of_obs = self.E[0].rvs(size=30)
        bern_obs, gamma_obs, _ = list(zip(*tuple_of_obs))
        self.assertTrue(len(bern_obs) == len(gamma_obs) == 30)

        tuple_of_obs1 = self.E[1].rvs(size=40)
        bern_obs1, gamma_obs1, _ = list(zip(*tuple_of_obs1))
        self.assertTrue(len(bern_obs1) == len(gamma_obs1) == 40)

    def test_pdf(self):
        """
        A unittest for the likelihood function.
        Here we generate one set of observation
        (the size == 1 which mean we just have one bernoulli, and one gamma).
        """
        # for stateDist0
        list_of_tuple_of_obs = self.E[0].rvs(size=1)
        tuple_of_obs = list_of_tuple_of_obs[0]
        likelihood = self.E[0].pdf(tuple_of_obs)
        self.assertTrue(0.0 <= likelihood <= 1.0)

        # for stateDist1
        list_of_tuple_of_obs1 = self.E[1].rvs(size=1)
        tuple_of_obs1 = list_of_tuple_of_obs1[0]
        likelihood1 = self.E[1].pdf(tuple_of_obs1)
        self.assertTrue(0.0 <= likelihood1 <= 1.0)

    def test_estimator(self):
        """
        A unittest for the estimator function, by generating 150 observatopns for each of the
        distribution functions, we use the estimator and compare. """
        tuples_of_obs = self.E[0].rvs(size=3000)
        gammas = np.array([1] * len(tuples_of_obs))
        estimator_obj = self.E[0].estimator(tuples_of_obs, gammas)

        # here we check the estimated parameters to be close
        self.assertTrue(0.0 <= abs(estimator_obj.bern_p - self.E[0].bern_p) <= 0.1)
        self.assertTrue(0.0 <= abs(estimator_obj.gamma_a - self.E[0].gamma_a) <= 3.0)
        self.assertTrue(0.0 <= abs(estimator_obj.gamma_scale - self.E[0].gamma_scale) <= 3.0)

    def test_censor(self):
        """
        A unittest for testing whether censoring is working
        as expected.
        """
        for lin in self.population:
            for cell in lin.output_lineage:
                if not cell.isRootParent:
                    if cell.parent.censored:
                        self.assertTrue(cell.censored)

    def test_get_experiment_time(self):
        """
        A unittest for obtaining the experiment time.
        """
        experiment_time2 = get_experiment_time(self.lineage2)
        experiment_time = get_experiment_time(self.lineage)
        self.assertLess(experiment_time2, experiment_time)

    def test_bernoulli_estimator(self):
        """
        Testing the bernoulli estimator,
        by comparing the result of the estimator
        to the result of scipy random variable generator.
        """
        bern_obs = sp.bernoulli.rvs(p=0.90, size=1000)  # bernoulli observations
        gammas = np.array([1] * len(bern_obs))
        self.assertTrue(0.87 <= bernoulli_estimator(bern_obs, (0.5,), gammas) <= 0.93)

    def test_gamma_estimator(self):
        """
        Testing the gamma estimator,
        by comparing the result of the estimator
        to the result of scipy random variable generator.
        """
        gamma_obs = sp.gamma.rvs(a=12.5, scale=3, size=1000)  # gamma observations
        gamma_censor_obs = np.ones_like(gamma_obs)
        gammas = [1] * len(gamma_obs)

        shape, scale = gamma_estimator(gamma_obs, gamma_censor_obs, gammas)

        self.assertTrue(10 <= shape <= 15)
        self.assertTrue(2 <= scale <= 4)

    def test_bern_pdf(self):
        """
        Testing the Bernoulli probability density function
        by comparing the result of the outputted likelihood
        against a known calculated value.
        """
        bern_ll = bern_pdf(x=1, p=1)
        self.assertTrue(bern_ll == 1)

    def test_gamma_pdf(self):
        """
        Testing the gamma probability density function
        by comparing the result of the outputted likelihood
        against a known calculated value.
        """
        gamma_ll = gamma_pdf(x=1, a=10, scale=5)
        self.assertTrue(gamma_ll <= 0.1)
