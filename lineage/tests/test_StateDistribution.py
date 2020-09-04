""" Unit test file. """
import unittest
import pytest
from copy import deepcopy
import numpy as np
import scipy.stats as sp
from ..states.StateDistributionGamma import StateDistribution
from ..states.StateDistributionGaPhs import StateDistribution as StateDistPhase
from ..states.stateCommon import (
    bernoulli_estimator,
    gamma_estimator,
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

        # Emissions
        self.E = [StateDistribution(0.99, 20, 5), StateDistribution(0.80, 10, 1)]
        self.E2 = [StateDistPhase(0.99, 0.9, 20, 5, 10, 3), StateDistPhase(0.8, 0.75, 10, 2, 15, 4)]

        # creating two lineages, one with False for pruning, one with True.
        self.lineage = LineageTree.init_from_parameters(self.pi, self.T, self.E, desired_num_cells=(2 ** 11) - 1)
        self.lineage2 = LineageTree.init_from_parameters(self.pi, self.T, self.E, desired_num_cells=(2 ** 5.5) - 1, censor_condition=2, desired_experiment_time=50)
        self.lineage3 = LineageTree.init_from_parameters(self.pi, self.T, self.E, desired_num_cells=(2 ** 11) - 1, censor_condition=3, desired_experiment_time=800)
        self.population = [LineageTree.init_from_parameters(self.pi, self.T, self.E, desired_num_cells=(2 ** 11) - 1, censor_condition=3, desired_experiment_time=800) for i in range(50)]

    def test_rvs(self):
        """
        A unittest for random generator function,
        given the number of random variables we want from each distribution,
        that each corresponds to one of the observation types
        """
        bern_obs, gamma_obs, _ = self.E[0].rvs(size=30)
        self.assertTrue(len(bern_obs) == len(gamma_obs) == 30)

        bern_obs1, gamma_obs1, _ = self.E[1].rvs(size=40)
        self.assertTrue(len(bern_obs1) == len(gamma_obs1) == 40)

        bern_obsG1, bern_obsG2, gamma_obsG1, gamma_obsG2, _, _ = self.E2[0].rvs(size=50)
        self.assertTrue(len(bern_obsG1) == len(bern_obsG2) == len(gamma_obsG1) == len(gamma_obsG2) == 50)

    def test_estimator(self):
        """
        A unittest for the estimator function, by generating 3000 observatons for each of the
        distribution functions, we use the estimator and compare. """
        # Gamma dist.
        tuples_of_obs = self.E[0].rvs(size=5000)
        tuples_of_obs = list(map(list, zip(*tuples_of_obs)))
        gammas = np.ones(len(tuples_of_obs))
        estimator_obj = deepcopy(self.E[0])
        estimator_obj.estimator(tuples_of_obs, gammas)

        # G1/G2 separated Gamma dist.
        tuples_of_obsPhase = self.E2[0].rvs(size=5000)
        tuples_of_obsPhase = list(map(list, zip(*tuples_of_obsPhase)))
        gammas = np.ones(len(tuples_of_obsPhase))
        estimator_objPhase = deepcopy(self.E2[0])
        estimator_objPhase.estimator(tuples_of_obsPhase, gammas, const=None)

        # here we check the estimated parameters to be close for Gamma distribution
        np.testing.assert_allclose(estimator_obj.params, self.E[0].params, rtol=0.1)

        # For StateDistPhase
        np.testing.assert_allclose(estimator_objPhase.G1.params, self.E2[0].G1.params, rtol=0.1)
        np.testing.assert_allclose(estimator_objPhase.G2.params, self.E2[0].G2.params, rtol=0.1)

    def test_censor(self):
        """
        A unittest for testing whether censoring is working
        as expected.
        """
        for lin in self.population:
            for cell in lin.output_lineage:
                if not cell.isRootParent:
                    if not cell.parent.observed:
                        self.assertFalse(cell.observed)

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
        self.assertTrue(0.87 <= bernoulli_estimator(bern_obs, gammas) <= 0.93)

    def test_gamma_estimator(self):
        """
        Testing the gamma estimator,
        by comparing the result of the estimator
        to the result of scipy random variable generator.
        """
        gamma_obs = sp.gamma.rvs(a=12.5, scale=3, size=1000)  # gamma observations
        gamma_censor_obs = np.ones_like(gamma_obs)
        gammas = np.ones_like(gamma_obs)

        shape, scale = gamma_estimator(gamma_obs, gamma_censor_obs, gammas, None)
        self.assertTrue(10 <= shape <= 15)
        self.assertTrue(2 <= scale <= 4)


@pytest.mark.parametrize("dist", [StateDistribution, StateDistPhase])
def test_self_dist_zero(dist):
    """ Test that the distance from a distribution to itself is zero. """
    dd = dist()
    assert dd.dist(dd) == 0.0
