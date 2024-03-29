""" Unit test file. """

import unittest
import pytest
from copy import deepcopy
import numpy as np
from ..states.StateDistributionGamma import StateDistribution
from ..states.StateDistributionGaPhs import StateDistribution as StateDistPhase
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
        self.E2 = [
            StateDistPhase(0.99, 0.9, 100, 1, 20, 1.5),
            StateDistPhase(0.8, 0.75, 100, 0.2, 60, 1),
        ]

        # creating two lineages, one with False for pruning, one with True.
        self.lineage = LineageTree.rand_init(
            self.pi, self.T, self.E, desired_num_cells=(2**11) - 1
        )
        self.lineage2 = LineageTree.rand_init(
            self.pi,
            self.T,
            self.E,
            desired_num_cells=(2**5.5) - 1,
            censor_condition=2,
            desired_experiment_time=50,
        )
        self.lineage3 = LineageTree.rand_init(
            self.pi,
            self.T,
            self.E,
            desired_num_cells=(2**11) - 1,
            censor_condition=3,
            desired_experiment_time=800,
        )
        self.population = [
            LineageTree.rand_init(
                self.pi,
                self.T,
                self.E,
                desired_num_cells=(2**11) - 1,
                censor_condition=3,
                desired_experiment_time=800,
            )
            for i in range(50)
        ]

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
        self.assertTrue(
            len(bern_obsG1)
            == len(bern_obsG2)
            == len(gamma_obsG1)
            == len(gamma_obsG2)
            == 50
        )

    def test_estimator(self):
        """
        A unittest for the estimator function, by generating 3000 observatons for each of the
        distribution functions, we use the estimator and compare."""
        # Gamma dist.
        tuples_of_obs = self.E[0].rvs(size=5000)
        tuples_of_obs = np.vstack(tuples_of_obs).T
        gammas = np.ones(tuples_of_obs.shape[0])
        estimator_obj = deepcopy(self.E[0])
        estimator_obj.estimator(tuples_of_obs, gammas)

        # G1/G2 separated Gamma dist.
        tuples_of_obsPhase = self.E2[0].rvs(size=5000)
        tuples_of_obsPhase = np.vstack(tuples_of_obsPhase).T
        gammas = np.ones(tuples_of_obsPhase.shape[0])
        estimator_objPhase = deepcopy(self.E2[0])
        estimator_objPhase.estimator(tuples_of_obsPhase, gammas)

        # here we check the estimated parameters to be close for Gamma distribution
        np.testing.assert_allclose(estimator_obj.params, self.E[0].params, rtol=0.1)

        # For StateDistPhase
        np.testing.assert_allclose(
            estimator_objPhase.G1.params, self.E2[0].G1.params, rtol=0.1
        )
        np.testing.assert_allclose(
            estimator_objPhase.G2.params, self.E2[0].G2.params, rtol=0.1
        )

    def test_censor(self):
        """
        A unittest for testing whether censoring is working
        as expected.
        """
        for lin in self.population:
            for cell in lin.output_lineage[1:]:
                if not cell.parent.observed:
                    self.assertFalse(cell.observed)


@pytest.mark.parametrize("dist", [StateDistribution, StateDistPhase])
def test_self_dist_zero(dist):
    """Test that the distance from a distribution to itself is zero."""
    dd = dist()
    assert dd.dist(dd) == 0.0
