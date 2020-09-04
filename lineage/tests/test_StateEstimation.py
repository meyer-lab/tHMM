""" Unit test file. """
import unittest
import numpy as np

from ..LineageTree import LineageTree
from ..tHMM import tHMM
from ..states.StateDistributionGamma import StateDistribution as gamma_state


class TestModel(unittest.TestCase):
    """
    Unit test class for state distributions.
    """

    def setUp(self):
        """
        Creates and fits one state lineages.
        """
        self.pi = np.array([1])
        self.T = np.array([[1]])
        self.E_gamma = [gamma_state(bern_p=1., gamma_a=7, gamma_scale=4.5)]
        # Setting the bern_p to 1. ensures that all cells live and censoring is only
        # due to living past the experiment time

    def test_estimationEvaluationGamma(self):
        """
        Evaluates the performance of fitting and the underlying estimator
        by comparing the parameter estimates to their true values.
        Gamma uncensored.
        """
        lineage_gamma = LineageTree.init_from_parameters(self.pi, self.T, self.E_gamma, 2**9)
        solver_gamma = tHMM([lineage_gamma], 1)  # evaluating for one state
        solver_gamma.fit()
        self.assertGreater(2., np.linalg.norm(solver_gamma.estimate.E[0].params - self.E_gamma[0].params))

    def test_estimationEvaluationGammaCensored(self):
        """
        Evaluates the performance of fitting and the underlying estimator
        by comparing the parameter estimates to their true values.
        Gamma censored.
        """

        def gen(): return LineageTree.init_from_parameters(self.pi, self.T, self.E_gamma, 2**9, censor_condition=3, desired_experiment_time=50)
        lineage_gamma_censored = [gen() for _ in range(20)]
        solver_gamma_censored = tHMM(lineage_gamma_censored, 1)  # evaluating for one state
        solver_gamma_censored.fit()
        self.assertGreater(5., np.linalg.norm(solver_gamma_censored.estimate.E[0].params - self.E_gamma[0].params))
