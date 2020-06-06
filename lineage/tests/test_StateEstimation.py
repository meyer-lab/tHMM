""" Unit test file. """
import unittest
import numpy as np

from ..LineageTree import LineageTree
from ..tHMM import tHMM
from ..states.StateDistributionGamma import StateDistribution as gamma_state
from ..states.StateDistributionExpon import StateDistribution as expon_state


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
        self.E_expon = [expon_state(bern_p=1., exp_beta=7.0)]
        # Setting the bern_p to 1. ensures that all cells live and censoring is only
        # due to living past the experiment time

        self.lineage_gamma = LineageTree(self.pi, self.T, self.E_gamma, 2**9)
        self.solver_gamma = tHMM([self.lineage_gamma], 1)  # evaluating for one state
        self.solver_gamma.fit()
        self.gamma_state_estimate = self.solver_gamma.estimate.E[0]

        self.lineage_expon = LineageTree(self.pi, self.T, self.E_expon, 2**9)
        self.solver_expon = tHMM([self.lineage_expon], 1)  # evaluating for one state
        self.solver_expon.fit()
        self.expon_state_estimate = self.solver_expon.estimate.E[0]

        self.lineage_gamma_censored = LineageTree(self.pi, self.T, self.E_gamma, 2**9, censor_condition=3, desired_experiment_time=30)
        assert not all([cell.obs[2] == 1 for cell in self.lineage_gamma_censored.output_lineage])  # ensures that at least some cells are censored
        self.solver_gamma_censored = tHMM([self.lineage_gamma_censored], 1)  # evaluating for one state
        self.solver_gamma_censored.fit()
        self.gamma_state_censored_estimate = self.solver_gamma_censored.estimate.E[0]

        self.lineage_expon_censored = LineageTree(self.pi, self.T, self.E_expon, 2**9, censor_condition=3, desired_experiment_time=30)
        assert not all([cell.obs[2] == 1 for cell in self.lineage_expon_censored.output_lineage])  # ensures that at least some cells are censored
        self.solver_expon_censored = tHMM([self.lineage_expon_censored], 1)  # evaluating for one state
        self.solver_expon_censored.fit()
        self.expon_state_censored_estimate = self.solver_expon_censored.estimate.E[0]

    def estimationEvaluationGamma(self):
        """
        Evaluates the performance of fitting and the underlying estimator
        by comparing the parameter estimates to their true values.
        Gamma uncensored.
        """
        self.assertGreater(1., abs(self.gamma_state_estimate.gamma_a - self.E_gamma.gamma_a))
        self.assertGreater(1., abs(self.gamma_state_estimate.gamma_scale - self.E_gamma.gamma_scale))

    def estimationEvaluationExpon(self):
        """
        Evaluates the performance of fitting and the underlying estimator
        by comparing the parameter estimates to their true values.
        Exponential uncensored.
        """
        self.assertGreater(1., abs(self.expon_state_estimate.exp_beta - self.E_expon.exp_beta))

    def estimationEvaluationGammaCensored(self):
        """
        Evaluates the performance of fitting and the underlying estimator
        by comparing the parameter estimates to their true values.
        Gamma censored.
        """
        self.assertGreater(1., abs(self.gamma_state_censored_estimate.gamma_a - self.E_gamma.gamma_a))
        self.assertGreater(1., abs(self.gamma_state_censored_estimate.gamma_scale - self.E_gamma.gamma_scale))

    def estimationEvaluationExponCensored(self):
        """
        Evaluates the performance of fitting and the underlying estimator
        by comparing the parameter estimates to their true values.
        Exponential censored.
        """
        self.assertGreater(1., abs(self.expon_state_censored_estimate.exp_beta - self.E_expon.exp_beta))
