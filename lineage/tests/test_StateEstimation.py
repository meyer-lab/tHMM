""" Unit test file. """
import unittest

from ..LineageTree import LineageTree
from ..states.StateDistributionGamma import StateDistribution as gamma_state
from ..states.StateDistributionExpon import StateDistribution as expon_state

class TestModel(unittest.TestCase):
    """
    Unit test class for state distributions.
    """

    def setUp(self):
        self.pi = np.array([1])
        self.T = np.array([[1]])
        self.E_gamma = [gamma_state()]
        self.E_expon = [expon_state()]

        self.lineage_gamma = LineageTree(self.pi, self.T, self.E_gamma, 2**9)
        self.solver_gamma = tHMM([self.lineage_gamma], 1) # evaluating for one state
        self.solver_gamma.fit()

        self.lineage_expon = LineageTree(self.pi, self.T, self.E_expon, 2**9)
        self.solver_expon = tHMM([self.lineage_expon], 1) # evaluating for one state
        self.solver_expon.fit()

        self.lineage_gamma_censored = LineageTree(self.pi, self.T, self.E_gamma, 2**9, censor_condition=3, desired_experiment_time=30)
        self.solver_gamma_censored = tHMM([self.lineage_gamma_censored], 1) # evaluating for one state
        self.solver_gamma_censored.fit()

        self.lineage_expon_censored = LineageTree(self.pi, self.T, self.E_expon, 2**9, censor_condition=3, desired_experiment_time=30)
        self.solver_expon_censored = tHMM([self.lineage_expon_censored], 1) # evaluating for one state
        self.solver_expon_censored.fit()