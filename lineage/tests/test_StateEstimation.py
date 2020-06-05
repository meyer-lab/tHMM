""" Unit test file. """
import unittest

from ..LineageTree import LineageTree
from ..states.StateDistributionGamma import StateDistribution as gamma_state
from ..states.StateDistributionExponential import StateDistribution as exp_state

class TestModel(unittest.TestCase):
    """
    Unit test class for state distributions.
    """

    def setUp(self):
        self.pi = np.array([1])
        self.T = np.array([[1]])
        self.E_gamma = [gamma_state()]
        self.E_exp = [exp_state()]
        
        self.lineage_gamma_uncensored = LineageTree(self.pi, self.T, self.E_gamma, 2**9)
        self.lineage_gamma = LineageTree(self.pi, self.T, self.E_gamma, 2**9, censor_condition=3, desired_experiment_time=30)
        
        self.lineage_exp_uncensored = LineageTree(self.pi, self.T, self.E_exp, 2**9)
        self.lineage_exp = LineageTree(self.pi, self.T, self.E_exp, 2**9, censor_condition=3, desired_experiment_time=30)