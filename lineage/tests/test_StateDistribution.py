""" Unit test file. """
import unittest
import numpy as np
import scipy.stats as sp
from ..StateDistribution import StateDistribution, bernoulli_estimator, exponential_estimator, gamma_estimator, report_time
from ..LineageTree import LineageTree
from ..CellVar import CellVar as c


class TestModel(unittest.TestCase):
    """Here are the unit tests."""

    def test_bernoulli_estimator(self):
        """ blah """
        bern_obs = sp.bernoulli.rvs(p=0.9, size=1000)  # bernoulli observations
        self.assertTrue(0.88 <= bernoulli_estimator(bern_obs) <= 0.92)

    def test_exponential_estimator(self):
        """ blah """
        exp_obs = sp.expon.rvs(scale=50, size=1000)  # exponential observations
        self.assertTrue(45 <= exponential_estimator(exp_obs) <= 55)  # +/- 5 of beta

    def test_gamma_estimator(self):
        """ blah """
        gamma_obs = sp.gamma.rvs(a=12.5, scale=3, size=1000)  # gamma observations
        shape, scale = gamma_estimator(gamma_obs)

        self.assertTrue(10 <= shape <= 15)
        self.assertTrue(2 <= scale <= 4)

    def test_report_time(self):

        T = np.array([[1.0, 0.0],
                      [0.0, 1.0]])

        pi = [0.0, 1.0]

        state0 = 0
        bern_p0 = 1.0
        expon_scale_beta0 = 20
        gamma_a0 = 5.0
        gamma_scale0 = 1.0

        # State 1 parameters "Susciptible"
        state1 = 1
        bern_p1 = 1.0
        expon_scale_beta1 = 60
        gamma_a1 = 10.0
        gamma_scale1 = 2.0

        state_obj0 = StateDistribution(state0, bern_p0, expon_scale_beta0, gamma_a0, gamma_scale0)
        state_obj1 = StateDistribution(state1, bern_p1, expon_scale_beta1, gamma_a1, gamma_scale1)

        E = [state_obj0, state_obj1]

        desired_num_cells = 2**2 - 1 
        prune_boolean = True # To get the full tree
        lineage1 = LineageTree(pi, T, E, desired_num_cells, prune_boolean)

        X = [lineage1]
        list_of_cells = []
        for num, lineageObj in enumerate(X):
            lineage = lineageObj.output_lineage
            for cells in lineage:
                list_of_cells.append(cells)
        
        self.assertTrue(len(list_of_cells == 3))
        print(list_of_cells)
        cell0_tau = list_of_cells[0].obs[1]
        cell0_to_left_tau = list_of_cells[1].obs[1] + cell0_tau
        cell0_to_right_tau = list_of_cells[2].obs[1] + cell0_tau

        cell0_tau_report = report_time(list_of_cells[0])
        self.assertTrue(cell0_tau == cell0_tau_report)

        cell0_left_report = report_time(list_of_cells[1])
        self.assertTrue(cell0_to_left_tau == cell0_left_report)

        cell0_right_report = report_time(list_of_cells[2])
        self.assertTrue(cell0_to_right_tau == cell0_right_report)
        