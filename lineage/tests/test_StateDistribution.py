""" Unit test file. """
import unittest
import numpy as np
import scipy.stats as sp
from ..StateDistribution import StateDistribution, bernoulli_estimator, exponential_estimator, gamma_estimator, prune_rule, report_time, get_experiment_time
from ..LineageTree import LineageTree
from ..CellVar import CellVar as c


class TestModel(unittest.TestCase):
    """Here are the unit tests."""
    def setUp(self):
        # observation parameters for state0
        self.state0 = 0
        self.bern0 = 1.0
        self.expon0 = 40.0
        self.gamma_a0 = 10.0
        self.gamma_b0 = 2.0
        self.stateDist0 = StateDistribution(self.state0, self.bern0, self.expon0, self.gamma_a0, self.gamma_b0)

        # ingredients for LineageTree!
        self.pi = np.array([0.75, 0.25])
        self.T = np.array([[0.85, 0.15],
                      [0.2, 0.8]])

        # observation parameters for state1
        self.state1 = 1
        self.bern1 = 0.8
        self.expon1 = 20
        self.gamma_a1 = 2.0
        self.gamma_b1 = 10.0
        self.stateDist1 = StateDistribution(self.state1, self.bern1, self.expon1, self.gamma_a1, self.gamma_b1)

        # observations object
        self.E = [self.stateDist0, self.stateDist1]

        # creating two lineages, one with False for pruning, one with True.
        self.lineage = LineageTree(self.pi, self.T, self.E, desired_num_cells=2**3 - 1, prune_boolean=False) # 7-cell lineage
        self.lineage2 = LineageTree(self.pi, self.T, self.E, desired_num_cells=2**2 - 1, prune_boolean=False)

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

    def test_prune_rule(self):
        """ A unittest for the prune_rule. """

        _, cells_in_state0, _, _ = self.lineage._full_assign_obs(self.state0)
        for cell in cells_in_state0:
            if cell.obs[0] == 0:
                self.assertTrue(prune_rule(cell) == True)

        _, cells_in_state1, _, _ = self.lineage._full_assign_obs(self.state1)
        for cell in cells_in_state1:
            if cell.obs[0] == 0:
                self.assertTrue(prune_rule(cell) == True)
                
    def test_report_time(self):
        """ Given a cell, the report_time function has to return the time since the start of the experiment to the time of this cell's time. """
        _, cells_in_state0, _, _ = self.lineage._full_assign_obs(self.state0)
        _, cells_in_state1, _, _ = self.lineage._full_assign_obs(self.state1)
        # bringing all the cells after assigning observations to them
        all_cells = cells_in_state0 + cells_in_state1

        # here we check this for the root parent, since the time has taken so far, equals to the lifetime of the cell
        for cell in all_cells:
            if cell._isRootParent():
                parent_tau = cell.obs[1]
                self.assertTrue(report_time(cell) == parent_tau)

        # here we check for the root parent and its left child
        for cell in all_cells:
            if cell._isRootParent():
                taus = cell.obs[1] + cell.left.obs[1]
                self.assertTrue(report_time(cell.left) == taus)

    def test_get_experiment_time(self):
        """ A unittest to check the experiment time is reported correctly. Here we use a lineage with 3 cells, self.lineage2 built in the setup function."""
        _, cells_in_state0, _, _ = self.lineage2._full_assign_obs(self.state0)
        _, cells_in_state1, _, _ = self.lineage2._full_assign_obs(self.state1)
        # bringing all the cells after assigning observations to them
        all_cells = cells_in_state0 + cells_in_state1

        # here we check this for the root parent, since the time has taken so far, equals to the lifetime of the cell
        for cell in all_cells:
            if cell._isRootParent():
                left = cell.obs[1] + cell.left.obs[1]
                right = cell.obs[1] + cell.right.obs[1]
        maximum = max(left, right)
        self.assertTrue(get_experiment_time(self.lineage2) == maximum)
                

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
