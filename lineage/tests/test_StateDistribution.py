""" Unit test file. """
import unittest
import numpy as np
import scipy.stats as sp
from ..StateDistribution import StateDistribution, bernoulli_estimator, exponential_estimator, gamma_estimator, prune_rule, report_time, get_experiment_time
from ..LineageTree import LineageTree


class TestModel(unittest.TestCase):
    """Here are the unit tests."""

    def setUp(self):
        # ingredients for LineageTree!
        self.pi = np.array([0.75, 0.25])
        self.T = np.array([[0.85, 0.15],
                           [0.20, 0.80]])

        # State 0 parameters "Resistant"
        self.state0 = 0
        bern_p0 = 0.99
        gamma_a0 = 20
        gamma_loc = 0.0
        gamma_scale0 = 5

        # State 1 parameters "Susceptible"
        self.state1 = 1
        bern_p1 = 0.8
        gamma_a1 = 10
        gamma_scale1 = 1

        self.stateDist0 = StateDistribution(self.state0, bern_p0, gamma_a0, gamma_loc, gamma_scale0)
        self.stateDist1 = StateDistribution(self.state1, bern_p1, gamma_a1, gamma_loc, gamma_scale1)

        self.E = [self.stateDist0, self.stateDist1]

        # creating two lineages, one with False for pruning, one with True.
        self.lineage = LineageTree(
            self.pi,
            self.T,
            self.E,
            desired_num_cells=2**3 - 1,
            prune_boolean=False)  # 7-cell lineage
        self.lineage2 = LineageTree(
            self.pi,
            self.T,
            self.E,
            desired_num_cells=2**2 - 1,
            prune_boolean=False)

    def test_rvs(self):
        """ A unittest for random generator function, given the number of random variables we want from each distribution, that each corresponds to one of the observation types. """
        tuple_of_obs = self.stateDist0.rvs(size=30)
        bern_obs, gamma_obs = list(zip(*tuple_of_obs))
        self.assertTrue(len(bern_obs) == len(gamma_obs) == 30)

        tuple_of_obs1 = self.stateDist1.rvs(size=40)
        bern_obs1, gamma_obs1 = list(zip(*tuple_of_obs1))
        self.assertTrue(len(bern_obs1) == len(gamma_obs1) == 40)

    def test_pdf(self):
        """
        A unittest for the likelihood function.
        Here we generate one set of observation
        (the size == 1 which mean we just have one bernoulli, and one gamma).
        """
        # for stateDist0
        list_of_tuple_of_obs = self.stateDist0.rvs(size=1)
        tuple_of_obs = list_of_tuple_of_obs[0]
        likelihood = self.stateDist0.pdf(tuple_of_obs)
        self.assertTrue(0.0 <= likelihood <= 1.0)

        # for stateDist1
        list_of_tuple_of_obs1 = self.stateDist1.rvs(size=1)
        tuple_of_obs1 = list_of_tuple_of_obs1[0]
        likelihood1 = self.stateDist1.pdf(tuple_of_obs1)
        self.assertTrue(0.0 <= likelihood1 <= 1.0)

    def test_estimator(self):
        """ A unittest for the estimator function, by generating 150 observatopns for each of the distribution functions, we use the estimator and compare. """
        tuples_of_obs = self.stateDist0.rvs(size=3000)
        estimator_obj = self.stateDist0.estimator(tuples_of_obs)

        # here we check the estimated parameters to be close
        self.assertEqual(estimator_obj.state, self.stateDist0.state)
        self.assertTrue(
            0.0 <= abs(
                estimator_obj.bern_p -
                self.stateDist0.bern_p) <= 0.1)
        self.assertTrue(
            0.0 <= abs(
                estimator_obj.gamma_a -
                self.stateDist0.gamma_a) <= 3.0)
        self.assertTrue(
            0.0 <= abs(
                estimator_obj.gamma_scale -
                self.stateDist0.gamma_scale) <= 3.0)

    def test_prune_rule(self):
        """ A unittest for the prune_rule. """

        for cell in self.lineage.lineage_stats[0].full_lin_cells:
            if cell.obs[0] == 0:
                self.assertTrue(prune_rule(cell))

        for cell in self.lineage.lineage_stats[1].full_lin_cells:
            if cell.obs[0] == 0:
                self.assertTrue(prune_rule(cell))

    def test_report_time(self):
        """
        Given a cell, the report_time function has to
        return the time since the start of the experiment
        to the time of this cell's time.
        """
        full_lin_cells_holder = []
        for state in range(2):
            full_lin_cells_holder.append(self.lineage.lineage_stats[state].full_lin_cells)

        # bringing all the cells after assigning observations to them
        all_cells = [cell for sub_statelist in full_lin_cells_holder for cell in sub_statelist]

        # here we check this for the root parent, since the time has taken
        # so far, equals to the lifetime of the cell
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
        """
        A unittest to check the experiment time is
        reported correctly. Here we use a lineage with 3 cells,
        self.lineage2 built in the setup function.
        """
        full_lin_cells_holder = []
        for state in range(2):
            full_lin_cells_holder.append(
                self.lineage2.lineage_stats[state].full_lin_cells)

        # bringing all the cells after assigning observations to them
        all_cells = [cell for sublist in full_lin_cells_holder for cell in sublist]

        # here we check this for the root parent, since the time has taken
        # so far, equals to the lifetime of the cell
        for cell in all_cells:
            if cell._isRootParent():
                left = cell.obs[1] + cell.left.obs[1]
                right = cell.obs[1] + cell.right.obs[1]
        maximum = max(left, right)
        self.assertTrue(get_experiment_time(self.lineage2) == maximum)

    def test_bernoulli_estimator(self):
        """
        Testing the bernoulli estimator,
        by comparing the result of the estimator
        to the result of scipy random variable generator.
        """
        bern_obs = sp.bernoulli.rvs(
            p=0.90, size=1000)  # bernoulli observations
        self.assertTrue(0.87 <= bernoulli_estimator(bern_obs) <= 0.93)

    def test_exponential_estimator(self):
        """
        Testing the exponential estimator,
        by comparing the result of the estimator
        to the result of scipy random variable generator.
        """
        exp_obs = sp.expon.rvs(
            scale=50, size=1000)  # exponential observations
        self.assertTrue(45 <= exponential_estimator(
            exp_obs) <= 55)  # +/- 5 of beta

    def test_gamma_estimator(self):
        """
        Testing the gamma estimator,
        by comparing the result of the estimator
        to the result of scipy random variable generator.
        """
        gamma_obs = sp.gamma.rvs(
            a=12.5, loc=0.0, scale=3, size=1000)  # gamma observations
        shape, loc, scale = gamma_estimator(gamma_obs)

        self.assertTrue(10 <= shape <= 15)
        self.assertTrue(2 <= scale <= 4)
        self.assertTrue(loc == 0.0)
