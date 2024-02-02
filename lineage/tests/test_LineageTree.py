""" Unit test file. """

import unittest
import numpy as np
from ..CellVar import CellVar as c
from ..LineageTree import LineageTree, max_gen
from ..states.StateDistributionGamma import StateDistribution


class TestModel(unittest.TestCase):
    """
    Unit test class for lineages.
    """

    def setUp(self):
        """
        Setting up lineages for testing.
        """
        # pi: the initial probability vector
        self.pi = np.array([0.75, 0.25])

        # T: transition probability matrix
        self.T = np.array([[0.85, 0.15], [0.20, 0.80]])

        # State 0 parameters "Resistant"
        self.state0 = 0
        bern_p0 = 0.99
        gamma_a0 = 20
        gamma_scale0 = 5

        # State 1 parameters "Susceptible"
        self.state1 = 1
        bern_p1 = 0.8
        gamma_a1 = 10
        gamma_scale1 = 1

        state_obj0 = StateDistribution(bern_p0, gamma_a0, gamma_scale0)
        state_obj1 = StateDistribution(bern_p1, gamma_a1, gamma_scale1)

        self.E = [state_obj0, state_obj1]

        # creating lineages with the various censor conditions
        self.lineage1 = LineageTree.rand_init(
            self.pi, self.T, self.E, desired_num_cells=(2**11) - 1
        )
        self.lineage2_fate_censored = LineageTree.rand_init(
            self.pi, self.T, self.E, desired_num_cells=(2**11) - 1, censor_condition=1
        )
        self.lineage3_time_censored = LineageTree.rand_init(
            self.pi,
            self.T,
            self.E,
            desired_num_cells=(2**11) - 1,
            censor_condition=2,
            desired_experiment_time=500,
        )
        self.lineage4_both_censored = LineageTree.rand_init(
            self.pi,
            self.T,
            self.E,
            desired_num_cells=(2**11) - 1,
            censor_condition=3,
            desired_experiment_time=500,
        )

        # creating 7 cells for 3 generations manually
        cell_1 = c(state=self.state0, parent=None)
        cell_2 = c(state=self.state0, parent=cell_1)
        cell_3 = c(state=self.state0, parent=cell_1)
        cell_4 = c(state=self.state0, parent=cell_2)
        cell_5 = c(state=self.state0, parent=cell_2)
        cell_6 = c(state=self.state0, parent=cell_3)
        cell_7 = c(state=self.state0, parent=cell_3)
        cell_1.left = cell_2
        cell_1.right = cell_3
        cell_2.left = cell_4
        cell_2.right = cell_5
        cell_3.left = cell_6
        cell_3.right = cell_7

        self.test_lineage = [cell_1, cell_2, cell_3, cell_4, cell_5, cell_6, cell_7]
        self.level1 = [0]
        self.level2 = [1, 2]
        self.level3 = [3, 4, 5, 6]
        # for test_get_subtrees
        self.cell_2 = cell_2
        self.subtree1 = [cell_2, cell_4, cell_5]
        self.cell_3 = cell_3
        self.subtree2 = [cell_3, cell_6, cell_7]
        # for test_find_two_subtrees
        self.cell_1 = cell_1
        # for test_get_mixed_subtrees
        self.mixed = [cell_2, cell_3, cell_4, cell_5, cell_6, cell_7]

    def test_max_gen(self):
        """
        A unittest for testing max_gen function by creating the lineage manually
        for 3 generations ==> total of 7 cells in the setup function.
        """
        list_by_gen = max_gen(self.test_lineage)
        np.testing.assert_array_equal(list_by_gen[0], self.level1)
        np.testing.assert_array_equal(list_by_gen[1], self.level2)
        np.testing.assert_array_equal(list_by_gen[2], self.level3)

    def test_get_parent_for_level(self):
        """A unittest for get_parent_for_level."""
        list_by_gen = max_gen(self.lineage1.output_lineage)
        parent_ind_holder = np.unique(self.lineage1.cell_to_parent[list_by_gen[3]])
        np.testing.assert_array_equal(parent_ind_holder, list_by_gen[2])
