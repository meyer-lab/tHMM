""" Unit test file. """
import unittest
import numpy as np
from ..CellVar import CellVar as c, get_subtrees, find_two_subtrees
from ..LineageTree import LineageTree, max_gen, get_leaves
from ..states.StateDistributionGamma import StateDistribution
from ..states.stateCommon import get_experiment_time


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
        self.lineage1 = LineageTree.init_from_parameters(self.pi, self.T, self.E, desired_num_cells=(2 ** 11) - 1)
        self.lineage2_fate_censored = LineageTree.init_from_parameters(self.pi, self.T, self.E, desired_num_cells=(2 ** 11) - 1, censor_condition=1)
        self.lineage3_time_censored = LineageTree.init_from_parameters(
            self.pi, self.T, self.E, desired_num_cells=(2 ** 11) - 1, censor_condition=2, desired_experiment_time=500
        )
        self.lineage4_both_censored = LineageTree.init_from_parameters(
            self.pi, self.T, self.E, desired_num_cells=(2 ** 11) - 1, censor_condition=3, desired_experiment_time=500
        )

        # creating 7 cells for 3 generations manually
        cell_1 = c(state=self.state0, parent=None, gen=1)
        cell_2 = c(state=self.state0, parent=cell_1, gen=2)
        cell_3 = c(state=self.state0, parent=cell_1, gen=2)
        cell_4 = c(state=self.state0, parent=cell_2, gen=3)
        cell_5 = c(state=self.state0, parent=cell_2, gen=3)
        cell_6 = c(state=self.state0, parent=cell_3, gen=3)
        cell_7 = c(state=self.state0, parent=cell_3, gen=3)
        cell_1.left = cell_2
        cell_1.right = cell_3
        cell_2.left = cell_4
        cell_2.right = cell_5
        cell_3.left = cell_6
        cell_3.right = cell_7

        self.test_lineage = [cell_1, cell_2, cell_3, cell_4, cell_5, cell_6, cell_7]
        self.level1 = [cell_1]
        self.level2 = [cell_2, cell_3]
        self.level3 = [cell_4, cell_5, cell_6, cell_7]
        # for test_get_subtrees
        self.cell_2 = cell_2
        self.subtree1 = [cell_2, cell_4, cell_5]
        self.cell_3 = cell_3
        self.subtree2 = [cell_3, cell_6, cell_7]
        # for test_find_two_subtrees
        self.cell_1 = cell_1
        # for test_get_mixed_subtrees
        self.mixed = [cell_2, cell_3, cell_4, cell_5, cell_6, cell_7]

    def test_censor_lineage(self):
        """
        A unittest for censor_lineage.
        """

        # checking all the cells in the censord version should have all the
        # bernoulli observations == 1 (dead cells have been removed.)
        self.assertGreater(get_experiment_time(self.lineage1), 500)
        for cell in self.lineage1.output_lineage:
            self.assertTrue(cell.observed)
        for cell in self.lineage2_fate_censored.output_lineage:
            if not cell.observed and not cell.get_sister().observed:
                self.assertTrue(cell.parent.isLeaf)
        for cell in self.lineage3_time_censored.output_lineage:
            if not cell.observed and not cell.get_sister().observed:
                self.assertTrue(cell.parent.isLeaf)
        for cell in self.lineage4_both_censored.output_lineage:
            if not cell.observed and not cell.get_sister().observed:
                self.assertTrue(cell.parent.isLeaf)

    def test_max_gen(self):
        """
        A unittest for testing max_gen function by creating the lineage manually
        for 3 generations ==> total of 7 cells in the setup function.
        """

        max_generation, list_by_gen = max_gen(self.test_lineage)
        self.assertTrue(max_generation == 3)
        self.assertTrue(list_by_gen[1] == self.level1)
        self.assertTrue(list_by_gen[2] == self.level2)
        self.assertTrue(list_by_gen[3] == self.level3)

    def test_get_parent_for_level(self):
        """ A unittest for get_parent_for_level. """
        _, list_by_gen = max_gen(self.lineage1.output_lineage)
        parent_ind_holder = self.lineage1.get_parents_for_level(list_by_gen[3])

        # making a list of parent cells using the indexes that
        # _get_parent_for_level returns
        parent_holder = []
        for ind in parent_ind_holder:
            parent_holder.append(self.lineage1.output_lineage[ind])

        self.assertTrue(parent_holder == list_by_gen[2])

    def test_get_leaves(self):
        """
        A unittest fot get_leaves function.
        """
        # getting the leaves and their indexes for lineage1
        leaf_index, leaf_cells = get_leaves(self.lineage1.output_lineage)

        # to check the leaf cells do not have daughters
        for cells in leaf_cells:
            self.assertTrue(cells.isLeaf())
            self.assertTrue(cells.isLeaf())

        # to check the indexes for leaf cells are true
        for i in leaf_index:
            self.assertTrue(self.lineage1.output_lineage[i].isLeaf())

    def test_get_subtrees(self):
        """
        A unittest to get the subtrees and the remaining lineage except for that subtree.
        Here we use the manually-built-7-cell lineage in the setup function.
        """
        subtree1, _ = get_subtrees(self.cell_2, self.test_lineage)
        self.assertTrue(subtree1 == self.subtree1)

        subtree2, _ = get_subtrees(self.cell_3, self.test_lineage)
        self.assertTrue(subtree2 == self.subtree2)

    def test_find_two_subtrees(self):
        """
        A unittest for find_two_subtrees, using the built-in-7-cell lineage in the setup function.
        """
        left_sub, right_sub, neither_subtree = find_two_subtrees(self.cell_1, self.test_lineage)
        self.assertTrue(left_sub == self.subtree1)
        self.assertTrue(right_sub == self.subtree2)
        self.assertTrue(neither_subtree == [self.cell_1])
