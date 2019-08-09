""" Unit test file. """
import unittest
import numpy as np
from ..CellVar import CellVar as c
from ..LineageTree import LineageTree, max_gen
from ..StateDistribution import StateDistribution


class TestModel(unittest.TestCase):

    def setUp(self):
        # pi: the initial probability vector
        self.pi = np.array([0.75, 0.25])

        # T: transition probability matrix
        self.T = np.array([[0.85, 0.15],
                      [0.2, 0.8]])

        # E: states are defined as StateDistribution objects
        # State 0 parameters "Resistant"
        self.state0 = 0
        self.bern_p0 = 0.99
        self.expon_scale_beta0 = 20
        self.gamma_a0 = 5.0
        self.gamma_scale0 = 1.0

        # State 1 parameters "Susciptible"
        self.state1 = 1
        self.bern_p1 = 0.8
        self.expon_scale_beta1 = 80
        self.gamma_a1 = 10.0
        self.gamma_scale1 = 2.0

        # creating the state object
        state_obj0 = StateDistribution(self.state0, self.bern_p0, self.expon_scale_beta0, self.gamma_a0, self.gamma_scale0)
        state_obj1 = StateDistribution(self.state1, self.bern_p1, self.expon_scale_beta1, self.gamma_a1, self.gamma_scale1)

        # observations object
        self.E = [state_obj0, state_obj1]

        # creating two lineages, one with False for pruning, one with True.
        self.lineage1 = LineageTree(self.pi, self.T, self.E, desired_num_cells=2**9 - 1, prune_boolean=False)
        self.lineage2_pruned = LineageTree(self.pi, self.T, self.E, desired_num_cells=2**9 - 1, prune_boolean=True)

        # creating 7 cells for 3 generations manually
        cell_1 = c(state=self.state0, left=None, right=None, parent=None, gen=1)
        cell_2 = c(state=self.state0, left=None, right=None, parent=cell_1, gen=2)
        cell_3 = c(state=self.state0, left=None, right=None, parent=cell_1, gen=2)
        cell_4 = c(state=self.state0, left=None, right=None, parent=cell_2, gen=3)
        cell_5 = c(state=self.state0, left=None, right=None, parent=cell_2, gen=3)
        cell_6 = c(state=self.state0, left=None, right=None, parent=cell_3, gen=3)
        cell_7 = c(state=self.state0, left=None, right=None, parent=cell_3, gen=3)
        self.test_lineage = [cell_1, cell_2, cell_3, cell_4, cell_5, cell_6, cell_7]
        self.level1 = [cell_1]
        self.level2 = [cell_2, cell_3]
        self.level3 = [cell_4, cell_5, cell_6, cell_7]
        

    def test_generate_lineage_list(self):
        """ A unittest for generate_lineage_list. """
        # checking the number of cells generated is equal to the desired number of cells given by the user.
        self.assertTrue(len(self.lineage1.full_lin_list) == 2**9 -1)
        self.assertTrue(self.lineage1.full_lin_list == self.lineage1.output_lineage), "The output lineage is wrong according to prune boolean"
        self.assertTrue(self.lineage2_pruned.pruned_lin_list == self.lineage2_pruned.output_lineage), "The output lineage is wrong according to prune boolean"

    def test_prune_lineage(self):
        """ A unittest for prune_lineage. """
        # checking the number of cells in the pruned version is smaller than the full version.
        self.assertTrue(len(self.lineage1.full_lin_list) >= len(self.lineage1.pruned_lin_list))

        # checking all the cells in the pruned version should have all the bernoulli observations == 1 (dead cells have been removed.)
        for cell in self.lineage1.pruned_lin_list:
            if cell._isLeaf():
                self.assertTrue(cell.left == None)
                self.assertTrue(cell.right == None)

        for cell in self.lineage2_pruned.pruned_lin_list:
            if cell._isLeaf():
                self.assertTrue(cell.left == None)
                self.assertTrue(cell.right == None)

    def test_get_full_state_count(self):
        """ A unittest for _get_full_state_count. """
        num_cells_in_state, cells_in_state, indices_of_cells_in_state = self.lineage1._get_full_state_count(self.state0)
        self.assertTrue(len(cells_in_state) == num_cells_in_state)
        self.assertTrue(num_cells_in_state <= 2**9 - 1), "The number of cells in one state is greater than the total number of cells!"
        self.assertTrue(max(indices_of_cells_in_state) <= 2**9 - 1), "something is wrong with the indices of the cells returned by the function"

        num_cells_in_state1, cells_in_state1, indices_of_cells_in_state1 = self.lineage1._get_full_state_count(self.state1)
        self.assertTrue(len(cells_in_state1) == num_cells_in_state1)
        self.assertTrue(num_cells_in_state1 <= 2**9 - 1), "The number of cells in one state is greater than the total number of cells!"
        self.assertTrue(max(indices_of_cells_in_state1) <= 2**9 - 1), "something is wrong with the indices of the cells returned by the function"


    def test_get_pruned_state_count(self):
        """ A unittest for _get_pruned_state_count. """
        num_cells_in_state, cells_in_state, list_of_tuples_of_obs, indices_of_cells_in_state = self.lineage1._get_pruned_state_count(self.state0)
        self.assertTrue(len(cells_in_state) == num_cells_in_state == len(list_of_tuples_of_obs))
        self.assertTrue(num_cells_in_state <= len(self.lineage1.pruned_lin_list)), "The number of cells in one state is greater than the total number of cells!"
        if len(indices_of_cells_in_state) > 0:
            self.assertTrue(max(indices_of_cells_in_state) <= (2**9 - 1)), "something is wrong with the indices of the cells returned by the function"

        num_cells_in_state1, cells_in_state1, list_of_tuples_of_obs1, indices_of_cells_in_state1 = self.lineage1._get_pruned_state_count(self.state1)
        self.assertTrue(len(cells_in_state1) == num_cells_in_state1 == len(list_of_tuples_of_obs1))
        self.assertTrue(num_cells_in_state1 <= len(self.lineage1.pruned_lin_list)), "The number of cells in one state is greater than the total number of cells!"
        if len(indices_of_cells_in_state) > 0:
            self.assertTrue(max(indices_of_cells_in_state1) <= (2**9 - 1)), "something is wrong with the indices of the cells returned by the function"


    def test_full_assign_obs(self):
        """ A unittest for checking the full_assign_obs function. """
        num_cells_in_state, cells_in_state, list_of_tuples_of_obs, indices_of_cells_in_state = self.lineage1._full_assign_obs(self.state0)
        unzipped_list_obs = list(zip(*list_of_tuples_of_obs))
        bern_obs = list(unzipped_list_obs[0])
        exp_obs = list(unzipped_list_obs[1])
        gamma_obs = list(unzipped_list_obs[2])
        self.assertTrue(len(bern_obs) == len(exp_obs) == len(gamma_obs))


        for i, cell in enumerate(cells_in_state):
            self.assertTrue(cell.obs == list_of_tuples_of_obs[i])

        num_cells_in_state1, cells_in_state1, list_of_tuples_of_obs1, indices_of_cells_in_state1 = self.lineage1._full_assign_obs(self.state1)
        unzipped_list_obs1 = list(zip(*list_of_tuples_of_obs1))
        bern_obs1 = list(unzipped_list_obs1[0])
        exp_obs1 = list(unzipped_list_obs1[1])
        gamma_obs1 = list(unzipped_list_obs1[2])
        self.assertTrue(len(bern_obs1) == len(exp_obs1) == len(gamma_obs1))

        for j, Cell in enumerate(cells_in_state1):
            self.assertTrue(Cell.obs == list_of_tuples_of_obs1[j])

        
    def test_max_gen(self):
        """ A unittest for testing max_gen function by creating the lineage manually for 3 generations ==> total of 7 cells. """

        max_generation, list_by_gen = max_gen(self.test_lineage)
        self.assertTrue(max_generation == 3)
        self.assertTrue(list_by_gen[1] == self.level1)
        self.assertTrue(list_by_gen[2] == self.level2)
        self.assertTrue(list_by_gen[3] == self.level3)

#     def test_get_parent_for_level(self):
#         """ A unittest for get_parent_for_level. """
#         self.output_lineage = self.test_lineage
#         parent_holder2 = self.test_lineage._get_parents_for_level(self.level2)
#         parent_holder3 = self.test_lineage._get_parents_for_level(self.level3)
#         self.assertTrue(parent_holder2 == self.level2)
#         self.assertTrue(parent_holder3 == self.level3)
        
        
        