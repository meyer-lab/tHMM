""" Unit test file. """
import unittest
import numpy as np
from ..CellVar import CellVar as c
from ..LineageTree import LineageTree, max_gen, get_leaves, get_subtrees, tree_recursion, find_two_subtrees, get_mixed_subtrees
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

        # unzipping the tuple of observations
        unzipped_list_obs = list(zip(*list_of_tuples_of_obs))
        bern_obs = list(unzipped_list_obs[0])
        exp_obs = list(unzipped_list_obs[1])
        gamma_obs = list(unzipped_list_obs[2])
        self.assertTrue(len(bern_obs) == len(exp_obs) == len(gamma_obs)), "The number of observations for different emissions don't match!"

        # making sure observations have been assigned properly
        for i, cell in enumerate(cells_in_state):
            self.assertTrue(cell.obs == list_of_tuples_of_obs[i])

        # checking the above tests for a lineage with prune_boolean == True
        num_cells_in_state1, cells_in_state1, list_of_tuples_of_obs1, indices_of_cells_in_state1 = self.lineage1._full_assign_obs(self.state1)
        unzipped_list_obs1 = list(zip(*list_of_tuples_of_obs1))
        bern_obs1 = list(unzipped_list_obs1[0])
        exp_obs1 = list(unzipped_list_obs1[1])
        gamma_obs1 = list(unzipped_list_obs1[2])
        self.assertTrue(len(bern_obs1) == len(exp_obs1) == len(gamma_obs1))

        for j, Cell in enumerate(cells_in_state1):
            self.assertTrue(Cell.obs == list_of_tuples_of_obs1[j])
        
        
    def test_max_gen(self):
        """ A unittest for testing max_gen function by creating the lineage manually for 3 generations ==> total of 7 cells in the setup  function. """

        max_generation, list_by_gen = max_gen(self.test_lineage)
        self.assertTrue(max_generation == 3)
        self.assertTrue(list_by_gen[1] == self.level1)
        self.assertTrue(list_by_gen[2] == self.level2)
        self.assertTrue(list_by_gen[3] == self.level3)

    def test_get_parent_for_level(self):
        """ A unittest for get_parent_for_level. """
        _, list_by_gen = max_gen(self.lineage1.output_lineage)
        parent_ind_holder = self.lineage1._get_parents_for_level(list_by_gen[3])

        # making a list of parent cells using the indexes that _get_parent_for_level returns
        parent_holder = []
        for ind in parent_ind_holder:
            parent_holder.append(self.lineage1.output_lineage[ind])

        self.assertTrue(parent_holder == list_by_gen[2])

    def test_get_leaves(self):
        """ A unittest fot get_leaves function. """
        leaf_index, leaf_cells = get_leaves(self.lineage1.output_lineage) # getting the leaves and their indexes for lineage1

        # to check the leaf cells do not have daughters
        for cells in leaf_cells:
            self.assertTrue(cells.left == None), " The leaf cell doesn't seems to be a leaf, it has a left daughter."
            self.assertTrue(cells.right == None), " The leaf cell doesn't seems to be a leaf, it has a right daughter."

        # to check the indexes for leaf cells are true
        for i in leaf_index:
            self.assertTrue(self.lineage1.output_lineage[i]._isLeaf() == True)

    def test_get_subtrees(self):
        """ A unittest to get the subtrees and the remaining lineage except for that subtree. Here we use the manually-built-7-cell lineage in the setup function. """
        subtree1, not_subtree1 = get_subtrees(self.cell_2, self.test_lineage)
        self.assertTrue(subtree1 == self.subtree1), " The subtree is not being specified correctly."

        subtree2, not_subtree2 = get_subtrees(self.cell_3, self.test_lineage)
        self.assertTrue(subtree2 == self.subtree2), " The subtree is not being specified correctly."

    def test_find_two_subtrees(self):
        """ A unittest for find_two_subtrees, using the built-in-7-cell lineage in the setup function.  """
        left_sub, right_sub, neither_subtree = find_two_subtrees(self.cell_1, self.test_lineage)
        self.assertTrue(left_sub == self.subtree1), "left subtree wrong"
        self.assertTrue(right_sub == self.subtree2), "right subtree wrong"
        self.assertTrue(neither_subtree == [self.cell_1]), "neither subtree is wrong"

    def test_get_mixed_subtrees(self):
        """ A unittest for get_mixed_subtrees, using the built-in-7-cell lineage in the setup function. """
        mixed_sub, not_mixed = get_mixed_subtrees(self.cell_2, self.cell_3, self.test_lineage)
        mixed = self.subtree2 + self.subtree1
        self.assertTrue(mixed_sub == mixed)
        self.assertTrue(not_mixed == [self.cell_1])
            
            
        
        
        