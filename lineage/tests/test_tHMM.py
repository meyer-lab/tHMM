""" Unit test file. """
import unittest
import numpy as np
from ..UpwardRecursion import get_leaf_Normalizing_Factors
from ..LineageTree import LineageTree
from ..tHMM import tHMM
from lineage.figures.figureCommon import pi, T, E


class TestModel(unittest.TestCase):
    """
    Unit test class for the tHMM model.
    """

    def setUp(self):
        """ This tests that one step of Baum-Welch increases the likelihood of the fit. """
        # Using an unpruned lineage to avoid unforseen issues
        self.X = [LineageTree(pi, T, E, desired_num_cells=(2 ** 11) - 1)]

    def test_init_paramlist(self):
        """
        Make sure paramlist has proper
        labels and sizes.
        """
        t = tHMM(self.X, num_states=2)  # build the tHMM class with X
        self.assertEqual(t.estimate.pi.shape[0], 2)  # make sure shape is num_states
        self.assertEqual(t.estimate.T.shape[0], 2)  # make sure shape is num_states
        self.assertEqual(t.estimate.T.shape[1], 2)  # make sure shape is num_states
        self.assertEqual(len(t.estimate.E), 2)  # make sure shape is num_states

    def test_get_MSD(self):
        """
        Calls get_Marginal_State_Distributions and
        ensures the output is of correct data type and
        structure.
        """
        t = tHMM(self.X, num_states=2)  # build the tHMM class with X
        MSD = t.get_Marginal_State_Distributions()
        self.assertLessEqual(len(MSD), 50)  # there are <=50 lineages in the population
        for _, MSDlin in enumerate(MSD):
            self.assertGreaterEqual(MSDlin.shape[0], 0)  # at least zero cells in each lineage
            self.assertEqual(MSDlin.shape[1], 2)  # there are 2 states for each cell
            for node_n in range(MSDlin.shape[0]):
                self.assertTrue(np.isclose(sum(MSDlin[node_n, :]), 1))  # the rows should sum to 1

    def test_get_EL(self):
        """
        Calls get_Emission_Likelihoods and ensures
        the output is of correct data type and structure.
        """
        t = tHMM(self.X, num_states=2)  # build the tHMM class with X
        EL = t.get_Emission_Likelihoods()
        self.assertLessEqual(len(EL), 50)  # there are <=50 lineages in the population
        for _, ELlin in enumerate(EL):
            self.assertGreaterEqual(ELlin.shape[0], 0)  # at least zero cells in each lineage
            self.assertEqual(ELlin.shape[1], 2)  # there are 2 states for each cell

    def test_get_leaf_NF(self):
        """
        Calls get_leaf_Normalizing_Factors and
        ensures the output is of correct data type and
        structure.
        """
        t = tHMM(self.X, num_states=2)  # build the tHMM class with X
        NF = get_leaf_Normalizing_Factors(t)
        self.assertLessEqual(len(NF), 50)  # there are <=50 lineages in the population
        for _, NFlin in enumerate(NF):
            self.assertGreaterEqual(NFlin.shape[0], 0)  # at least zero cells in each lineage
