""" Unit test file. """
import unittest
import numpy as np
from ..StateDistribution import StateDistribution
from ..UpwardRecursion import get_leaf_Normalizing_Factors, get_leaf_betas, get_nonleaf_NF_and_betas, calculate_log_likelihood
from ..BaumWelch import fit
from ..LineageTree import LineageTree
from ..tHMM import tHMM


class TestModel(unittest.TestCase):

    def setUp(self):
        """ This tests that one step of Baum-Welch increases the likelihood of the fit. """

        # pi: the initial probability vector
        pi = np.array([0.6, 0.4], dtype="float")

        # T: transition probability matrix
        T = np.array([[0.85, 0.15],
                      [0.15, 0.85]], dtype="float")
        # State 0 parameters "Resistant"
        state0 = 0
        bern_p0 = 0.95
        gamma_a0 = 20
        gamma_loc = 0
        gamma_scale0 = 5

        # State 1 parameters "Susciptible"
        state1 = 1
        bern_p1 = 0.85
        gamma_a1 = 10
        gamma_scale1 = 1

        state_obj0 = StateDistribution(state0, bern_p0, gamma_a0, gamma_loc, gamma_scale0)
        state_obj1 = StateDistribution(state1, bern_p1, gamma_a1, gamma_loc, gamma_scale1)
        self.E = [state_obj0, state_obj1]
        # Using an unpruned lineage to avoid unforseen issues
        self.X = [LineageTree(pi, T, self.E, desired_num_cells=(2**11) - 1, desired_experiment_time=500, prune_condition='die',
                              prune_boolean=False)]
        tHMMobj = tHMM(self.X, numStates=2)  # build the tHMM class with X

        # Test cases below
        # Get the likelihoods before fitting
        NF_before = get_leaf_Normalizing_Factors(tHMMobj)
        betas_before = get_leaf_betas(tHMMobj, NF_before)
        get_nonleaf_NF_and_betas(tHMMobj, NF_before, betas_before)
        LL_before = calculate_log_likelihood(tHMMobj, NF_before)
        self.assertTrue(np.isfinite(LL_before))

        # Get the likelihoods after fitting
        tHMMobj_after, NF_after, _, _, new_LL_list_after = fit(tHMMobj, max_iter=4)
        LL_after = calculate_log_likelihood(tHMMobj_after, NF_after)
        self.assertTrue(np.isfinite(LL_after))
        self.assertTrue(np.isfinite(new_LL_list_after))

        self.assertGreater(LL_after, LL_before)

    def test_init_paramlist(self):
        '''
        Make sure paramlist has proper
        labels and sizes.
        '''
        t = tHMM(self.X, numStates=2)  # build the tHMM class with X
        print(t.estimate.pi.shape)
        self.assertEqual(t.estimate.pi.shape[0], 2)  # make sure shape is numStates
        self.assertEqual(t.estimate.T.shape[0], 2)  # make sure shape is numStates
        self.assertEqual(t.estimate.T.shape[1], 2)  # make sure shape is numStates
        self.assertEqual(len(t.estimate.E), 2)  # make sure shape is numStates

    def test_get_MSD(self):
        '''
        Calls get_Marginal_State_Distributions and
        ensures the output is of correct data type and
        structure.
        '''
        t = tHMM(self.X, numStates=2)  # build the tHMM class with X
        MSD = t.get_Marginal_State_Distributions()
        self.assertLessEqual(len(MSD), 50)  # there are <=50 lineages in the population
        for _, MSDlin in enumerate(MSD):
            self.assertGreaterEqual(MSDlin.shape[0], 0)  # at least zero cells in each lineage
            self.assertEqual(MSDlin.shape[1], 2)  # there are 2 states for each cell
            for node_n in range(MSDlin.shape[0]):
                self.assertTrue(np.isclose(sum(MSDlin[node_n, :]), 1))  # the rows should sum to 1

    def test_get_EL(self):
        '''
        Calls get_Emission_Likelihoods and ensures
        the output is of correct data type and structure.
        '''
        t = tHMM(self.X, numStates=2)  # build the tHMM class with X
        EL = t.get_Emission_Likelihoods()
        self.assertLessEqual(len(EL), 50)  # there are <=50 lineages in the population
        for _, ELlin in enumerate(EL):
            self.assertGreaterEqual(ELlin.shape[0], 0)  # at least zero cells in each lineage
            self.assertEqual(ELlin.shape[1], 2)  # there are 2 states for each cell

    ##################################
    # UpwardRecursion.py tests below #
    ##################################

    def test_get_leaf_NF(self):
        '''
        Calls get_leaf_Normalizing_Factors and
        ensures the output is of correct data type and
        structure.
        '''
        t = tHMM(self.X, numStates=2)  # build the tHMM class with X
        NF = get_leaf_Normalizing_Factors(t)
        self.assertLessEqual(len(NF), 50)  # there are <=50 lineages in the population
        for _, NFlin in enumerate(NF):
            self.assertGreaterEqual(NFlin.shape[0], 0)  # at least zero cells in each lineage
