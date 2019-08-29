""" Unit test file. """
import unittest
import numpy as np
from ..StateDistribution import StateDistribution
from ..UpwardRecursion import get_leaf_Normalizing_Factors, get_leaf_betas, get_nonleaf_NF_and_betas, calculate_log_likelihood
from ..BaumWelch import fit
from ..Viterbi import get_leaf_deltas, get_nonleaf_deltas, get_delta_parent_child_prod, delta_parent_child_func, Viterbi
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
        gamma_scale0 = 5

        # State 1 parameters "Susciptible"
        state1 = 1
        bern_p1 = 0.85
        gamma_a1 = 10
        gamma_scale1 = 1

        state_obj0 = StateDistribution(state0, bern_p0, gamma_a0, gamma_scale0)
        state_obj1 = StateDistribution(state1, bern_p1, gamma_a1, gamma_scale1)
        self.E = [state_obj0, state_obj1]
        num = 2**7 - 1
        # Using an unpruned lineage to avoid unforseen issues
        self.X = [LineageTree(pi, T, self.E, num, prune_boolean=False)]
        tHMMobj = tHMM(self.X, numStates=2)  # build the tHMM class with X

    ##########################
    # Viterbi.py tests below #
    ##########################

    def test_viterbi(self):
        '''
        Builds the tHMM class and calls
        the Viterbi function to find
        the optimal hidden states.
        '''
        t = tHMM(self.X, numStates=2)  # build the tHMM class with X
        deltas, state_ptrs = get_leaf_deltas(t)  # gets the deltas matrix
        self.assertLessEqual(len(deltas), 50)  # there are <=50 lineages in X
        self.assertLessEqual(len(state_ptrs), 50)  # there are <=50 lineages in X
        get_nonleaf_deltas(t, deltas, state_ptrs)
        self.assertLessEqual(len(deltas), 50)  # there are <=50 lineages in X
        self.assertLessEqual(len(state_ptrs), 50)  # there are <=50 lineages in X
        all_states = Viterbi(t, deltas, state_ptrs)
        self.assertLessEqual(len(all_states), 50)  # there are <=50 lineages in X

    def test_viterbi2(self):
        '''
        Builds the tHMM class and calls
        the Viterbi function to find
        the optimal hidden states.
        Now trying to see if altering the parameters
        gives one different optimal state
        trees.
        '''
        numStates = 2
        t = tHMM(self.X, numStates=numStates)  # build the tHMM class with X
        fake_param_list = []
        numLineages = t.numLineages
        temp_params = {"pi": np.ones((numStates), dtype=int),  # inital state distributions [K] initialized to 1/K
                       "T": np.ones((numStates, numStates), dtype=int) / (numStates),  # state transition matrix [KxK] initialized to 1/K
                       "E": np.ones((numStates, 2))}  # sequence of emission likelihood distribution parameters [Kx2]
        temp_params["pi"][1] = 0  # the hidden state for the second node should always be 1
        to_state_one = np.zeros((numStates, numStates), dtype=int)
        to_state_one[:, 1] = np.ones((numStates), dtype=int)
        temp_params["T"] = to_state_one  # should always end up in state 1 regardless of previous state
        # since transition matrix is a dependent matrix (0 is now a trivial state)
        temp_params["E"][:, 0] *= 0.5  # initializing all Bernoulli p parameters to 0.5
        temp_params["E"][:, 1] *= 50  # initializing all Gompoertz s(cale) parameters to 50

        for lineage_num in range(numLineages):  # for each lineage in our population
            fake_param_list.append(temp_params.copy())  # create a new dictionary holding the parameters and append it
            assert len(fake_param_list) == lineage_num + 1
        t.paramlist = fake_param_list
        t.MSD = t.get_Marginal_State_Distributions()  # rerun these with new parameters
        t.EL = t.get_Emission_Likelihoods()  # rerun these with new parameters
        # run Viterbi with new parameter list
        deltas, state_ptrs = get_leaf_deltas(t)  # gets the deltas matrix
        self.assertLessEqual(len(deltas), 50)  # there are <=50 lineages in X
        self.assertLessEqual(len(state_ptrs), 50)  # there are <=50 lineages in X
        get_nonleaf_deltas(t, deltas, state_ptrs)
        self.assertLessEqual(len(deltas), 50)  # there are <=50 lineages in X
        self.assertLessEqual(len(state_ptrs), 50)  # there are <=50 lineages in X
        all_states = Viterbi(t, deltas, state_ptrs)
        self.assertLessEqual(len(all_states), 50)  # there are <=50 lineages in X
        for num in range(numLineages):
            curr_all_states = all_states[num]
            self.assertEqual(curr_all_states[0], 0)  # the first state should always be 0
            all_ones = curr_all_states[1:]  # get all the items in the list except the first item
            # this list should now be all ones since everything will always transition to 1
            self.assertTrue(all(all_ones))

    def test_viterbi3(self):
        '''
        Builds the tHMM class and calls
        the Viterbi function to find
        the optimal hidden states.
        Now trying to see if altering the parameters
        gives one different optimal state
        trees. For this example, we have two
        homogeneous populations. Using parameter sets that
        describe those homogenous populations.
        '''
        numStates = 2
        t = tHMM(self.X, numStates=numStates)  # build the tHMM class with X

        fake_param_list = []
        numLineages = t.numLineages

        temp_params = {"pi": np.ones((numStates), dtype=float) / (numStates),  # inital state distributions [K] initialized to 1/K
                       "T": np.eye(2, dtype=int),  # state transition matrix [KxK] initialized to identity (no transitions)
                       # should always end up in state 1 regardless of previous state
                       "E": np.ones((numStates, 3))}  # sequence of emission likelihood distribution parameters [Kx2]

        temp_params["pi"][0] = 4 / 5  # the population is distributed as such 2/5 is of state 0
        temp_params["pi"][1] = 1 / 5  # state 1 occurs 3/5 of the time

        temp_params["E"][0, 0] *= 0.999  # initializing all Bernoulli p parameters to 0.5
        temp_params["E"][0, 1] *= 40  # initializing all Exponential parameters to 50

        temp_params["E"][1, 0] *= 0.6  # initializing all Bernoulli p parameters to 0.5
        temp_params["E"][1, 1] *= 50  # initializing all exponential parameters to 50

        for lineage_num in range(numLineages):  # for each lineage in our population
            fake_param_list.append(temp_params.copy())  # create a new dictionary holding the parameters and append it
            assert len(fake_param_list) == lineage_num + 1
        t.paramlist = fake_param_list

        t.MSD = t.get_Marginal_State_Distributions()  # rerun these with new parameters
        t.EL = t.get_Emission_Likelihoods()  # rerun these with new parameters
        # run Viterbi with new parameter list
        deltas, state_ptrs = get_leaf_deltas(t)  # gets the deltas matrix
        self.assertLessEqual(len(deltas), 50)  # there are <=50 lineages in X
        self.assertLessEqual(len(state_ptrs), 50)  # there are <=50 lineages in X
        get_nonleaf_deltas(t, deltas, state_ptrs)
        self.assertLessEqual(len(deltas), 50)  # there are <=50 lineages in X
        self.assertLessEqual(len(state_ptrs), 50)  # there are <=50 lineages in X
        all_states = Viterbi(t, deltas, state_ptrs)
        self.assertLessEqual(len(all_states), 50)  # there are <=50 lineages in X
        num_of_zeros = 0  # counts how many lineages were all of the 0 states
        num_of_ones = 0  # counts how many lineages were all of the 1 states
        for num in range(numLineages):
            curr_all_states = all_states[num]
            if curr_all_states[0] == 0:
                all_zeros = curr_all_states
                self.assertFalse(all(all_zeros))
                # this should be true since the homogenous lineage is all of state 0
                num_of_zeros += 1
            else:
                all_ones = curr_all_states
                self.assertTrue(all(all_ones))
                # this should be true since the homogenous lineage is all of state 1
                num_of_ones += 1
        self.assertGreater(num_of_zeros, num_of_ones)
        # there should be a greater number of lineages with all zeros than all ones as hidden states
