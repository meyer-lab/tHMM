""" Unit test file for Viterbi. """
import unittest
import numpy as np
from numpy.random import randint
from lineage.StateDistribution import StateDistribution
from lineage.Analyze import Analyze, LLFunc
from lineage.LineageTree import LineageTree

class TestViterbi(unittest.TestCase):
    """ Unit tests for Viterbi. """

    def test_vt(self):
        """ This tests that state assignments by Viterbi are maximum likelihood. """

        # pi: the initial probability vector
        pi = np.array([0.6, 0.4], dtype="float")

        # T: transition probability matrix
        T = np.array([[0.85, 0.15],
                      [0.15, 0.85]], dtype="float")

        # resistant
        state_obj0 = StateDistribution(0.95, 20, 5)
        # susceptible
        state_obj1 = StateDistribution(0.85, 10, 1)

        E = [state_obj0, state_obj1]

        X = LineageTree(pi, T, E, desired_num_cells=(2**11) - 1, desired_experiment_time=500, prune_condition='fate', prune_boolean=False)
        tHMMobj, pred_states_by_lineage, _ = Analyze([X], num_states=2)
        all_LLs = []
        vitLL = LLFunc(T, pi, tHMMobj, pred_states_by_lineage)
        for i in range(10):
            rand = randint(0, 2, (2**11) - 1)
            temp = LLFunc(T, pi, tHMMobj, [rand])
            all_LLs.append(temp)
        self.assertTrue(all(all_LLs <= vitLL))
            
        