""" Unit test file for Viterbi. """
import unittest
import numpy as np
from numpy.random import randint
from lineage.StateDistribution import StateDistribution
from lineage.Analyze import Analyze
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
        vitLL = stateLikelihood(tHMMobj)
        length = pred_states_by_lineage[0].shape[0]
        LLTot = []
        for i in range(2):
            random.seed(123)
            temp = randint(0, 2, length)
            for ind, cell in enumerate(tHMMobj.X[0].full_lin_list):
                cell.state = temp[ind]
            ll = stateLikelihood(tHMMobj)
            LLTot.append(ll.sum())