""" Unit test file for Viterbi. """
import unittest
from numpy.random import randint
from ..Analyze import LLFunc
from ..Viterbi import get_leaf_deltas, get_nonleaf_deltas, Viterbi
from ..tHMM import tHMM
from ..LineageTree import LineageTree
from ..figures.figureCommon import pi, T, E


class TestViterbi(unittest.TestCase):
    """ Unit tests for Viterbi. """

    def test_vt(self):
        """ This tests that state assignments by Viterbi are maximum likelihood. """
        X = LineageTree(pi, T, E, desired_num_cells=(2 ** 9) - 1)
        tHMMobj = tHMM([X], num_states=2)

        deltas, state_ptrs = get_leaf_deltas(tHMMobj)
        get_nonleaf_deltas(tHMMobj, deltas, state_ptrs)
        pred_states_by_lineage = Viterbi(tHMMobj, deltas, state_ptrs)

        vitLL = LLFunc(T, pi, tHMMobj, pred_states_by_lineage)

        for _ in range(10):
            rand = randint(0, 2, (2 ** 9) - 1)
            temp = LLFunc(T, pi, tHMMobj, [rand])
            self.assertTrue(temp <= vitLL)
