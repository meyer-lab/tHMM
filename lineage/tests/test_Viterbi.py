""" Unit test file for Viterbi. """
import unittest
from numpy.random import randint
from lineage.Analyze import Analyze, LLFunc
from lineage.LineageTree import LineageTree
from lineage.figures.figureCommon import pi, T, E


class TestViterbi(unittest.TestCase):
    """ Unit tests for Viterbi. """

    def test_vt(self):
        """ This tests that state assignments by Viterbi are maximum likelihood. """
        X = LineageTree(pi, T, E, desired_num_cells=(2 ** 11) - 1)
        tHMMobj, pred_states_by_lineage, _ = Analyze([X], num_states=2)
        vitLL = LLFunc(T, pi, tHMMobj, pred_states_by_lineage)

        for _ in range(10):
            rand = randint(0, 2, (2 ** 11) - 1)
            temp = LLFunc(T, pi, tHMMobj, [rand])
            self.assertTrue(temp <= vitLL)
