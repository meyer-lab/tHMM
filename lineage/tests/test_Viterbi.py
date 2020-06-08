""" Unit test file for Viterbi. """
import unittest
from numpy.random import randint
from ..tHMM import tHMM
from ..LineageTree import LineageTree
from ..figures.figureCommon import pi, T, E


class TestViterbi(unittest.TestCase):
    """ Unit tests for Viterbi. """

    def test_vt(self):
        """ This tests that state assignments by Viterbi are maximum likelihood. """
        X = LineageTree(pi, T, E, desired_num_cells=(2 ** 9) - 1)
        tHMMobj = tHMM([X], num_states=2)

        _, _, _, _, _, _ = tHMMobj.fit()
        pred_states_by_lineage = tHMMobj.predict()

        true_log_scores = tHMMobj.log_score(pred_states_by_lineage)

        for _ in range(10):
            rand = randint(0, 2, (2 ** 9) - 1)
            random_log_scores = tHMMobj.log_score([rand])
            self.assertTrue(random_log_scores[0] <= true_log_scores[0])
