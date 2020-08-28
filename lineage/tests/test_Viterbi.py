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
        X = LineageTree.init_from_parameters(pi, T, E, desired_num_cells=(2 ** 9) - 1)
        tHMMobj = tHMM([X], num_states=2, fpi=pi, fT=T, fE=E)
        model_log_score = tHMMobj.log_score(tHMMobj.predict())[0]

        for _ in range(5):
            # Generate a random sequence
            random_log_scores = tHMMobj.log_score([randint(0, 2, (2 ** 9) - 1)])[0]
            self.assertLessEqual(random_log_scores, model_log_score)
