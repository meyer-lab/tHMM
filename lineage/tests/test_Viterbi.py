""" Unit test file for Viterbi. """
import unittest
import numpy as np
from sklearn.metrics import rand_score
from ..tHMM import tHMM
from ..LineageTree import LineageTree
from ..figures.figureCommon import pi, T, E


class TestViterbi(unittest.TestCase):
    """ Unit tests for Viterbi. """

    def test_vt(self):
        """ This tests that state assignments by Viterbi are maximum likelihood. """
        X = LineageTree.init_from_parameters(pi, T, E, desired_num_cells=(2 ** 9) - 1)
        tHMMobj = tHMM([X], num_states=2, fpi=pi, fT=T, fE=E)
        true_states = [[cell.state for cell in lineage.output_lineage] for lineage in tHMMobj.X][0]
        pred_states = tHMMobj.predict()[0]

        for _ in range(5):
            # Generate a random sequence
            random_scores = rand_score(np.random.randint(0, 2, (2 ** 9) - 1), true_states)
            pred_scores = rand_score(pred_states, true_states)
            # compare similarity of predicted states and random sequence vs. true states
            self.assertLessEqual(random_scores, pred_scores)
