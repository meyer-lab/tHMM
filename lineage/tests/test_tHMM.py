""" Unit test file. """
import unittest
import pytest
import numpy as np
from ..UpwardRecursion import (
    get_Marginal_State_Distributions,
    get_Emission_Likelihoods,
)
from ..LineageTree import LineageTree
from ..tHMM import tHMM
from ..states.StateDistributionGaPhs import StateDistribution as StateDistPhase
from ..figures.common import pi, T, E
from ..Analyze import Analyze, Results, run_Analyze_over


class TestModel(unittest.TestCase):
    """
    Unit test class for the tHMM model.
    """

    def setUp(self):
        """ This tests that one step of Baum-Welch increases the likelihood of the fit. """
        # Using an unpruned lineage to avoid unforseen issues
        self.X = [LineageTree.init_from_parameters(pi, T, E, desired_num_cells=(2 ** 11) - 1)]
        self.pi = np.array([0.55, 0.35, 0.10])
        self.T = np.array([[0.75, 0.20, 0.05], [0.1, 0.85, 0.05], [0.1, 0.1, 0.8]])

        # Emissions
        self.E = [StateDistPhase(0.99, 0.9, 20, 5, 10, 3), StateDistPhase(0.88, 0.75, 10, 2, 15, 4), StateDistPhase(0.77, 0.85, 15, 7, 20, 5)]
        self.X3 = [LineageTree.init_from_parameters(self.pi, self.T, self.E, desired_num_cells=(2 ** 11) - 1)]

        self.t = tHMM(self.X, num_states=2)  # build the tHMM class with X
        self.t3 = tHMM(self.X3, num_states=3)  # build the tHMM class for 3 states

        self.MSD = get_Marginal_State_Distributions(self.t)
        self.MSD3 = get_Marginal_State_Distributions(self.t3)

    def test_init_paramlist(self):
        """
        Make sure paramlist has proper
        labels and sizes.
        """
        t = self.t
        self.assertEqual(t.estimate.pi.shape[0], 2)  # make sure shape is num_states
        self.assertEqual(t.estimate.T.shape[0], 2)  # make sure shape is num_states
        self.assertEqual(t.estimate.T.shape[1], 2)  # make sure shape is num_states
        self.assertEqual(len(t.estimate.E), 2)  # make sure shape is num_states

        t3 = self.t3
        self.assertEqual(t3.estimate.pi.shape[0], 3)  # make sure shape is num_states
        self.assertEqual(t3.estimate.T.shape[0], 3)  # make sure shape is num_states
        self.assertEqual(t3.estimate.T.shape[1], 3)  # make sure shape is num_states
        self.assertEqual(len(t3.estimate.E), 3)  # make sure shape is num_states

    def test_get_MSD(self):
        """
        Calls get_Marginal_State_Distributions and
        ensures the output is of correct data type and
        structure.
        """
        MSD = self.MSD
        MSD3 = self.MSD3
        for ind, MSDlin in enumerate(MSD):
            self.assertGreaterEqual(MSDlin.shape[0], 0)  # at least zero cells in each lineage
            self.assertGreaterEqual(MSD3[ind].shape[0], 0)  # at least zero cells in each lineage
            self.assertEqual(MSDlin.shape[1], 2)  # there are 2 states for each cell
            self.assertEqual(MSD3[ind].shape[1], 3)  # there are 3 states for each cell
            for node_n in range(MSDlin.shape[0]):
                self.assertTrue(np.isclose(sum(MSDlin[node_n, :]), 1))  # the rows should sum to 1

    def test_get_EL(self):
        """
        Calls get_Emission_Likelihoods and ensures
        the output is of correct data type and structure.
        """
        EL = get_Emission_Likelihoods(self.t)
        EL3 = get_Emission_Likelihoods(self.t3)

        for ind, ELlin in enumerate(EL):
            self.assertGreaterEqual(ELlin.shape[0], 0)  # at least zero cells in each lineage
            self.assertGreaterEqual(EL3[ind].shape[0], 0)  # at least zero cells in each lineage
            self.assertEqual(ELlin.shape[1], 2)  # there are 2 states for each cell
            self.assertEqual(EL3[ind].shape[1], 3)  # there are 3 states for each cell


def test_fit_performance():
    """ Really defined states should get an accuracy >95%.
    Lineages used should be large and distinct. """
    X = [LineageTree.init_from_parameters(pi, T, E, desired_num_cells=(2 ** 9) - 1)]
    first = Results(*Analyze(X, 2, fpi=pi))["state_similarity"]
    second = Results(*Analyze(X, 2, fpi=pi))["state_similarity"]
    assert max(first, second) > 95.0


@pytest.mark.parametrize("sizze", [1, 3])
@pytest.mark.parametrize("stateNum", [1, 2, 3])
def test_small_lineages(sizze, stateNum):
    """ To test lineages with 3 cells in them for simple gamma. """
    # test with 2 state model
    lin = [LineageTree.init_from_parameters(pi, T, E, sizze) for _ in range(2)]

    _, LL1 = Analyze(lin, stateNum)
    assert np.all(np.isfinite(LL1))


def test_BIC():
    """
    To test the BIC function. One a 1-state population, we run the BIC for 1, 2, and 3 states.
    We run it 20 times and make sure it got the right answer for more than half the times.
    """
    # create 1-state lineages
    pi1 = np.array([1.0, 0.0])
    T1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    E1 = [StateDistPhase(0.99, 0.9, 200, 0.5, 100, 0.6), StateDistPhase(0.99, 0.9, 200, 0.5, 100, 0.6)]
    lin = [[LineageTree.init_from_parameters(pi1, T1, E1, 1)] for _ in range(3)]
    desired_num_states = np.arange(1, 4)

    nums = 0
    for lins in lin:
        for _ in lins[0].output_lineage:
            nums += 1
    # run a few times and make sure it gives one state as the answer more than half the time.
    BIC = np.empty((len(desired_num_states), 20))
    for j in range(20):
        output = run_Analyze_over(lin, desired_num_states)

        for idx in range(len(desired_num_states)):
            BIC[idx, j], _ = output[idx][0].get_BIC(output[idx][1], nums)
        BIC[:, j] = BIC[:, j] - np.min(BIC[:, j])
    assert np.count_nonzero(BIC[0, :] == 0) > 10
