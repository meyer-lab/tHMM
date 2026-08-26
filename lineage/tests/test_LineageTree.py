"""Unit test file."""

import unittest

import numpy as np

from ..CellVar import CellVar as c
from ..LineageTree import LineageTree
from ..states.StateDistributionGamma import StateDistribution


class TestModel(unittest.TestCase):
    """
    Unit test class for lineages.
    """

    def setUp(self):
        """
        Setting up lineages for testing.
        """
        # pi: the initial probability vector
        self.pi = np.array([0.75, 0.25])

        # T: transition probability matrix
        self.T = np.array([[0.85, 0.15], [0.20, 0.80]])

        # State 0 parameters "Resistant"
        self.state0 = 0
        bern_p0 = 0.99
        gamma_a0 = 20
        gamma_scale0 = 5

        # State 1 parameters "Susceptible"
        self.state1 = 1
        bern_p1 = 0.8
        gamma_a1 = 10
        gamma_scale1 = 1

        state_obj0 = StateDistribution(bern_p0, gamma_a0, gamma_scale0)
        state_obj1 = StateDistribution(bern_p1, gamma_a1, gamma_scale1)

        self.E = [state_obj0, state_obj1]

        # creating lineages with the various censor conditions
        self.lineage1 = LineageTree.rand_init(self.pi, self.T, self.E, desired_num_cells=(2**11) - 1)
        self.lineage2_fate_censored = LineageTree.rand_init(
            self.pi, self.T, self.E, desired_num_cells=(2**11) - 1, censor_condition=1
        )
        self.lineage3_time_censored = LineageTree.rand_init(
            self.pi,
            self.T,
            self.E,
            desired_num_cells=(2**11) - 1,
            censor_condition=2,
            desired_experiment_time=500,
        )
        self.lineage4_both_censored = LineageTree.rand_init(
            self.pi,
            self.T,
            self.E,
            desired_num_cells=(2**11) - 1,
            censor_condition=3,
            desired_experiment_time=500,
        )

        # creating 7 cells for 3 generations manually
        cell_1 = c(state=self.state0, parent=None)
        cell_2 = c(state=self.state0, parent=cell_1)
        cell_3 = c(state=self.state0, parent=cell_1)
        cell_4 = c(state=self.state0, parent=cell_2)
        cell_5 = c(state=self.state0, parent=cell_2)
        cell_6 = c(state=self.state0, parent=cell_3)
        cell_7 = c(state=self.state0, parent=cell_3)
        cell_1.left = cell_2
        cell_1.right = cell_3
        cell_2.left = cell_4
        cell_2.right = cell_5
        cell_3.left = cell_6
        cell_3.right = cell_7

        self.test_lineage = [cell_1, cell_2, cell_3, cell_4, cell_5, cell_6, cell_7]

    def test_lineage_csr_structure(self):
        """Test that LineageTree correctly constructs the CSR representation."""
        lin = LineageTree(self.test_lineage, self.E)
        self.assertEqual(lin.tree.shape, (7, 7))
        self.assertEqual(lin.tree.nnz, 6)

        # Root cell 0 has children 1 and 2
        root_children = lin.tree.indices[lin.tree.indptr[0] : lin.tree.indptr[1]]
        np.testing.assert_array_equal(sorted(root_children), [1, 2])

        # Cell 1 has children 3 and 4
        cell1_children = lin.tree.indices[lin.tree.indptr[1] : lin.tree.indptr[2]]
        np.testing.assert_array_equal(sorted(cell1_children), [3, 4])

        # Cell 2 has children 5 and 6
        cell2_children = lin.tree.indices[lin.tree.indptr[2] : lin.tree.indptr[3]]
        np.testing.assert_array_equal(sorted(cell2_children), [5, 6])

        # Leaves (3, 4, 5, 6) have no children
        for leaf in [3, 4, 5, 6]:
            self.assertEqual(lin.tree.indptr[leaf + 1] - lin.tree.indptr[leaf], 0)

        np.testing.assert_array_equal(lin.leaves_idx, [3, 4, 5, 6])

    def test_cell_to_daughters_compatibility(self):
        """Test that cell_to_daughters property returns the expected (N, 2) array."""
        lin = LineageTree(self.test_lineage, self.E)
        c2d = lin.cell_to_daughters
        self.assertEqual(c2d.shape, (7, 2))
        np.testing.assert_array_equal(c2d[0], [1, 2])
        np.testing.assert_array_equal(c2d[1], [3, 4])
        np.testing.assert_array_equal(c2d[2], [5, 6])
        for leaf in [3, 4, 5, 6]:
            np.testing.assert_array_equal(c2d[leaf], [-1, -1])

    def test_rand_init_csr_lineages(self):
        """Test rand_init produces valid CSR trees across censor conditions."""
        for lin in [
            self.lineage1,
            self.lineage2_fate_censored,
            self.lineage3_time_censored,
            self.lineage4_both_censored,
        ]:
            n = len(lin)
            self.assertEqual(lin.tree.shape, (n, n))
            self.assertEqual(len(lin.tree.indptr), n + 1)
            # Check leaves indexing
            diffs = np.diff(lin.tree.indptr)
            np.testing.assert_array_equal(lin.leaves_idx, np.nonzero(diffs == 0)[0])
            # Check edge count = n - 1 (for connected lineage with 1 root)
            self.assertEqual(lin.tree.nnz, n - 1)
