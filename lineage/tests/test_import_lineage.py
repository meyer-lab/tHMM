""" Unit test for the new AU565 data. """
import unittest
import math
import pandas as pd
import numpy as np

from ..CellVar import CellVar as c
from ..import_lineage import read_lineage_data


class TestModel(unittest.TestCase):
    """
    Unit test class for importing data.
    """

    def setUp(self):
        """ Manually setting up the first two lineages from the new AU565 data. """
        self.cell1 = c(parent=None, gen=1)
        self.cell1.left = c(parent=self.cell1, gen=2)
        self.cell1.right = c(parent=self.cell1, gen=2)

        self.cell1.obs = [1, 3.0, 11.32, 1]
        self.cell1.left.obs = [0, 6.0, 11.83, 0]
        self.cell1.right.obs = [0, 1.5, 1.41, 0]
        self.cell2 = self.cell1.left
        self.cell3 = self.cell1.right

        self.lin1 = [self.cell1, self.cell2, self.cell3]

        self.cell4 = c(parent=None, gen=1)
        self.cell4.obs = [1, 14.5, 13.21, 1]
        self.cell4.left = c(parent=self.cell4, gen=2)
        self.cell4.left.obs = [np.nan, 9.5, 11.15, 1]
        self.cell4.right = c(parent=self.cell4, gen=2)
        self.cell4.right.obs = [np.nan, 9.5, 11.02, 1]
        self.lin2 = [self.cell4, self.cell4.left, self.cell4.right]

    def test_data(self):
        """ import and test. """
        lineages = read_lineage_data("lineage/data/LineageData/AU02101_A3_field_1_RP_50_CSV-Table.csv")
        lin1 = lineages[0]  # lineageID = 2
        lin2 = lineages[2]  # lineageID = 3

        assert len(lin1) == 3
        assert len(lin2) == 3

        for i, cell in enumerate(lin1):
            np.testing.assert_allclose(cell.obs, self.lin1[i].obs, rtol=1e-2)
            assert cell.lineageID == 2
        for j, cells in enumerate(lin2):
            np.testing.assert_allclose(cells.obs, self.lin2[j].obs, rtol=1e-2)
            assert cells.lineageID == 3
