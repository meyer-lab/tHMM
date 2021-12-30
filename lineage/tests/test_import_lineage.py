""" Unit test for MCF10A data. """
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
        """ Manually setting up the lineages from the test2.csv file which is fake data similar to MCF10A data. """
        self.cell1 = c(parent=None, gen=1)
        self.cell1.left = c(parent=self.cell1, gen=2)
        self.cell1.right = c(parent=self.cell1, gen=2)

        self.cell1.obs = [1, 4.0, 10.0, 1]
        self.cell1.left.obs = [1, 13.5, 12.0, 0]
        self.cell1.right.obs = [0, 13.5, 14.0, 0]
        self.cell2 = self.cell1.left
        self.cell3 = self.cell1.right

        self.cell2.left = c(parent=self.cell2, gen=3)
        self.cell2.left.obs = [np.nan, 6.0, 13.0, 1]
        self.cell2.right = c(parent=self.cell2, gen=3)
        self.cell2.right.obs = [np.nan, 6.0, 15.0, 1] 

        self.lin1 = [self.cell1, self.cell2, self.cell3, self.cell2.left, self.cell2.right]

        self.cell4 = c(parent=None, gen=1)
        self.cell4.obs = [1, 6.5, 11.0, 1]
        self.cell4.left = c(parent=self.cell4, gen=2)
        self.cell4.left.obs = [0, 1.5, 10.0, 0]
        self.cell4.right = c(parent=self.cell4, gen=2)
        self.cell4.right.obs = [0, 1.5, 12.0, 0]
        self.lin2 = [self.cell4, self.cell4.left, self.cell4.right]

    def test_data(self):
        """ import and test. """
        lineages = read_lineage_data("lineage/data/test.csv")
        lin1 = lineages[0]
        lin2 = lineages[1]
        assert len(lineages) == 2
        assert len(lin1) == 5
        assert len(lin2) == 3
        for i, cell in enumerate(lin1):
            assert np.all(cell.obs == self.lin1[i].obs)
        for j, cells in enumerate(lin2):
            print(cells.obs)
            print("now", self.lin2[j].obs)
            assert np.all(cells.obs == self.lin2[j].obs)
