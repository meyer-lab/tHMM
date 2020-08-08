"""" Unit test file. """
import unittest
import pandas as pd
import math

from ..LineageInputOutput import import_Heiser, tryRecursion
from ..CellVar import CellVar as c


class TestModel(unittest.TestCase):
    """
    Unit test class for importing data.
    """

    def setUp(self):
        """
        Gets the path to synthetic data.
        This data is formatted similarly to Heiser lab data,
        but contains known mistakes/exceptions that the functions
        should be able to handle.
        """

        self.path_to_synthetic_data = r"lineage/data/heiser_data/Synth_data.xlsx"

        # manually setting up trees from Synth_data
        # 1
        self.parent1 = c(parent=None, gen=1, synthetic=False)
        self.parent1.obs = [1, 1, 10, 10, 0, 1]
        self.left1 = c(parent=self.parent1, gen=2, synthetic=False)
        self.left1.obs = [1, 0, 10, 10, 1, 1]
        self.right1 = c(parent=self.parent1, gen=2, synthetic=False)
        self.right1.obs = [1, float('nan'), 20, 105, 1, 0]
        self.parent1.left = self.left1
        self.parent1.right = self.right1
        self.lin1 = [self.left1, self.right1, self.parent1]

        # 2
        self.parent2 = c(parent=None, gen=1, synthetic=False)
        self.parent2.obs = [1, 1, 10, 10, 0, 1]
        self.left2 = c(parent=self.parent2, gen=2, synthetic=False)
        self.left2.obs = [float('nan'), float(
            'nan'), 125, float('nan'), 0, float('nan')]
        self.right2 = c(parent=self.parent2, gen=2, synthetic=False)
        self.right2.obs = [1, 0, 10, 10, 1, 1]
        self.parent2.left = self.left2
        self.parent2.right = self.right2
        self.lin2 = [self.left2, self.right2, self.parent2]

        # 3
        self.parent3 = c(parent=None, gen=1, synthetic=False)
        self.parent3.obs = [1, 1, float('nan'), 30, float('nan'), 0]
        self.left3_1 = c(parent=self.parent3, gen=2, synthetic=False)
        self.left3_1.obs = [1, 1, 30, 30, 1, 1]
        self.right3_1 = c(parent=self.parent3, gen=2, synthetic=False)
        self.right3_1.obs = [1, 0, 10, 80, 1, 1]
        self.parent3.left = self.left3_1
        self.parent3.right = self.right3_1
        self.left3_2 = c(parent=self.left3_1, gen=3, synthetic=False)
        self.left3_2.obs = [1, float('nan'), 30, 25, 1, 0]
        self.right3_2 = c(parent=self.left3_1, gen=3, synthetic=False)
        self.right3_2.obs = [1, float('nan'), 25, 30, 1, 0]
        self.lin3 = [self.left3_2, self.right3_2,
                     self.left3_1, self.right3_1, self.parent3]
        self.lin = [self.lin1, self.lin2, self.lin3]

    def test_import_Heiser(self):
        """
        Tests the main import function for Heiser lab data.
        """
        path2use = self.path_to_synthetic_data
        lineages = import_Heiser(path2use)
        self.assertTrue(len(lineages) == 3)
        self.assertTrue(len(lineages[0]) == 3)
        self.assertTrue(len(lineages[1]) == 3)
        self.assertTrue(len(lineages[2]) == 5)

        # This won't work if the order the cells are stored is changed
        for i in range(len(lineages)):
            # soft check that the order is probably the same
            assert lineages[i][len(lineages[i]) - 1].gen == 1
            for j in range(len(lineages[i])):
                for k in range(6):
                    self.assertTrue(lineages[i][j].obs[k] == self.lin[i][j].obs[k] or (
                        math.isnan(lineages[i][j].obs[k]) and math.isnan(self.lin[i][j].obs[k])))

    def test_tryRecursion(self):
        """
        Tests the recursion function used to recurse acros Excel cells
        in Heiser lab data.
        """
        path2use = self.path_to_synthetic_data
        excel_file = pd.read_excel(path2use, header=None)
        data = excel_file.to_numpy()
        cLin = []
        _ = tryRecursion(1, 45, 37, self.parent3, cLin, data, 30, 145)
        self.assertTrue(len(cLin) == 3)
        i = 0
        while i < len(cLin) and cLin[i].gen != 2:
            i += 1
        assert i < len(cLin)
        for j in range(6):
            self.assertTrue(cLin[i].obs[j] == self.left3_1.obs[j] or (
                math.isnan(self.left3_1.obs[j]) and math.isnan(cLin[i].obs[j])))
            self.assertTrue(cLin[i].right.obs[j] == self.right3_2.obs[j] or (
                math.isnan(self.right3_2.obs[j]) and math.isnan(cLin[i].right.obs[j])))
            self.assertTrue(cLin[i].left.obs[j] == self.left3_2.obs[j] or (
                math.isnan(self.left3_2.obs[j]) and math.isnan(cLin[i].left.obs[j])))
            self.assertTrue(cLin[i].parent.obs[j] == self.parent3.obs[j] or (
                math.isnan(self.parent3.obs[j]) and math.isnan(cLin[i].parent.obs[j])))
