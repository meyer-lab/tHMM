""" Unit test file. """
import unittest
import math
import numpy as np
from ..Lineage import Population as p, generatePopulationWithTime as gpt
from ..tHMM_start import tHMM, remove_NaNs, max_gen, get_gen
from ..CellNode import CellNode

class TestModel(unittest.TestCase):
    """ Here are the unit tests. """
    def setUp(self):
        """ Small trees that can be used to run unit tests and check for edge cases. Some cells are labeled as self so they can be accessed in unit tests for comparisons. """
        # 3 level tree with no gaps
        cell1 = CellNode(startT=0, linID=1)
        cell2, cell3 = cell1.divide(10)
        self.cell4, self.cell5 = cell2.divide(15)
        self.cell6, self.cell7 = cell3.divide(20)
        self.lineage1 = [cell1, cell2, cell3, self.cell4, self.cell5, self.cell6, self.cell7]

        # 3 level tree where only one of the 2nd generation cells divides
        cell10 = CellNode(startT=0, linID=2)
        self.cell11, self.cell12 = cell10.divide(10)
        self.cell13, self.cell14 = self.cell11.divide(15)
        self.lineage2 = [cell10, self.cell11, self.cell12, self.cell13, self.cell14]

        # 3 level tree where one of the 2nd generation cells doesn't divide and the other only has one daughter cell
        cell20 = CellNode(startT=0, linID=3)
        cell21, cell22 = cell20.divide(10)
        self.cell23, _ = cell21.divide(15)
        cell21.right = None # reset the right pointer to None to effectively delete the cell from the lineage
        self.lineage3 = [cell20, cell21, cell22, self.cell23]
        
        # example where lineage is just one cell
        self.cell30 = CellNode(startT=0, linID=4)
        self.lineage4 = [self.cell30]
        
        
    def test_remove_NaNs(self):
        """ Checks to see that cells with a NaN of tau are eliminated from a population list. """
        experimentTime = 100.
        initCells = [50, 50]
        locBern = [0.6, 0.8]
        cGom = [2, 0.5]
        scaleGom = [40, 50]
        X = gpt(experimentTime, initCells, locBern, cGom, scaleGom) # generate a population
        X = remove_NaNs(X) # remove unfinished cells
        num_NAN = 0
        for cell in X:
            if cell.isUnfinished():
                num_NAN += 1

        self.assertEqual(num_NAN, 0) # there should be no unfinished cells left

    def test_get_numLineages(self):
        """ Checks to see that the initial number of cells created is the number of lineages. """
        experimentTime = 50.
        initCells = [50] # there should be 50 lineages b/c there are 50 initial cells
        locBern = [0.6]
        cGom = [2]
        scaleGom = [40]
        X = gpt(experimentTime, initCells, locBern, cGom, scaleGom) # generate a population

        t = tHMM(X) # build the tHMM class with X
        numLin = t.get_numLineages()
        self.assertEqual(numLin, 50) # call func

        # case where the lineages follow different parameter sets
        initCells = [50, 42, 8] # there should be 100 lineages b/c there are 100 initial cells
        locBern = [0.6, 0.8, 0.7]
        cGom = [2, 0.5, 1]
        scaleGom = [40, 50, 45]
        X = gpt(experimentTime, initCells, locBern, cGom, scaleGom) # generate a population

        t = tHMM(X) # build the tHMM class with X
        numLin = t.get_numLineages()
        self.assertEqual(numLin, 100) # call func

    def test_init_Population(self):
        """ Tests that populations are lists of lineages and each cell in a lineage has the correct linID. """
        experimentTime = 50.
        initCells = [50] # there should be 50 lineages b/c there are 50 initial cells
        locBern = [0.6]
        cGom = [2]
        scaleGom = [40]
        X = gpt(experimentTime, initCells, locBern, cGom, scaleGom) # generate a population

        t = tHMM(X) # build the tHMM class with X
        pop = t.init_Population()
        self.assertEqual(len(pop), initCells[0]) # len(pop) corresponds to the number of lineages

        # check that all cells in a lineage have same linID
        for i, lineage in enumerate(pop): # for each lineage
            for cell in lineage: # for each cell in said lineage
                self.assertEqual(i, cell.linID) # linID should correspond with i

    def test_max_gen(self):
        """ Calls lineages 1 through 4 and ensures that the maximimum number of generations is correct in each case. """
        # lineages 1-3 have 3 levels/generations
        self.assertEqual(max_gen(self.lineage1), 3)
        self.assertEqual(max_gen(self.lineage2), 3)
        self.assertEqual(max_gen(self.lineage3), 3)
        self.assertEqual(max_gen(self.lineage4), 1) # lineage 4 is just one cell

    def test_get_gen(self):
        """ Checks to make sure get_gen of a certain lineage returns the proper cells. """
        temp1 = get_gen(3, self.lineage1) # bottom row of lineage row
        self.assertIn(self.cell4, temp1)
        self.assertIn(self.cell5, temp1)
        self.assertIn(self.cell6, temp1)
        self.assertIn(self.cell7, temp1)
        self.assertEqual(len(temp1), 4)

        temp2 = get_gen(2, self.lineage2) # 2nd generation of lineage 2
        self.assertIn(self.cell11, temp2)
        self.assertIn(self.cell12, temp2)
        self.assertEqual(len(temp2), 2)

        temp3 = get_gen(3, self.lineage2) # 3rd generation of lineage 2
        self.assertIn(self.cell13, temp3)
        self.assertIn(self.cell14, temp3)
        self.assertEqual(len(temp3), 2)

        temp4 = get_gen(3, self.lineage3)
        self.assertIn(self.cell23, temp4)
        self.assertEqual(len(temp4), 1)

        temp5 = get_gen(1, self.lineage4)
        self.assertIn(self.cell30, temp5)
        self.assertEqual(len(temp5), 1)

    def test_init_paramlist(self):
        """ Make sure paramlist has proper labels and sizes. """
        experimentTime = 50.
        initCells = [50] # there should be 50 lineages b/c there are 50 initial cells
        locBern = [0.6]
        cGom = [2]
        scaleGom = [40]
        X = gpt(experimentTime, initCells, locBern, cGom, scaleGom) # generate a population
        X = remove_NaNs(X)
        t = tHMM(X, numStates=2) # build the tHMM class with X
        self.assertEqual(t.paramlist[0]["pi"].shape[0], 2) # make sure shape is numStates
        self.assertEqual(t.paramlist[0]["T"].shape[0], 2) # make sure shape is numStates
        self.assertEqual(t.paramlist[0]["T"].shape[1], 2) # make sure shape is numStates
        self.assertEqual(t.paramlist[0]["E"].shape[0], 2) # make sure shape is numStates
