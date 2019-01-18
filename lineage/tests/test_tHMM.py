""" Unit test file. """
import unittest
import math
import numpy as np
from ..Lineage import Population as p, generatePopulationWithTime as gpt
from ..tHMM_start import tHMM, remove_NaNs, max_gen
from ..CellNode import CellNode

class TestModel(unittest.TestCase):
    """ Here are the unit tests. """
    def setUp(self):
        """ small trees that can be used to run unit tests and check for edge cases. """
        # 3 level tree with no gaps
        cell1 = CellNode(startT=0, linID=1)
        cell2, cell3 = cell1.divide(10)
        cell4, cell5 = cell2.divide(15)
        cell6, cell7 = cell3.divide(20)
        self.lineage1 = [cell1, cell2, cell3, cell4, cell5, cell6, cell7]

        # 3 level tree where only one of the 2nd generation cells divides
        cell10 = CellNode(startT=0, linID=2)
        cell11, cell12 = cell10.divide(10)
        cell13, cell14 = cell11.divide(15)
        self.lineage2 = [cell10, cell11, cell12, cell13, cell14]

        # 3 level tree where one of the 2nd generation cells doesn't divide and the other only has one daughter cell
        cell20 = CellNode(startT=0, linID=3)
        cell21, cell22 = cell20.divide(10)
        cell23, _ = cell21.divide(15)
        cell21.right = None # reset the right pointer to None to effectively delete the cell from the lineage
        self.lineage3 = [cell20, cell21, cell22, cell23]
        
        # example where lineage is just one cell
        cell30 = CellNode(startT=0, linID=4)
        self.lineage4 = [cell30]
        
        
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

        self.assertTrue(num_NAN == 0) # there should be no unfinished cells left

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
        self.assertTrue(numLin == 50) # call func

        # case where the lineages follow different parameter sets
        initCells = [50, 42, 8] # there should be 100 lineages b/c there are 100 initial cells
        locBern = [0.6, 0.8, 0.7]
        cGom = [2, 0.5, 1]
        scaleGom = [40, 50, 45]
        X = gpt(experimentTime, initCells, locBern, cGom, scaleGom) # generate a population

        t = tHMM(X) # build the tHMM class with X
        numLin = t.get_numLineages()
        self.assertTrue(numLin == 100) # call func

    def test_get_Population(self):
        """ Tests that populations are lists of lineages and each cell in a lineage has the correct linID. """
        experimentTime = 50.
        initCells = [50] # there should be 50 lineages b/c there are 50 initial cells
        locBern = [0.6]
        cGom = [2]
        scaleGom = [40]
        X = gpt(experimentTime, initCells, locBern, cGom, scaleGom) # generate a population

        t = tHMM(X) # build the tHMM class with X
        pop = t.get_Population()
        self.assertTrue(len(pop) == initCells[0]) # len(pop) corresponds to the number of lineages

        # check that all cells in a lineage have same linID
        for i, lineage in enumerate(pop): # for each lineage
            for cell in lineage: # for each cell in said lineage
                self.assertTrue(i == cell.linID) # linID should correspond with i

    def test_max_gen(self):
        """ Calls lineages 1 through 4 and ensures that the maximimum number of generations is correct in each case. """
        # lineages 1-3 have 3 levels/generations
        self.assertTrue(max_gen(self.lineage1) == 3)
        self.assertTrue(max_gen(self.lineage2) == 3)
        self.assertTrue(max_gen(self.lineage3) == 3)
        self.assertTrue(max_gen(self.lineage4) == 1) # lineage 4 is just one cell
