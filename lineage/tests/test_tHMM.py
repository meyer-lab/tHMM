""" Unit test file. """
import unittest
import math
import numpy as np
from ..Lineage import Population as p, generatePopulationWithTime as gpt
from ..tHMM_start import tHMM, remove_NaNs

class TestModel(unittest.TestCase):
    """ Here are the unit tests. """
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
        print(numLin)
        self.assertTrue(numLin == 50) # call func

        # case where the lineages follow different parameter sets
        initCells = [50, 50] # there should be 100 lineages b/c there are 100 initial cells
        locBern = [0.6, 0.8]
        cGom = [2, 0.5]
        scaleGom = [40, 50]
        X = gpt(experimentTime, initCells, locBern, cGom, scaleGom) # generate a population

        t = tHMM(X) # build the tHMM class with X
        numLin = t.get_numLineages()
        print(numLin)
        self.assertTrue(numLin == 100) # call func
        
    def test_get_Population(self):
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
                self.assertTrue(i == cell.linID) # linID should correspond with I