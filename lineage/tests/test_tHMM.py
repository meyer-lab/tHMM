""" Unit test file. """
import unittest
import math
import numpy as np
from ..Lineage import Population as p, generatePopulationWithTime as gpt
from ..tHMM_start import tHMM as t, remove_NaNs

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
                
        