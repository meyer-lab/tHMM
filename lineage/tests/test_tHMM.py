""" Unit test file. """
import unittest
import math
import numpy as np
from ..Lineage import Population as p, generatePopulationWithTime as gpt
from ..tHMM_start import tHMM as t, remove_NaNs

class TestModel(unittest.TestCase):
    """ Here are the unit tests. """
    def test_remove_NaNs(self):
        """ Checks to see that cells with a NaN of tau are eliminated from the list. """
        experimentTime = 100.
        initCells = [100, 100]
        locBern = [0.6, 0.8]
        cGom = [2, 0.5]
        scaleGom = [40, 50]
        X = gpt(experimentTime, initCells, locBern, cGom, scaleGom)
        num_NAN = 0
        for cell in X:
            if cell.tau is float('nan'):
                num_NAN += 1
        print("Number of initial NANs: " + str(num_NAN))

        remove_NaNs(X)
        for cell in X:
            if cell.tau is float('nan'):
                print("this cell should've been deleted")
                
        