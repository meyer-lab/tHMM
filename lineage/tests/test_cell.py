""" Unit test file. """
import unittest
import math
import numpy as np
from ..Lineage import Population as p, generatePopulationWithTime as gpt
from ..CellNode import CellNode as c, generateLineageWithTime, doublingTime

class TestModel(unittest.TestCase):
    """ Here are the unit tests. """

    def test_lifetime(self):
        """ Make sure the cell isUnfinished before the cell dies and then make sure the cell's lifetime (tau) is calculated properly after it dies. """
        cell1 = c(startT=20)

        # nan before setting death time
        self.assertTrue(math.isnan(cell1.tau))
        self.assertTrue(cell1.isUnfinished())

        # correct life span after setting endT
        cell1.die(500)
        self.assertTrue(cell1.tau == 480)
        self.assertFalse(cell1.isUnfinished()) # cell is dead

    def test_divide(self):
        """ Make sure cells divide properly with proper parent/child member variables. """
        cell1 = c(startT=20)
        cell2, cell3 = cell1.divide(40)

        # cell divides at correct time & parent dies
        self.assertFalse(cell1.isUnfinished())
        self.assertTrue(cell1.tau == 20)
        self.assertTrue(cell2.startT == 40)
        self.assertTrue(cell2.isUnfinished())

        # left and right children exist for cell1 with proper linking
        self.assertTrue(cell1.left is cell2)
        self.assertTrue(cell1.right is cell3)
        self.assertTrue(cell2.parent is cell1)
        self.assertTrue(cell3.parent is cell1)

    def test_generate_fate(self):
        """ Make sure we can generate fake data properly when tuning the Bernoulli parameter for cell fate. """
        

    def test_generate_lifetime(self):
        """ Make sure generated fake data behaves properly when tuning the Gompertz parameters. """        
        print("testing lifetime")
        # average and stdev are both larger when c = 0.5 compared to c = 3
        out_c05 = generateLineageWithTime(10, 100, 0.8, 0.5, 50) 
        print("made out_c05")
        out_c3 = generateLineageWithTime(10, 100, 0.8, 3.0, 50)
        print("made out_c3")

        tau_c05 = [] # create an empty list 
        tau_c3 = tau_c05.copy()
        for n in range(pop_size):
            if not out_c05[n].isUnfinished():  # if cell has died, append tau to list
                tau_c05.append(out_c05[n].tau)
            if not out_c3[n].isUnfinished():  # if cell has died, append tau to list
                tau_c3.append(out_c3[n].tau)

        print("done appending taus")
        self.assertGreater(np.mean(tau_c05), np.mean(tau_c3))
        self.assertGreater(np.std(tau_c05), np.std(tau_c3))
        
        # average and stdev are both larger when scale = 3 compared to scale = 0.5
        out_scale40 = generateLineageWithTime(10, 100, 0.8, 2, 40) 
        print("made out_scale05")
        out_scale50 = generateLineageWithTime(10, 100, 0.8, 2, 50)
        print("made out_scale3")

        tau_scale40 = [] # create an empty list 
        tau_scale50 = tau_scale40.copy()
        for n in range(pop_size):
            if not out_scale40[n].isUnfinished():  # if cell has died, append tau to list
                tau_scale05.append(out_scale05[n].tau)
            if not out_scale50[n].isUnfinished():  # if cell has died, append tau to list
                tau_scale3.append(out_scale3[n].tau)

        print("done appending taus")
        self.assertGreater(np.mean(tau_scale50), np.mean(tau_scale40))
        self.assertGreater(np.std(tau_scale50), np.std(tau_scale40))

    def test_MLE_bern(self):
        """ Generate multiple lineages and estimate the bernoulli parameter with MLE. """
        experimentTime = 168 # we can now set this to be a value (in hours) that is experimentally useful (a week's worth of hours)
        locBern = [0.6]
        cGom = [2]
        scaleGom = [0.5e2]
        initCells = [100]
        popTime = p(experimentTime, initCells, locBern, cGom, scaleGom) # initialize "pop" as of class Population

        # both estimators must be within +/- 0.08 of true locBern for popTime
        self.assertTrue(0.52 <= p.bernoulliParameterEstimatorAnalytical(popTime) <= 0.68)
        self.assertTrue(0.52 <= p.bernoulliParameterEstimatorNumerical(popTime) <= 0.68)

    def test_MLE_gomp(self):
        """ Generate multiple lineages and estimate the gompertz parameters with MLE. """
        experimentTime = 168 # we can now set this to be a value (in hours) that is experimentally useful (a week's worth of hours)
        locBern = [0.6]
        cGom = [2]
        scaleGom = [0.5e2]
        initCells = [100]
        popTime = p(experimentTime, initCells, locBern, cGom, scaleGom) # initialize "pop" as of class Population

        # test populations w.r.t. time
        out = p.gompertzParameterEstimatorNumerical(popTime) # out[0] is cGom and out[1] is scaleGom
        self.assertTrue(1 <= out[0] <= 3) # +/- 1.0 of true cGom
        self.assertTrue(35 <= out[1] <= 65) # +/- 15 of scaleGom

    def test_doubleT(self):
        """ Check for basic functionality of doubleT. """
        base = doublingTime(100, 0.7, 2, 50)

        # doubles quicker when cells divide 90% of the time
        self.assertGreater(base, doublingTime(100, 0.9, 2, 50))

        # doubles quicker when cell lifetime is shorter (larger c & lower scale == shorter life)
        self.assertGreater(base, doublingTime(100, 0.7, 3, 50))
        self.assertGreater(base, doublingTime(100, 0.7, 2, 40))
