""" Unit test file. """
import unittest
import math
import numpy as np
from ..Lineage import Lineage as l, Population as p, generatePopulationWithNum as gpn, generatePopulationWithTime as gpt
from ..CellNode import CellNode as c, generateLineageWithNum, generateLineageWithTime

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
        # if cell always divides it will stop at the maximum cell count when odd and one cell above when even (you can't divide and produce only 1 cell)
        out1 = generateLineageWithNum(7, 1.0, 0.6, 1)
        self.assertTrue(len(out1) == 7)
        out1 = generateLineageWithNum(10, 1.0, 0.6, 1)
        self.assertTrue(len(out1) == 11)

        # only 1 cell no matter numCells when cells always die
        out1 = generateLineageWithNum(7, 0.0, 0.6, 1)
        self.assertTrue(len(out1) == 1)

        # when locBern is 0.5 the initial cell divides ~1/2 the time
        nDiv = 0
        for i in range(1000):
            out1 = generateLineageWithNum(3, 0.5, 0.6, 1) # allow for 1 division max
            if len(out1) == 3:
                nDiv += 1
        self.assertTrue(450 <= nDiv <= 550) # assert that it divided ~500 times

    def test_generate_lifetime(self):
        """ Make sure generated fake data behaves properly when tuning the Gompertz parameters. """
        pop_size = 499 # cell number will always be odd
        
        # average and stdev are both larger when c = 0.5 compared to c = 3
        out_c05 = generateLineageWithNum(pop_size, 1.0, 0.5, 1) 
        out_c3 = generateLineageWithNum(pop_size, 1.0, 3.0, 1)

        tau_c05 = [] # create an empty list 
        tau_c3 = tau_c05.copy()
        for n in range(pop_size):
            if not out_c05[n].isUnfinished():  # if cell has died, append tau to list
                tau_c05.append(out_c05[n].tau)
            if not out_c3[n].isUnfinished():  # if cell has died, append tau to list
                tau_c3.append(out_c3[n].tau)

        self.assertGreater(np.mean(tau_c05), np.mean(tau_c3))
        self.assertGreater(np.std(tau_c05), np.std(tau_c3))
        
        # average and stdev are both larger when scale = 3 compared to scale = 0.5
        out_scale05 = generateLineageWithNum(pop_size, 1.0, 0.75, 0.5) 
        out_scale3 = generateLineageWithNum(pop_size, 1.0, 0.75, 3)

        tau_scale05 = [] # create an empty list 
        tau_scale3 = tau_scale05.copy()
        for n in range(pop_size):
            if not out_scale05[n].isUnfinished():  # if cell has died, append tau to list
                tau_scale05.append(out_scale05[n].tau)
            if not out_scale3[n].isUnfinished():  # if cell has died, append tau to list
                tau_scale3.append(out_scale3[n].tau)

        self.assertGreater(np.mean(tau_scale3), np.mean(tau_scale05))
        self.assertGreater(np.std(tau_scale3), np.std(tau_scale05))

    def test_MLE_bern(self):
        """ Generate multiple lineages and estimate the bernoulli parameter with MLE. """
        experimentTime = 168 # we can now set this to be a value (in hours) that is experimentally useful (a week's worth of hours)
        # division time of a cancer cell is about 20 hours
        locBern = 0.6
        cGom = 2
        scaleGom = 0.5e2
        numLineages = 20
        numCells = 75
        pop = p() # initialize "pop" as of class Population

        # generate a population of lineages w.r.t. time
        pop.group = gpt(numLineages, experimentTime, locBern, cGom, scaleGom)
        # both estimators must be within +/- 0.08 of true locBern
        self.assertTrue(0.52 <= p.bernoulliParameterEstimatorAnalytical(pop) <= 0.68)
        self.assertTrue(0.52 <= p.bernoulliParameterEstimatorNumerical(pop) <= 0.68)

        # generate a population of lineages w.r.t. number
        pop.group = gpn(numLineages, numCells, locBern, cGom, scaleGom)
        # both estimators must be within +/- 0.08 of true locBern
        self.assertTrue(0.52 <= p.bernoulliParameterEstimatorAnalytical(pop) <= 0.68)
        self.assertTrue(0.52 <= p.bernoulliParameterEstimatorNumerical(pop) <= 0.68)

    def test_MLE_gomp(self):
        """ Generate multiple lineages and estimate the gompertz parameters with MLE. """
        experimentTime = 168 # we can now set this to be a value (in hours) that is experimentally useful (a week's worth of hours)
        # division time of a cancer cell is about 20 hours
        locBern = 0.6
        cGom = 2
        scaleGom = 0.5e2
        numLineages = 20
        numCells = 75
        pop = p() # initialize "pop" as of class Population

        # generate a population of lineages w.r.t. time
        pop.group = gpt(numLineages, experimentTime, locBern, cGom, scaleGom)
        out = p.gompertzParameterEstimatorNumerical(pop) # out[0] is cGom and out[1] is scaleGom
        print(out[0])
        self.assertTrue(0.75 <= out[0] <= 3.25)
        self.assertTrue(35 <= out[1] <= 65)
        