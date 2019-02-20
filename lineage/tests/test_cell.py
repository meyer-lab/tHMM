""" Unit test file. """
import unittest
import math
import numpy as np
from ..Lineage_utils import generatePopulationWithTime, gompertzParameterEstimatorNumerical, bernoulliParameterEstimatorAnalytical, gompertzAnalytical
from ..CellNode import CellNode as c, generateLineageWithTime, doublingTime

class TestModel(unittest.TestCase):
    """Here are the unit tests."""
    def test_lifetime(self):
        """Make sure the cell isUnfinished before the cell dies and then make sure the cell's lifetime (tau) is calculated properly after it dies."""
        cell1 = c(startT=20)

        # nan before setting death time
        self.assertTrue(math.isnan(cell1.tau))
        self.assertTrue(cell1.isUnfinished())

        # correct life span after setting endT
        cell1.die(500)
        self.assertTrue(cell1.tau == 480)
        self.assertFalse(cell1.isUnfinished()) # cell is dead

    def test_divide(self):
        """Make sure cells divide properly with proper parent/child member variables."""
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

    def test_generate_endT(self):
        """Make sure experiment ends at proper time when using generateLineageWithTime."""
        out = generateLineageWithTime(100, 100, 0.5, 2, 50)
        for cell in out:
            if cell.isUnfinished(): # if cell is alive
                self.assertTrue(math.isnan(cell.endT)) # don't know final lifetime
                self.assertTrue(math.isnan(cell.tau))
                self.assertLess(cell.startT, 100) # was created before end of experiment
                self.assertTrue(cell.fate is None) # fate is none
            else:
                self.assertLess(cell.endT, 100) # endT is before end of experiment
                self.assertFalse(math.isnan(cell.tau)) # tau is not NaN
                self.assertLess(cell.startT, cell.endT) # start time is before endT
                self.assertTrue(cell.fate is not None) # fate is none

    def test_generate_fate(self):
        """There are more live cells at end of 100 hour experiment when bernoulli param is larger."""
        out_5 = generateLineageWithTime(100, 100, 0.5, 2, 50)
        count_5 = 0
        for cell in out_5:
            if cell.isUnfinished():
                count_5 += 1
        out_8 = generateLineageWithTime(100, 100, 0.8, 2, 50)
        count_8 = 0
        for cell in out_8:
            if cell.isUnfinished():
                count_8 += 1

        self.assertGreater(count_8, count_5)

    def test_generate_lifetime(self):
        """Make sure generated fake data behaves properly when tuning the Gompertz parameters."""
        # average and stdev are both larger when c = 0.5 compared to c = 3
        out_c05 = generateLineageWithTime(10, 100, 0.8, 0.5, 50)
        out_c3 = generateLineageWithTime(10, 100, 0.8, 3.0, 50)

        tau_c05 = [] # create an empty list
        tau_c3 = tau_c05.copy()
        for n in out_c05:
            if not n.isUnfinished():  # if cell has died, append tau to list
                tau_c05.append(n.tau)
        for n in out_c3:
            if not n.isUnfinished():  # if cell has died, append tau to list
                tau_c3.append(n.tau)

        self.assertGreater(np.mean(tau_c05), np.mean(tau_c3))
        self.assertGreater(np.std(tau_c05), np.std(tau_c3))

        # average and stdev are both larger when scale = 3 compared to scale = 0.5
        out_scale40 = generateLineageWithTime(10, 100, 0.8, 2, 40)
        out_scale50 = generateLineageWithTime(10, 100, 0.8, 2, 50)

        tau_scale40 = [] # create an empty list
        tau_scale50 = tau_scale40.copy()
        for n in out_scale40:
            if not n.isUnfinished():  # if cell has died, append tau to list
                tau_scale40.append(n.tau)
        for n in out_scale50:
            if not n.isUnfinished():  # if cell has died, append tau to list
                tau_scale50.append(n.tau)

        self.assertGreater(np.mean(tau_scale50), np.mean(tau_scale40))

    def test_MLE_bern(self):
        """Generate multiple lineages and estimate the bernoulli parameter with MLE."""
        experimentTime = 168 # we can now set this to be a value (in hours) that is experimentally useful (a week's worth of hours)
        locBern = [0.6]
        cGom = [2]
        scaleGom = [0.5e2]
        initCells = [100]
        popTime = generatePopulationWithTime(experimentTime, initCells, locBern, cGom, scaleGom) # initialize "pop" as of class Population

        # both estimators must be within +/- 0.08 of true locBern for popTime
        self.assertTrue(0.52 <= bernoulliParameterEstimatorAnalytical(popTime) <= 0.68)

    def test_MLE_gomp(self):
        """Generate multiple lineages and estimate the gompertz parameters with MLE."""
        experimentTime = 168 # we can now set this to be a value (in hours) that is experimentally useful (a week's worth of hours)
        locBern = [0.6]
        cGom = [2]
        scaleGom = [0.5e2]
        initCells = [100]
        popTime = generatePopulationWithTime(experimentTime, initCells, locBern, cGom, scaleGom) # initialize "pop" as of class Population

        # test populations w.r.t. time
        out = gompertzParameterEstimatorNumerical(popTime) # out[0] is cGom and out[1] is scaleGom
        self.assertTrue(0 <= out[0] <= 5) # +/- 2.0 of true cGom
        self.assertTrue(30 <= out[1] <= 70) # +/- 20 of scaleGom
        
    def test_MLE_gomp_analytical(self):
        """ Use the analytical shortcut to estimate the gompertz parameters. """
        experimentTime = 168 # we can now set this to be a value (in hours) that is experimentally useful (a week's worth of hours)
        locBern = [0.6]
        cGom = [2]
        scaleGom = [50.]
        initCells = [100]
        popTime = generatePopulationWithTime(experimentTime, initCells, locBern, cGom, scaleGom) # initialize "pop" as of class Population

        # test populations w.r.t. time
        c_out, scale_out = gompertzAnalytical(popTime) # out[0] is cGom and out[1] is scaleGom
        print("c_out: " + str(c_out))
        print("scale_out: " + str(scale_out))
        self.assertTrue(0 <= c_out <= 5) # +/- 2.0 of true cGom
        self.assertTrue(30 <= scale_out <= 70) # +/- 20 of scaleGom

    def test_doubleT(self):
        """Check for basic functionality of doubleT."""
        base = doublingTime(100, 0.7, 2, 50)

        # doubles quicker when cells divide 90% of the time
        self.assertGreater(base, doublingTime(100, 0.9, 2, 50))

        # doubles quicker when cell lifetime is shorter (larger c & lower scale == shorter life)
        self.assertGreater(base, doublingTime(100, 0.7, 3, 50))
        self.assertGreater(base, doublingTime(100, 0.7, 2, 40))

    def test_hetergeneous_pop(self):
        """ Calls generatePopulationWithTime when there is a switch in parameters over the course of the experiment's time. """
        experimentTime = 168 # we can now set this to be a value (in hours) that is experimentally useful (a week's worth of hours)
        # first set of parameters (from t=0 to t=100)
        locBern = [0.6]
        cGom = [2]
        scaleGom = [0.5e2]
        initCells = [100]
        switchT = 100 # switch at t=100
        # second set of parameters (from t=100 to t=experimentTime)
        bern2 = [0.85]
        cG2 = [2]
        scaleG2 = [40]
        popTime = generatePopulationWithTime(experimentTime, initCells, locBern, cGom, scaleGom, switchT, bern2, cG2, scaleG2) # initialize "pop" as of class Populations
        bernEstimate = bernoulliParameterEstimatorAnalytical(popTime)

        # the Bernoulli parameter estimate should be greater than than locBern since bern2>locBern
        self.assertTrue(bernEstimate > 0.7)
