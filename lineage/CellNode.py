# author : shakthi visagan (shak360), adam weiner (adamcweiner)
# description: a file to hold the cell class

import sys
import math
import scipy.stats as sp
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

class CellNode:
    def __init__(self, gen=1, startT=0, endT=float('nan'), fate=None, left=None, right=None, parent=None, plotVal=0):
        ''' Instantiates a cell node.'''
        self.gen = gen
        self.startT = startT
        self.endT = endT
        self.tau = self.endT - self.startT # avoiding self.t, since that is a common function (i.e. transposing matrices)
        self.fate = fate
        self.left = left
        self.right = right
        self.parent = parent
        self.plotVal = plotVal

    def isParent(self):
        """ Return the parent of the current cell. """
        return self.left.parent is self and self.right.parent is self

    def isChild(self):
        """ Returns true if this cell has a known parent. """
        return self.parent.isParent()

    def isRootParent(self):
        """Returns true if this cell has no documented parent. """
        return (self.gen == 1 and self.parent is None)

    def calcTau(self):
        """ Find the cell's lifetime. """
        self.tau = self.endT - self.startT   # calculate tau here
        if math.isnan(self.tau):
            print("Warning: your cell lifetime {} is a nan".format(self.tau))

    def isUnfinished(self):
        """ See if the cell is living or has already died/divided. """
        return math.isnan(self.endT) and self.fate is None   # returns true when cell is still alive

    def setUnfinished(self):
        """ Set a finished cell back to being unfinished. """
        self.endT = float('nan')
        self.fate = None
        self.tau = float('nan')
    
    def die(self, endT):
        """ Cell dies without dividing. """
        self.fate = False   # no division
        self.endT = endT    # mark endT
        self.calcTau()      # calculate Tau when cell dies

    def divide(self, endT):
        """ Cell life ends through division. """
        self.endT = endT
        self.fate = True    # division
        self.calcTau()      # calculate Tau when cell dies

        if self.isRootParent():
            self.left = CellNode(gen=self.gen+1, startT=endT, parent=self, plotVal=self.plotVal+0.75)
            self.right = CellNode(gen=self.gen+1, startT=endT, parent=self, plotVal=self.plotVal-0.75)
        else:
            self.left = CellNode(gen=self.gen+1, startT=endT, parent=self, plotVal=self.plotVal+(0.5**(self.gen))*(1.35**(self.gen))*self.plotVal)
            self.right = CellNode(gen=self.gen+1, startT=endT, parent=self, plotVal=self.plotVal-(0.5**(self.gen))*(1.35**(self.gen))*self.plotVal)

        return (self.left, self.right)

def generateLineageWithTime(initCells, experimentTime, locBern, cGom, scaleGom):
    ''' generates list given an experimental end time, a Bernoulli parameter for dividing/dying and a Gompertz parameter for cell lifetime'''
    #create an empty lineage
    lineage = []

    # initialize the list with cells
    for ii in range(initCells):
        lineage.append(CellNode(startT=0))

    # have cell divide/die according to distribution
    for cell in lineage:   # for all cells (cap at numCells)
        if cell.isUnfinished():
            cell.tau = sp.gompertz.rvs(cGom, scale=scaleGom)
            cell.endT = cell.startT + cell.tau
            if cell.endT < experimentTime: # determine fate only if endT is within range
                cell.fate = sp.bernoulli.rvs(locBern) # assign fate
                if cell.fate:
                    temp1, temp2 = cell.divide(cell.endT) # cell divides
                    # append to list
                    lineage.append(temp1)
                    lineage.append(temp2)
                else:
                    cell.die(cell.endT)
            else: # if the endT is past the experimentTime
                cell.setUnfinished() # reset cell to be unfinished and move to next cell

    # return the list at end
    return lineage

def doublingTime(initCells, locBern, cGom, scaleGom):
    ''' calculates the doubling time of a homogeneous cell population given the three parameters and an initial cell count. '''
    numAlive = [] # list that stores the number of alive cells for each experimentTime
    experimentTimes = np.logspace(start=0, stop=2, num=49)
    experimentTimes = [0] + experimentTimes

    for experimentTime in experimentTimes:
        lineage = generateLineageWithTime(initCells, experimentTime, locBern, cGom, scaleGom)
        count = 0
        for cell in lineage:
            if cell.isUnfinished():
                count += 1
        numAlive.append(count)

    # Fit to exponential curve and find exponential coefficient. 
    def expFunc(experimentTimes, *expParam):
        """ Calculates the exponential."""
        return(initCells * np.exp(expParam[0] * experimentTimes))

    expY = lambda experimentTimes, expParam: expFunc(experimentTimes, expParam)

    fitExpParam, _ = curve_fit(expY, experimentTimes, numAlive, p0=[0]) # fit an exponential curve to generated data

    doubleT = np.log(2) / fitExpParam[0] # relationship between doubling time and exponential function

    return doubleT