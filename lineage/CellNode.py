""" author : shakthi visagan (shak360), adam weiner (adamcweiner)
description: a file to hold the cell class
"""
import math
import numpy as np
from scipy import optimize, stats as sp

class CellNode:
    """ Each cell in our tree will consist of a node containing these traits. """

    def __init__(self, gen=1, linID=0, startT=0, endT=float('nan'), fate=None, left=None, right=None, parent=None, trackID=None, true_state=None):
        ''' Instantiates a cell node.'''
        self.gen = gen # the generation of the cell, root cells are of generation 1, each division adds 1 to the previous generation
        self.linID = linID # the lineage identity of the cell, keeps track of what lineage a cell belongs to
        self.startT = startT # the starting time of the cell, the point at which it spawned into existence
        self.endT = endT # the end time of the cell, the point at which it either divided or died, can be NaN
        self.tau = self.endT - self.startT # avoiding self.t, since that is a common function (i.e. transposing matrices)
        # tau is how long the cell lived
        self.fate = fate # the fate at the endT of the cell, 0 is death, 1 is division
        self.left = left # the left daughter of the cell, either returns a CellNode or NoneType object
        self.right = right # the right daughter of the cell, either returns a CellNode or NoneType object
        self.parent = parent # the parent of the cell, returns a CellNode object (except at the root node)

        self.trackID = trackID # ID (integer) of the cell used during image tracking
        self.true_state = true_state # indicates whether cell is PC9 (0) or H1299 (1)

    def isParent(self):
        """ Return true if the cell has at least one daughter. """
        return self.left or self.right


    def isChild(self):
        """ Returns true if this cell has a known parent. """
        return self.parent.isParent()

    def isRootParent(self):
        """ Returns whether this is a starting cell with no parent. """
        bool_parent = False
        if self.parent is None:
            assert self.gen == 1
            bool_parent = True
        return bool_parent

    def isLeaf(self):
        '''Returns wheter a cell is a leaf with no children)'''
        return self.left is None and self.right is None

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

    def divide(self, endT, trackID_d1=None, trackID_d2=None):
        """ Cell life ends through division. """
        self.endT = endT
        self.fate = True    # division
        self.calcTau()      # calculate Tau when cell dies

        if self.isRootParent():

            self.left = CellNode(gen=self.gen+1, trackID=trackID_d1, linID=self.linID, startT=endT, parent=self, true_state=self.true_state)
            self.right = CellNode(gen=self.gen+1, trackID=trackID_d2, linID=self.linID, startT=endT, parent=self, true_state=self.true_state)
        else:
            self.left = CellNode(gen=self.gen+1, trackID=trackID_d1, linID=self.linID, startT=endT, parent=self, true_state=self.true_state)
            self.right = CellNode(gen=self.gen+1, trackID=trackID_d2, linID=self.linID, startT=endT, parent=self, true_state=self.true_state)

        return (self.left, self.right)

    def get_root_cell(self):
        '''Gets the root cell associated with the cell.'''
        cell_linID = self.linID
        curr_cell = self
        while curr_cell.gen > 1:
            curr_cell = curr_cell.parent
            assert cell_linID == curr_cell.linID
        assert cell_linID == curr_cell.linID
        return curr_cell

def generateLineageWithTime(initCells, experimentTime, locBern, cGom, scaleGom, switchT=None, bern2=None, cG2=None, scaleG2=None, FOM='G', betaExp=None, betaExp2=None):
    ''' generates list given an experimental end time, a Bernoulli parameter for dividing/dying and a Gompertz parameter for cell lifetime'''
    #create an empty lineage
    lineage = []

    # initialize the list with cells
    for ii in range(initCells):
        lineage.append(CellNode(startT=0, linID = ii))

    # have cell divide/die according to distribution
    for cell in lineage:   # for all cells (cap at numCells)
        if cell.isUnfinished():
            if switchT is not None and cell.startT > switchT: # when the cells should abide by the second set of parameters
                cell.true_state=1
                if FOM=='G':
                    cell.tau = sp.gompertz.rvs(cG2, scale=scaleG2)
                elif FOM=='E':
                    cell.tau = np.random.exponential(scale=betaExp2)
            else: # use first set of parameters for non-heterogeneous lineages or before the switch time
                cell.true_state=0
                if FOM=='G':
                    cell.tau = sp.gompertz.rvs(cGom, scale=scaleGom)
                elif FOM=='E':
                    cell.tau = np.random.exponential(scale=betaExp)                
            cell.endT = cell.startT + cell.tau
            if cell.endT < experimentTime: # determine fate only if endT is within range
                # assign cell fate
                if switchT is not None and cell.startT > switchT: # when the cells should abide by the second set of parameters
                    cell.fate = sp.bernoulli.rvs(bern2)
                else: # use first set of parameters for non-heterogeneous lineages or before the switch time
                    cell.fate = sp.bernoulli.rvs(locBern) # assign fate
                # divide or die based on fate
                if cell.fate:
                    temp1, temp2 = cell.divide(cell.endT) # cell divides
                    # append to list
                    lineage.append(temp1)
                    lineage.append(temp2)
                else:
                    cell.die(cell.endT)
            else: # if the endT is past the experimentTime
                cell.tauFake = experimentTime-cell.startT 
                cell.fateFake = sp.bernoulli.rvs(locBern) # assign fate
                cell.endT = experimentTime
                cell.setUnfinished() # reset cell to be unfinished and move to next cell

    # return the list at end
    return lineage

def doublingTime(initCells, locBern, cGom, scaleGom, FOM='G', betaExp=None):
    ''' calculates the doubling time of a homogeneous cell population given the three parameters and an initial cell count. '''
    numAlive = [] # list that stores the number of alive cells for each experimentTime
    experimentTimes = np.logspace(start=0, stop=2, num=49)
    experimentTimes = [0] + experimentTimes

    for experimentTime in experimentTimes:
        lineage = []
        if FOM=='G':
            lineage = generateLineageWithTime(initCells, experimentTime, locBern, cGom, scaleGom)
        elif FOM=='E':
            lineage = generateLineageWithTime(initCells, experimentTime, locBern, cGom, scaleGom, FOM='E', betaExp=betaExp)
        count = 0
        for cell in lineage:
            if cell.isUnfinished():
                count += 1
        numAlive.append(count)

    # Fit to exponential curve and find exponential coefficient.
    def expFunc(experimentTimes, *expParam):
        """ Calculates the exponential."""
        return initCells * np.exp(expParam[0] * experimentTimes)

    fitExpParam, _ = optimize.curve_fit(expFunc, experimentTimes, numAlive, p0=[0]) # fit an exponential curve to generated data

    doubleT = np.log(2) / fitExpParam[0] # relationship between doubling time and exponential function

    return doubleT