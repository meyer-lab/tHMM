""" author : shakthi visagan (shak360), adam weiner (adamcweiner)
description: a file to hold the cell class
"""
import math
import numpy as np
from scipy import optimize, stats as sp


class CellNode:
    """Each cell in our tree will consist of a node containing these traits.

    This class includes many functions that assign the cell its properties, i.e.,
    the cell's generation, lifetime, true_state, etc."""

    def __init__(self, gen=1, linID=0, startT=0, endT=float('nan'), fate=None, left=None, right=None, parent=None, trackID=None, true_state=None):
        """
        Args:
        ----------
            gen (int): the generation of the cell, root cells are of generation 1,
                each division adds 1 to the previous generation.

            linID (int): the lineage identity of the cell, keeps track of what
            lineage a cell belongs to.

            startT (float): the starting time of the cell, the point at which
            it spawned into existence.

            endT (float): the end time of the cell, the point at which it either
            divided or died, can be NaN.

            tau (float): [avoiding self.t, since that is a common function
            (i.e. transposing matrices)] tau is how long the cell lived.

            fate (0/1): the fate at the endT of the cell, 0 is death, 1 is division.

            left (obj): the left daughter of the cell, either returns a CellNode
            or NoneType object.

            right (obj): the right daughter of the cell, either returns a CellNode
            or NoneType object.

            parent (obj): the parent of the cell, returns a CellNode object
            (except at the root node)

            trackID (int): ID of the cell used during image tracking

            true_state (0/1): indicates whether cell is PC9 (0) or H1299 (1)

            fateObserved (T/F): marks whether the cell reached the true end
            of its lifetime (has truely died or divided)

        """
        self.gen = gen
        self.linID = linID
        self.startT = startT
        self.endT = endT
        self.tau = self.endT - self.startT
        self.fate = fate
        self.left = left
        self.right = right
        self.parent = parent

        self.trackID = trackID
        self.true_state = true_state
        self.fateObserved = False

    def isParent(self):
        """ Returns true if the cell has at least one daughter; i.e., if either of the left or right daughter cells exist, it returns True. """
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
        """ Returns True when a cell is a leaf with no children. """
        return self.left is None and self.right is None

    def calcTau(self):
        """
        Find the cell's lifetime by subtracting its endTime from startTime
        it makes sure the cell's lifetime is not NaN.
        """
        self.tau = self.endT - self.startT   # calculate tau here
        assert np.isfinite(self.tau), "Warning: your cell lifetime, {}, is a nan".format(self.tau)

    def isUnfinished(self):
        """ See if the cell is living or has already died/divided. """
        return math.isnan(self.endT) and self.fate is None   # returns true when cell is still alive

    def setUnfinished(self):
        """ Set a finished cell back to being unfinished. """
        self.endT = float('nan')
        self.fate = None
        self.tau = float('nan')

    def die(self, endT):
        """
        Cell dies without dividing.

        If the cell dies, the endTime is reached so we calculate the lifetime (tau)
        and the death is observed.
        Args:
            ----------
            endT (float): end time of the cell.

        This function doesn't return
        """
        self.fate = False   # no division
        self.endT = endT    # mark endT
        self.calcTau()      # calculate Tau when cell dies
        self.fateObserved = True  # this cell has truly died

    def divide(self, endT, trackID_d1=None, trackID_d2=None):
        """
        Cell life ends through division. The two optional trackID arguments
        represents the trackIDs given to the two daughter cells.

        Args:
            ---------
            endT (float): end time of the cell

        kwargs:
            trackID_d1 & trackID_d2: since trackID is an attribute of experimental data,
            so when we generate the lineage we need to assing "None" to the trackID
            of the left and right daughter cell

        Returns:
            ----------
            self.left (obj): left new born daughter cell
            self.right (obj): right new born daughter cell

        """
        self.endT = endT
        self.fate = True    # division
        self.calcTau()      # calculate Tau when cell dies
        self.fateObserved = True  # this cell has truly divided

        if self.isRootParent():

            self.left = CellNode(gen=self.gen + 1, trackID=trackID_d1, linID=self.linID, startT=endT, parent=self, true_state=self.true_state)
            self.right = CellNode(gen=self.gen + 1, trackID=trackID_d2, linID=self.linID, startT=endT, parent=self, true_state=self.true_state)
        else:
            self.left = CellNode(gen=self.gen + 1, trackID=trackID_d1, linID=self.linID, startT=endT, parent=self, true_state=self.true_state)
            self.right = CellNode(gen=self.gen + 1, trackID=trackID_d2, linID=self.linID, startT=endT, parent=self, true_state=self.true_state)

        return (self.left, self.right)

    def get_root_cell(self):
        """
        Gets the root cell associated with the cell.

        it keeps going up to the root by jumping from daughter cells to their
        mother cells by keeping track of their lineage ID (linID) until it reaches
        the first generation and then returns the last hold cell which is the
        ancestor of the lineage.

        Returns:
            ---------
            curr_cell (obj): the ancestor cell if a lineage

        """
        cell_linID = self.linID
        curr_cell = self
        while curr_cell.gen > 1:
            curr_cell = curr_cell.parent
            assert cell_linID == curr_cell.linID
        assert cell_linID == curr_cell.linID
        return curr_cell


def generateLineageWithTime(initCells, experimentTime, locBern, cGom, scaleGom, switchT=None, bern2=None, cG2=None, scaleG2=None, FOM='E', betaExp=None, betaExp2=None):
    """
    generates a list of objects (cells) in a lineage.

    Given an experimental end time, a Bernoulli distribution for
    dividing/dying and a Gompertz/exponential parameter for cell lifetime,
    it creates objects as cells and puts them in a list.
    If the switchT is not None, then after a while it will switch
    to the new bernoulli, Gompertz/Exponential parameters and creates lineages
    based on the new distribution.


    Bernoulli distribution:
        It has one parameter (locBern)
        If locBern = 0.80 then 80% of the times the cells will divide and 20%
        of the times they die; meaning, for every cell that is generated,
        a random number (either 1:divide, or 0:die) is picked from the
        distribution with this parameter and the fate of the cell is assigned.

    Gompertz distribution:
        It has two parameters (cGom, scaleGom)
        Given these two parameters, for every cell, it randomly picks a number
        from the distribution and assigns it to the cell as its lifetime. The unit
        of the lifetime would be in [hours].

    Exponential distribution:
        It has one parameter (betaExp)
        It is a replacement for Gompertz to produce cell's lifetime, given the
        beta parameter, every time we pick a random number from an exponential
        distributions with parameter betaExp, and assign it to be the lifetime
        of the cell.


    Args:
        ----------
        initCells (int): the number of initial cells to initiate the tree with
        experimentTime (int) [hours]: the time that the experiment will be running
        to allow for the cells to grow
        locBern (float): the Bernoulli distribution parameter
        (p = success) for fate assignment (either the cell dies or divides)
        range = [0, 1]
        cGom (float): shape parameter of the Gompertz distribution,
        the normal range: [0.5, 5] outside this boundary simulation
        time becomes very long
        scaleGom (float): scale parameter of Gompertz distribution,
        normal range: [20, 50] outside this boundary simulation
        time becomes very long
        switchT (int): the time (assuming the beginning of experiment is 0) that
        we want to switch to the new set of parameters of distributions.
        bern2 (float): second Bernoulli distribution parameter.
        cG2 (float): second shape parameter of Gompertz distribution
        scaleG2 (float): second scale parameter of Gompertz distrbution
        FOM (str): this determines the type of distribution we want to use for
        lifetime here it is either "G": Gompertz, or "E": Exponential.
        betaExp (float): the parameter of Exponential distribution
        betaExp2 (float): second parameter of Exponential distribution

    Returns:
        ----------
        lineage (list): A list of objects (cells) that creates the tree.
    """

    # create an empty lineage
    lineage = []

    # initialize the list with cells
    for ii in range(initCells):
        lineage.append(CellNode(startT=0, linID=ii))

    # have cell divide/die according to distribution
    for cell in lineage:   # for all cells (cap at numCells)
        if cell.isUnfinished():
            if switchT and cell.startT > switchT:  # when the cells should abide by the second set of parameters
                cell.true_state = 1
                if FOM == 'G':
                    cell.tau = sp.gompertz.rvs(cG2, scale=scaleG2)
                elif FOM == 'E':
                    cell.tau = sp.expon.rvs(scale=betaExp2)

            else:  # use first set of parameters for non-heterogeneous lineages or before the switch time
                cell.true_state = 0
                if FOM == 'G':
                    cell.tau = sp.gompertz.rvs(cGom, scale=scaleGom)
                elif FOM == 'E':
                    cell.tau = sp.expon.rvs(scale=betaExp)
            cell.endT = cell.startT + cell.tau
            if cell.endT < experimentTime:  # determine fate only if endT is within range
                # assign cell fate
                if switchT is not None and cell.startT > switchT:  # when the cells should abide by the second set of parameters
                    cell.fate = sp.bernoulli.rvs(bern2)
                else:  # use first set of parameters for non-heterogeneous lineages or before the switch time
                    cell.fate = sp.bernoulli.rvs(locBern)  # assign fate
                # divide or die based on fate
                if cell.fate:
                    temp1, temp2 = cell.divide(cell.endT)  # cell divides
                    # append to list
                    lineage.append(temp1)
                    lineage.append(temp2)
                else:
                    cell.die(cell.endT)
            else:  # if the endT is past the experimentTime
                cell.tauFake = experimentTime - cell.startT
                cell.setUnfinished()  # reset cell to be unfinished and move to next cell

    # return the list at end
    return lineage


def doublingTime(initCells, locBern, cGom, scaleGom, FOM='G', betaExp=None):
    """
    Calculates the doubling time of a homogeneous cell population,
    given the three parameters and an initial cell count.

    Using the 'generateLineageWithTime' function, it generates a lineage,
    and keeps track of the number of alive cells during the experiment time, then
    fits an Exponential curve to it, and finds the parameter of this exponential (lambda).

    the doubling time == ln(2)/lambda

    Args:
        ----------
        initCells (int): number of initial cells to initiate the tree
        locBern (float): parameter of Bernoulli distribution
        cGom (float): shape parameter of Gompertz distribution
        scaleGom (float): scale parameter of Gompertz distribution
        FOM (str): either 'G': Gompertz, or 'E': Exponential. Decides on the
        distribution for setting the cell's lifetime
        betExp (float): the parameter of Exponential distribution.

    Returns:
        ----------
        doubleT (float): the doubling time of the population of the cells
        in the lineage.

        This function works for homogeneous population of cells yet.

    """
    numAlive = []  # list that stores the number of alive cells for each experimentTime
    experimentTimes = np.logspace(start=0, stop=2, num=49)
    experimentTimes = [0] + experimentTimes

    for experimentTime in experimentTimes:
        lineage = []
        if FOM == 'G':
            lineage = generateLineageWithTime(initCells, experimentTime, locBern, cGom, scaleGom)
        elif FOM == 'E':
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

    fitExpParam, _ = optimize.curve_fit(expFunc, experimentTimes, numAlive, p0=[0])  # fit an exponential curve to generated data

    doubleT = np.log(2) / fitExpParam[0]  # relationship between doubling time and exponential function

    return doubleT
