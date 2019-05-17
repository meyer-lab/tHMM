""" author : shakthi visagan (shak360), adam weiner (adamcweiner)
some changes: Farnaz Mohammadi
description: a file to hold the cell class
"""
import math
import numpy as np
from scipy import optimize, stats as sp


class CellNode:
    """Each cell in our tree will consist of a node containing these traits.

    This class includes many functions that assign the cell its properties, i.e.,
    the cell's generation, lifetime, true_state, etc."""

    def __init__(self, gen=1, linID=0, startT=0, endT=float('nan'), fate=None, left=None, right=None, parent=None, true_state=None, g1=float('nan'), g2=float('nan')):
        """
        Args:
        -----
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

        self.true_state = true_state
        self.fateObserved = False
        self.g1 = g1
        self.g2 = g2

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

    def start_G2(self):
        """
        Returns the start time point of cell's G2 phase by adding the start time and the duration of G1.
        """
        return self.startT + self.g1

    def die(self, endT):
        """
        Cell dies without dividing.

        If the cell dies, the endTime is reached so we calculate the lifetime (tau)
        and the death is observed.
        Args:
        -----
            endT (float): end time of the cell.

        This function doesn't return
        """
        self.fate = False   # no division
        self.endT = endT    # mark endT
        self.calcTau()      # calculate Tau when cell dies
        self.fateObserved = True  # this cell has truly died

    def divide(self, endT):
        """
        Cell life ends through division. The two optional trackID arguments
        represents the trackIDs given to the two daughter cells.

        Args:
        -----
            endT (float): end time of the cell

        kwargs:
            trackID_d1 & trackID_d2: since trackID is an attribute of experimental data,
            so when we generate the lineage we need to assing "None" to the trackID
            of the left and right daughter cell

        Returns:
        --------
            self.left (obj): left new born daughter cell
            self.right (obj): right new born daughter cell

        """
        self.endT = endT
        self.fate = True    # division
        self.calcTau()      # calculate Tau when cell dies
        self.fateObserved = True  # this cell has truly divided

        if self.isRootParent():

            self.left = CellNode(gen=self.gen + 1, linID=self.linID, startT=endT, parent=self, true_state=self.true_state)
            self.right = CellNode(gen=self.gen + 1, linID=self.linID, startT=endT, parent=self, true_state=self.true_state)
        else:
            self.left = CellNode(gen=self.gen + 1, linID=self.linID, startT=endT, parent=self, true_state=self.true_state)
            self.right = CellNode(gen=self.gen + 1, linID=self.linID, startT=endT, parent=self, true_state=self.true_state)

        return (self.left, self.right)

    def get_root_cell(self):
        """
        Gets the root cell associated with the cell.

        it keeps going up to the root by jumping from daughter cells to their
        mother cells by keeping track of their lineage ID (linID) until it reaches
        the first generation and then returns the last hold cell which is the
        ancestor of the lineage.

        Returns:
        --------
            curr_cell (obj): the ancestor cell if a lineage

        """
        cell_linID = self.linID
        curr_cell = self
        while curr_cell.gen > 1:
            curr_cell = curr_cell.parent
            assert cell_linID == curr_cell.linID
        assert cell_linID == curr_cell.linID
        return curr_cell


##----------------------------------- Create Lineage ------------------------------------##

def generateLineageWithTime(initCells, experimentTime, locBern, g1_a=None, g1_b=None, g2_a=None, g2_b=None):
    """
    generates a list of objects (cells) in a lineage.

    Given an experimental end time, a Bernoulli distribution for
    dividing/dying and two Gamma parameters for cell lifetime,
    it creates objects as cells and puts them in a list.


    Bernoulli distribution:
        It has one parameter (locBern)
        If locBern = 0.80 then 80% of the times the cells will divide and 20%
        of the times they die; meaning, for every cell that is generated,
        a random number (either 1:divide, or 0:die) is picked from the
        distribution with this parameter and the fate of the cell is assigned.


    Gamma distribution:
        It has two parameters(shape , scale) {here are a and b}
        Used as alifetime generator for cells. Here to generate the cells we
        specify the two parameters and it will return a number that we assign to cell's
        lifetime, as G1 phase and G2 phase of the cell cycle.

    Args:
    -----
        initCells (int): the number of initial cells to initiate the tree with
        experimentTime (int) [hours]: the time that the experiment will be running
        to allow for the cells to grow
        locBern (float): the Bernoulli distribution parameter
        (p = success) for fate assignment (either the cell dies or divides)
        range = [0, 1]
        g1_a, g2_a: shape parameters of Gamma for G1 and G2 phase of the cell cycle.
        g1_b, g2_b: scale parameters of Gamma for G1 and G2 phase of the cell cycle.

    Returns:
    --------
        lineage (list): A list of objects (cells) that creates the tree.
    """

    # create an empty lineage
    lineage = []

    # initialize the list with cells
    for ii in range(initCells):
        lineage.append(CellNode(startT=0, linID=ii))

    # have cell divide/die according to distribution
    for cell in lineage:
        cell.g1 = sp.gamma.rvs(g1_a, scale = g1_b)
        cell.g2 = sp.gamma.rvs(g2_a, scale = g2_b)
        
        if cell.isUnfinished():
            cell.tau = cell.g1 + cell.g2
            cell.endT = cell.startT + cell.tau
            
            if cell.endT < experimentTime:   # determine fate only if endT is within range
                # assign cell fate
                cell.fate = sp.bernoulli.rvs(locBern)  # assign fate
                
                # divide or die based on fate
                if cell.fate:
                    temp1, temp2 = cell.divide(cell.endT)  # cell divides
                    # append the children to the list
                    lineage.append(temp1)
                    lineage.append(temp2)
                else:
                    cell.die(cell.endT)

    return lineage

##------------------------- How many cells are in G1 or G2? --------------------------------##

def inG1_or_G2(X, time):
    """
    This function determines whether the cell is in G1 phase or in G2 phase.
    
    Args:
    -----
        X (list): is the lineage, a list of objects representing cells.
        time (list): a list -- could be np.linspace() -- including time points of 
        duration of the time experiment is being conducted. 

    Returns:
    --------
        num_G1 (list):  a list of # of cells in G1 at each time point
        num_G2 (list):  a list of # of cells in G2 at each time point
        num_cell (list):  a list of total # of cells at each time point
    """
    
    num_G1=[]
    num_G2=[]
    num_cell=[]
    
    for t in time:
        count_G1 = 0
        count_G2 = 0
        count_numCell = 0
        
        for cell in X:
            g2=cell.start_G2()
            
            # if the time point is between the cell's start time and the end of G1 phase, then count it as being in G1.
            if cell.startT <= t <= g2:
                count_G1+=1

            # if the time point is between the start of the cell's G1 phase and the end time, then count it as being in G2.
            if g2 <= t <= cell.endT:
                count_G2+=1
    
            # if the time point is within the cell's lifetime, count it as being alive.
            if cell.startT <= t <= cell.endT:
                count_numCell+=1
                
        num_G1.append(count_G1)
        num_G2.append(count_G2)
        num_cell.append(count_numCell)
        
    return num_G1, num_G2, num_cell


##--------------------- Separate different lineages by their root parent -------------------------##

def separate_pop(numLineages, X):
    """
    This function separates each lineage by their root parent, using their linID.

    Args:
    -----
        numLineages (int): the number of lineages, which here basically is the number of initial cells.
        X (list): is the lineage, a list of objects representing cells.

    Returns:
    --------
        population (list of lists): a list that holds lists of cells that belong tothe same parent.
    """
    

    population = []
    for i in range(numLineages):
        list_cell = []

        for cell in X:
            if cell.linID == i:
                list_cell.append(cell)
        population.append(list_cell)
            
    return population

##------------------------------- Estimate parameters analytically ----------------------------------##

def GAnalytical(g):  # for G1 and G2
    """
    This function estimates two parameters of Gamma distribution 'a' and 'b' analytically, given data.

    Args:
    -----
        g (1D np.array): an array holding the data points

    Returns:
    --------
        a_hat_new (float): estimated shape parameter of Gamma distribution.
        b_hat_new (float): estimated shape parameter of Gamma distriubtion.
    """

    # calculate required mean and log_mean of data
    tau_mean = np.mean(g)
    tau_logmean = np.log(tau_mean)
    tau_meanlog = np.mean(np.log(g))

    # initialization step
    a_hat0 = 0.5 / (tau_logmean - tau_meanlog)  # shape
    b_hat0 = tau_mean / a_hat0  # scale
    psi_0 = np.log(a_hat0) - 1 / (2 * a_hat0)  # psi is the derivative of log of gamma function, which has been approximated as this term
    psi_prime0 = 1 / a_hat0 + 1 / (a_hat0 ** 2)  # this is the derivative of psi
    assert a_hat0 != 0, "the first parameter has been set to zero!"

    # updating the parameters
    for i in range(100):
        a_hat_new = (a_hat0 * (1 - a_hat0 * psi_prime0)) / (1 - a_hat0 * psi_prime0 + tau_meanlog - tau_logmean + np.log(a_hat0) - psi_0)
        b_hat_new = tau_mean / a_hat_new

        a_hat0 = a_hat_new
        psi_prime0 = 1 / a_hat0 + 1 / (a_hat0 ** 2)
        psi_0 = np.log(a_hat0) - 1 / (2 * a_hat0)
        psi_prime0 = 1 / a_hat0 + 1 / (a_hat0 ** 2)

        if np.abs(a_hat_new - a_hat0) <= 0.01:
            return [a_hat_new, b_hat_new]
        else:
            pass

    assert np.abs(a_hat_new - a_hat0) <= 0.01, "a_hat has not converged properly, a_hat_new - a_hat0 = {}".format(np.abs(a_hat_new - a_hat0))

    result = [a_hat_new, b_hat_new]
    return result
