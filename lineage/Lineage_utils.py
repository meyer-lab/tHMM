'''utility and helper functions for cleaning up input populations and lineages and other needs in the tHMM class'''

import numpy as np
from scipy.optimize import root
from .CellNode import generateLineageWithTime

def generatePopulationWithTime(experimentTime, initCells, locBern, cGom, scaleGom, switchT=None, bern2=None, cG2=None, scaleG2=None, FOM='G', betaExp=None, betaExp2=None):
    ''' generates a population of lineages that abide by distinct parameters. '''

    assert len(initCells) == len(locBern) == len(cGom) == len(scaleGom) # make sure all lists have same length
    numLineages = len(initCells)
    population = [] # create empty list

    if switchT is None: # when there is no heterogeneity over time
        for ii in range(numLineages):
            if FOM == 'G':
                temp = generateLineageWithTime(initCells[ii], experimentTime, locBern[ii], cGom[ii], scaleGom[ii], FOM='G') # create a temporary lineage
            elif FOM == 'E':
                temp = generateLineageWithTime(initCells[ii], experimentTime, locBern[ii], cGom[ii], scaleGom[ii], FOM='E', betaExp=betaExp[ii]) # create a temporary lineage
            for cell in temp:
                sum_prev = 0
                j = 0
                while j < ii:
                    sum_prev += initCells[j]
                    j += 1
                cell.linID += sum_prev # shift the lineageID so there's no overlap with populations of different parameters
                cell.true_state = 0
                population.append(cell) # append all individual cells into a population
    else: # when the second set of parameters is defined
        for ii in range(numLineages):
            if FOM == 'G':
                temp = generateLineageWithTime(initCells[ii], experimentTime, locBern[ii], cGom[ii], scaleGom[ii], switchT, bern2[ii], cG2[ii], scaleG2[ii], FOM='G')
            elif FOM == 'E':
                print("making a heterogeneous exponential lineage")
                temp = generateLineageWithTime(initCells[ii], experimentTime, locBern[ii], cGom[ii], scaleGom[ii], switchT, bern2[ii], cG2[ii], scaleG2[ii], FOM='E', betaExp=betaExp[ii], betaExp2=betaExp2[ii])
            # create a temporary lineage
            for cell in temp:
                sum_prev = 0
                j = 0
                while j < ii:
                    sum_prev += initCells[j]
                    j += 1
                cell.linID += sum_prev # shift the lineageID so there's no overlap with populations of different parameters
                population.append(cell) # append all individual cells into a population

    return population

def remove_NaNs(X):
    '''Removes unfinished cells in Population and root cells with no daughters'''
    ii = 0 # establish a count outside of the loop
    while ii in range(len(X)): # for each cell in X
        if X[ii].isUnfinished(): # if the cell has NaNs in its times
            if X[ii].parent is None: # do nothing if the parent pointer doesn't point to a cell
                pass
            elif X[ii].parent.left is X[ii]: # if it is the left daughter of the parent cell
                X[ii].parent.left = None # replace the cell with None
            elif X[ii].parent.right is X[ii]: # or if it is the right daughter of the parent cell
                X[ii].parent.right = None # replace the cell with None
            X.pop(ii) # pop the unfinished cell at the current position
        else:
            ii += 1 # only move forward in the list if you don't delete a cell
    return X

def remove_singleton_lineages(X):
    '''Removes lineages that are only a single root cell that does not divide'''
    ii = 0
    while ii in range(len(X)): # for each cell in X
        if X[ii].isRootParent(): # if the cell is a root parent
            if X[ii].left is None and X[ii].right is None:
                X.pop(ii) # pop the unfinished cell at the current position
            else:
                ii += 1
        else:
            ii += 1 # only move forward in the list if you don't delete a cell
    return X

def get_numLineages(X):
    ''' Outputs total number of cell lineages in given Population. '''
    X = remove_singleton_lineages(X)
    root_cell_holder = [] # temp list to hold the root cells in the population
    root_cell_linID_holder = [] # temporary list to hold all the linIDs of the root cells in the population
    for cell in X: # for each cell in the population
        if cell.isRootParent():
            root_cell_holder.append(cell)
            root_cell_linID_holder.append(cell.linID) # append the linID of each cell
    assert len(root_cell_holder) == len(root_cell_linID_holder)
    numLineages = len(root_cell_holder) # the number of lineages is the number of root cells
    return numLineages

def init_Population(X, numLineages):
    '''Creates a full population list of lists which contain each lineage in the Population.'''
    X = remove_singleton_lineages(X)
    root_cell_holder = [] # temp list to hold the root cells in the population
    for cell in X: # for each cell in the population
        if cell.isRootParent() and cell.isParent:
            root_cell_holder.append(cell)
    population = []
    for lineage_num in range(numLineages): # iterate over the number of lineages in the population
        temp_lineage = [] # temporary list to hold the cells of a certain lineage with a particular linID
        for cell in X: # for each cell in the population
            if cell.get_root_cell() is root_cell_holder[lineage_num]: # if the cell's root cell is the root cell we're on
                assert cell.linID == cell.get_root_cell().linID
                temp_lineage.append(cell) # append the cell to that certain lineage
        if len(temp_lineage) > 1: # want to avoid lineages with <= 1 cell
            population.append(temp_lineage) # append the lineage to the Population holder
    return population

def bernoulliParameterEstimatorAnalytical(X):
    '''Estimates the Bernoulli parameter for a given population using MLE analytically'''
    fate_holder = [] # instantiates list to hold cell fates as 1s or 0s
    for cell in X: # go through every cell in the population
        if not cell.isUnfinished(): # if the cell has lived a meaningful life and matters
            fate_holder.append(cell.fate*1) # append 1 for dividing, and 0 for dying

    result = (sum(fate_holder)+1e-10)/ (len(fate_holder)+2e-10) # add up all the 1s and divide by the total length (finding the average)

    return result


def exponentialAnalytical(X):
    '''Estimates the Exponential beta parameter for a given population using MLE analytically'''
     # create list of all our taus
    tau_holder = []
    tauFake_holder = []
    for cell in X: # go through every cell in the population
        if not cell.isUnfinished(): # if the cell has lived a meaningful life and matters
            tau_holder.append(cell.tau) # append the cell lifetime
        elif cell.isUnfinished():
            tauFake_holder.append(cell.tauFake)

    result = (sum(tau_holder) + sum(tauFake_holder) + 50) / (len(tau_holder) + 1)

    return result

def gompertzAnalytical(X):
    """
    Uses analytical solution for one of the two gompertz parameters.
    See Pg. 14 of The Gompertz distribution and Maximum Likelihood Estimation of its parameters - a revision
    by Adam Lenart
    November 28, 2011
    """
    # create list of all our taus
    tau_holder = []
    tauFake_holder = []
    for cell in X: # go through every cell in the population
        if not cell.isUnfinished(): # if the cell has lived a meaningful life and matters
            tau_holder.append(cell.tau) # append the cell lifetime
        elif cell.isUnfinished():
            tauFake_holder.append(cell.tauFake)

    N = len(tau_holder) + len(tauFake_holder) # number of cells
    D = 0.5
    if N != 0:
        D = len(tau_holder)/N
    total_tau_holder = tau_holder+tauFake_holder
    delta_holder = [1]*len(tau_holder) + [0]*len(tauFake_holder)

    def help_exp(b):
        """ Returns an expression commonly used in the analytical solution. """
        temp = []
        for ii in range(N):
            temp.append(np.exp(b*total_tau_holder[ii]))
        return sum(temp)

    def left_term(b):
        """ Returns one of the two expressions used in the MLE for b. """
        temp = []
        denom = (help_exp(b) / N) - 1.0 # denominator is not dependent on ii
        for ii in range(N):
            numer = D * total_tau_holder[ii] * np.exp(b*total_tau_holder[ii])
            temp.append(numer/denom)
        return sum(temp)

    def right_term(b):
        """ Returns the other expression used in the MLE for b. """
        temp = []
        denom = ((b/N) * help_exp(b)) - b
        for ii in range(N):
            numer = D*(np.exp(b*total_tau_holder[ii]) - 1.0)
            temp.append((numer/denom) + delta_holder[ii]*total_tau_holder[ii])
        return sum(temp)

    def error_b(scale):
        """ Returns the square root of the squared error between left and right terms. """
        error = left_term(1./scale) - right_term(1./scale)

        return error

    result = [2., 50.] # dummy estimate
    if N != 0:
        #res = minimize(error_b, x0=[(45.)], method="Nelder-Mead", options={'maxiter': 1e10})
        res = root(error_b, x0=result[1])
        b = 1. / (res.x)
        # solve for a in terms of b
        a = D*b / ((help_exp(b) / N) - 1.0)

        # convert from their a and b to our cGom and scale
        c = a / b
        scale = res.x
        result = [c, scale] # true estimate with non-empty sequence of data

    return result
