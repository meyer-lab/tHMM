'''utility and helper functions for cleaning up input populations and lineages and other needs in the tHMM class'''

import numpy as np
import scipy.stats as sp
from scipy.optimize import minimize
from .CellNode import generateLineageWithTime

def generatePopulationWithTime(experimentTime, initCells, locBern, cGom, scaleGom):
    ''' generates a population of lineages that abide by distinct parameters. '''

    assert len(initCells) == len(locBern) == len(cGom) == len(scaleGom) # make sure all lists have same length
    numLineages = len(initCells)
    population = [] # create empty list

    for ii in range(numLineages):
        temp = generateLineageWithTime(initCells[ii], experimentTime, locBern[ii], cGom[ii], scaleGom[ii]) # create a temporary lineage
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

def get_numLineages(Y):
    ''' Outputs total number of cell lineages in given Population. '''
    X = remove_NaNs(Y)
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
    X = remove_NaNs(X)
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
        if len(temp_lineage)>1: # want to avoid lineages with <= 1 cell
            population.append(temp_lineage) # append the lineage to the Population holder
    return population

def bernoulliParameterEstimatorAnalytical(X):
    '''Estimates the Bernoulli parameter for a given population using MLE analytically'''
    fate_holder = [1] # instantiates list to hold cell fates as 1s or 0s
    for cell in X: # go through every cell in the population
        if not cell.isUnfinished(): # if the cell has lived a meaningful life and matters
            fate_holder.append(cell.fate*1) # append 1 for dividing, and 0 for dying

    return (sum(fate_holder) + 1)/ (len(fate_holder) + 2) # add up all the 1s and divide by the total length (finding the average)

def gompertzParameterEstimatorNumerical(X):
    '''Estimates the Gompertz parameters for a given population using MLE numerically'''
    tau_holder = [20] # instantiates list with a dummy cell
    for cell in X: # go through every cell in the population
        if not cell.isUnfinished(): # if the cell has lived a meaningful life and matters
            tau_holder.append(cell.tau) # append the cell lifetime

    def negLogLikelihoodGomp(gompParams, tau_holder):
        """ Calculates the log likelihood for gompertz. """
        return -1*np.sum(sp.gompertz.logpdf(x=tau_holder,c=gompParams[0], scale=gompParams[1]))

    res = minimize(negLogLikelihoodGomp, x0=[2,40], bounds=((0,10),(0,100)), method="SLSQP", options={'maxiter': 1e7}, args=(tau_holder))

    return res.x
