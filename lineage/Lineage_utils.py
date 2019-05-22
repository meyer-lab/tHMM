'''utility and helper functions for cleaning up input populations and lineages and other needs in the tHMM class'''

import numpy as np
import scipy as sp
from scipy.optimize import root
from scipy.special import logsumexp
from .CellNode import generateLineageWithTime

##------------------------ Generating population of cells ---------------------------##


def generatePopulationWithTime(experimentTime, initCells, locBern, betaExp, switchT=None, bern2=None, betaExp2=None,
                               FOM='E', shape_gamma1=None, scale_gamma1=None, shape_gamma2=None, scale_gamma2=None):
    """
    Generates a population of lineages that abide by distinct parameters.

    This function uses the same parameters as "GenerateLineageWithTime", along with
    experimentTime to create a population of cells including different lineages.
    The number of lineages would be the same as the number of initial cells we start with,
    Every initial cell and its descendants have the same linID.

    Args:
        ----------
        experimentTime (int): The duration time of experiment which the cells will
        continue growing.

        initCells (int): the number of initial cells to initiate the tree with.

        experimentTime (int) [hours]: the time that the experiment will be running
        to allow for the cells to grow.

        locBern (float): the Bernoulli distribution parameter
        (p = success) for fate assignment (either the cell dies or divides)
        range = [0, 1]

        betaExp (float): the parameter of Exponential distribution.

        switchT (int): the time (assuming the beginning of experiment is 0) that
        we want to switch to the new set of parameters of distributions.

        bern2 (float): second Bernoulli distribution parameter.

        FOM (str): this determines the type of distribution we want to use for
        lifetime here it is either "G": Gompertz, or "E": Exponential.

        betaExp2 (float): the parameter of Exponential distribution for the second
        population's distribution.

    Returns:
        ----------
        population (list): a list of objects that contain cells.

    """

    assert len(initCells) == len(locBern)   # make sure all lists have same length
    numLineages = len(initCells)
    population = []

    if switchT is None:  # when there is no heterogeneity over time
        for ii in range(numLineages):
            if FOM == 'E':
                temp = generateLineageWithTime(initCells[ii], experimentTime, locBern[ii], betaExp=betaExp[ii], FOM='E')
            elif FOM == 'Ga':
                temp = generateLineageWithTime(initCells[ii], experimentTime, locBern[ii], betaExp=betaExp[ii], FOM='Ga', shape_gamma1=shape_gamma1[ii], scale_gamma1=scale_gamma1[ii])
            for cell in temp:
                sum_prev = 0
                j = 0
                while j < ii:
                    sum_prev += initCells[j]
                    j += 1
                cell.linID += sum_prev  # shift the lineageID so there's no overlap with populations of different parameters
                cell.true_state = 0
                population.append(cell)  # append all individual cells into a population
    else:  # when the second set of parameters is defined
        for ii in range(numLineages):
            if FOM == 'E':
                temp = generateLineageWithTime(initCells[ii], experimentTime, locBern[ii], betaExp=betaExp[ii], switchT=switchT, bern2=bern2[ii], betaExp2=betaExp2[ii], FOM='E')
            elif FOM == 'Ga':
                temp = generateLineageWithTime(
                    initCells[ii],
                    experimentTime,
                    locBern[ii],
                    betaExp=betaExp[ii],
                    switchT=switchT,
                    bern2=bern2[ii],
                    betaExp2=betaExp2[ii],
                    FOM='Ga',
                    shape_gamma1=shape_gamma1[ii],
                    scale_gamma1=scale_gamma1[ii],
                    shape_gamma2=shape_gamma2[ii],
                    scale_gamma2=scale_gamma2[ii])
            # create a temporary lineage
            for cell in temp:
                sum_prev = 0
                j = 0
                while j < ii:
                    sum_prev += initCells[j]
                    j += 1
                cell.linID += sum_prev  # shift the lineageID so there's no overlap with populations of different parameters
                population.append(cell)  # append all individual cells into a population

    return population

##-------------------------------- Removing Unfinished Cells -------------------------##


def remove_unfinished_cells(X):
    """
    Removes unfinished cells in Population and root cells with no daughters.
     This Function checks every object in the list and if it includes NaN, then
    it replaces the cell with None which essentially removes the cell, and returns
    the new list of cells that does not inclue any NaN.
     Args:
        ----------
        X (list): list that holds cells as objects.
     Returns:
        ----------
        X (list): a list of objects (cells) in which the NaNs have been removed.
     """
    ii = 0  # establish a count outside of the loop
    while ii in range(len(X)):  # for each cell in X
        if X[ii].isUnfinished():  # if the cell has NaNs in its times
            if X[ii].parent is None:  # do nothing if the parent pointer doesn't point to a cell
                pass
            elif X[ii].parent.left is X[ii]:  # if it is the left daughter of the parent cell
                X[ii].parent.left = None  # replace the cell with None
            elif X[ii].parent.right is X[ii]:  # or if it is the right daughter of the parent cell
                X[ii].parent.right = None  # replace the cell with None
            X.pop(ii)  # pop the unfinished cell at the current position
        else:
            ii += 1  # only move forward in the list if you don't delete a cell
    return X

##------------------------- Removing Singleton Lineages ---------------------------##


def remove_singleton_lineages(X):
    """
    Removes lineages that are only a single root cell that does not divide or just dies

    Args:
        ----------
        X (list): list that holds cells as objects.

    Returns:
        ----------
        X (list): a list of objects (cells) in which the root cells that do not
        make a lineage, have been removed.

    """
    ii = 0
    while ii in range(len(X)):  # for each cell in X
        if X[ii].isRootParent():  # if the cell is a root parent
            if X[ii].left is None and X[ii].right is None:
                X.pop(ii)  # pop the unfinished cell at the current position
            else:
                ii += 1
        else:
            ii += 1  # only move forward in the list if you don't delete a cell
    return X

##------------------------ Find the number of Lineages ---------------------------##


def get_numLineages(X):
    """
    Outputs total number of cell lineages in a given Population.

    This function first removes those initial cells that do no make any lineages,
    and then keeps track of the cells that are root, and counts the number of them

    Args:
        ----------
        X (list): list of objects (cells)

    Returns:
        ----------
        numLineages (int): the number of lineages in the given population

    """
    X = remove_singleton_lineages(X)
    root_cell_holder = []  # temp list to hold the root cells in the population
    root_cell_linID_holder = []  # temporary list to hold all the linIDs of the root cells in the population
    for cell in X:  # for each cell in the population
        if cell.isRootParent():
            root_cell_holder.append(cell)
            root_cell_linID_holder.append(cell.linID)  # append the linID of each cell
    assert len(root_cell_holder) == len(root_cell_linID_holder), "Something wrong with your unique number of lineages. Check the number of root cells and the number of lineages in your data."
    numLineages = len(root_cell_holder)  # the number of lineages is the number of root cells
    return numLineages

##---------------------- creating a population out of lineages -------------------##


def init_Population(X, numLineages):
    """
    Creates a full population list of lists which contain each lineage in the Population.

    This function first removes the singleton cells, then finds the root cells,
    and tracks their lineage, and puts them in a list, then appends the list of
    lineages into another list to make the whole population.

    Args:
        ---------
        X (list): a list of objects (cells)
        numLineages (int): the number of lineages which is essentially the number
        of initial cells

    Returns:
        ---------
        population (list): a list of lists -- a list of lineages including cells
    """
    X = remove_singleton_lineages(X)
    root_cell_holder = []  # temp list to hold the root cells in the population
    for cell in X:  # for each cell in the population
        if cell.isRootParent() and cell.isParent:
            root_cell_holder.append(cell)
    population = []
    for lineage_num in range(numLineages):  # iterate over the number of lineages in the population
        temp_lineage = []  # temporary list to hold the cells of a certain lineage with a particular linID
        for cell in X:  # for each cell in the population
            if cell.get_root_cell() is root_cell_holder[lineage_num]:  # if the cell's root cell is the root cell we're on
                assert cell.linID == cell.get_root_cell().linID, "Your root cells have a different lineage ID than the lineages they are associated with. Check the number of root cells and the number of lineages in your data."
                temp_lineage.append(cell)  # append the cell to that certain lineage
        if len(temp_lineage) > 1:  # want to avoid lineages with <= 1 cell
            population.append(temp_lineage)  # append the lineage to the Population holder
    return population

##-------------------------Estimating Bernoulli Parameter -------------------------##


def bernoulliParameterEstimatorAnalytical(X):
    """
    Estimates the Bernoulli parameter for a given population using MLE analytically.

    This function keeps track of the number of cells that have been divided and
    died by appending 1s and 0s, respectively. Then it calculates the average
    number of times that cells have divided. This will be the success rate (p)
    in Bernoulli distribution in an analytical way.


    Args:
        ---------
        X (list): a list of objects (cells)

    Returns:
        ---------
        result (float): the success probability or the Bernoulli parameter

    """
    fate_holder = []  # instantiates list to hold cell fates as 1s or 0s
    for cell in X:  # go through every cell in the population
        if not cell.isUnfinished():  # if the cell has lived a meaningful life and matters
            fate_holder.append(cell.fate * 1)  # append 1 for dividing, and 0 for dying

    result = (sum(fate_holder) + 1e-10) / (len(fate_holder) + 2e-10)  # add up all the 1s and divide by the total length (finding the average)

    return result

##--------------------- Estimating Exponential Parameter ----------------------##


def exponentialAnalytical(X):
    """
    Estimates the Exponential beta parameter for a given population using MLE analytically

    Args:
        ----------
        X (list): list of objects (cells)

    Returns:
        result (float): average of the lifetime of those cells that have been lived and divided

    In this function to avoid getting weird results, 62.5 as an offset has been added
    to the sum (in averaging), and to avoid getting inf due to zeros at denomenator, 1 has been added.

    """
    # create list of all our taus
    tau_holder = []
    tauFake_holder = []
    for cell in X:  # go through every cell in the population
        if not cell.isUnfinished():  # if the cell has lived a meaningful life and matters
            tau_holder.append(cell.tau)  # append the cell lifetime
        elif cell.isUnfinished():
            tauFake_holder.append(cell.tauFake)

    result = (sum(tau_holder) + sum(tauFake_holder) + 62.5) / (len(tau_holder) + 1)

    return result

##------------------ Estimating Gamma Distribution Parameters --------------------##


def gammaAnalytical(X):
    """
    An analytical estimator for two parameters of the Gamma distribution. Based on Thomas P. Minka, 2002 "Estimating a Gamma distribution".

    The likelihood function for Gamma distribution is:
    p(x | a, b) = Gamma(x; a, b) = x^(a-1)/(Gamma(a) * b^a) * exp(-x/b)
    Here we intend to find "a" and "b" given x as a sequence of data -- in this case
    the data is the cells' lifetime.
    To find the best estimate we find the value that maximizes the likelihood function.

    b_hat = x_bar / a
    using Newton's method to find the second parameter:
    a_hat =~ 0.5 / (log(x_bar) - log(x)_bar)

    Here x_bar means the average of x.

    Args:
        ----------
        X (obj): The object holding cell's attributes, including lifetime, to be used as data.

    Returns:
        ----------
        a_hat (float): The estimated value for shape parameter of the Gamma distribution
        b_hat (float): The estimated value for scale parameter of the Gamma distribution
    """

    # store the lifetime of every cell in a list, only if it is finished by the end of the experiment
    tau1 = []
    for cell in X:
        if not cell.isUnfinished():
            tau1.append(cell.tau)

    tau_mean = np.mean(tau1)
    tau_logmean = np.log(tau_mean)
    tau_meanlog = np.mean(np.log(tau1))

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


##------------------------------ Select the population up to some time point -----------------------------------##

def select_population(lineage, experimentTime):
    
    leaf_taus = []
    new_population = []

    for cell in lineage:
        if cell.isLeaf():
            leaf_taus.append(cell.tau)

    intended_interval = max(leaf_taus) + 1
    intended_end_time = experimentTime - intended_interval

    for cell in lineage:
        if cell.startT <= intended_end_time:
            new_population.append(cell)

    for cell in new_population:
        assert cell.startT <= intended_end_time, "Something is wrong in aquiring cells for intended end time"

    return new_population



