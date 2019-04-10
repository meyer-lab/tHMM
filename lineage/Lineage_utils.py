'''utility and helper functions for cleaning up input populations and lineages and other needs in the tHMM class'''

import numpy as np
from scipy.optimize import root
from .CellNode import generateLineageWithTime

##------------------------ Generating population of cells ---------------------------##


def generatePopulationWithTime(experimentTime, initCells, locBern, cGom, scaleGom, switchT=None, bern2=None, cG2=None, scaleG2=None, FOM='G', betaExp=None, betaExp2=None):
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

        cGom (float): shape parameter of the Gompertz distribution,
        the normal range: [0.5, 5] outside this boundary simulation
        time becomes very long.

        scaleGom (float): scale parameter of Gompertz distribution,
        normal range: [20, 50] outside this boundary simulation
        time becomes very long.

        switchT (int): the time (assuming the beginning of experiment is 0) that
        we want to switch to the new set of parameters of distributions.

        bern2 (float): second Bernoulli distribution parameter.

        cG2 (float): second shape parameter of Gompertz distribution.

        scaleG2 (float): second scale parameter of Gompertz distrbution.

        FOM (str): this determines the type of distribution we want to use for
        lifetime here it is either "G": Gompertz, or "E": Exponential.

        betaExp (float): the parameter of Exponential distribution.

        betaExp2 (float): second parameter of Exponential distribution.

    Returns:
        ----------
        population (list): a list of objects that contain cells.

    """

    assert len(initCells) == len(locBern) == len(cGom) == len(scaleGom)  # make sure all lists have same length
    numLineages = len(initCells)
    population = []

    if switchT is None:  # when there is no heterogeneity over time
        for ii in range(numLineages):
            if FOM == 'G':
                temp = generateLineageWithTime(initCells[ii], experimentTime, locBern[ii], cGom[ii], scaleGom[ii], FOM='G')
            elif FOM == 'E':
                temp = generateLineageWithTime(initCells[ii], experimentTime, locBern[ii], cGom[ii], scaleGom[ii], FOM='E', betaExp=betaExp[ii])
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
            if FOM == 'G':
                temp = generateLineageWithTime(initCells[ii], experimentTime, locBern[ii], cGom[ii], scaleGom[ii], switchT, bern2[ii], cG2[ii], scaleG2[ii], FOM='G')
            elif FOM == 'E':
                temp = generateLineageWithTime(initCells[ii], experimentTime, locBern[ii], cGom[ii], scaleGom[ii], switchT,
                                               bern2[ii], cG2[ii], scaleG2[ii], FOM='E', betaExp=betaExp[ii], betaExp2=betaExp2[ii])
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

##-------------------------------- Removing NaNs -----------------------------------##


def remove_NaNs(X):
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

##-------------------------Estimating Gompertz Parameter -------------------------##


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
    for cell in X:  # go through every cell in the population
        if not cell.isUnfinished():  # if the cell has lived a meaningful life and matters
            tau_holder.append(cell.tau)  # append the cell lifetime
        elif cell.isUnfinished():
            tauFake_holder.append(cell.tauFake)

    N = len(tau_holder) + len(tauFake_holder)  # number of cells
    D = 1
    if N != 0:
        D = len(tau_holder) / N
    total_tau_holder = tau_holder + tauFake_holder
    delta_holder = [1] * len(tau_holder) + [0] * len(tauFake_holder)

##------------------Helper functions for gompertzAnalytical---------------------##
    def help_exp(b):
        """
        Returns an expression commonly used in the analytical solution.

        Here the function is:
            helper_exp(b) = sum ( exp(b*X_i) )
        in which X_i is the cell's lifetime (tau)

        Args:
            ---------
            b (float): the coefficient of X_i s in exponential function

        Returns:
            ---------
            sum(temp): which is a float number and is the sum over all exp(b*X_i)
            in which tau == (X_i)

        """
        temp = []
        for ii in range(N):
            temp.append(np.exp(b * total_tau_holder[ii]))
        return sum(temp)

    def left_term(b):
        """ Returns one of the two expressions used in the MLE for b.

        the expression this function calculates is:
            left_term(b) = sum[(D * exp(b * X_i) * X_i) / (1/n * sum[exp(b * X_i)]) - 1]

        Args:
            ---------
            b (float): the coefficient of X_i s in exponential function

        Returns:
            ---------
            sum(temp): it returns the expression written above (left_term(b))

        """
        temp = []
        denom = (help_exp(b) / N) - 1.0  # denominator is not dependent on ii
        for ii in range(N):
            numer = D * total_tau_holder[ii] * np.exp(b * total_tau_holder[ii])
            temp.append(numer / denom)
        return sum(temp)

    def right_term(b):
        """
        Returns the other expression used in the MLE for b.

        right_term(b) = sum[(D * (exp(b * X_i) - 1))/(b/n * sum(exp(b * X_i)) - b) + delta_i * X_i]

        Args:
            ----------
            b (float): the coefficient of X_i s in exponential function

        Returns:
            ----------
            sum(temp): it returns the expression written above (right_term(b))

        """
        temp = []
        denom = ((b / N) * help_exp(b)) - b
        for ii in range(N):
            numer = D * (np.exp(b * total_tau_holder[ii]) - 1.0)
            temp.append((numer / denom) + delta_holder[ii] * total_tau_holder[ii])
        return sum(temp)

    def error_b(scale):
        """
        Returns the difference between right_term(b) and left_term(b).

        To find the maximum likelihood estimate for b, the error between the two functions
        left_term(b) and right_term(b) is calculated.
        In this case b = 1/scale.

        Args:
            ---------
            scale (float): is the scale parameter of teh Gompertz distribution

        Returns:
            ---------
            error (float): is the difference between the two mentioned expressions.

        """
        error = left_term(1. / scale) - right_term(1. / scale)

        return error

    result = [2, 62.5]  # dummy estimate
    if N != 0:
        #res = minimize(error_b, x0=[(45.)], method="Nelder-Mead", options={'maxiter': 1e10})
        res = root(error_b, x0=result[1])
        b = 1. / (res.x)
        # solve for a in terms of b
        a = D * b / ((help_exp(b) / N) - 1.0)

        # convert from their a and b to our cGom and scale
        c = a / b
        scale = res.x
        result = [c, scale]  # true estimate with non-empty sequence of data

    return result
