'''utility and helper functions for cleaning up input populations and lineages and other needs in the tHMM class'''

import math
import numpy as np
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



