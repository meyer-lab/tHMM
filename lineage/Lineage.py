""" author : shakthi visagan (shak360), adam weiner (adamcweiner)
description: a file to hold the population class and functions that simulate a population """

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

class Population:
    """ This class holds populations of cells and estimates the parameters of how they behave. """
    def __init__(self, experimentTime, initCells, locBern, cGom, scaleGom):
        """ Builds the appropriate population according to option and args. """
        self.group = generatePopulationWithTime(experimentTime, initCells, locBern, cGom, scaleGom)

    def loadPopulation(self, csv_file):
        """ Write a function that imports a population from an external file. """
        pass

    def plotPopulation(self):
        ''' Write a function that plots a population whether it was imported or generated. '''
        pass
