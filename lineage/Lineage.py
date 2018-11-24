# author : shakthi visagan (shak360), adam weiner (adamcweiner)
# description: a file to hold the lineage class and population class

import sys
import math
import numpy as np
import scipy.stats as sp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from .CellNode import CellNode as c, generateLineageWithTime

def generatePopulationWithTime(experimentTime, initCells, locBern, cGom, scaleGom):
    ''' generates a population of lineages that abide by distinct parameters. '''

    assert(len(initCells) == len(locBern) == len(cGom) == len(scaleGom)) # make sure all lists have same length
    numLineages = len(initCells)
    population = [] # create empty list

    for ii in range(numLineages):
        temp = generateLineageWithTime(initCells[ii], experimentTime, locBern[ii], cGom[ii], scaleGom[ii]) # create a temporary lineage
        for cell in temp:
            population.append(cell) # append all individual cells into a population

    return(population)


class Population:
    def __init__(self, experimentTime, initCells, locBern, cGom, scaleGom):
        """ Builds the appropriate population according to option and args. """
        self.group = generatePopulationWithTime(experimentTime, initCells, locBern, cGom, scaleGom)

    def loadPopulation(self, csv_file):
        #TODO: write function to import a population from external file
        pass

    def plotPopulation(self):
        '''plots a population growth'''
        #TODO
        pass

    def bernoulliParameterEstimatorAnalytical(self):
        '''Estimates the Bernoulli parameter for a given population using MLE analytically'''
        population = self.group # assign population to a variable
        fate_holder = [] # instantiates list to hold cell fates as 1s or 0s
        for cell in population: # go through every cell in the population
            if not cell.isUnfinished(): # if the cell has lived a meaningful life and matters
                fate_holder.append(cell.fate*1) # append 1 for dividing, and 0 for dying

        return ( sum(fate_holder) / len(fate_holder) ) # add up all the 1s and divide by the total length (finding the average)

    def bernoulliParameterEstimatorNumerical(self):
        '''Estimates the Bernoulli parameter for a given population using MLE numerically'''
        population = self.group # assign population to a variable
        fate_holder = [] # instantiates list to hold cell fates as 1s or 0s
        for cell in population: # go through every cell in the population
            if not cell.isUnfinished(): # if the cell has lived a meaningful life and matters
                fate_holder.append(cell.fate*1) # append 1 for dividing, and 0 for dying

        def LogLikelihoodBern(locBernGuess, fate_holder):
            """ Calculates the log likelihood for bernoulli. """
            return(np.sum(sp.bernoulli.logpmf(k=fate_holder, p=locBernGuess)))

        nllB = lambda *args: -LogLikelihoodBern(*args)

        res = minimize(nllB, x0=0.5, bounds=((0,1),), method="SLSQP", args=(fate_holder))

        return(res.x[0])

    def gompertzParameterEstimatorNumerical(self):
        '''Estimates the Gompertz parameters for a given population using MLE numerically'''
        population = self.group # assign population to a variable
        tau_holder = [] # instantiates list
        for cell in population: # go through every cell in the population
            if not cell.isUnfinished(): # if the cell has lived a meaningful life and matters
                tau_holder.append(cell.tau) # append the cell lifetime

        def LogLikelihoodGomp(gompParams, tau_holder):
            """ Calculates the log likelihood for gompertz. """
            return(np.sum(sp.gompertz.logpdf(x=tau_holder,c=gompParams[0], scale=gompParams[1])))

        nllG = lambda *args: -LogLikelihoodGomp(*args)

        res = minimize(nllG, x0=[1,1e3], bounds=((0,5),(0,None)), method="SLSQP", options={'maxiter': 1e6}, args=(tau_holder))

        return(res.x)
