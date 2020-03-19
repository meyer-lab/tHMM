"""
File: figure3.py
Purpose: Generates figure 3.
Figure 3 analyzes heterogeneous (2 state), censored (by both time and fate), 
single lineages (no more than one lineage per population)
with at least 16 cells 
over increasing experimental times.
"""
import numpy as np

from .figureCommon import getSetup, subplotLabel, commonAnalyze, figureMaker, pi, T, E, max_desired_num_cells, min_experiment_time, max_experiment_time
from ..LineageTree import LineageTree


def makeFigure():
    """
    Makes figure 3.
    """

    # Get list of axis objects
    ax, f = getSetup((7, 6), (2, 3))

    figureMaker(ax, *accuracy_increased_cells())

    subplotLabel(ax)

    return f


def accuracy_increased_cells():
    """
    Calculates accuracy and parameter estimation 
    over an increasing number of cells in a lineage for 
    a censored two-state model.
    We increase the desired number of cells in a lineage by
    the experiment time. 
    """

    # Creating a list of populations to analyze over
    times = np.linspace(min_experiment_time, max_experiment_time, num_data_points)
    list_of_populations = []
    for experiment_time in times:
        population = []
        
        
        population.append(LineageTree(pi, T, E, max_desired_num_cells, censor_condition=3, desired_experiment_time=experiment_time))
       
        # Adding populations into a holder for analysing
        list_of_populations.append(population)

    return commonAnalyze(list_of_populations)
