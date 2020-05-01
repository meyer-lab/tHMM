"""
File: figure20.py
Purpose: Generates figure 20.
"""
import numpy as np

from .figureCommon import (
    getSetup,
    subplotLabel,
    commonAnalyze,
    figureMaker,
    pi,
    T,
    max_desired_num_cells,
    lineage_good_to_analyze,
    num_data_points,
    max_experiment_time,
)
from ..LineageTree import LineageTree
from ..states.StateDistribution import StateDistribution

def makeFigure():
    """
    Makes figure 20.
    """

    # Get list of axis objects
    ax, f = getSetup((7, 6), (2, 3))

    figureMaker(ax, *accuracy())

    subplotLabel(ax)

    return f


def accuracy():
    """
    Calculates accuracy and parameter estimation
    over an increasing number of cells in a lineage for
    a uncensored two-state model.
    We increase the desired number of cells in a lineage by
    the experiment time.
    """

    # Creating a list of populations to analyze over
    list_of_Es = [[StateDistribution(a, 7, 1), StateDistribution(a, 7, 7)] for a in np.linspace(0.8, 1, num_data_points)]
    list_of_populations = []
    list_of_fpi = []
    list_of_fT = []
    list_of_fE = []
    for E in list_of_Es:
        population = []

        good2go = False
        while not good2go:
            tmp_lineage = LineageTree(pi, T, E, max_desired_num_cells, censor_condition=3, desired_experiment_time=max_experiment_time)
            good2go = lineage_good_to_analyze(tmp_lineage)

        population.append(tmp_lineage)

        # Adding populations into a holder for analysing
        list_of_populations.append(population)
        list_of_fpi.append(pi)
        list_of_fT.append(T)
        list_of_fE.append(E)

    return commonAnalyze(list_of_populations, list_of_fpi=list_of_fpi)
