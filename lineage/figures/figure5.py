"""
File: figure5.py
Purpose: Generates figure 5.
"""
import numpy as np

from .figureCommon import getSetup, subplotLabel
from .figure4 import figure_maker, E, piiii, figFourCommon
from ..LineageTree import LineageTree


def makeFigure():
    """
    Makes figures 3.
    """

    # Get list of axis objects
    ax, f = getSetup((7, 6), (2, 3))

    figure_maker(ax, *accuracy_increased_cells())

    subplotLabel(ax)

    return f


def accuracy_increased_cells():
    """
    Calculates accuracy and transition rate estimation over an increasing number of cells in a lineage for an pruned two-state model.
    """

    # Creating a list of populations to analyze over
    list_of_populations = []
    
    # Create a list of transition matrices to transition over called list_of_Ts
    
    # Iterate through the transition matrices
    for T in list_of_Ts:
        # Creating an unpruned and pruned lineage
        lineage = LineageTree(piiii, T, E, (2**12) - 1, experiment_time, prune_condition='both', prune_boolean=True)

        while len(lineage.output_lineage) < 16:
            del lineage
            lineage = LineageTree(piiii, T, E, (2**12) - 1, experiment_time, prune_condition='both', prune_boolean=True)

        # Adding populations into a holder for analysing
        list_of_populations.append([lineage])

    return figFourCommon(list_of_populations)