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
    list_of_Ts = makeTs()
    
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

# Add function to generate transition matrices below
def makeTs(increment=0.1):
    
    """ Generates transition matrices """
    
    list_of_Ts = [np.array([[0.5, 0.5],[0.5, 0.5]])]
    new_arr = np.copy(list_of_Ts[0])
    while 0 < new_arr[0][0] < 1:
        new_arr[:,0] += increment
        new_arr[:,1] -= increment
        if (abs(1 - new_arr[0][0]) <= 0.1):
            break
        else:
            list_of_Ts.append(np.copy(new_arr))
    list_of_Ts.append(np.array([[1.,0.],[1.,0.]]))
    return list_of_Ts