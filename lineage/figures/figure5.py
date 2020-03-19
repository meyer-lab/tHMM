"""
File: figure5.py
Purpose: Generates figure 5.
Figure 5 analyzes heterogeneous (2 state), NOT censored, 
single lineages (no more than one lineage per population)
with different proportions of cells in states.
"""
import numpy as np

from .figureCommon import getSetup, subplotLabel, commonAnalyze, figureMaker, pi, E, max_desired_num_cells
from ..LineageTree import LineageTree


def makeFigure():
    """
    Makes figure 3.
    """

    # Get list of axis objects
    ax, f = getSetup((7, 6), (2, 3))

    figureMaker(ax, *accuracy_increased_cells(xlabel="Cells in State 0 [$\%$]"))

    subplotLabel(ax)

    return f


def accuracy_increased_cells():
    """
    Calculates accuracy and parameter estimation 
    over an increasing number of cells in a lineage for 
    a uncensored two-state model.
    We increase the proportion of cells in a lineage by
    fixing the Transition matrix to be biased towards state 0. 
    """

    # Creating a list of populations to analyze over
    list_of_populations = []

    # Create a list of transition matrices to transition over called list_of_Ts
    list_of_Ts = makeTs(num_data_points)

    # Iterate through the transition matrices
    for T in list_of_Ts:
        # Creating an unpruned and pruned lineage
        lineage = LineageTree(pi, T, E, max_desired_num_cells)

        # Adding populations into a holder for analysing
        list_of_populations.append([lineage])

    return figFourCommon(list_of_populations, xtype='prop')


def makeTs(num):
    """
    Generates transition matrices
    """   
    return [np.array([[i,1.0-i],[i,1.0-i]]) for i in np.linspace(0.9, 1.0, num)]
