"""
File: figure2.py
Purpose: Generates figure 2.
Figure 3 analyzes heterogeneous (2 state), uncensored,
single lineages (no more than one lineage per population).
"""
import numpy as np

from .figureCommon import (
    getSetup,
    subplotLabel,
    commonAnalyze,
    figureMaker,
    pi,
    T,
    E,
    max_desired_num_cells,
    num_data_points,
)
from ..LineageTree import LineageTree


def makeFigure():
    """
    Makes figure 3.
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
    a censored two-state model.
    We increase the desired number of cells in a lineage by
    the experiment time.
    """

    # Creating a list of populations to analyze over
    cells = np.linspace(16, 128, num_data_points)
    list_of_populations = []
    for cell_num in cells:
        population = []

        population.append(LineageTree(pi, T, E, cell_num))

        # Adding populations into a holder for analysing
        list_of_populations.append(population)

    return commonAnalyze(list_of_populations)
