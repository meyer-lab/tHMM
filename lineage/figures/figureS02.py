"""
File: figureS02.py
Purpose: Generates figure S02.
Figure S02 analyzes heterogeneous (2 state), uncensored,
populations of lineages (more than one lineage per populations).
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
    min_desired_num_cells,
    min_num_lineages,
    max_num_lineages,
    num_data_points,
)
from ..LineageTree import LineageTree


def makeFigure():
    """
    Makes figure 4.
    """

    # Get list of axis objects
    ax, f = getSetup((10, 10), (3, 3))

    figureMaker(ax, *accuracy())

    subplotLabel(ax)

    return f


def accuracy():
    """
    Calculates accuracy and parameter estimation
    over an increasing number of lineages in a population for
    a uncensored two-state model.
    We increase the desired number of cells in a lineage by
    the experiment time.
    """

    # Creating a list of populations to analyze over
    num_lineages = np.linspace(min_num_lineages, max_num_lineages, num_data_points, dtype=int)
    list_of_populations = [[LineageTree.init_from_parameters(pi, T, E, min_desired_num_cells) for _ in range(num)] for num in num_lineages]

    return commonAnalyze(list_of_populations, 2)
