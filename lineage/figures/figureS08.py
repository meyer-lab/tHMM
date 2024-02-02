"""
File: figureS08.py
Purpose: Generates figure S08.
Figure S08 analyzes heterogeneous (2 state), uncensored,
single lineages (no more than one lineage per population).
"""

import numpy as np

from .common import (
    getSetup,
    subplotLabel,
    commonAnalyze,
    pi,
    T,
    E2,
    max_desired_num_cells,
    num_data_points,
    min_desired_num_cells,
    figureMaker,
)
from ..LineageTree import LineageTree


def makeFigure():
    """
    Makes figure 2.
    """

    # Get list of axis objects
    ax, f = getSetup((10, 10), (4, 3))

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
    cells = np.linspace(min_desired_num_cells, max_desired_num_cells, num_data_points)

    # Adding populations into a holder for analysing
    list_of_populations = [
        [LineageTree.rand_init(pi, T, E2, cell_num)] for cell_num in cells
    ]
    list_of_fpi = [pi for _ in cells]

    return commonAnalyze(list_of_populations, 2, list_of_fpi=list_of_fpi)
