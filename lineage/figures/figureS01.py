"""
File: figureS01.py
Purpose: Generates figure S01.
Figure S01 analyzes heterogeneous (2 state), uncensored,
single lineages (no more than one lineage per population).
"""

import numpy as np

from ..LineageTree import LineageTree
from .common import (
    E,
    T,
    commonAnalyze,
    figureMaker,
    getSetup,
    max_desired_num_cells,
    min_desired_num_cells,
    num_data_points,
    pi,
    subplotLabel,
)

# Creating a list of populations to analyze over
cells = np.linspace(min_desired_num_cells, max_desired_num_cells, num_data_points, dtype=int)
list_of_fpi = [pi] * cells.size

# Generate populations
list_of_populations = [[LineageTree.rand_init(pi, T, E, int(cell_num))] for cell_num in cells]


def makeFigure():
    """
    Makes figure 2.
    """

    # Get list of axis objects
    ax, f = getSetup((10, 10), (3, 3))

    figureMaker(ax, *commonAnalyze(list_of_populations, 2, list_of_fpi=list_of_fpi))

    subplotLabel(ax)

    return f
