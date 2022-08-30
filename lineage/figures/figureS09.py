"""
File: figure23.py
Purpose: Generates figure 23.
Figure 23 analyzes heterogeneous (2 state), uncensored,
populations of lineages (more than one lineage per populations).
"""
import numpy as np

from .common import (
    getSetup,
    subplotLabel,
    commonAnalyze,
    figureMaker,
    pi,
    T,
    E2,
    min_desired_num_cells,
    min_num_lineages,
    max_num_lineages,
    num_data_points,
)
from ..LineageTree import LineageTree

# Creating a list of populations to analyze over
num_lineages = np.linspace(min_num_lineages, max_num_lineages, num_data_points, dtype=int)
list_of_populations = [[LineageTree.rand_init(pi, T, E2, min_desired_num_cells) for _ in range(num)] for num in num_lineages]


def makeFigure():
    """
    Makes figure 4.
    """
    # Get list of axis objects
    ax, f = getSetup((10, 10), (4, 3))

    figureMaker(ax, *commonAnalyze(list_of_populations, 2), num_lineages=num_lineages)

    subplotLabel(ax)
    return f
