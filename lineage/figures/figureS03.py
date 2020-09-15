"""
File: figureS03.py
Purpose: Generates figure S03.
Figure S03 analyzes heterogeneous (2 state), censored (by both time and fate),
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
    min_experiment_time,
    min_num_lineages,
    max_num_lineages,
    num_data_points,
)
from ..LineageTree import LineageTree

# Creating a list of populations to analyze over
num_lineages = np.linspace(min_num_lineages, max_num_lineages, num_data_points, dtype=int)
list_of_populations = []

for num in num_lineages:
    population = []

    for _ in range(num):
        tmp_lineage = LineageTree.init_from_parameters(pi, T, E, min_desired_num_cells, censor_condition=3, desired_experiment_time=min_experiment_time)
        population.append(tmp_lineage)

    # Adding populations into a holder for analysing
    list_of_populations.append(population)


def makeFigure():
    """
    Makes figure 5.
    """

    # Get list of axis objects
    ax, f = getSetup((10, 10), (3, 3))

    figureMaker(ax, *commonAnalyze(list_of_populations, 2), num_lineages=num_lineages)

    subplotLabel(ax)

    return f
