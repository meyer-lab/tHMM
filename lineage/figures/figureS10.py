"""
File: figureS10.py
Purpose: Generates figure S10.
Figure S10 analyzes heterogeneous (2 state), censored (by both time and fate),
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
    E2,
    min_desired_num_cells,
    max_experiment_time,
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
        tmp_lineage = LineageTree.init_from_parameters(pi, T, E2, min_desired_num_cells, censor_condition=3, desired_experiment_time=2 * max_experiment_time)
        if len(tmp_lineage.output_lineage) < 3:
            pass
        else:
            population.append(tmp_lineage)

    # Adding populations into a holder for analysing
    list_of_populations.append(population)

def makeFigure():
    """
    Makes figure 5.
    """

    # Get list of axis objects
    ax, f = getSetup((10, 13.333), (4, 3))

    figureMaker(ax, *commonAnalyze(list_of_populations, 2), xlabel="Number of Cells")

    subplotLabel(ax)

    return f
