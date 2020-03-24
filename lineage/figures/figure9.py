"""
File: figure8.py
Purpose: Generates figure 8.
Figure 8 analyzes heterogeneous (2 state), censored (by both time and fate),
single lineages (no more than one lineage per population)
with similar proportions of cells in states but
of varying distributions.
"""
import numpy as np

from .figureCommon import (
    getSetup,
    subplotLabel,
    commonAnalyze,
    figureMaker,
    pi,
    max_desired_num_cells,
    max_experiment_time,
    num_data_points,
    state1,
)
from ..LineageTree import LineageTree
from ..StateDistribution import StateDistribution


def makeFigure():
    """
    Makes figure 6.
    """

    # Get list of axis objects
    ax, f = getSetup((7, 6), (2, 3))

    figureMaker(ax, *accuracy(), xlabel="Wasserstein Divergence")

    subplotLabel(ax)

    return f


def accuracy():
    """
    Calculates accuracy and parameter estimation
    over an increasing number of cells in a lineage for
    a uncensored two-state model but differing state distribution.
    We vary the distribution by
    increasing the Wasserstein divergence between the two states.
    """

    # Creating a list of populations to analyze over
    list_of_Es = [[StateDistribution(0.88, a, 1), state1] for a in np.logspace(1, 2, num_data_points, base=10)]
    list_of_populations = []
    for E in list_of_Es:
        population = []

        population.append(LineageTree(pi, np.array([[0,1],[1,0]]), E, max_desired_num_cells, censor_condition=3, desired_experiment_time=max_experiment_time))

        # Adding populations into a holder for analysing
        list_of_populations.append(population)

    return commonAnalyze(list_of_populations, xtype="wass")
