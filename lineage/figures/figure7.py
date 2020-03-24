"""
File: figure6.py
Purpose: Generates figure 6.
Figure 6 analyzes heterogeneous (2 state), NOT censored,
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
    num_data_points,
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
    list_of_Es = [[StateDistribution(0.99, 4, 3), StateDistribution(0.99, a, 3)] for a in np.logspace(2, 5, num_data_points, base=2)]
    list_of_populations = []
    for E in list_of_Es:
        population = []

        population.append(LineageTree(pi, np.array([[0, 1], [1, 0]]), E, max_desired_num_cells))

        # Adding populations into a holder for analysing
        list_of_populations.append(population)

    return commonAnalyze(list_of_populations, xtype="wass")
