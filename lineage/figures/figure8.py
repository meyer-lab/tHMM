"""
File: figure8.py
Purpose: Generates figure 8.
Figure 8 analyzes heterogeneous (2 state), NOT censored,
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
    T,
    max_desired_num_cells,
    num_data_points,
    state1,
)
from ..LineageTree import LineageTree
from ..states.StateDistribution import StateDistribution


def makeFigure():
    """
    Makes figure 8.
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

        population.append(LineageTree(pi, T, E, max_desired_num_cells))

        # Adding populations into a holder for analysing
        list_of_populations.append(population)

    return commonAnalyze(list_of_populations, xtype="wass")
