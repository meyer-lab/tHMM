"""
File: figureS06.py
Purpose: Generates figure S06.
Figure S06 analyzes heterogeneous (2 state), NOT censored,
single lineages (more than one lineage per population)
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
from ..states.StateDistributionGamma import StateDistribution


def makeFigure():
    """
    Makes figure 8.
    """

    # Get list of axis objects
    ax, f = getSetup((10, 10), (3, 3))

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
    list_of_Es = [[StateDistribution(0.99, 7, a), state1] for a in np.linspace(1, 8, num_data_points)]
    list_of_populations = []
    list_of_fpi = []
    for E in list_of_Es:
        population = []
        for _ in range(4):
            tmp_lineage = LineageTree.init_from_parameters(pi, T, E, max_desired_num_cells)
            population.append(tmp_lineage)

        # Adding populations into a holder for analysing
        list_of_populations.append(population)
        list_of_fpi.append(pi)

    return commonAnalyze(list_of_populations, 2, xtype="wass", list_of_fpi=list_of_fpi)
