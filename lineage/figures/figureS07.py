"""
File: figureS07.py
Purpose: Generates figure S07.
Figure S07 analyzes heterogeneous (2 state), censored (by both time and fate),
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
    Makes figure 6.
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
    list_of_fT = []
    list_of_fE = []
    for E in list_of_Es:
        population = []

        for _ in range(5):
            tmp_lineage = LineageTree.init_from_parameters(pi, T, E, 0.5 * max_desired_num_cells, censor_condition=3, desired_experiment_time=500)
            if len(tmp_lineage.output_lineage) < 3:
                pass
            else:
                population.append(tmp_lineage)

        # Adding populations into a holder for analysing
        list_of_populations.append(population)
        list_of_fpi.append(pi)
        list_of_fT.append(T)
        list_of_fE.append(E)

    return commonAnalyze(list_of_populations, 2, xtype="wass", list_of_fpi=list_of_fpi)
