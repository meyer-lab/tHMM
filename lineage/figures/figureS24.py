"""
File: figure24.py
Purpose: Generates figure 24.
Figure 24 analyzes heterogeneous (2 state), censored (by both time and fate),
single lineages (no more than one lineage per population).
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
    max_desired_num_cells,
    min_experiment_time,
    lineage_good_to_analyze,
    max_experiment_time,
    num_data_points,
)
from ..LineageTree import LineageTree


def makeFigure():
    """
    Makes figure 3.
    """

    # Get list of axis objects
    ax, f = getSetup((10, 13.333), (4, 3))

    figureMaker(ax, *accuracy())

    subplotLabel(ax)

    return f


def accuracy():
    """
    Calculates accuracy and parameter estimation
    over an increasing number of cells in a lineage for
    a censored two-state model.
    We increase the desired number of cells in a lineage by
    the experiment time.
    """

    # Creating a list of populations to analyze over
    times = np.linspace(max_experiment_time, 2 * max_experiment_time, num_data_points)
    list_of_populations = []
    list_of_fpi = []
    list_of_fT = []
    list_of_fE = []
    for experiment_time in times:
        population = []
        good2go = False
        while not good2go:
            tmp_lineage = LineageTree.init_from_parameters(pi, T, E2, max_desired_num_cells, censor_condition=3, desired_experiment_time=experiment_time)
            good2go = lineage_good_to_analyze(tmp_lineage)

        population.append(tmp_lineage)

        # Adding populations into a holder for analysing
        list_of_populations.append(population)
        list_of_fpi.append(pi)
        list_of_fT.append(T)
        list_of_fE.append(E2)

    return commonAnalyze(list_of_populations, list_of_fpi=list_of_fpi)
