"""
File: figure5.py
Purpose: Generates figure 5.
Figure 5 analyzes heterogeneous (2 state), NOT censored,
single lineages (no more than one lineage per population)
with different proportions of cells in states by
changing the values in the transition matrices.
Includes G1 and G2 phases separately.
"""
from ..LineageTree import LineageTree
import numpy as np

from .figureCommon import (
    getSetup,
    subplotLabel,
    commonAnalyze,
    figureMaker2,
    pi,
    T,
    E2,
    min_desired_num_cells,
    min_experiment_time,
    lineage_good_to_analyze,
    min_num_lineages,
    max_num_lineages,
    num_data_points,
)


def makeFigure():
    """
    Makes figure 5.
    """

    # Get list of axis objects
    ax, f = getSetup((11, 6), (2, 4))

    figureMaker2(ax, *accuracy())

    subplotLabel(ax)

    return f


def accuracy():
    """
    Calculates accuracy and parameter estimation
    over an increasing number of lineages in a population for
    a censored two-state model.
    We increase the desired number of cells in a lineage by
    the experiment time.
    """

    # Creating a list of populations to analyze over
    num_lineages = np.linspace(min_num_lineages, max_num_lineages, num_data_points, dtype=int)
    list_of_populations = []
    list_of_fpi = []
    list_of_fT = []
    list_of_fE = []
    for num in num_lineages:
        population = []

        for _ in range(num):
            good2go = False
            while not good2go:
                tmp_lineage = LineageTree(pi, T, E2, min_desired_num_cells, censor_condition=0, desired_experiment_time=min_experiment_time)
                good2go = lineage_good_to_analyze(tmp_lineage)
            population.append(tmp_lineage)

        # Adding populations into a holder for analysing
        list_of_populations.append(population)
        list_of_fpi.append(pi)
        list_of_fT.append(T)
        list_of_fE.append(E2)

    return commonAnalyze(list_of_populations)