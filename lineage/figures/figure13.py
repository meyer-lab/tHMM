"""
File: figure3.py
Purpose: Generates figure 3.
Figure 3 analyzes heterogeneous (2 state), uncensored,
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
    E1,
    min_desired_num_cells,
    lineage_good_to_analyze,
    min_num_lineages,
    max_num_lineages,
    num_data_points,
)
from ..LineageTree import LineageTree


def makeFigure():
    """
    Makes figure 4.
    """

    # Get list of axis objects
    ax, f = getSetup((7, 7), (3, 3))

    figureMaker(ax, *accuracy())

    subplotLabel(ax)

    return f


def accuracy():
    """
    Calculates accuracy and parameter estimation
    over an increasing number of lineages in a population for
    a uncensored two-state model.
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
                tmp_lineage = LineageTree(pi, T, E1, min_desired_num_cells)
                good2go = lineage_good_to_analyze(tmp_lineage)

            population.append(tmp_lineage)

        # Adding populations into a holder for analysing
        list_of_populations.append(population)
        list_of_fpi.append(pi)
        list_of_fT.append(T)
        list_of_fE.append(E1)

    return commonAnalyze(list_of_populations)
