"""
File: figure3.py
Purpose: Generates figure 3.
Figure 3 analyzes heterogeneous (2 state), pruned (by both time and fate), single lineages
(no more than one lineage per population) with at least 16 cells over increasing experimental
times.
"""
import numpy as np

from .figureCommon import getSetup, subplotLabel
from .figure4 import figure_maker, E, piiii, T
from ..Analyze import run_Analyze_over, run_Results_over
from ..LineageTree import LineageTree


def makeFigure():
    """
    Makes figures 3.
    """

    # Get list of axis objects
    ax, f = getSetup((7, 6), (2, 3))

    figure_maker(ax, *accuracy_increased_cells())

    subplotLabel(ax)

    return f


def accuracy_increased_cells():
    """
    Calculates accuracy and transition rate estimation over an increasing number of cells in a lineage for an pruned two-state model.
    """

    # Creating a list of populations to analyze over
    times = np.linspace(100, 1000, 50)
    list_of_populations = []
    for experiment_time in times:
        # Creating an unpruned and pruned lineage
        lineage = LineageTree(piiii, T, E, (2**12) - 1, experiment_time, prune_condition='both', prune_boolean=True)

        while len(lineage.output_lineage) < 16:
            del lineage
            lineage = LineageTree(piiii, T, E, (2**12) - 1, experiment_time, prune_condition='both', prune_boolean=True)

        # Adding populations into a holder for analysing
        list_of_populations.append([lineage])

    # Analyzing the lineages in the list of populations (parallelized function)
    output = run_Analyze_over(list_of_populations, 2, parallel=True)

    # Collecting the results of analyzing the lineages
    results_holder = run_Results_over(output)

    dictOut = {}

    for key in results_holder[0].keys():
        dictOut[key] = []

    for results_dict in results_holder:
        for key, val in results_dict.items():
            dictOut[key].append(val)

    paramEst = np.array(dictOut["param_estimates"])

    return dictOut["total_number_of_cells"], paramEst, dictOut["accuracy_after_switching"], dictOut["transition_matrix_norm"], dictOut["pi_vector_norm"]
