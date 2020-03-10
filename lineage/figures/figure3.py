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

    # Collect necessary things to plot
    x = []
    bern_p0_est = []
    bern_p1_est = []
    gamma_a0_est = []
    gamma_a1_est = []
    gamma_scale0_est = []
    gamma_scale1_est = []
    accuracies = []
    tr = []
    pi = []

    for results_dict in results_holder:
        x.append(results_dict["total_number_of_cells"])
        accuracies.append(results_dict["accuracy_after_switching"])
        tr.append(results_dict["transition_matrix_norm"])
        pi.append(results_dict["pi_vector_norm"])
        bern_p0_est.append(results_dict["param_estimates"][0][0])
        bern_p1_est.append(results_dict["param_estimates"][1][0])
        gamma_a0_est.append(results_dict["param_estimates"][0][1])
        gamma_a1_est.append(results_dict["param_estimates"][1][1])
        gamma_scale0_est.append(results_dict["param_estimates"][0][2])
        gamma_scale1_est.append(results_dict["param_estimates"][1][2])

    return x, bern_p0_est, bern_p1_est, gamma_a0_est, gamma_a1_est, gamma_scale0_est, gamma_scale1_est, accuracies, tr, pi
