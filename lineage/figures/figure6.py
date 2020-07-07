""" This file contains functions for plotting the performance of the model for censored data. """

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
    lineage_good_to_analyze,
    num_data_points,
    min_desired_num_cells,
    max_experiment_time,
    min_experiment_time,
    min_num_lineages,
    max_num_lineages
)
from ..LineageTree import LineageTree
from ..plotTree import plotLineage

def makeFigure():
    """
    Makes fig 6.
    """
    x_Sim, x_Cen, Accuracy_Sim, Accuracy_Cen, _, _ = accuracy()

    lineage_uncensored = LineageTree.init_from_parameters(pi, T, E2, desired_num_cells=2**8 - 1)
    plotLineage(lineage_uncensored, 'lineage/figures/cartoons/lineage_notcen.svg')

    lineage_censored = LineageTree.init_from_parameters(pi, T, E2, desired_num_cells=2**8 - 1, censor_condition=3, desired_experiment_time=400)
    plotLineage(lineage_censored, 'lineage/figures/cartoons/lineage_cen.svg')
    # Get list of axis objects
    ax, f = getSetup((6, 5), (2, 2))

    figureMaker(ax, x_Sim, x_Cen, Accuracy_Sim, Accuracy_Cen)

    subplotLabel(ax)

    return f

def accuracy():
    """
    Calculates accuracy and parameter estimation
    over an increasing number of cells in a lineage for
    a uncensored two-state model.
    We increase the desired number of cells in a lineage by
    the experiment time.
    """

    # Creating a list of populations to analyze over
    num_lineages = np.linspace(min_num_lineages, max_num_lineages, num_data_points, dtype=int)
    list_of_populations = []
    list_of_populationsSim = []
    list_of_fpi = []
    list_of_fT = []
    list_of_fE2 = []
    for num in num_lineages:
        population = []
        populationSim = []

        for _ in range(num):
            good2go = False
            while not good2go:
                tmp_lineage = LineageTree.init_from_parameters(pi, T, E2, 48, censor_condition=3, desired_experiment_time=200)
                tmp_lineageSim = LineageTree.init_from_parameters(pi, T, E2, 24)
                good2go1 = lineage_good_to_analyze(tmp_lineage)
                good2go2 = lineage_good_to_analyze(tmp_lineageSim)
                good2go = good2go1 and good2go2
            population.append(tmp_lineage)
            populationSim.append(tmp_lineageSim)

        # Adding populations into a holder for analysing
        list_of_populations.append(population)
        list_of_populationsSim.append(populationSim)
        list_of_fpi.append(pi)
        list_of_fT.append(T)
        list_of_fE2.append(E2)

    x_Sim, _, Accuracy_Sim, _, _, _ = commonAnalyze(list_of_populationsSim, list_of_fpi=list_of_fpi)
    x_Cen, _, Accuracy_Cen, _, _, _ = commonAnalyze(list_of_populations, list_of_fpi=list_of_fpi)
    return x_Sim, x_Cen, Accuracy_Sim, Accuracy_Cen, list_of_populationsSim, list_of_populations

def figureMaker(ax, x_Sim, x_Cen, Accuracy_Sim, Accuracy_Cen, xlabel="Number of Cells"):
    """
    Makes a 2 panel figures displaying state accuracy estimation across lineages
    of different censoring states.
    """

    i = 0
    ax[i].axis('off')

    i += 1
    ax[i].axis('off')

    i += 1
    ax[i].set_xlabel(xlabel)
    ax[i].set_ylim(bottom=0, top=101)
    ax[i].scatter(x_Sim, Accuracy_Sim, c="k", marker="o", edgecolors="k", alpha=0.25)
    ax[i].set_ylabel(r"State Accuracy [%]")
    ax[i].axhline(y=100, linestyle="--", linewidth=2, color="k", alpha=1)
    ax[i].set_title("Full lineage data")

    i += 1
    ax[i].set_xlabel(xlabel)
    ax[i].set_ylim(bottom=0, top=101)
    ax[i].scatter(x_Cen, Accuracy_Cen, c="k", marker="o", edgecolors="k", alpha=0.25)
    ax[i].set_ylabel(r"State Accuracy [%]")
    ax[i].axhline(y=100, linestyle="--", linewidth=2, color="k", alpha=1)
    ax[i].set_title("Censored Data")
