""" This file contains functions for plotting the performance of the model for censored data. """

import numpy as np
import pandas as pd
import seaborn as sns

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
    max_num_lineages,
)
from ..LineageTree import LineageTree
from ..plotTree import plotLineage

scatter_state_1_kws = {
    "alpha": 0.33,
    "marker": "+",
    "s": 20,
}


def makeFigure():
    """
    Makes fig 3.
    """
    x_Sim, x_Cen, Accuracy_Sim, Accuracy_Cen, _, _ = accuracy()

    lineage_uncensored1 = LineageTree.init_from_parameters(
        pi, T, E2, desired_num_cells=2**5 - 1)
    plotLineage(lineage_uncensored1,
                'lineage/figures/cartoons/uncen_fig3_1.svg', censore=False)
    lineage_uncensored2 = LineageTree.init_from_parameters(
        pi, T, E2, desired_num_cells=2**5 - 1)
    plotLineage(lineage_uncensored2,
                'lineage/figures/cartoons/uncen_fig3_2.svg', censore=False)
    lineage_uncensored3 = LineageTree.init_from_parameters(
        pi, T, E2, desired_num_cells=2**5 - 1)
    plotLineage(lineage_uncensored3,
                'lineage/figures/cartoons/uncen_fig3_3.svg', censore=False)

    lineage_censored1 = LineageTree.init_from_parameters(
        pi, T, E2, desired_num_cells=2**6 - 1, censor_condition=3, desired_experiment_time=300)
    plotLineage(lineage_censored1,
                'lineage/figures/cartoons/cen_fig3_1.svg', censore=True)
    lineage_censored2 = LineageTree.init_from_parameters(
        pi, T, E2, desired_num_cells=2**6 - 1, censor_condition=3, desired_experiment_time=300)
    plotLineage(lineage_censored2,
                'lineage/figures/cartoons/cen_fig3_2.svg', censore=True)
    lineage_censored3 = LineageTree.init_from_parameters(
        pi, T, E2, desired_num_cells=2**6 - 1, censor_condition=3, desired_experiment_time=300)
    plotLineage(lineage_censored3,
                'lineage/figures/cartoons/cen_fig3_3.svg', censore=True)

    # Get list of axis objects
    ax, f = getSetup((5, 6), (3, 2))
    figureMaker3(ax, x_Sim, x_Cen, Accuracy_Sim, Accuracy_Cen)

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
    num_lineages = np.linspace(
        1, 20, 20, dtype=int)
    list_of_populations = []
    list_of_populationsSim = []
    list_of_fpi = []
    list_of_fT = []
    list_of_fE2 = []
    for num in num_lineages:
        population = []
        populationSim = []

        for _ in range(num):
            good2go1 = False
            good2go2 = False
            while not good2go1:
                tmp_lineage = LineageTree.init_from_parameters(
                    pi, T, E2, 2**4 - 1)
                good2go1 = lineage_good_to_analyze(tmp_lineage)
            while not good2go2:
                tmp_lineageSim = LineageTree.init_from_parameters(
                    pi, T, E2, 2**4 - 1)
                good2go2 = lineage_good_to_analyze(tmp_lineageSim)
            population.append(tmp_lineage)
            populationSim.append(tmp_lineageSim)

        # Adding populations into a holder for analysing
        list_of_populations.append(population)
        list_of_populationsSim.append(populationSim)
        list_of_fpi.append(pi)
        list_of_fT.append(T)
        list_of_fE2.append(E2)

    x_Sim, _, Accuracy_Sim, _, _, _ = commonAnalyze(
        list_of_populationsSim, list_of_fpi=list_of_fpi, parallel=True)
    x_Cen, _, Accuracy_Cen, _, _, _ = commonAnalyze(
        list_of_populations, list_of_fpi=list_of_fpi, parallel=True)
    return x_Sim, x_Cen, Accuracy_Sim, Accuracy_Cen, list_of_populationsSim, list_of_populations


def figureMaker3(ax, x_Sim, x_Cen, Accuracy_Sim, Accuracy_Cen, xlabel="Number of Cells"):
    """
    Makes a 2 panel figures displaying state accuracy estimation across lineages
    of different censoring states.
    """
    accuracy_sim_df = pd.DataFrame(
        columns=["Cell number", "State Assignment Accuracy"])
    accuracy_sim_df["Cell number"] = x_Sim
    accuracy_sim_df["State Assignment Accuracy"] = Accuracy_Sim

    accuracy_cen_df = pd.DataFrame(
        columns=["Cell number", "State Assignment Accuracy"])
    accuracy_cen_df["Cell number"] = x_Cen
    accuracy_cen_df["State Assignment Accuracy"] = Accuracy_Cen

    i = 0
    ax[i].axis('off')

    i += 1
    ax[i].axis('off')

    i += 1
    ax[i].axis('off')

    i += 1
    ax[i].axis('off')

    i += 1
    ax[i].axhline(y=100, ls='--', c='k', alpha=0.5)
    sns.regplot(x="Cell number", y="State Assignment Accuracy", data=accuracy_sim_df,
                ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_state_1_kws)
    ax[i].set_xlabel(xlabel)
    ax[i].set_ylim(bottom=50, top=101)
    ax[i].set_ylabel(r"State Accuracy [%]")
    ax[i].set_title("Full lineage data")

    i += 1
    ax[i].axhline(y=100, ls='--', c='k', alpha=0.5)
    sns.regplot(x="Cell number", y="State Assignment Accuracy", data=accuracy_cen_df,
                ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_state_1_kws)
    ax[i].set_xlabel(xlabel)
    ax[i].set_ylim(bottom=50, top=101)
    ax[i].set_ylabel(r"State Accuracy [%]")
    ax[i].set_title("Censored Data")
