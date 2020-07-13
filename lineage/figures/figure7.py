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
    max_num_lineages
)
from .figureS53 import return_closest
from ..LineageTree import LineageTree
from ..plotTree import plotLineage


def makeFigure():
    """
    Makes fig 6.
    """
    x_Sim, x_Cen, Accuracy_Sim, Accuracy_Cen, _, _ = accuracy()

    lineage_uncensored = LineageTree.init_from_parameters(pi, T, E2, desired_num_cells=2**7 - 1)
    plotLineage(lineage_uncensored, 'lineage/figures/cartoons/lineage_notcen.svg')

    lineage_censored = LineageTree.init_from_parameters(pi, T, E2, desired_num_cells=2**7 - 1, censor_condition=3, desired_experiment_time=400)
    plotLineage(lineage_censored, 'lineage/figures/cartoons/lineage_cen.svg')
    # Get list of axis objects
    ax, f = getSetup((5, 5), (2, 2))
    number_of_columns = 25
    figureMaker7(ax, x_Sim, x_Cen, Accuracy_Sim, Accuracy_Cen, number_of_columns)

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
                tmp_lineage = LineageTree.init_from_parameters(pi, T, E2, 2**7-1, censor_condition=3, desired_experiment_time=400)
                tmp_lineageSim = LineageTree.init_from_parameters(pi, T, E2, 2**7-1)
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


def figureMaker7(ax, x_Sim, x_Cen, Accuracy_Sim, Accuracy_Cen, number_of_columns, xlabel="Number of Cells"):
    """
    Makes a 2 panel figures displaying state accuracy estimation across lineages
    of different censoring states.
    """
    accuracy_sim_df = pd.DataFrame(columns=["Cell number", "Approximate cell number", "State Assignment Accuracy"])
    accuracy_sim_df["Cell number"] = x_Sim
    accuracy_sim_df["State Assignment Accuracy"] = Accuracy_Sim
    maxx = np.max(x_Sim)
    cell_num_columns = [int(maxx * (2 * i + 1) / 2 / number_of_columns) for i in range(number_of_columns)]
    assert len(cell_num_columns) == number_of_columns
    for indx, num in enumerate(x_Sim):
        accuracy_sim_df.loc[indx, 'Approximate cell number'] = return_closest(num, cell_num_columns)

    accuracy_cen_df = pd.DataFrame(columns=["Cell number", "Approximate cell number", "State Assignment Accuracy"])
    accuracy_cen_df["Cell number"] = x_Cen
    accuracy_cen_df["State Assignment Accuracy"] = Accuracy_Cen
    maxx = np.max(x_Cen)
    cell_num_columns = [int(maxx * (2 * i + 1) / 2 / number_of_columns) for i in range(number_of_columns)]
    assert len(cell_num_columns) == number_of_columns
    for indx, num in enumerate(x_Cen):
        accuracy_cen_df.loc[indx, 'Approximate cell number'] = return_closest(num, cell_num_columns)

    i = 0
    ax[i].axis('off')

    i += 1
    ax[i].axis('off')

    i += 1
    ax[i].axhline(y=100, ls='--', c='k', alpha=0.5)
    sns.lineplot(x="Approximate cell number", y="State Assignment Accuracy", data=accuracy_sim_df, ax=ax[i])
    ax[i].set_xlabel(xlabel)
    ax[i].set_ylim(bottom=50, top=101)
    ax[i].set_ylabel(r"State Accuracy [%]")
    ax[i].set_title("Full lineage data")

    i += 1
    ax[i].axhline(y=100, ls='--', c='k', alpha=0.5)
    sns.lineplot(x="Approximate cell number", y="State Assignment Accuracy", data=accuracy_cen_df, ax=ax[i])
    ax[i].set_xlabel(xlabel)
    ax[i].set_ylim(bottom=50, top=101)
    ax[i].set_ylabel(r"State Accuracy [%]")
    ax[i].set_title("Censored Data")
