""" This file contains functions for plotting the performance of the model for censored data. """

from string import ascii_lowercase
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
    num_data_points,
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


def regGen(num):
    tmp = LineageTree.init_from_parameters(pi, T, E2, desired_num_cells=num)
    while len(tmp.output_lineage) < 5:
        tmp = LineageTree.init_from_parameters(pi, T, E2, desired_num_cells=num)
    return tmp


def cenGen(num):
    tmp = LineageTree.init_from_parameters(pi, T, E2, desired_num_cells=num, censor_condition=3, desired_experiment_time=200)
    while len(tmp.output_lineage) < 5:
        tmp = LineageTree.init_from_parameters(pi, T, E2, desired_num_cells=num, censor_condition=3, desired_experiment_time=200)
    return tmp


def makeFigure():
    """
    Makes fig 4.
    """
    x_Sim, x_Cen, Accuracy_Sim, Accuracy_Cen, _, _ = accuracy()

    # Get list of axis objects
    ax, f = getSetup((13, 6), (2, 3))
    figureMaker3(ax, x_Sim, x_Cen, Accuracy_Sim, Accuracy_Cen)
    ax[0].text(-0.2, 1.22, ascii_lowercase[0], transform=ax[0].transAxes, fontsize=16, fontweight="bold", va="top")
    ax[2].text(-0.2, 1.22, ascii_lowercase[1], transform=ax[2].transAxes, fontsize=16, fontweight="bold", va="top")
    ax[3].text(-0.2, 1.7, ascii_lowercase[2], transform=ax[3].transAxes, fontsize=16, fontweight="bold", va="top")
    ax[5].text(-0.2, 1.7, ascii_lowercase[3], transform=ax[5].transAxes, fontsize=16, fontweight="bold", va="top")

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
    num_cells = np.linspace(5, 31, num_data_points, dtype=int)
    list_of_fpi = [pi] * num_lineages.size

    # Adding populations into a holder for analysing
    list_of_populations = [[regGen(num_cells[i]) for _ in range(num)] for i, num in enumerate(num_lineages)]
    list_of_populationsSim = [[cenGen(num_cells[i]) for _ in range(num)] for i, num in enumerate(num_lineages)]

    x_Sim, _, output_Sim, _ = commonAnalyze(list_of_populationsSim, 2, list_of_fpi=list_of_fpi)
    x_Cen, _, output_Cen, _ = commonAnalyze(list_of_populations, 2, list_of_fpi=list_of_fpi)
    return x_Sim, output_Sim, x_Cen, output_Cen, list_of_populationsSim, list_of_populations


def figureMaker3(ax, x_Sim, output_Sim, x_Cen, output_Cen, xlabel="Number of Cells"):
    """
    Makes a 2 panel figures displaying state accuracy estimation across lineages
    of different censoring states.
    """
    Accuracy_Sim = output_Sim["balanced_accuracy_score"]
    Accuracy_Cen = output_Cen["balanced_accuracy_score"]
    accuracy_sim_df = pd.DataFrame(columns=["Cell number", "State Assignment Accuracy"])
    accuracy_sim_df["Cell number"] = x_Sim
    accuracy_sim_df["State Assignment Accuracy"] = Accuracy_Sim

    accuracy_cen_df = pd.DataFrame(columns=["Cell number", "State Assignment Accuracy"])
    accuracy_cen_df["Cell number"] = x_Cen
    accuracy_cen_df["State Assignment Accuracy"] = Accuracy_Cen

    i = 0
    plotLineage(regGen(31), axes=ax[i], censore=False)
    ax[i].axis('off')

    i = 1
    plotLineage(regGen(31), axes=ax[i], censore=False)
    ax[i].axis('off')

    i = 2
    ax[i].axhline(y=100, ls='--', c='k', alpha=0.5)
    sns.regplot(x="Cell number", y="State Assignment Accuracy", data=accuracy_sim_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_state_1_kws)
    ax[i].set_xlabel(xlabel)
    ax[i].set_ylim(bottom=50, top=101)
    ax[i].set_ylabel(r"State Accuracy [%]")
    ax[i].set_title("Full lineage data")

    i = 3
    plotLineage(cenGen(31), axes=ax[i], censore=True)
    ax[i].axis('off')

    i = 4
    plotLineage(cenGen(31), axes=ax[i], censore=True)
    ax[i].axis('off')

    i = 5
    ax[i].axhline(y=100, ls='--', c='k', alpha=0.5)
    sns.regplot(x="Cell number", y="State Assignment Accuracy", data=accuracy_cen_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_state_1_kws)
    ax[i].set_xlabel(xlabel)
    ax[i].set_ylim(bottom=50, top=101)
    ax[i].set_ylabel(r"State Accuracy [%]")
    ax[i].set_title("Censored Data")
