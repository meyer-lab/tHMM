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

regGen = lambda: LineageTree.init_from_parameters(pi, T, E2, desired_num_cells=2**5 - 1)
cenGen = lambda: LineageTree.init_from_parameters(pi, T, E2, desired_num_cells=2**6 - 1, censor_condition=3, desired_experiment_time=250)


def makeFigure():
    """
    Makes fig 3.
    """
    x_Sim, x_Cen, Accuracy_Sim, Accuracy_Cen, _, _ = accuracy()

    plotLineage(regGen(), 'lineage/figures/cartoons/uncen_fig4_1.svg', censore=False)
    plotLineage(regGen(), 'lineage/figures/cartoons/uncen_fig4_2.svg', censore=False)
    plotLineage(regGen(), 'lineage/figures/cartoons/uncen_fig4_3.svg', censore=False)

    plotLineage(cenGen(), 'lineage/figures/cartoons/cen_fig4_1.svg', censore=True)
    plotLineage(cenGen(), 'lineage/figures/cartoons/cen_fig4_2.svg', censore=True)
    plotLineage(cenGen(), 'lineage/figures/cartoons/cen_fig4_3.svg', censore=True)

    # Get list of axis objects
    ax, f = getSetup((6.5, 4), (2, 2))
    figureMaker3(ax, x_Sim, x_Cen, Accuracy_Sim, Accuracy_Cen)
    ax[0].text(-0.2, 1.22, ascii_lowercase[0], transform=ax[0].transAxes, fontsize=16, fontweight="bold", va="top")
    ax[1].text(-0.2, 1.22, ascii_lowercase[1], transform=ax[1].transAxes, fontsize=16, fontweight="bold", va="top")

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
    list_of_fpi = [pi] * num_lineages.size

    # Adding populations into a holder for analysing
    list_of_populations = [[regGen() for _ in range(num)] for num in num_lineages]
    list_of_populationsSim = [[cenGen() for _ in range(num)] for num in num_lineages]

    x_Sim, _, Accuracy_Sim, _, _, _ = commonAnalyze(list_of_populationsSim, 2, list_of_fpi=list_of_fpi)
    x_Cen, _, Accuracy_Cen, _, _, _ = commonAnalyze(list_of_populations, 2, list_of_fpi=list_of_fpi)
    return x_Sim, x_Cen, Accuracy_Sim, Accuracy_Cen, list_of_populationsSim, list_of_populations


def figureMaker3(ax, x_Sim, x_Cen, Accuracy_Sim, Accuracy_Cen, xlabel="Number of Cells"):
    """
    Makes a 2 panel figures displaying state accuracy estimation across lineages
    of different censoring states.
    """
    accuracy_sim_df = pd.DataFrame(columns=["Cell number", "State Assignment Accuracy"])
    accuracy_sim_df["Cell number"] = x_Sim
    accuracy_sim_df["State Assignment Accuracy"] = Accuracy_Sim

    accuracy_cen_df = pd.DataFrame(columns=["Cell number", "State Assignment Accuracy"])
    accuracy_cen_df["Cell number"] = x_Cen
    accuracy_cen_df["State Assignment Accuracy"] = Accuracy_Cen

    i = 0
    ax[i].axis('off')

    i += 1
    ax[i].axis('off')

    i += 1
    ax[i].axhline(y=100, ls='--', c='k', alpha=0.5)
    sns.regplot(x="Cell number", y="State Assignment Accuracy", data=accuracy_sim_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_state_1_kws)
    ax[i].set_xlabel(xlabel)
    ax[i].set_ylim(bottom=50, top=101)
    ax[i].set_ylabel(r"State Accuracy [%]")
    ax[i].set_title("Full lineage data")

    i += 1
    ax[i].axhline(y=100, ls='--', c='k', alpha=0.5)
    sns.regplot(x="Cell number", y="State Assignment Accuracy", data=accuracy_cen_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_state_1_kws)
    ax[i].set_xlabel(xlabel)
    ax[i].set_ylim(bottom=50, top=101)
    ax[i].set_ylabel(r"State Accuracy [%]")
    ax[i].set_title("Censored Data")
