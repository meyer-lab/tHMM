""" This file contains functions for plotting the performance of the model for censored data. """

import numpy as np
import pandas as pd
import seaborn as sns
from copy import deepcopy
from .common import (
    getSetup,
    subplotLabel,
    commonAnalyze,
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
    tmp = LineageTree.rand_init(pi, T, E2, desired_num_cells=num)
    while len(tmp.output_lineage) < 5:
        tmp = LineageTree.rand_init(pi, T, E2, desired_num_cells=num)
    return tmp


def cenGen(num):
    tmp = LineageTree.rand_init(pi, T, E2, desired_num_cells=num, censor_condition=3, desired_experiment_time=250)
    while len(tmp.output_lineage) < 5:
        tmp = LineageTree.rand_init(pi, T, E2, desired_num_cells=num, censor_condition=3, desired_experiment_time=250)
    return tmp


def makeFigure():
    """
    Makes fig 4.
    """
    x_Sim, output_Sim, x_Cen, output_Cen = accuracy()

    # Get list of axis objects
    ax, f = getSetup((9, 6), (2, 2))

    figureMaker3(ax, x_Sim, output_Sim, x_Cen, output_Cen)
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
    num_cells = np.linspace(5, 31, num_data_points, dtype=int)
    list_of_fpi = [pi] * num_lineages.size

    # Adding populations into a holder for analysing
    list_of_populationsSim = [[cenGen(num_cells[i]) for _ in range(num)] for i, num in enumerate(num_lineages)]

    SecondPopulation = deepcopy(list_of_populationsSim)
    for lin_list in SecondPopulation:
        for lins in lin_list:
            for cells in lins.output_lineage:
                if cells.obs[4] == 0.0:
                    cells.obs[4] = 1.0
                    assert np.isfinite(cells.obs[2])
                if cells.obs[5] == 0.0:
                    cells.obs[5] = 1.0
                    assert np.isfinite(cells.obs[3])

    x_Sim, _, output_Sim, _ = commonAnalyze(SecondPopulation, 2, list_of_fpi=list_of_fpi)
    x_Cen, _, output_Cen, _ = commonAnalyze(list_of_populationsSim, 2, list_of_fpi=list_of_fpi)
    return x_Sim, output_Sim, x_Cen, output_Cen


def figureMaker3(ax, x_Sim, output_Sim, x_Cen, output_Cen, xlabel="Number of Cells"):
    """
    Makes a 2 panel figures displaying state accuracy estimation across lineages
    of different censoring states.
    """
    Accuracy_Sim = output_Sim["state_similarity"]
    Accuracy_Cen = output_Cen["state_similarity"]
    accuracy_sim_df = pd.DataFrame(columns=["Cell number", "Adjusted Rand Index Accuracy"])
    accuracy_sim_df["Cell number"] = x_Sim
    accuracy_sim_df["Adjusted Rand Index Accuracy"] = Accuracy_Sim

    accuracy_cen_df = pd.DataFrame(columns=["Cell number", "Adjusted Rand Index Accuracy"])
    accuracy_cen_df["Cell number"] = x_Cen
    accuracy_cen_df["Adjusted Rand Index Accuracy"] = Accuracy_Cen

    i = 0
    plotLineage(regGen(45), axes=ax[i], censor=False)
    ax[i].axis('off')

    i += 1
    plotLineage(cenGen(45), axes=ax[i], censor=True)
    ax[i].axis('off')

    i += 1
    ax[i].axhline(y=100, ls='--', c='k', alpha=0.5)
    sns.regplot(x="Cell number", y="Adjusted Rand Index Accuracy", data=accuracy_sim_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_state_1_kws)
    ax[i].set_xlabel(xlabel)
    ax[i].set_ylim(bottom=0, top=101)
    ax[i].set_ylabel(r"Adjusted Rand Index Accuracy [%]")
    ax[i].set_title("Censored data, uncensored model")

    i += 1
    ax[i].axhline(y=100, ls='--', c='k', alpha=0.5)
    sns.regplot(x="Cell number", y="Adjusted Rand Index Accuracy", data=accuracy_cen_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_state_1_kws)
    ax[i].set_xlabel(xlabel)
    ax[i].set_ylim(bottom=0, top=101)
    ax[i].set_ylabel(r"Adjusted Rand Index Accuracy [%]")
    ax[i].set_title("Censored data, censored model")
