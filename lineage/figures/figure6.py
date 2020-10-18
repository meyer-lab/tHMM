""" This file contains figures related to how far the states need to be,
which is shown by Wasserestein distance. """
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score
import itertools

from .figureCommon import (
    getSetup,
    subplotLabel,
    commonAnalyze,
    pi,
    T,
    E2,
    max_desired_num_cells,
    num_data_points,
)
from ..tHMM import tHMM
from ..LineageTree import LineageTree
from ..states.StateDistributionGaPhs import StateDistribution


def makeFigure():
    """
    Makes fig 6.
    """

    # Get list of axis objects
    ax, f = getSetup((9, 6), (2, 2))
    figureMaker5(ax, *accuracy())

    subplotLabel(ax)

    return f


def accuracy():
    """ A Helper function to create more random copies of a population. """
    # Creating a list of populations to analyze over
    list_of_Es = [[StateDistribution(E2[1].params[0], E2[1].params[1], E2[1].params[2], a, E2[1].params[4], E2[1].params[5]), E2[1]] for a in np.linspace(4.0, 20.0, num_data_points)]
    list_of_populations = [[LineageTree.init_from_parameters(pi, T, E, max_desired_num_cells)] for E in list_of_Es]
    # for the violin plots
    list_of_Es2 = [[StateDistribution(E2[1].params[0], E2[1].params[1], E2[1].params[2], a, E2[1].params[4], E2[1].params[5]), E2[1]] for a in np.linspace(4.0, 20.0, num_data_points)]
    list_of_populations2 = [[LineageTree.init_from_parameters(pi, T, E, 3*max_desired_num_cells)] for E in list_of_Es2]

    balanced_score = np.empty(len(list_of_populations))

    for ii, pop in enumerate(list_of_populations):
        ravel_true_states = np.array([cell.state for lineage in pop for cell in lineage.output_lineage])
        all_cells = np.array([cell.obs for lineage in pop for cell in lineage.output_lineage])

        kmeans = KMeans(n_clusters=2).fit(all_cells).labels_
        balanced_score[ii] = 100 * balanced_accuracy_score(ravel_true_states, kmeans)

    # replace x with 1-x if the accuracy is less than 50%
    balanced_score[balanced_score < 50.0] = 100.0 - balanced_score[balanced_score < 50.0]

    wass, _, dict_out, _ = commonAnalyze(list_of_populations, 2, xtype="wass", list_of_fpi=[pi] * num_data_points, list_of_fT=[T] * num_data_points, parallel=True)
    accuracy = dict_out["balanced_accuracy_score"]
    distribution_df = pd.DataFrame(columns=["Distribution type", "G1 lifetime", "State"])
    lineages = [list_of_populations2[int(num_data_points * i / 4.)][0] for i in range(4)]
    len_lineages = [len(lineage) for lineage in lineages]
    distribution_df["G1 lifetime"] = [(cell.obs[1] + cell.obs[2]) for lineage in lineages for cell in lineage.output_lineage]
    distribution_df["State"] = ["State 1" if cell.state == 0 else "State 2" for lineage in lineages for cell in lineage.output_lineage]
    distribution_df["Distribution type"] = len_lineages[0] * ["Same\n" + str(0) + "-" + str(wass[-1] / 4)] +\
        len_lineages[1] * ["Similar\n" + str(wass[-1] / 4) + "-" + str(wass[-1] / 2)] +\
        len_lineages[2] * ["Different\n" + str(wass[-1] / 2) + "-" + str(wass[-1] * 0.75)] +\
        len_lineages[3] * ["Distinct\n>" + str(wass[-1] * 0.75)]

    # for the violin plot (distributions)
    wasser_df = pd.DataFrame(columns=["Wasserstein distance", "State Assignment Accuracy"])
    wasser_df["Wasserstein distance"] = wass
    wasser_df["State Assignment Accuracy"] = accuracy
    wasser_df["KMeans accuracy"] = balanced_score
    return distribution_df, wasser_df


def figureMaker5(ax, distribution_df, wasser_df):
    """
    This makes figure 5.
    """
    # cartoon to show different shapes --> similar shapes
    i = 0
    ax[i].axis('off')

    i += 1
    ax[i].axis('off')

    i += 1

    sns.violinplot(x="Distribution type", y="G1 lifetime", hue="State", palette={"State 2": "g", "State 1": "b"}, split=True, data=distribution_df, ax=ax[i])

    i += 1
    # state accuracy
    sns.regplot(x="Wasserstein distance", y="State Assignment Accuracy", data=wasser_df, label="tHMM", ax=ax[i], lowess=True, marker='+')
    sns.regplot(x="Wasserstein distance", y="KMeans accuracy", data=wasser_df, ax=ax[i], label="K-means", lowess=True, marker='+')
    ax[i].set_title("State Assignment Accuracy")
    ax[i].set_ylabel("Accuracy [%]")
    ax[i].set_ylim(bottom=10.0, top=101)
    ax[i].legend()
