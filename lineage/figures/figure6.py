""" This file contains figures related to how far the states need to be,
which is shown by Wasserestein distance. """
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import rand_score

from .common import (
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
    ax, f = getSetup((12, 5), (1, 3))
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
    list_of_populations2 = [[LineageTree.init_from_parameters(pi, T, E, 3 * max_desired_num_cells)] for E in list_of_Es2]

    balanced_score = np.empty(len(list_of_populations))

    for ii, pop in enumerate(list_of_populations):
        ravel_true_states = np.array([cell.state for lineage in pop for cell in lineage.output_lineage])
        all_cells = np.array([cell.obs[2] for lineage in pop for cell in lineage.output_lineage])

        shape, scale1, scale2 = list_of_Es[ii][0].params[2], list_of_Es[ii][0].params[3], list_of_Es[ii][1].params[3]
        thresh = classification_threshold(shape, scale1, scale2)
        pred_st = np.zeros(all_cells.shape)
        for j, obs in enumerate(all_cells):
            if obs <= thresh:
                pred_st[j] = 1
        balanced_score[ii] = 100 * rand_score(ravel_true_states, pred_st)

    # replace x with 1-x if the accuracy is less than 50%
    balanced_score[balanced_score < 50.0] = 100.0 - balanced_score[balanced_score < 50.0]

    wass, _, dict_out, _ = commonAnalyze(list_of_populations, 2, xtype="wass", list_of_fpi=[pi] * num_data_points, list_of_fT=[T] * num_data_points, parallel=True)
    accuracy = dict_out["state_similarity"]
    distribution_df = pd.DataFrame(columns=["Distribution Similarity", "G1 lifetime", "State"])
    lineages = [list_of_populations2[int(num_data_points * i / 4.)][0] for i in range(4)]
    len_lineages = [len(lineage) for lineage in lineages]
    distribution_df["G1 lifetime"] = [(cell.obs[1] + cell.obs[2]) for lineage in lineages for cell in lineage.output_lineage]
    distribution_df["State"] = ["State 1" if cell.state == 0 else "State 2" for lineage in lineages for cell in lineage.output_lineage]
    distribution_df["Distribution Similarity"] = len_lineages[0] * ["Same\n" + str(0) + "-" + str(wass[-1] / 4)] +\
        len_lineages[1] * ["Similar\n" + str(wass[-1] / 4) + "-" + str(wass[-1] / 2)] +\
        len_lineages[2] * ["Different\n" + str(wass[-1] / 2) + "-" + str(wass[-1] * 0.75)] +\
        len_lineages[3] * ["Distinct\n>" + str(wass[-1] * 0.75)]

    # for the violin plot (distributions)
    wasser_df = pd.DataFrame(columns=["Wasserstein distance", "Adjusted Rand Index Accuracy"])
    wasser_df["Wasserstein distance"] = wass
    wasser_df["Adjusted Rand Index Accuracy"] = accuracy
    wasser_df["Baseline Accuracy"] = balanced_score
    return distribution_df, wasser_df


def figureMaker5(ax, distribution_df, wasser_df):
    """
    This makes figure 5.
    """
    # cartoon to show different shapes --> similar shapes
    i = 0
    ax[i].axis('off')
    ax[i].set_title("state difference")

    i += 1

    sns.violinplot(x="G1 lifetime", y="Distribution Similarity", hue="State", palette={"State 1": "b", "State 2": "g"}, split=True, data=distribution_df, ax=ax[i])

    i += 1
    # state accuracy
    sns.regplot(x="Wasserstein distance", y="Adjusted Rand Index Accuracy", data=wasser_df, label="tHMM", ax=ax[i], lowess=True, marker='+')
    sns.regplot(x="Wasserstein distance", y="Baseline Accuracy", data=wasser_df, ax=ax[i], label="Best threshold", lowess=True, marker='+')
    ax[i].set_title("State Assignment Accuracy")
    ax[i].set_ylabel("Adjusted Rand Index Accuracy [%]")
    ax[i].set_ylim(bottom=10.0, top=101)
    ax[i].legend()


def classification_threshold(shape, scale1, scale2):
    """ Given the parameters of the gamma distribution, it provides an analytical threshold for classification.
    This function is specific to this figure, as the shape parameter is shared and only the scale varies.
    """
    if scale1 == scale2:
        return shape * scale1
    else:
        numer = shape * np.log(scale2 / scale1)
        denom = (1 / scale1) - (1 / scale2)
        return numer / denom
