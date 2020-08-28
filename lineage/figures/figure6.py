""" This file contains figures related to how far the states need to be,
which is shown by Wasserestein distance. """
import numpy as np
import pandas as pd
import seaborn as sns
from .figureCommon import (
    getSetup,
    subplotLabel,
    commonAnalyze,
    pi,
    T,
    max_desired_num_cells,
    num_data_points,
    scatter_kws_list,
)
from ..LineageTree import LineageTree
from ..states.StateDistributionGaPhs import StateDistribution


def makeFigure():
    """
    Makes fig 6.
    """

    # Get list of axis objects
    ax, f = getSetup((7, 5), (2, 2))
    figureMaker5(ax, *accuracy())

    subplotLabel(ax)

    return f


def accuracy():
    """ A Helper function to create more random copies of a population. """
    # Creating a list of populations to analyze over
    list_of_Es = [[StateDistribution(0.99, 0.9, 12, a, 4, 5), StateDistribution(0.99, 0.8, 12, 1.5, 8, 5)] for a in np.linspace(1.5, 4, num_data_points)]
    list_of_populations = []
    list_of_fpi = []
    list_of_fT = []
    list_of_fE = []
    for E in list_of_Es:
        population = LineageTree.init_from_parameters(pi, T, E, max_desired_num_cells)

        # Adding populations into a holder for analysing
        list_of_populations.append([population])
        list_of_fpi.append(pi)
        list_of_fT.append(T)
        list_of_fE.append(E)

    wass, _, accuracy_after_switching, _, _, _ = commonAnalyze(list_of_populations, 2, xtype="wass", list_of_fpi=list_of_fpi, parallel=True)
#     for indx, a in enumerate(accuracy_after_switching):
#         if a <= 60:
#             print(list_of_populations[indx])

    distribution_df = pd.DataFrame(columns=["Distribution type", "G1 lifetime", "State"])
    lineages = [list_of_populations[int(num_data_points * i / 4.)][0] for i in range(4)]
    len_lineages = [len(lineage) for lineage in lineages]
    distribution_df["G1 lifetime"] = [(cell.obs[1] + cell.obs[2]) for lineage in lineages for cell in lineage.output_lineage]
    distribution_df["State"] = ["State 1" if cell.state == 0 else "State 2" for lineage in lineages for cell in lineage.output_lineage]
    distribution_df["Distribution type"] = len_lineages[0] * ["Same"] +\
        len_lineages[1] * ["Similar"] +\
        len_lineages[2] * ["Different"] +\
        len_lineages[3] * ["Distinct"]

    # for the violin plot (distributions)
    wasser_df = pd.DataFrame(columns=["Wasserstein distance", "State Assignment Accuracy"])
    wasser_df["Wasserstein distance"] = wass
    wasser_df["State Assignment Accuracy"] = accuracy_after_switching
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
    sns.violinplot(x="Distribution type", y="G1 lifetime", hue="State", split=True, data=distribution_df, ax=ax[i])

    i += 1
    # state accuracy
    sns.regplot(x="Wasserstein distance", y="State Assignment Accuracy", data=wasser_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[0])
    ax[i].set_title("State Assignment Accuracy")
    ax[i].set_ylabel("Accuracy [%]")
    ax[i].set_ylim(bottom=10.0, top=101)
