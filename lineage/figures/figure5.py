""" This file contains figures related to how far the states need to be,
which is shown by Wasserestein distance. """
import itertools
import numpy as np
import pandas as pd
import scipy.stats as sp
import seaborn as sns
from .figureCommon import (
    getSetup,
    subplotLabel,
    commonAnalyze,
    pi,
    T,
    max_desired_num_cells,
    lineage_good_to_analyze,
    num_data_points,
    return_closest,
)
from ..LineageTree import LineageTree
from ..states.StateDistributionGaPhs import StateDistribution


def makeFigure():
    """
    Makes fig 5.
    """

    # Get list of axis objects
    ax, f = getSetup((5, 5), (2, 2))
    number_of_columns = 25
    figureMaker4(ax, *accuracy(number_of_columns))

    subplotLabel(ax)

    return f


def accuracy(number_of_columns):
    """ A Helper function to create more random copies of a population. """
    # Creating a list of populations to analyze over
    list_of_Es = [[StateDistribution(0.99, 0.8, 12, a, 10, 5), StateDistribution(0.99, 0.8, 12, 1, 10, 5)] for a in np.linspace(1, 2.5, num_data_points)]
    list_of_populations = []
    list_of_fpi = []
    list_of_fT = []
    list_of_fE = []
    for E in list_of_Es:
        population = []

        good2go = False
        while not good2go:
            tmp_lineage = LineageTree.init_from_parameters(pi, T, E, max_desired_num_cells)
            good2go = lineage_good_to_analyze(tmp_lineage)

        population.append(tmp_lineage)

        # Adding populations into a holder for analysing
        list_of_populations.append(population)
        list_of_fpi.append(pi)
        list_of_fT.append(T)
        list_of_fE.append(E)

    wass, _, accuracy_after_switching, _, _, paramTrues = commonAnalyze(list_of_populations, xtype="wass", list_of_fpi=list_of_fpi)

    distribution_df = pd.DataFrame(columns=["Distribution type", "G2 Lifetime", "State"])
    lineages = [list_of_populations[int(num_data_points * i / 4.)][0] for i in range(4)]
    len_lineages = [len(lineage) for lineage in lineages]
    distribution_df["G1 lifetime"] = [cell.obs[2] for lineage in lineages for cell in lineage.output_lineage]
    distribution_df["State"] = ["State 1" if cell.state == 0 else "State 2" for lineage in lineages for cell in lineage.output_lineage]
    distribution_df["Distribution type"] = len_lineages[0] * ["Same"] +\
        len_lineages[1] * ["Similar"] +\
        len_lineages[2] * ["Different"] +\
        len_lineages[3] * ["Distinct"]

    # for the violin plot (distributions)
    wasser_df = pd.DataFrame(columns=["Wasserstein distance", "Approximate Wasserstein distance", "State Assignment Accuracy"])
    wasser_df["Wasserstein distance"] = wass
    wasser_df["State Assignment Accuracy"] = accuracy_after_switching
    maxx = np.max(wass)
    wass_columns = [int(maxx * (2 * i + 1) / 2 / number_of_columns) for i in range(number_of_columns)]
    assert len(wass_columns) == number_of_columns
    for indx, num in enumerate(wass):
        wasser_df.loc[indx, 'Approximate Wasserstein distance'] = return_closest(num, wass_columns)

    return distribution_df, wasser_df


def figureMaker4(ax, distribution_df, wasser_df):
    """
    This makes figure 4.
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
    sns.lineplot(x="Approximate Wasserstein distance", y="State Assignment Accuracy", data=wasser_df, ax=ax[i])
    ax[i].set_title("State Assignment Accuracy")
    ax[i].set_ylabel("Accuracy [%]")
    ax[i].set_ylim(bottom=50.0, top=101)
