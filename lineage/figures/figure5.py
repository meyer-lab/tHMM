""" This file is to show the model works in case we have rare phenotypes. """
import numpy as np
import pandas as pd
import seaborn as sns
from .figureCommon import (
    getSetup,
    subplotLabel,
    commonAnalyze,
    pi,
    E2,
    lineage_good_to_analyze,
    max_desired_num_cells,
    num_data_points
)
from ..LineageTree import LineageTree
from .figureS53 import return_closest


def makeFigure():
    """
    Makes fig 5.
    """

    # Get list of axis objects
    ax, f = getSetup((5, 5), (2, 1)) # each figure will take twice its normal size horizontally
    number_of_columns = 25
    figureMaker5(ax, accuracy(number_of_columns))

    subplotLabel(ax)

    return f


def accuracy(number_of_columns):
    """
    Calculates accuracy and parameter estimation
    over an similar number of cells in a lineage for
    a uncensored two-state model but differing state distribution.
    We increase the proportion of cells in a lineage by
    fixing the Transition matrix to be biased towards state 0.
    """

    # Creating a list of populations to analyze over
    list_of_Ts = [np.array([[i, 1.0 - i], [i, 1.0 - i]]) for i in np.linspace(0.1, 0.9, num_data_points)]
    list_of_populations = []
    list_of_fpi = []
    list_of_fT = []
    list_of_fE = []
    for T in list_of_Ts:
        population = []

        good2go = False
        while not good2go:
            tmp_lineage = LineageTree.init_from_parameters(pi, T, E2, max_desired_num_cells)
            good2go = lineage_good_to_analyze(tmp_lineage)

        population.append(tmp_lineage)

        # Adding populations into a holder for analysing
        list_of_populations.append(population)
        list_of_fpi.append(pi)
        list_of_fT.append(T)
        list_of_fE.append(E2)

    percentageS1, _, acc, _, _, _ = commonAnalyze(list_of_populations, xtype="prop", list_of_fpi=list_of_fpi)

    accuracy_df = pd.DataFrame(columns=["Proportions", "Approximate proportions", "State Assignment Accuracy"])

    accuracy_df["Proportions"] = percentageS1
    accuracy_df["State Assignment Accuracy"] = acc

    maxx = np.max(percentageS1)
    prop_columns = [int(maxx * (2 * i + 1) / 2 / number_of_columns) for i in range(number_of_columns)]
    assert len(prop_columns) == number_of_columns
    for indx, num in enumerate(percentageS1):
        accuracy_df.loc[indx, 'Approximate proportions'] = return_closest(num, prop_columns)

    return accuracy_df


def figureMaker5(ax, accuracy_df):
    """
    This makes figure 5.
    """
    # cartoon to show different shapes --> similar shapes
    i = 0
    ax[i].axis('off')

    i += 1
    # state assignment accuracy
    sns.lineplot(x="Approximate proportions", y="State Assignment Accuracy", data=accuracy_df, ax=ax[i])
    ax[i].set_title("Accuracy relative to presence of state")
    ax[i].set_ylabel("Accuracy [%]")
    ax[i].set_xlabel("Approximate percentage of cells in state 1 [%]")
    ax[i].set_ylim(bottom=50.0, top=105.0)
