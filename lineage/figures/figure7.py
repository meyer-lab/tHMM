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
    max_desired_num_cells,
    num_data_points,
    scatter_kws_list,
)
from ..LineageTree import LineageTree
from ..states.StateDistributionGaPhs import StateDistribution



def makeFigure():
    """
    Makes fig 7.
    """

    # Get list of axis objects
    ax, f = getSetup((6.9, 5.5), (2, 2))  # each figure will take twice its normal size horizontally
    figureMaker6(ax, *accuracy())

    subplotLabel(ax)

    return f
Em2 = [StateDistribution(E2[1].params[0], E2[1].params[1], E2[1].params[2], 12, E2[1].params[4], E2[1].params[5]), E2[1]]

def accuracy():
    """
    Calculates accuracy and parameter estimation
    over an similar number of cells in a lineage for
    a uncensored two-state model but differing state distribution.
    We increase the proportion of cells in a lineage by
    fixing the Transitions matrix to be biased towards state 0.
    """

    # Creating a list of populations to analyze over
    list_of_Ts = [np.array([[i, 1.0 - i], [i, 1.0 - i]]) for i in np.linspace(0.1, 0.9, num_data_points)]
    list_of_fpi = [pi] * len(list_of_Ts)

    def genF(x): return LineageTree.init_from_parameters(pi, x, Em2, max_desired_num_cells)
    def genC(x): return LineageTree.init_from_parameters(pi, x, Em2, max_desired_num_cells, censor_condition=3, desired_experiment_time=500)

    # Generate populations
    list_of_uncen_populations = [[genF(T) for _ in range(3)] for T in list_of_Ts]
    list_of_populations = [[genC(T) for _ in range(3)] for T in list_of_Ts]

    percentageS1un, _, acc_un, _, _, _ = commonAnalyze(list_of_uncen_populations, 2, xtype="prop", list_of_fpi=list_of_fpi)
    percentageS1, _, acc, _, _, _ = commonAnalyze(list_of_populations, 2, xtype="prop", list_of_fpi=list_of_fpi)

    un_accuracy_df = pd.DataFrame(columns=["Proportions", "State Assignment Accuracy"])
    un_accuracy_df["Proportions"] = percentageS1un
    un_accuracy_df["State Assignment Accuracy"] = acc_un

    accuracy_df = pd.DataFrame(columns=["Proportions", "State Assignment Accuracy"])
    accuracy_df["Proportions"] = percentageS1
    accuracy_df["State Assignment Accuracy"] = acc

    return un_accuracy_df, accuracy_df


def figureMaker6(ax, un_accuracy_df, accuracy_df):
    """
    This makes figure 7.
    """
    # cartoon to show different shapes --> similar shapes
    i = 0
    ax[i].axis('off')

    i += 1
    ax[i].axis('off')

    i += 1
    # state assignment accuracy
    sns.regplot(x="Proportions", y="State Assignment Accuracy", data=un_accuracy_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[0])
    ax[i].set_title("Uncensored Data")
    ax[i].set_ylabel("Accuracy [%]")
    ax[i].set_xlabel("Approximate percentage of cells in state 1 [%]")
    ax[i].set_ylim(bottom=50.0, top=105.0)

    i += 1
    # state assignment accuracy
    sns.regplot(x="Proportions", y="State Assignment Accuracy", data=accuracy_df, ax=ax[i], lowess=True, marker='+', scatter_kws=scatter_kws_list[0])
    ax[i].set_title("Censored Data")
    ax[i].set_ylabel("Accuracy [%]")
    ax[i].set_xlabel("Approximate percentage of cells in state 1 [%]")
    ax[i].set_ylim(bottom=50.0, top=105.0)
