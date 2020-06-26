""" This file is to show the model works in case we have rare phenotypes. """
import numpy as np
import pandas as pd
import scipy.stats as sp

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
from ..states.StateDistPhase import StateDistribution

def makeFigure():
    """
    Makes fig 4.
    """

    # Get list of axis objects
    ax, f = getSetup((3.0, 5.0), (2, 1))

    figureMaker2(ax, *accuracy())

    subplotLabel(ax)

    return f


def accuracy():
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
            tmp_lineage = LineageTree(pi, T, E2, max_desired_num_cells)
            good2go = lineage_good_to_analyze(tmp_lineage)

        population.append(tmp_lineage)

        # Adding populations into a holder for analysing
        list_of_populations.append(population)
        list_of_fpi.append(pi)
        list_of_fT.append(T)
        list_of_fE.append(E2)

    percentageS1, _, _, _, _, _, accuracy = commonAnalyze(list_of_populations, xtype="prop", list_of_fpi=list_of_fpi)

    dataframe = pd.DataFrame(columns=['% in S1', 'state acc.'])
    maxx = len(percentageS1)
    newperc = np.zeros(len(percentageS1))
    for indx, _ in enumerate(percentageS1):
        if 0 <= indx <= maxx / 4:
            newperc[indx] = np.round(np.mean(percentageS1[0:int(maxx / 4)]), 2)
        elif maxx / 4 < indx <= maxx / 2:
            newperc[indx] = np.round(np.mean(percentageS1[int(maxx / 4):int(maxx / 2)]), 2)
        elif maxx / 2 < indx <= maxx * 3 / 4:
            newperc[indx] = np.round(np.mean(percentageS1[int(maxx / 2):int(maxx * 3 / 4)]), 2)
        elif indx >= maxx * 3 / 4:
            newperc[indx] = np.round(np.mean(percentageS1[int(maxx * 3 / 4):int(maxx)]), 2)
    dataframe['state acc.'] = accuracy
    dataframe['% in S1'] = newperc
    return dataframe, accuracy

def figureMaker2(ax, dataframe, accuracy):
    """
    This makes figure 4.
    """
    # cartoon to show different shapes --> similar shapes
    i = 0
    ax[i].axis('off')
    i += 1
    # state assignment accuracy
    sns.boxplot(x="% in S1", y="state acc.", data=dataframe, ax=ax[i], palette="deep")
    ax[i].set_title("state assignemnt accuracy")
    ax[i].set_ylabel("accuracy (%)")
    ax[i].grid(linestyle="--")
    ax[i].set_ylim(bottom=10.0, top=105.0)
    ax[i].tick_params(axis="both", which="major", grid_alpha=0.25)
