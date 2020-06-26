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
    E2,
    T,
    max_desired_num_cells,
    lineage_good_to_analyze
)
from ..LineageTree import LineageTree
from ..states.StateDistPhase import StateDistribution


def makeFigure():
    """
    Makes fig 3B.
    """

    # Get list of axis objects
    ax, f = getSetup((4.0, 7.5), (3, 1))

    figureMaker2(ax, *accuracy())

    subplotLabel(ax)

    return f


def accuracy():
    """
    Calculates accuracy and parameter estimation
    over an increasing number of cells in a lineage for
    a uncensored two-state model but differing state distribution.
    We vary the distribution by
    increasing the Wasserstein divergence between the two states.
    """

    # Creating a list of populations to analyze over
    list_of_Es = [[StateDistribution(0.99, 0.8, 12, a, 10, 5), StateDistribution(0.99, 0.75, 12, 1, 9, 4)] for a in np.linspace(1, 10, 40)]
    list_of_populations = []
    list_of_fpi = []
    list_of_fT = []
    list_of_fE = []
    for E in list_of_Es:
        population = []

        good2go = False
        while not good2go:
            tmp_lineage = LineageTree(pi, T, E, max_desired_num_cells)
            good2go = lineage_good_to_analyze(tmp_lineage)

        population.append(tmp_lineage)

        # Adding populations into a holder for analysing
        list_of_populations.append(population)
        list_of_fpi.append(pi)
        list_of_fT.append(T)
        list_of_fE.append(E)

    wass, _, Accuracy, _, _, paramTrues = commonAnalyze(list_of_populations, xtype="wass")
    total = []
    for i in range(4):
        tmp1 = list(sp.gamma.rvs(a=paramTrues[i, 0, 3], loc=0.0,
                                 scale=paramTrues[i, 0, 5], size=200))
        total.append(tmp1)
        tmp2 = list(sp.gamma.rvs(a=paramTrues[i, 1, 3], loc=0.0,
                                 scale=paramTrues[i, 1, 5], size=200))
        total.append(tmp2)

    violinDF = pd.DataFrame(columns=['G2 lifetime', 'state', 'distributions'])
    violinDF['G2 lifetime'] = list(itertools.chain.from_iterable(total))
    violinDF['state'] = 200 * [1] + 200 * [2] + 200 * [1] + 200 * [2] + 200 * [1] + 200 * [2] + 200 * [1] + 200 * [2]
    violinDF['distributions'] = 400 * ['very similar'] + 400 * ['similar'] + 400 * ['different'] + 400 * ['very different']

    dataframe = pd.DataFrame(columns=['Wasserestein distance', 'state acc.'])
    maxx = len(wass)
    newwass = np.zeros(len(wass))
    for indx, _ in enumerate(wass):
        if 0 <= indx <= maxx / 4:
            newwass[indx] = np.round(np.mean(wass[0:int(maxx / 4)]), 2)
        elif maxx / 4 < indx <= maxx / 2:
            newwass[indx] = np.round(np.mean(wass[int(maxx / 4):int(maxx / 2)]), 2)
        elif maxx / 2 < indx <= maxx * 3 / 4:
            newwass[indx] = np.round(np.mean(wass[int(maxx / 2):int(maxx * 3 / 4)]), 2)
        elif indx >= maxx * 3 / 4:
            newwass[indx] = np.round(np.mean(wass[int(maxx * 3 / 4):int(maxx)]), 2)

    dataframe['state acc.'] = Accuracy
    dataframe['Wasserestein distance'] = newwass

    return dataframe, violinDF


def figureMaker2(ax, dataframe, violinDF):
    """
    This makes figure 3B.
    """
    # cartoon to show different shapes --> similar shapes
    i = 0
    ax[i].axis('off')
    i += 1
    sns.violinplot(x="distributions", y="G2 lifetime",
                   palette="muted", split=True, hue="state",
                   data=violinDF, ax=ax[i])
    sns.despine(left=True, ax=ax[i])
    i += 1
    # state accuracy
    sns.boxplot(x="Wasserestein distance", y="state acc.", data=dataframe, ax=ax[i], palette="deep")
    ax[i].set_title("state assignemnt accuracy")
    ax[i].set_ylabel("accuracy (%)")
    ax[i].grid(linestyle="--")
    ax[i].set_ylim(bottom=10.0, top=105.0)
    ax[i].tick_params(axis="both", which="major", grid_alpha=0.25)
