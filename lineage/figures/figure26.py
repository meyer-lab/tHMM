""" This file contains figures related to how big the experment needs to be. """
import itertools
import random
import numpy as np
import pandas as pd
import scipy.stats as sp
import seaborn as sns
from scipy.stats import wasserstein_distance
from .figureCommon import (
    getSetup,
    subplotLabel,
    pi,
    E2,
    T,
    max_desired_num_cells,
    min_num_lineages,
    max_num_lineages,
    lineage_good_to_analyze,
    num_data_points
)
from ..LineageTree import LineageTree
from ..states.StateDistPhase import StateDistribution
from ..Analyze import run_Analyze_over, run_Results_over


def makeFigure():
    """
    Makes fig 3b.
    """

    # Get list of axis objects
    ax, f = getSetup((4.0, 10.0), (3, 1))

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
    list_of_Es = [[StateDistribution(0.99, 0.8, 12, a, 10, 5), StateDistribution(0.99, 0.75, 12, 1, 9, 4)] for a in np.linspace(1, 10, num_data_points)]
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

    output = run_Analyze_over(list_of_populations, 2, list_of_fpi=list_of_fpi, list_of_fT=list_of_fT, list_of_fE=list_of_fE)

    # Collecting the results of analyzing the lineages
    results_holder, all_states = run_Results_over(output)

    dictOut = {}

    for key in results_holder[0].keys():
        dictOut[key] = []

    for results_dict in results_holder:
        for key, val in results_dict.items():
            dictOut[key].append(val)

    paramTrues = np.array(dictOut["param_trues"])
    obs_by_state_rand_sampled = []
    for state in range(output[0][0].num_states):
        full_list = [cell.obs[3] for cell in output[0][0].X[0].output_lineage if cell.state == state]
        obs_by_state_rand_sampled.append(full_list)

    num2use = min(len(obs_by_state_rand_sampled[0]), len(obs_by_state_rand_sampled[1]))
    if num2use == 0:
        results_dict["wasserstein"] = float("inf")
    else:
        results_dict["wasserstein"] = wasserstein_distance(
            random.sample(obs_by_state_rand_sampled[0], num2use), random.sample(obs_by_state_rand_sampled[1], num2use)
        )

    accuracy = dictOut["accuracy_after_switching"]
    wass = dictOut["wasserstein"]

    total = []
    for i in range(4):
        tmp1 = list(sp.gamma.rvs(a=paramTrues[i, 0, 3], loc=0.0,
                              scale=paramTrues[i, 0, 5],
                              size=200))
        total.append(tmp1)
        tmp2 = list(sp.gamma.rvs(a=paramTrues[i, 1, 3], loc=0.0,
                              scale=paramTrues[i, 1, 5],
                              size=200))
        total.append(tmp2)

    violinDF = pd.DataFrame(columns=['G2 lifetime', 'state', 'distributions'])
    violinDF['G2 lifetime'] = list(itertools.chain.from_iterable(total))
    violinDF['state'] = 200 * [1] + 200 * [2] + 200 * [1] + 200 * [2] + 200 * [1] + 200 * [2] + 200 * [1] + 200 * [2]
    violinDF['distributions'] = 400 * ['very different'] + 400 * ['different'] + 400 * ['similar'] + 400 * ['very similar']

    dataframe = pd.DataFrame(columns=['Wasserestein distance', 'state acc.'])
    dataframe['state acc.'] = accuracy

    maxx = len(dataframe['Wasserestein distance'])
    for i in range(4):
        dataframe['Wasserestein distance'][i] = np.mean(dataframe['Wasserestein distance'][(i-1)*maxx/4:int((i+1)*maxx/4)])

    return dataframe, violinDF


def figureMaker2(ax, dataframe, violinDF):
    """
    This makes figure 3B.
    """
    # cartoon to show different shapes --> similar shapes
    i = 0
    i += 1
    sns.violinplot(x="distributions", y="G2 lifetime",
                   palette="muted", split=True, hue="state",
                   data=violinDF, ax=ax[i])
    sns.despine(left=True, ax=ax[i])
    i += 1
    # state accuracy
    sns.boxplot(x="Wasserestein distance", y="state acc.", data=dataframe, ax=ax[i], palette="deep")
    ax[i].set_ylabel("accuracy")
    ax[i].set_title("state assignemnt accuracy")
    ax[i].set_ylabel("accuracy (%)")
    ax[i].grid(linestyle="--")
    ax[i].set_ylim(bottom=10.0, top=105.0)
    ax[i].tick_params(axis="both", which="major", grid_alpha=0.25)
